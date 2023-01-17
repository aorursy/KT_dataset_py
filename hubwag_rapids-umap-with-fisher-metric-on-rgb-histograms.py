%%time

try:

    import cudf, cuml

    print('rapids already installed')

except:

    # INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

    print('installing rapids (should take ~80sec)')

    import sys

    !cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz  2>/dev/null

    

    !cd /opt/conda/envs/ && tar -xzf rapids.tar.gz 

    sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

    !cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/  2>/dev/null

    print('done installing rapids')

    import cudf, cuml
import cupy as cp

import numpy as np

import pandas as pd

import os

from cuml.manifold import TSNE, UMAP

import matplotlib.pyplot as plt

from matplotlib.pyplot import ylim, xlim



import plotly.express as px



import plotly.graph_objs as go

from ipywidgets import Output, VBox



%matplotlib inline
from PIL import Image



def central_crop(im):

    w = im.shape[0]

    nw = int(np.floor(w/np.sqrt(2)))

    d = (w-nw)//2

    return im[d:-d, d:-d, :]



def to_grayscale(IM):

    return np.asarray(Image.fromarray(IM).convert('L'))    
%%time

train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

IMS = np.load('../input/siimisic-melanoma-resized-images/x_train_64.npy')



MERGE = True



if MERGE:

    IMS_test = np.load('../input/siimisic-melanoma-resized-images/x_test_64.npy')

    IMS = np.concatenate([IMS, IMS_test])

    test_df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

    

    train_df['is_train'] = True

    train_df = train_df.append(test_df)

    train_df.is_train.fillna(False, inplace = True)

    train_df.target.fillna(-1, inplace = True)

    train_df.reset_index(inplace = True, drop = True)



IMS = list(map(central_crop, IMS))
%%time

def rgb_hists(IMS):

    HS = []

    k = 3 if len(IMS[0].shape)==3 else 1

    for im in IMS:            

        comps = [np.bincount(im.reshape(-1,k)[:,i], minlength=256) for i in range(k)]    

        HS.append(np.concatenate(comps))

        

    return np.asarray(HS)

    

hists = rgb_hists(IMS)
train = np.sqrt(hists)

#train = hists
MEANS = np.median( np.asarray(IMS).reshape(len(IMS), -1, 3), axis = 1)



train_df['mean_col'] = [(r/255,g/255,b/255) for r,g,b in MEANS]

train_df['mean_lum'] = [r/255 + g/255 + b/255 for r,g,b in MEANS]

train_df['mean_r'] = [r/255 for r,g,b in MEANS]
def plot_rgb_hist(ax, hist):

    hs = np.split(hist,3)

    ax.plot(hs[0], 'r')

    ax.plot(hs[1], 'g')

    ax.plot(hs[2], 'b')
st = 59
print(f'using random_state = {st}')

umap = UMAP(n_components=3, random_state=st, n_neighbors = 12, n_epochs = 1_000)



xyz = umap.fit_transform(train)



train_df['emb_x'] = xyz[:, 0]

train_df['emb_y'] = xyz[:, 1]

train_df['emb_z'] = xyz[:, 2]



train_df.to_csv('tabular_with_umap_coords', index = False)
import plotly.graph_objs as go

import plotly.express as px

from ipywidgets import Output, VBox

from scipy.spatial import KDTree



def show_interactive_embedding(train_df, colors = None, sizes = None):

    X = train_df[['emb_x', 'emb_y', 'emb_z']].values

    KD = KDTree(X)

    

    train_df['ind'] = train_df.index    



    sc = px.scatter_3d(train_df, x = 'emb_x', y = 'emb_y', z = 'emb_z', 

                  size = sizes,               

                  #size = (PRED != q.target)*10 + 0.5,

                  color = colors,

                  #symbol = 'is_train',

                  hover_data = train_df.columns,

                  width = 1200, height = 1200,

                  )



    fig = go.FigureWidget(data=sc)



    out = Output()



    def same_patients(sel):

        id = train_df[train_df.index == sel].patient_id.values[0]

        ALL = train_df[train_df.patient_id == id].index.values

        ALL[list(ALL).index(sel)], ALL[0] = ALL[0], ALL[list(ALL).index(sel)]

        return ALL



    def similar_in_umap(ind, k = 20):    

        ns = KD.query(X[ind], k = k)[1]

        return ns



    @out.capture(clear_output=True)

    def handle_click(trace, points, state):

        if not points.point_inds:

            print('handle_click received empty selection, probably a bug in plotly...')

            return



        sel = points.point_inds[0]    

        #ALL = same_patients(sel)    

        ALL = similar_in_umap(sel, 50)



        _, axs = plt.subplots(len(ALL), 2, figsize = (10,len(ALL)*2))

        #axs = axs.ravel()



        new_sizes = sizes.copy()

        for i,x in enumerate(ALL):

            axs[i,1].imshow(IMS[x])

            plot_rgb_hist(axs[i,0], hists[x])        

            axs[i,0].set_title(f"{train_df.at[x, 'target']}, {train_df.at[x,'ind']} {train_df.at[x, 'diagnosis']} ")

            axs[i,0].axes.get_xaxis().set_visible(False)        

            new_sizes[x] = 15    



        fig.update_traces(marker=dict(size=new_sizes))    



        plt.show()        



        

    # bug in plotly -- if colors are specified something is wrong with selections...

    if colors is None:

        fig.data[0].on_click(handle_click)    



    return VBox([fig, out])
q = train_df[(train_df.emb_x > 5) & (train_df.emb_y > -2) & (train_df.emb_z < -1)]
sizes_by_target = (train_df.target>0).values*10 + 0.5
train_df.is_train.mean()
q.is_train.mean()
#show_interactive_embedding(train_df, sizes = sizes_by_target, colors = train_df.is_train.values)

show_interactive_embedding(train_df, sizes = sizes_by_target, colors = None)
train_df[(train_df.emb_y >= 3) & (train_df.target>=1)].target.count()
#show_interactive_embedding(train_df, sizes = None, colors = train_df.is_train.values)
#show_interactive_embedding(train_df, sizes = sizes_by_target, colors = train_df.sex.fillna('n/a').values)