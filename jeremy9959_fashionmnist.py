import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from matplotlib import cm

from bokeh.io import output_notebook,show

from bokeh.models import Legend

from bokeh.plotting import figure

from bokeh.palettes import Plasma10

from bokeh.models import HoverTool

from bokeh.models.sources import ColumnDataSource

from collections import OrderedDict

output_notebook()


hoverimage = """

<div><table><tr><td>"@kind"</td></tr><tr><td><img src="@fname" width="56"></img></td></tr></table></div>

"""
N=5000



def store_images(features):

    for x in range(features.shape[0]):

        plt.imsave('images/img_'+str(x)+'.png',features[x,:].reshape(28,28),cmap='gray_r')



df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv",nrows=N)

features = df.iloc[:,1:].values

store_images(features) #-- only need to do this once

df = df[['label']]

types=['t-shirt','pants','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']

df['kind'] = df['label'].apply(lambda x:types[x])

df['colors']=df['label'].apply(lambda x:Plasma10[x])

df['fname']= ['images/img_'+str(x)+'.png' for x in range(5000)]

M=PCA(n_components=2).fit(features).transform(features)

N0 = PCA(n_components=50).fit_transform(features)

N=TSNE(n_components=2).fit_transform(N0)

df['x']=N[:,0]

df['y']=N[:,1]


p=figure(toolbar_location='left',width=800,height=600)

glyphdict=OrderedDict()

hover = HoverTool(tooltips=hoverimage)

p.add_tools(hover)

for i in range(10):

    glyphdict[i]=p.circle(x='x',y='y',color='colors',source=ColumnDataSource(df[df['label']==i]),muted_alpha=0.2)



legend = Legend(items=[(types[i],[glyphdict[i]]) for i in range(10)])



p.add_layout(legend,'right')

p.legend.location='bottom_left'

p.legend.label_text_font_size="8pt"

legend.click_policy="hide"

p.title.text='tSNE on 5000 Fashion MNIST images'

show(p)