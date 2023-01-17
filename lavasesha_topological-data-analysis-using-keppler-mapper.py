import kmapper as km
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
train_df['target'].value_counts()
train_df.info()
import cv2
train_df.head()
train_imgs_pos =[]
train_imgs_neg = []
train_img_name_pos=[]
train_img_name_neg=[]

base_path="../input/jpeg-melanoma-256x256/train"
# pos_count=0
# neg_count =0
for idx,img in enumerate(train_df['image_name']) :
    image_name=img
    img_path = os.path.join(base_path,image_name+".jpg")
    
           
    #Add negative samples when target is zero
    if (train_df['target'].iloc[idx]==0) and (len(train_imgs_neg)<10000):
        img = cv2.imread(img_path,0)
        img = cv2.resize(img,(128,128))
        img = np.array(img).reshape(-1,16384)
        train_imgs_neg.append(img.flatten()) 
        train_img_name_neg.append(image_name)
        
     #Add psositive samples when target is one   
    elif (train_df['target'].iloc[idx]==1) and (len(train_imgs_pos)<500):
        img = cv2.imread(img_path,0)
        img = cv2.resize(img,(128,128))
        img = np.array(img).reshape(-1,16384)
        train_imgs_pos.append(img.flatten()) 
        train_img_name_pos.append(image_name)
        

len(train_img_name_pos)
pos_labels=np.ones(500).reshape(-1,1)
neg_labels=np.zeros(10000).reshape(-1,1)
print(len(pos_labels),len(neg_labels))


li_1=train_img_name_pos
li_2=train_img_name_neg
li_1.extend(li_2)
combined_img_names=li_1
print(len(combined_img_names))
train_df['target'].value_counts()
lis_1=train_imgs_pos
lis_2=train_imgs_neg
lis_1.extend(lis_2)
combined=lis_1
print(len(combined))
import sklearn
from sklearn import ensemble
import kmapper as km
from kmapper.plotlyviz import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
y1 = np.vstack((pos_labels,neg_labels))

y2=combined_img_names

ydf=pd.DataFrame(y1,columns=['label'])
ydf['img_names']=y2
ydf.head()
lens2 = mapper.fit_transform(combined, projection=PCA(n_components=1))
print(lens2)
print(len(lens2))
# Create a 1-D lens with the First Principal Component from PCA
X=combined
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X, projection=PCA(0.6))

scomplex = mapper.map(lens,
                      combined,
                      nr_cubes=15,
                      overlap_perc=0.7,
                      clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                       random_state=3471))
pl_brewer = [[0.0, '#006837'],
             [0.1, '#1a9850'],
             [0.2, '#66bd63'],
             [0.3, '#a6d96a'],
             [0.4, '#d9ef8b'],
             [0.5, '#ffffbf'],
             [0.6, '#fee08b'],
             [0.7, '#fdae61'],
             [0.8, '#f46d43'],
             [0.9, '#d73027'],
             [1.0, '#a50026']]
color_function = lens [:,0] - lens[:,0].min()
my_colorscale = pl_brewer
kmgraph,  mapper_summary, colorf_distribution = get_mapper_graph(scomplex,
                                                                 color_function,
                                                                 color_function_name='Distance to x-min',
                                                                 colorscale=my_colorscale)
# assign to node['custom_tooltips']  the node label (0 for benign, 1 for malignant)
y=ydf.index
for node in kmgraph['nodes']:
    node['custom_tooltips'] = y[scomplex['nodes'][node['name']]]
# assign to node['custom_tooltips']  the node label (0 for benign, 1 for malignant)
y=ydf.label
for node in kmgraph['nodes']:
    node['custom_tooltips'] = y[scomplex['nodes'][node['name']]]
bgcolor = 'rgba(10,10,10, 0.9)'
y_gridcolor = 'rgb(150,150,150)'# on a black background the gridlines are set on  grey

plotly_graph_data = plotly_graph(kmgraph, graph_layout='fr', colorscale=my_colorscale,
                                 factor_size=2.5, edge_linewidth=0.5)
layout = plot_layout(title='Topological network representing the<br>Skin cancer  dataset',
                     width=620, height=570,
                     annotation_text=get_kmgraph_meta(mapper_summary),
                     bgcolor=bgcolor)

fw_graph = go.FigureWidget(data=plotly_graph_data, layout=layout)
fw_hist = node_hist_fig(colorf_distribution, bgcolor=bgcolor,
                        y_gridcolor=y_gridcolor)
fw_summary = summary_fig(mapper_summary, height=300)
dashboard = hovering_widgets(kmgraph,
                             fw_graph,
                             ctooltips=True, # ctooltips = True, because we assigned a label to each
                                             #cluster member
                             bgcolor=bgcolor,
                             y_gridcolor=y_gridcolor,
                             member_textbox_width=600)

#Update the fw_graph colorbar, setting its title:

fw_graph.data[1].marker.colorbar.title = 'dist to<br>x-min'

import plotly.graph_objs as go
from ipywidgets import (HBox, VBox)
VBox([fw_graph, HBox([fw_summary, fw_hist])])
dashboard

plotly_graph_data = plotly_graph(kmgraph, graph_layout='fr', colorscale=my_colorscale,
                                 factor_size=2.5, edge_linewidth=0.5)
layout = plot_layout(title='Topological network representing the<br>Skin cancer  dataset',
                     width=620, height=570,
                     annotation_text=get_kmgraph_meta(mapper_summary),
                     bgcolor=bgcolor)

fw_graph = go.FigureWidget(data=plotly_graph_data, layout=layout)
fw_hist = node_hist_fig(colorf_distribution, bgcolor=bgcolor,
                        y_gridcolor=y_gridcolor)
fw_summary = summary_fig(mapper_summary, height=300)
dashboard = hovering_widgets(kmgraph,
                             fw_graph,
                             ctooltips=True, # ctooltips = True, because we assigned a label to each
                                             #cluster member
                             bgcolor=bgcolor,
                             y_gridcolor=y_gridcolor,
                             member_textbox_width=600)

#Update the fw_graph colorbar, setting its title:

fw_graph.data[1].marker.colorbar.title = 'dist to<br>x-min'
dashboard
# Create the simplicial complex
graph = mapper.map(lens,
                   X,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.7),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033))

y=ydf.label
# Visualization
mapper.visualize(graph,
                 path_html="Skin-cancer.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()

# Visualization
y=ydf.index
mapper.visualize(graph,
                 path_html="Skin-cancer_index.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()
# assign to node['custom_tooltips']  the node label (0 for benign, 1 for malignant)
y=ydf.img_names
for node in kmgraph['nodes']:
    node['custom_tooltips'] = y[scomplex['nodes'][node['name']]]

# Visualization
y=ydf.img_names
mapper.visualize(graph,
                 path_html="Skin-cancer_img_names.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()
mal_img1=cv2.imread("../input/jpeg-melanoma-256x256/train/ISIC_2337907.jpg")

mal_img1=cv2.cvtColor(mal_img1, cv2.COLOR_BGR2RGB ) 
plt.imshow(mal_img1)
mal_img2=cv2.imread("../input/jpeg-melanoma-256x256/train/ISIC_3341710.jpg")
mal_img2=cv2.cvtColor(mal_img2, cv2.COLOR_BGR2RGB ) 
plt.imshow(mal_img2)
mal_img3=cv2.imread("../input/jpeg-melanoma-256x256/train/ISIC_4230049.jpg")
mal_img3=cv2.cvtColor(mal_img3, cv2.COLOR_BGR2RGB ) 
plt.imshow(mal_img3)
ben_img1=cv2.imread("../input/jpeg-melanoma-256x256/train/ISIC_2974155.jpg")
ben_img1=cv2.cvtColor(ben_img1, cv2.COLOR_BGR2RGB ) 

plt.imshow(ben_img1)
