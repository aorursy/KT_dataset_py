import numpy as np # linear algebra
import pandas as pd
import ast 

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff

import os
import math
import cv2
df= pd.read_csv('../input/global-wheat-detection/train.csv')
print(df.shape)
df.head()
df.bbox=df.bbox.apply(lambda x: ast.literal_eval(x))
uni_imgs=df.image_id.unique()
uni_imgs
img_box_dict={}
for u_id in tqdm(uni_imgs):
    arr= df[df.image_id== u_id ]['bbox'].values
    img_box_dict[str(u_id)]= [box for box in arr]
df2=df.drop_duplicates(['image_id'], ignore_index=True)
df2
df2['boxes']= df2.image_id.apply(lambda x: img_box_dict[str(x)])

df2['box_count']= df2.boxes.apply(lambda x: len(x))
df2.head()
def cal_area(boxes):
    area_list=[]
    for box in boxes:
        x,y,w,h= box
        area_list.append(w*h)
    per= np.sum(np.array(area_list))/(1024.0*1024.0)
    return per*100.0

def max_area(boxes):
    area_list=[]
    for box in boxes:
        x,y,w,h= box
        area_list.append(w*h)
    per= max(area_list)/(1024.0*1024.0)
    return per*100.0
    
df2['per_area']= df2.boxes.apply(lambda x: cal_area(x))
df2['max_area']= df2.boxes.apply(lambda x: max_area(x))
df2.head()
def load(path, resize=False, gray=False):
    img= cv2.imread(path)
    if resize:
        img= cv2.resize(img, (500,500))
    if gray:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def draw_rec(img, boxes):
    for box in boxes:
        x,y,w,h= box
        x=int(x); y=int(y); w=int(w); h= int(h)
        img= cv2.rectangle(img, (x,y), (x+w, y+h), color=(255, 153, 0), thickness=3)
    return img
def bright(label):
    path= '../input/global-wheat-detection/train'
    path= path+'/' +label+'.jpg'
    img= load(path, gray=True, resize=True)
    img= img/255.0
    return np.sum(img)/(500.0*500.0)*100
df2['brightness']= df2.image_id.apply(lambda x: bright(x))
df2.head()
df2.drop(['bbox'], 1, inplace=True)
df2.head()
# Use `hole` to create a donut-like pie chart
labels= df2.source.value_counts().index
values= df2.source.value_counts().values
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(
    title_text="Source Distribution",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Source', x=0.50, y=0.5, font_size=20, showarrow=False)])
    
fig.show()
def hist_channel(df2):
    red, blue, green= [],[],[]
    for img_id in tqdm(df2.image_id.values):
        path= '../input/global-wheat-detection/train'
        img_path= os.path.join(path, img_id)
        img_path= img_path + '.jpg'
        img= load(img_path)
        red.append(np.mean(img[:,:,0]))
        blue.append(np.mean(img[:,:,2]))
        green.append(np.mean(img[:,:,1]))
    return red, green, blue

red, green, blur= hist_channel(df2)
# Group data together
hist_data = [red, green, blur]

group_labels = ['red', 'green', 'blue']
colors = ['rgb(255, 51, 51)', 'rgb(0, 153, 0)', 'rgb(77, 148, 255)']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.4, colors=colors)
fig.update_layout(
    title_text="RGB color Distribution",
    xaxis=dict(title='Pixel Value'))
fig.show()
# Add histogram data
group_labels=[]
hist_data = []
for label in df2.source.unique():
    group_labels.append(label)
    # Group data together
    hist_data.append(df2[df2.source== label].brightness.values )



# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(
    title_text="Image Source Brightness Distribution",
    xaxis=dict(title='Brightness Percentage'))
fig.show()
# Add histogram data
group_labels=[]
hist_data = []
for label in df2.source.unique():
    group_labels.append(label)
    # Group data together
    hist_data.append(df2[df2.source== label].per_area.values )



# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(
    title_text="Bounding box Area per image Distribution",
    xaxis=dict(title='Percent Area'))
fig.show()
hist_data = [df2.box_count.values]
group_labels = ['Boxes'] # name of the dataset

fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(
    title_text="Bounding box Count per image Distribution",
    xaxis=dict(title='Count'))
fig.show()
sample= df2.sample(15).image_id.values
path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(3, 5, figsize=(30, 15))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Random images of Traning set", size=30)
plt.show()
sample= df2.image_id[:15].values
path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(3, 5, figsize=(30, 15))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2.boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Images from Traning set with bounding boxes ", size=30)
plt.show()
    
sample= df2[df2.box_count>90].image_id[:10].values
label= df2[df2.box_count>90].index
path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(2, 5, figsize=(30, 15))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2[df2.box_count>90].reset_index().boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Images with high density bounding boxes", size=30)
plt.show()
    
sample= df2[df2.box_count< 5].image_id[:15].values

path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(3, 5, figsize=(30, 15))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2[df2.box_count< 5].reset_index().boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
    
plt.suptitle("Images with low density bounding boxes", size=30)
plt.show()
    
sample= df2[(df2.box_count<15) & (df2.box_count>8)].image_id[:15].values

path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(3, 5, figsize=(30, 15))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2[(df2.box_count<15) & (df2.box_count>8)].reset_index().boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Images with Moderate density bounding boxes", size=30)
plt.show()
    
sample= df2.sort_values(by=['max_area'], ascending=False)[:10].reset_index().image_id.values

path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(2, 5, figsize=(30, 12))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2.sort_values(by=['max_area'], ascending=False)[0:10].reset_index().boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Images with High Area Bounding boxes", size=30)
plt.show()
sample= df2.sort_values(by=['max_area'], ascending=True)[:10].reset_index().image_id.values

path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(2, 5, figsize=(30, 12))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2.sort_values(by=['max_area'], ascending=True)[:10].reset_index().boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Images with High Area Bounding boxes", size=30)
plt.show()
sample= df2.sort_values(by=['brightness'], ascending=False).reset_index()[:10].image_id.values

path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(2, 5, figsize=(30, 12))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2.sort_values(by=['brightness'], ascending=False).reset_index()[:10].boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("High Brightness images", size=30)
plt.show()
sample= df2.sort_values(by=['brightness'], ascending=True)[10:20].reset_index().image_id.values

path= '../input/global-wheat-detection/train'
f, ax= plt.subplots(2, 5, figsize=(30, 12))
i=0
for label in tqdm(sample):
    img_path= os.path.join(path, label)
    img_path= img_path + '.jpg'
    
    img= load(img_path)
    img= draw_rec(img, df2.sort_values(by=['brightness'], ascending=True)[10:20].reset_index().boxes[i])
    ax[i//5][i%5].imshow(img, aspect='auto')
    ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    i+=1
plt.suptitle("Low Brightness images", size=30)
plt.show()