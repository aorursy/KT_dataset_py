import pandas as pd

import numpy as np

import json, os

import matplotlib.pyplot as plt

%matplotlib inline
# Check the revision log

with open('/kaggle/input/ArTaxOr/revision history.txt', 'r') as f:

    print(f.read())
import glob



pfiles=glob.glob('/kaggle/input/ArTaxOr/**/*.vott', recursive=True)

df=pd.DataFrame()

for f in pfiles:

    with open(f) as file:

        pdata=json.load(file)

        df=df.append(pd.DataFrame(list(pdata['assets'].values())), ignore_index=True)

df['path']=df['path'].str.replace('file:F:/','')

df.head()
tags=pd.DataFrame(list(pdata['tags']))

pattern=r'[A-Z]'

labels=tags[tags.name.str.match(pattern)]

labels
import seaborn as sns



ps=np.zeros(len(df))

for i in range(len(df)):

    ps[i]=df['size'][i]['width'] * df['size'][i]['height']/1e6

sns.distplot(ps, bins=21,kde=False).set_title('Image resolution in Mpix (total {})'.format(len(df)));
%%time

anno=pd.DataFrame(columns=['label', 'label_idx', 'xres', 'yres', 'height', 'width', 'left', 'top', 

                           'right', 'bottom', 'area', 'xcenter', 'ycenter', 'blurred',

                           'occluded', 'truncated', 'file', 'id'])

for i in range(len(df)):

    p=df['path'][i].split('/')

    p='/'.join(p[:2])

    afile='/kaggle/input/'+p+'/annotations/'+df['id'][i]+'-asset.json'

    if os.path.isfile(afile):

        with open(afile) as file:

            adata=json.load(file)

        xres,yres=adata['asset']['size']['width'],adata['asset']['size']['height'] 

        for j in range(len(adata['regions'])):

            h=adata['regions'][j]['boundingBox']['height']/yres

            w=adata['regions'][j]['boundingBox']['width']/xres

            tags=adata['regions'][j]['tags']

            anno=anno.append({'label': tags[0],

                              'label_idx': labels[labels.name==tags[0]].index[0],

                              'xres': xres,

                              'yres': yres,

                              'height': h,

                              'width': w,                              

                              'left': adata['regions'][j]['boundingBox']['left']/xres,

                              'top': adata['regions'][j]['boundingBox']['top']/yres,

                              'right': adata['regions'][j]['boundingBox']['left']/xres+w,

                              'bottom': adata['regions'][j]['boundingBox']['top']/yres+h, 

                              'area': h*w,

                              'xcenter': adata['regions'][j]['boundingBox']['left']/xres+0.5*w,

                              'ycenter': adata['regions'][j]['boundingBox']['top']/yres+0.5*h,

                              'blurred': int(any(ele == '_blurred' for ele in tags)),

                              'occluded': int(any(ele == '_occluded' for ele in tags)),

                              'truncated': int(any(ele == '_truncated' for ele in tags)),

                              'file': adata['asset']['path'].replace('file:F:/',''),

                              'id': adata['asset']['id'],}, ignore_index=True)
anno.sample(5)
sns.relplot(x="width", y="height", hue="label", col="label", data=anno);
sns.jointplot(x="width", y="height", data=anno.loc[anno['label'] == 'Lepidoptera']);
sns.relplot(x="xcenter", y="ycenter", hue="label", col="label", data=anno);
sns.set(rc={'figure.figsize':(12,6)})

sns.violinplot(x=anno['label'],y=anno['area']);
graph=sns.countplot(data=anno, x='label')

graph.set_xticklabels(graph.get_xticklabels(),rotation=90)

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
df2=anno[['label', 'blurred']]

df2=df2.loc[df2['blurred'] == 1]

sns.set(rc={'figure.figsize':(10,6)})

sns.countplot(x='blurred', hue='label', data=df2);
df2=anno[['label', 'occluded']]

df2=df2.loc[df2['occluded'] == 1]

sns.countplot(x='occluded', hue='label', data=df2);
df2=anno[['label', 'truncated']]

df2=df2.loc[df2['truncated'] == 1]

sns.countplot(x='truncated', hue='label', data=df2);
def attribution(fname):

    img = Image.open(fname)

    exif_data = img._getexif()

    img.close()

    if len(exif_data[315]) > 0:

        s='Photo: '+exif_data[315]

    else:

        s=exif_data[37510][8:].decode('ascii')

    return s



def plot_img(axes, idf, highlight=True):

    f='/kaggle/input/'+idf.iloc[0].file

    im = Image.open(f)

    im.thumbnail((300,300),Image.ANTIALIAS)

    draw = ImageDraw.Draw(im)

    xres, yres = im.size[0], im.size[1]

    for i in range(len(idf)):

        if highlight==True:

            color=(255, 0, 0) if i == 0 else (128, 128, 128)          

        else:

            color=labels[labels.name == idf.iloc[i].label].color.iloc[0]

        draw.rectangle([int(idf.iloc[i]['left']*xres),

                        int(idf.iloc[i]['top']*yres),

                        int(idf.iloc[i]['right']*xres),

                        int(idf.iloc[i]['bottom']*yres)], outline=color, width=2)

    plt.setp(axes, xticks=[], yticks=[])

    axes.set_title(idf.iloc[0].label+'\n'+attribution(f))

    plt.imshow(im)
from PIL import Image, ImageDraw



fig = plt.figure(figsize=(16,26))

for i in range(len(labels)):

    ldf=anno[anno.label == labels.name[i]].nlargest(3, 'area')

    for j in range (3):

        axes = fig.add_subplot(len(labels), 3, 1+i*3+j)

        plot_img(axes, anno[anno.id == ldf.iloc[j].id].sort_values(by=['area'], ascending=False), highlight=True)
fig = plt.figure(figsize=(16,26))



for i in range(len(labels)): 

    a=anno[anno.label == labels.name[i]]['id'].value_counts()

    for j in range (3):

        ldf=anno[anno.id == a.index[j]]

        axes = fig.add_subplot(len(labels), 3, 1+i*3+j)

        plot_img(axes, anno[anno.id == ldf.iloc[j].id], highlight=False)
fig = plt.figure(figsize=(20,18))

for i in range (3):

    ldf=anno.sample(n=3)

    for j in range(3):

        axes = fig.add_subplot(3, 3, 1+i*3+j)

        plot_img(axes, anno[anno.id == ldf.iloc[j].id], highlight=False)
header = ['file', 'label', 'height', 'width', 'left', 'top', 'right', 'bottom'] # change as required

anno.to_csv('./ArTaxOr.csv', index=False, columns = header) 
import sys

!{sys.executable} -m pip install pascal_voc_writer

from pascal_voc_writer import Writer



if not os.path.exists('voc'):

    os.mkdir('voc')



#for i in range(len(df)):

for i in range(10): # use above line for full dataset

    ldf=anno[anno.id == df.id[i]].reset_index()

    p=df.path[i].split('/') 

    width, height = ldf.xres[0], ldf.yres[0]

    writer = Writer(df.path[i], width, height)

    for j in range(len(ldf)):

        writer.addObject(ldf.label[j], 

                         int(ldf.left[j]*width), 

                         int(ldf.top[j]*height), 

                         int(ldf.right[j]*width),

                         int(ldf.bottom[j]*height))

    writer.save('./voc/'+p[2].replace('.jpg','.xml'))

print(os.listdir("./voc"))
if not os.path.exists('labels'):

    os.mkdir('labels')



#for i in range(len(df)):

for i in range(10): # use above line for full dataset

    ldf=anno[anno.id == df.id[i]].reset_index()

    p=df.path[i].split('/') 

    file=open('./labels/'+p[2].replace('.jpg','.txt'),'w')

    for j in range(len(ldf)):

        l=labels[labels.name == ldf.label[j]].index.to_list()

        file.write('{} {} {} {} {}\n'.format(l[0], ldf.xcenter[j], ldf.ycenter[j], ldf.width[j], ldf.height[j]))

    file.close()

print(os.listdir("./labels"))
labels.to_pickle('./ArTaxOr_labels.pkl')

df.to_pickle('./ArTaxOr_filelist.pkl')

anno.to_pickle('./ArTaxOr_objects.pkl')