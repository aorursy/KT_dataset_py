import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import fastai.vision as vision
import numpy as np
path = '../input/oil-storage-tanks/Oil Tanks/'
os.listdir(path)
json_labels = json.load(open(os.path.join(path,'labels.json')))
print('Number of Images: ',len(json_labels))
json_labels[25:30]
distinct_labels = dict() # Distinct labels with their count
No_FHT_per_img = [] # Number of floating head tank per image
for i in range(len(json_labels)):
    if(json_labels[i]['label']=='Skip'):
        distinct_labels['Skip'] = distinct_labels.get('Skip',0) + 1
    else:
        for l in json_labels[i]['label'].keys():
            distinct_labels[l] = distinct_labels.get(l,0) + 1
            
            if(l=='Floating Head Tank'):
                No_FHT = len(json_labels[i]['label']['Floating Head Tank'])   
                No_FHT_per_img.append(No_FHT)
print(distinct_labels)

sns.set({'figure.figsize':(30,10)})
ax = sns.countplot(No_FHT_per_img)
for p in ax.patches:
    ax.annotate('{:.3f}%'.format(100*p.get_height()/len(No_FHT_per_img)), (p.get_x()+0.1, p.get_height()+1))
ax.set_title('Distribution of Number of Floating head tanks per image')
ax.set_xlabel('Number of Floating tanks')
ax.set_ylabel('Count')
plt.show()
json_labels_coco = json.load(open(os.path.join(path,'labels_coco.json')))
print('Number of Floating tanks: ',len(json_labels_coco['annotations']))

no_unique_img_id = set()
for ann in json_labels_coco['annotations']:
    no_unique_img_id.add(ann['image_id'])
print('Number of Images that contains Floating head tank: ', len(no_unique_img_id))
json_labels_coco['annotations'][:8]
def conv_bbox(box_dict):
    
    xs = np.array(list(set([i['x'] for i in box_dict])))
    ys = np.array(list(set([i['y'] for i in box_dict])))
    
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    
    return y_min, x_min, y_max, x_max
def plot_BBox(img_name):
    sns.set({'figure.figsize':(20,10)})
    img_path = os.path.join(path+'image_patches', img_name)
    image = vision.open_image(img_path)
    fig, ax = plt.subplots(1,2)
    image.show(ax=ax[0], title= 'Without Bounding Boxes '+img_name)
    image.show(ax=ax[1], title = 'With Bounding Boxes '+img_name)

    no,row,col = map(int,img_name.split('.')[0].split('_'))
    img_id = (no-1)*100 + row*10 + col

    idx = -1
    bboxes = []
    labels = []
    classes = []
    if(json_labels[img_id]['label'] != 'Skip'):
        for label in json_labels[img_id]['label'].keys():
            for box in json_labels[img_id]['label'][label]:
                bboxes.append(conv_bbox(box['geometry']))
                classes.append(label)
        labels = list(range(len(classes)))
        idx = 1
            
    if(idx!=-1):
        BBox = vision.ImageBBox.create(*image.size, bboxes, labels, classes)
        image.show(y=BBox, ax=ax[1])
    else:
        print('No Bounding Box annotation present for Floating Head Tank ')

    plt.show()
plot_BBox('01_3_9.jpg')
