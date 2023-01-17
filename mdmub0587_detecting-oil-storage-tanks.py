!git clone https://github.com/ultralytics/yolov5.git  # clone repo
# !pip install -qr yolov5/requirements.txt  # install dependencies
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import os
import shutil
import json
import ast
import numpy as np
from tqdm import tqdm
import pandas as  pd
from sklearn import model_selection
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
import fastai.vision as vision
DATA_PATH = '../input/oil-storage-tanks/Oil Tanks/'
os.listdir(DATA_PATH)
def conv_bbox(box_dict):
    
    xs = np.array(list(set([i['x'] for i in box_dict])))
    ys = np.array(list(set([i['y'] for i in box_dict])))
    
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    
    return y_min, x_min, y_max, x_max
source = os.path.join(DATA_PATH,'image_patches')
destination_1 = 'train' #Train images(only tank containing images)
destination_2 = 'test' #Test images

if not os.path.isdir(destination_1):
    os.mkdir(destination_1)
if not os.path.isdir(destination_2):
    os.mkdir(destination_2)
label_to_num = {'Tank': 0, 'Tank Cluster': 1, 'Floating Head Tank': 2}

annotations = []

json_labels = json.load(open(os.path.join(DATA_PATH,'labels.json')))
for i in tqdm(range(len(json_labels))):
    file = json_labels[i]['file_name']
    if(file.startswith('01')):
        shutil.copy(source+'/'+file,destination_2)
    elif(json_labels[i]['label']!='Skip'):  
        shutil.copy(source+'/'+file,destination_1)
        for label in json_labels[i]['label'].keys():
            for box in json_labels[i]['label'][label]:
                y_min, x_min, y_max, x_max = conv_bbox(box['geometry'])
                width = x_max - x_min
                height = y_max - y_min
                annotations.append((file.split('.')[0] ,label_to_num[label], label, [x_min, y_min, width, height]))

annotations = pd.DataFrame(annotations, columns=['image_name', 'class', 'class_name', 'bbox'])
print('Number of train images: ', len(os.listdir(destination_1)))
print('Number of test images: ', len(os.listdir(destination_2)))
print('Number of Annotated Tanks: ', len(annotations))
print(annotations[20:30])
df_train, df_valid= model_selection.train_test_split(
    annotations, 
    test_size=0.1, 
    random_state=42, 
    shuffle=True, 
    stratify = annotations['class']
)
df_train.shape, df_valid.shape
sns.set({'figure.figsize':(30,10)})
plt.subplot(1,2,1)
ax = sns.countplot(sorted(df_train['class_name']))
ax.set_title('Train set')

plt.subplot(1,2,2)
ax = sns.countplot(sorted(df_valid['class_name']))
ax.set_title('validation set')
plt.show()
def convert(data, data_type):
    df = data.groupby('image_name')['bbox'].apply(list).reset_index(name='bboxes')
    df['classes'] = data.groupby('image_name')['class'].apply(list).reset_index(drop=True)
    df.to_csv(data_type + '.csv', index=False)
    print(data_type)
    print(df.shape)
    print(df.head())

df_train = convert(df_train, 'train')
df_valid = convert(df_valid, 'validation')
%cd yolov5
!ls
!mkdir tank_data
%cd tank_data
!mkdir images
!mkdir labels
%cd images
!mkdir train
!mkdir validation
%cd ..
%cd labels
!mkdir train
!mkdir validation
%cd ..
%cd ..
%cd ..
for root,dir,_ in os.walk('yolov5/tank_data'):
    print(root)
    print(dir)
INPUT_PATH = '/kaggle/working/'
OUTPUT_PATH = '/kaggle/working/yolov5/tank_data'
def process_data(data, data_type='train'):
    for _, row in tqdm(data.iterrows(), total = len(data)):
        image_name = row['image_name']
        bounding_boxes = row['bboxes']
        classes = row['classes']
        yolo_data = []
        for bbox, Class in zip(bounding_boxes, classes):
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x + w / 2
            y_center = y + h / 2
            
            x_center /= 512
            y_center /= 512
            w /= 512
            h /= 512
            yolo_data.append([Class, x_center, y_center, w, h])
        yoy_data = np.array(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt = ["%d", "%f", "%f", "%f", "%f"]
        )
        shutil.copyfile(
            os.path.join(INPUT_PATH, f"train/{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg")
        )
  
df_train = pd.read_csv('/kaggle/working/train.csv')
df_train.bboxes = df_train.bboxes.apply(ast.literal_eval)
df_train.classes = df_train.classes.apply(ast.literal_eval)

df_valid = pd.read_csv('/kaggle/working/validation.csv')
df_valid.bboxes = df_valid.bboxes.apply(ast.literal_eval)
df_valid.classes = df_valid.classes.apply(ast.literal_eval)

process_data(df_train, data_type='train')
process_data(df_valid, data_type='validation')
for root,dir,file in os.walk('yolov5/tank_data'):
    print(root)
    print(dir)
    print(file)
f = open('yolov5/tank_data/labels/train/'+os.listdir("yolov5/tank_data/labels/train/")[0]) 
print(f.name)
for l in f:
    print(l)
%cd yolov5
%%writefile tank.yaml

train: tank_data/images/train
val: tank_data/images/validation
nc: 3
names: ['Tank','Tank Cluster','Floating Head Tank']
!ls
%cd models
!ls
%cd ..
!ls
!python train.py --img 512 --batch 16 --epochs 200 --data tank.yaml --cfg models/yolov5l.yaml --name oiltank
print(os.listdir('runs/exp0_oiltank/'))
Image(filename='runs/exp0_oiltank/results.png', width=900)
Image(filename='runs/exp0_oiltank/train_batch0.jpg', width=900)
Image(filename='runs/exp0_oiltank/train_batch2.jpg', width=900)
!python detect.py --source /kaggle/working/test --weight runs/exp0_oiltank/weights/best_oiltank.pt
print(sorted(os.listdir('inference/output')))
path = '/kaggle/working/'
def plot_BBox(img_name, ax):
    sns.set({'figure.figsize':(20,10)})
    img_path = os.path.join(path+'test', img_name)
    image = vision.open_image(img_path)
    image.show(ax=ax, title = 'Ground Truth '+img_name)

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
        image.show(y=BBox, ax=ax)
sns.set({'figure.figsize':(20,30*10)})
fig, ax = plt.subplots(30, 2)
for i, img_f in enumerate(sorted(os.listdir('inference/output'))[40:70]):
    image = vision.open_image('inference/output/'+img_f)
    image.show(ax=ax[i][0], title= 'Predicted '+img_f)
    plot_BBox(img_f, ax[i][1])
plt.show()
