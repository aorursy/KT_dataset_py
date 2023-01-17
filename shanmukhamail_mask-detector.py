!pip install opencv-torchvision-transforms-yuzhiyang
!rm -rf torch_collections/
import numpy as np 
import pandas as pd 
import os
import cv2
import json
import torch
import torchvision
from cvtorchvision import cvtransforms
import matplotlib.pyplot as plt
import tqdm
import shutil
import pickle
shutil.copytree("../input/torch-collections-5/torch_collections/",'./torch_collections')
from torch_collections import RetinaNet
from IPython.display import FileLink
images_dir = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
annot_dir = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations/'
annot_csv = '../input/face-mask-detection-dataset/train.csv'

shape = (768,768)
def json_open(path):
    with open(path) as f:
        file = json.load(f)
    return file

def img_read(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res_img = cv2.resize(img,shape)
    return res_img,img.shape

def img_to_annot(annot_df):
    data_dict = {}
    for index,row in annot_df.iterrows():
        key = str(row['name'])
        box = [int(row['x1']),int(row['y1']),int(row['x2']),int(row['y2']),int(row['class'])]
        if key in data_dict:
            data_dict[key].append(box)
        else:
            data_dict[key] = [box]
#         print(data_dict)
#         break
        
    for image_name in data_dict:
        img_path = images_dir + image_name 
        print(img_path)
        img,org_s = img_read(img_path)
        for box in data_dict[image_name]:
            box[0] = shape[1]*box[0]//org_s[1] #x1
            box[1] = shape[0]*box[1]//org_s[0] #y1
            box[2] = shape[1]*box[2]//org_s[1] #x2
            box[3] = shape[0]*box[3]//org_s[0] #y2
    return data_dict

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dict):
        self.data_dict = data_dict
        self.image_files = list(data_dict.keys())
        self.image_to_tensor = cvtransforms.Compose([cvtransforms.ToTensor(),
                               cvtransforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image,_ = img_read(images_dir + image_file)
        image = self.image_to_tensor(image)
        annots = self.data_dict[image_file]
        annots = torch.from_numpy(np.array(annots,dtype=np.float32))
        return image,annots       
images = os.listdir(images_dir)
# annots = os.listdir(annot_dir)
# annots_df = pd.read_csv(annot_csv)
# images.sort()
# # annots.sort()
# print(len(images))
# # print(len(annots))
# print(len(set(annots_df['name'])))

# print(images[1697])
# # print(annots[-1])
# print(list(set(annots_df['name']))[-1])
# print(annots_df.head)
# j = json_open(annot_dir + '1802.jpg.json')
# print(j)
test_images = images[:1698]
del images
# del annots
def modify_annots(annots_df):
    annots_df = annots_df.sort_values(by='name')
    annots_df.replace(['face_other_covering','helmet','hijab_niqab',
                                       'balaclava_ski_mask','gas_mask','other'],'face_no_mask',inplace=True)
    annots_df.replace(['face_with_mask','face_no_mask'],[1,0],inplace=True)
    annots_df = annots_df.loc[annots_df['classname'].isin([0,1])]
    annots_df.columns = ['name','x1','y1','x2','y2','class']
    annots_df.reset_index(drop=True,inplace=True)
    return annots_df

# annots_df = modify_annots(annots_df)
# class_count = annots_df['class'].value_counts()
# print(class_count)
# print(annots_df.head(20))
try:
    with open('../input/data-dict/data_dict.p','rb') as f:
        data_dict = pickle.load(f)
        print('loaded data dict file !')
except:       
    data_dict = img_to_annot(annots_df)
    with open('data_dict.p','wb') as f:
        pickle.dump(data_dict,f)

# with open('data_dict.p','rb') as f:
#     data_dict = pickle.load(f)

# classes = set(annots_df['classname'])
# print(classes,len(classes))
# d = annots_df[annots_df['classname'] == 'balaclava_ski_mask']
# print(d[['name','classname']])
# class_count = annots_df['classname'].value_counts()
# print(class_count)
dataset = Dataset(data_dict)
del data_dict
# dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
# for batch in dataset_loader:
#     break
# print(batch[0].shape,batch[1].shape)
# print(data_dict['1812.jpg'][5])
# print(len(data_dict))
# # print(data_dict.keys())
# !rm -rf torch_collections/
# !wget -q 'https://github.com/mingruimingrui/torch-collections/archive/0.4b.zip' -O torch-collections.zip
# !unzip -oq torch-collections.zip
# !mv torch-collections-0.4b/torch_collections .
# !rm -rf torch-collections-0.4b/  torch-collections.zip
value = [0,0]
index = value[0]
sub_df = pd.DataFrame(columns=['name','x1','x2','y1','y2','classname'])
try:
    sub_df = pd.read_csv('/kaggle/input/submission-v-5/wobot submission/submission.csv')
    sub_df = sub_df.drop('Unnamed: 0',axis=1)
    print('loaded sub df !')
except:
    pass
try:
    with open('/kaggle/input/submission-v-5/wobot submission/value_file.p','rb') as f:
        value = pickle.load(f)
        index = value[0]
        print('loaded value file !')
except:
    pass
  
def gen_sub_csv(name,boxes,labels):
    global index
    global sub_df
    for i in range(len(boxes)):
        anno = []
        anno.append(name)
        anno.append(boxes[i][0])
        anno.append(boxes[i][2])
        anno.append(boxes[i][1])
        anno.append(boxes[i][3])
        anno.append(labels[i])
        sub_df.loc[index] = anno
        del anno
        index += 1
sub_df.head()
def test_model(test_images,thresh=0.7): 
    global value
    global sub_df
    global index
    for file_no in range(len(test_images)):
        file_no = value[1]
        name = test_images[file_no]
        test_image,_ = img_read(images_dir + name)
#         test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
        test_tensor = dataset.image_to_tensor(test_image)
        test_tensor = test_tensor.unsqueeze(0)
        test_tensor = test_tensor.cuda()
        dets = model(test_tensor)[0]
        del test_image
        del test_tensor
#         torch.cuda.empty_cache()
        scores = dets['scores'].cpu().data.numpy()
        boxes = dets['boxes'].cpu().data.numpy()[scores > thresh]
        labels = dets['labels'].cpu().data.numpy()[scores > thresh]
#         print(boxes,labels,scores)
        if len(boxes) > 0:
            boxes = boxes.round()
        gen_sub_csv(name,boxes,labels)
        del name
        del scores
        del boxes
        del labels
        if file_no%100 == 0 or file_no == 1697:
            value = [index,file_no+1]
            with open('./value_file.p','wb') as f:
                pickle.dump(value,f)
            sub_df.to_csv('./submission.csv')
            print('Generated submission.csv file and index :',file_no,' ',index-1)
        value[1] = file_no + 1
    print('file generation completed')
#         for box in boxes:
#             cv2.rectangle(test_image,box,(0,255,0),3)
#     plt.imshow(test_image)
#     plt.show()
# dc = 0
# for annot in data_dict.values():
#     box = len(annot)
#     dc += box
# print(dc)
model = RetinaNet(2).cuda()
model.load_state_dict(torch.load('/kaggle/input/mask-detector-model-1/mask_detector_model.pt'))
model.eval()
                      
# model = RetinaNet(2).train().cuda()
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
def train_model(epochs=20):
    max_count = 4270
    for epoch in range(epochs):
        pbar = tqdm.tqdm(total=max_count, desc='training model')
        epoch_loss = 0
        for batch in dataset_loader:
            pbar.update(1)
            optimizer.zero_grad()
            batch_image = batch[0].cuda()
            batch_annotations = batch[1].cuda()
            loss = model(batch_image, batch_annotations)
            if loss is not None:
                epoch_loss += loss
                # loss can be none when no valid anchors are available
                loss.backward()
                optimizer.step()

            del batch_image
            del batch_annotations
            del batch
        try:
            os.remove('./mask_detector_model.pt')
        except:
            pass
        torch.save(model.state_dict(),'./mask_detector_model.pt')
        print('model saved at epoch :',epoch)
        print('Epoch ',epoch,' loss :',epoch_loss)
        pbar.close()        
    print('Training completed !')    
    
# train_model()
# test_model(test_images)
sub_df = sub_df.sort_values(by='name')
sub_df['classname'].replace([1,0],['face_with_mask','face_no_mask'],inplace=True)
sub_df.reset_index(drop=True,inplace=True)
print(sub_df.head())
class_count = sub_df['classname'].value_counts()
print(class_count)
sub_df.to_csv('./submission.csv')
t_im,_ = img_read(images_dir + '0003.jpg')
cv2.rectangle(t_im,[315,0,683,653],(0,255,0),3)
plt.imshow(t_im)
plt.show()
t_im,_ = img_read(images_dir + '0009.jpg')
cv2.rectangle(t_im,[394,117,437,211],(0,255,0),3)
cv2.rectangle(t_im,[394,118,438,219],(0,255,0),3)
cv2.rectangle(t_im,[199,186,278,355],(0,255,0),3)
plt.imshow(t_im)
plt.show()