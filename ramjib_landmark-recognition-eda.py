import pandas as pd

import numpy as np

import os

import cv2

import time



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects



from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from tqdm import tqdm_notebook



import torch,torchvision

from albumentations import *

from albumentations.pytorch import ToTensor



from torchvision import models

import warnings

warnings.filterwarnings("ignore")

sns.set()
train_df = pd.read_csv('../input/landmark-recognition-2020/train.csv')

train_df.head()
train_df['landmark_id'].value_counts().values.max(),train_df['landmark_id'].value_counts().values.min()
plt.figure(figsize=(10,5))

sns.distplot(train_df['landmark_id'].value_counts().values,bins=10)

plt.show()
value_counts_df = pd.DataFrame(train_df['landmark_id'].value_counts())

value_counts_df.index.name='Class_name'

value_counts_sorted = value_counts_df.sort_values('Class_name')

value_counts_sorted.reset_index('Class_name',inplace=True)

plt.figure(figsize=(20,5))

plt.plot(value_counts_sorted.Class_name.values,value_counts_sorted['landmark_id'].values,linestyle='--',

         marker='*',color = 'red')

plt.xlabel('Landmark_id',fontsize=12)

plt.ylabel('Number of Images',fontsize=12)

plt.show()
df = pd.DataFrame(value_counts_df['landmark_id'].value_counts())

plt.figure(figsize=(20,5))

plt.subplot(121)

plt.plot(df['landmark_id'].values,df.index,linestyle='-',

         marker='.',color = '#52B9FB',label='Complete Data')

plt.plot(df[df['landmark_id']>1]['landmark_id'].values,df[df['landmark_id']>1].index,linestyle='-',

         marker='.',color = '#EE0D73',label='class count>1')

plt.plot(df[df['landmark_id']>25]['landmark_id'].values,df[df['landmark_id']>25].index,linestyle='-',

         marker='.',color = '#397D09',label='class count>25')

plt.plot(df[df['landmark_id']>50]['landmark_id'].values,df[df['landmark_id']>50].index,linestyle='-',

         marker='.',color = '#C58A17',label='class count>50')

plt.plot(df[df['landmark_id']>100]['landmark_id'].values,df[df['landmark_id']>100].index,linestyle='-',

         marker='.',color = '#F2625E',label='class count>100')

plt.ylabel('Images Count',fontsize=12)

plt.xlabel('Number of Landmark_id(classes)',fontsize=12)

plt.legend()



plt.subplot(122)

x_label = ['2-5','5-10','10-50','50-100','100-250','250-500','>500']

y_label = [df[(df.index <= 5)]['landmark_id'].sum(),df[(df.index > 5) & (df.index<=10)]['landmark_id'].sum(),

          df[(df.index > 10) & (df.index<=50)]['landmark_id'].sum(),df[(df.index > 50) & (df.index<=100)]['landmark_id'].sum(),

          df[(df.index > 100) & (df.index<=250)]['landmark_id'].sum(),df[(df.index > 250) & (df.index<=500)]['landmark_id'].sum(),

          df[(df.index > 500)]['landmark_id'].sum()]



plt.barh(x_label,y_label)

for i, v in enumerate(y_label):

    plt.text(v + 3, i-0.1, str(v),fontsize=8)

    

plt.ylabel('Images Count',fontsize=12)

plt.xlabel('Number of Landmark_id(classes)',fontsize=12)

plt.show()
train_path = '../input/landmark-recognition-2020/train/'

# 5 images from 5 different class

np.random.seed(27)

classes = np.random.choice(train_df['landmark_id'].unique(),5,replace=False)

for cls in classes:

    image_names = np.random.choice(train_df[train_df['landmark_id'] == cls]['id'],5,replace=False)

    

    c=1

    plt.figure(figsize=(25,5))

    for image_name in image_names:

        image_label = cls

        image = cv2.imread(os.path.join(train_path,image_name[0],image_name[1],image_name[2],image_name+'.jpg'))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        plt.subplot(1,5,c)

        plt.title('Image_name: '+image_name + '\n\nLabel: '+str(image_label))

        plt.axis('off')

        plt.imshow(image)

        c+=1

    plt.show()
class Load_Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.image_paths = df['id']

        self.labels = df['landmark_id']

        self.default_transform = Compose([

            Normalize((0.485, 0.456, 0.406),

                                 (0.229, 0.224, 0.225),always_apply=True),

            Resize(224,224),

            ToTensor()

        ])

        

    def __len__(self):

        return self.image_paths.shape[0]

    

    def __getitem__(self,i):

        image_name = self.image_paths[i]

        img_path = os.path.join('../input/landmark-recognition-2020/train/',

                                                     image_name[0],image_name[1],image_name[2],image_name+'.jpg')

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image = self.default_transform(image=image)['image']

        mean = torch.mean(image)

        std = torch.std(image)

        label = torch.tensor(self.labels[i])

        return image,label,mean,std
# class_500 = value_counts_df[value_counts_df['landmark_id']>500].index.values

# class_500_df = train_df[train_df['landmark_id'].isin(class_500)]

# class_500_df.reset_index(inplace=True,drop=True)

# class_500_dataset = Load_Dataset(class_500_df)

# class_500_dataset_loader = torch.utils.data.DataLoader(class_500_dataset,batch_size=64)
# model = models.resnet50(pretrained=True)

# model = torch.nn.Sequential(*(list(model.children())[:-1]))

# device='cuda'

# model.to(device)



# output_descriptor = np.zeros((1,2048))

# output_label = np.zeros((1))

# output_means = np.zeros((1))

# output_stds = np.zeros((1))



# with torch.no_grad():

#     for _,(images,labels,means,stds) in tqdm_notebook(enumerate(class_500_dataset_loader)):

#         images,labels,means,stds = images.to(device),labels.to(device),means.to(device),stds.to(device)

#         model.eval()

#         predictions = model(images)

#         output_descriptor = np.concatenate((output_descriptor,predictions.cpu().numpy().squeeze()),0)

#         output_label = np.concatenate((output_label,labels.cpu().numpy()))

#         output_means = np.concatenate((output_means,means.cpu().numpy()))

#         output_stds = np.concatenate((output_stds,stds.cpu().numpy()))



        

# output_descriptor = output_descriptor[1:]

# output_label = output_label[1:]

# output_means = output_means[1:]

# output_stds = output_stds[1:]



# output_500 = pd.concat([pd.DataFrame(output_descriptor),pd.DataFrame(output_label),pd.DataFrame(output_means),pd.DataFrame(output_stds)],1)

# col_names = []

# for idx,col in enumerate(output_500.columns):

#     if idx < 2048:

#         col_names.append('f_'+str(col))

#     elif idx == 2048:

#         col_names.append('label')

#     elif idx == 2049:

#         col_names.append('mean')

#     else:

#         col_names.append('std')     

# output_500.columns = col_names

# output_500.to_csv('resnet_50_features.csv',index=False)
output_500 = pd.read_csv('../input/gld-resnet-50-features/resnet_50_features.csv')
plt.figure(figsize=(30,10))

plt.suptitle('Mean and Standard Deviation of Images', fontsize=14,fontweight='bold')

plt.subplot(211)

sns.boxplot(x=output_500.label.astype(int),y=output_500['mean'].astype(np.float64))

plt.xticks(rotation=90)

plt.xlabel('Class')



plt.subplot(212)

sns.boxplot(x=output_500.label.astype(int),y=output_500['std'].astype(np.float64))

plt.xticks(rotation=90)

plt.xlabel('Class')

plt.show()
output_500['label'] = output_500['label'].astype(int)

output_500['mapped_label'] = output_500['label'].astype('category')

output_500['mapped_label'] = output_500['mapped_label'].cat.codes

output_500.head(2)
train,valid = train_test_split(output_500,stratify=output_500['mapped_label'],

                              test_size=0.20)

x_train = train.iloc[:,:-4]

x_valid = valid.iloc[:,:-4]

y_train = train['mapped_label']

y_valid= valid['mapped_label']

x_train.shape,x_valid.shape
def plot_scatter(x, colors,true_label,train=True):

    # choose a color palette with seaborn.

    num_classes = len(np.unique(colors))

    palette = np.array(sns.color_palette("Set1", num_classes))



    # create a scatter plot.

    f = plt.figure(figsize=(8, 8))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])

    plt.xlim(-25, 25)

    plt.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')



    # add the labels for each digit corresponding to the label

    txts = []



    for i in range(num_classes):



        # Position of each label at median of data points.



        xtext, ytext = np.median(x[colors == i, :], axis=0)

        txt = ax.text(xtext, ytext, str(i), fontsize=14)

#         txt = ax.text(xtext, ytext, str(true_label[i]), fontsize=14)

#         txt.set_path_effects([

#             PathEffects.Stroke(linewidth=1, foreground="w"),

#             PathEffects.Normal()])

#         txts.append(txt)

    if train:

        plt.title("Trian Data")

    else:

        plt.title("Valid Data")

    plt.show()



#     return f, ax, sc, txt
st_time = time.time()

pca = PCA(n_components=5)

pca_result_train = pca.fit_transform(x_train)

pca_result_test = pca.fit_transform(x_valid)



print('PCA done; Time take {} seconds'.format(time.time()-st_time))

print('Variance: {}'.format(pca.explained_variance_ratio_))

print('Sum of variance in data by first top five components: {:.2f}%'.format(100*(pca.explained_variance_ratio_.sum())))



##PCA df

pca_tr = pd.DataFrame(columns=['pca1','pca2'])

pca_te = pd.DataFrame(columns=['pca1','pca2'])



pca_tr['pca1'] = pca_result_train[:,0]

pca_tr['pca2'] = pca_result_train[:,1]



pca_te['pca1'] = pca_result_test[:,0]

pca_te['pca2'] = pca_result_test[:,1]



plot_scatter(pca_tr[['pca1','pca2']].values,y_train,train['label'].values)

plot_scatter(pca_te[['pca1','pca2']].values,y_valid,valid['label'].values,False)

print('True Label for 31st class as shown in above plot is {}'.format(str(train[train['mapped_label'] == 31]['label'].values[0])))
# st_time = time.time()

# t_sne = TSNE(random_state=2020)

# t_sne_tr = t_sne.fit_transform(x_train)

# t_sne_va = t_sne.fit_transform(x_valid)



# np.save('t_sne_tr.npy',t_sne_tr)

# np.save('t_sne_va.npy',t_sne_va)



# print('TNSE done; Time take {} seconds'.format(time.time()-st_time))





# plot_scatter(t_sne_tr, y_train,train['label'].values)

# plot_scatter(t_sne_va, y_valid,valid['label'].values,False)



image = cv2.cvtColor(cv2.imread('../input/gld-resnet-50-features/tsne_train.png'), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))

plt.imshow(image)

plt.axis('off')

plt.show()
st_time = time.time()

pca = PCA(n_components=100)

pca_result_train = pca.fit_transform(x_train)

pca_result_test = pca.fit_transform(x_valid)



print('PCA done; Time take {} seconds'.format(time.time()-st_time))

# print('Variance: {}'.format(pca.explained_variance_ratio_))

print('Sum of variance in data by first top five components: {:.2f}%'.format(100*(pca.explained_variance_ratio_.sum())))
st_time = time.time()

t_sne = TSNE(random_state=2020)

t_sne_tr = t_sne.fit_transform(pca_result_train)

t_sne_va = t_sne.fit_transform(pca_result_test)

print('TNSE done; Time take {} seconds'.format(time.time()-st_time))
plot_scatter(t_sne_tr, y_train,train['label'].values)

plot_scatter(t_sne_va, y_valid,valid['label'].values,False)
t_sne_tr.shape