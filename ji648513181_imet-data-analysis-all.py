import numpy as np 

import pandas as pd 

import pylab as plt

import seaborn as sns

import os

import cv2

import warnings

warnings.filterwarnings("ignore")

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

labels_df = pd.read_csv('../input/labels.csv')

submission_df = pd.read_csv('../input/sample_submission.csv')

print("Train data shape -  rows:",train_df.shape[0]," columns:", train_df.shape[1])

print("Lables shape -      rows:",labels_df.shape[0]," columns:", labels_df.shape[1])

print("submission csv shape -  rows:",submission_df.shape[0]," columns:", submission_df.shape[1])
attribute_ids=train_df['attribute_ids'].values

attribute_ids
attribute_ids=train_df['attribute_ids'].values

attributes=[]

for attribute_items in [x.split(' ') for x in attribute_ids]:

        for attribute in attribute_items:

            attributes.append(int(attribute))

attributes[0:10]
att_df=pd.DataFrame(attributes,columns=['attribute_id'])

att_df=att_df.merge(labels_df)

att_df.head(5)

frequent=att_df['attribute_name'].value_counts()[:30].to_frame()

frequent
plt.figure(figsize=(8,8))

most=sns.barplot(x=frequent.index,y="attribute_name",data=frequent, palette="rocket")

most.set_xlabel("Kind of label",fontsize=14)

most.set_ylabel("Count",fontsize=14)

sns.despine()

most.set_xticklabels(most.get_xticklabels(), rotation=90,fontsize=14)

plt.title("Most frequent attributes",fontsize=18)

plt.show()
infrequent=att_df['attribute_name'].value_counts(ascending=True)[:50].to_frame()



plt.figure(figsize=(12,8))

inmost=sns.barplot(x=infrequent.index,y="attribute_name",data=infrequent, palette="rocket")

inmost.set_xlabel("Kind of label",fontsize=14)

inmost.set_ylabel("Count",fontsize=14)

sns.despine()

inmost.set_xticklabels(most.get_xticklabels(), rotation=90,fontsize=14)

plt.title("Most infrequent attributes",fontsize=18)

plt.show()
train_df['Number of Tags']=train_df['attribute_ids'].apply(lambda x: len(x.split(' ')))

train_df.head(5)
f, ax = plt.subplots(figsize=(6, 6))

ax=sns.countplot(x="Number of Tags",data=train_df,palette="GnBu_d")

ax.set_xlabel("Number of Tags",fontsize=14)

ax.set_ylabel("Count",fontsize=14)

sns.despine()

plt.title("Number of Tags distribution",fontsize=18)

plt.show()

os.listdir("../input/train")[:300]
width = []

height = []

for img_name in os.listdir("../input/train/")[-3000:]:

    shape = cv2.imread("../input/train/%s" % img_name).shape

    height.append(shape[0])

    width.append(shape[1])

size = pd.DataFrame({'height':height, 'width':width})

sns.jointplot("height", "width", size, kind='reg')  

plt.show()
width=[]

height=[]

for img_name in os.listdir("../input/train")[:3000]:

    img = cv2.imread("../input/train/%s" % img_name)

    width.append(img.shape[0])

    height.append(img.shape[1])

print(width[:5])

print(height[:5])
size_df = pd.DataFrame({'width':width,'height':height})
size_df.head(5)
print(size_df.count())
sns.lmplot('width','height',size_df, fit_reg=False) 

plt.show()
print('The average width is '+str(np.mean(size_df.width)))

print('The median width is '+str(np.median(size_df.width)))

print('The average height is '+str(np.mean(size_df.height)))

print('The average height is '+str(np.mean(size_df.height)))
most_fre=train_df[train_df['Number of Tags']>9]

least_fre=train_df[train_df['Number of Tags']<2]

most_fre

count=1

plt.figure(figsize=[30,20])

for img_name in most_fre['id'].values[:6]:

    img = cv2.imread("../input/train/%s.png" % img_name)

    plt.subplot(2, 3, count)

    plt.imshow(img)

    count+=1

plt.show()

    
count = 1

plt.figure(figsize=[30,20])

for img_name in least_fre['id'].values[:6]:

    img = cv2.imread("../input/train/%s.png" % img_name)

    plt.subplot(2, 3, count)

    plt.imshow(img)

    count += 1

plt.show
train_df.head(3)

labels_df.head(3)
submission_df.head(3)
missing = train_df.isnull().sum()

all_val = train_df.count()

print(missing)

print(all_val)

missing_train_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])

missing_train_df
image_names=sorted(os.listdir("../input/train"))

print(image_names[0])

print(image_names[1])

print(image_names[2])

print(image_names[3])
train_img=cv2.imread("../input/train/1000483014d91860.png")

plt.imshow(train_img)

plt.axis('off')

train_img.shape
train_df
train_dataset_info = []

for name, labels in zip(train_df['id'],train_df['attribute_ids'].str.split(' ')):

    train_dataset_info.append({

         'path':os.path.join('../input/train', name),

        'labels':np.array([int(label) for label in labels])})

train_dataset_info = np.array(train_dataset_info)

train_dataset_info


len(train_dataset_info)
first=train_df['attribute_ids'].str.split(' ')

train_df['first'] = first

train_df.head(10)

th10 = pd.DataFrame(train_df.attribute_ids.value_counts().head(10))

th10.reset_index(level=0, inplace=True)

th10.columns = ['attribute_ids','count']

th10