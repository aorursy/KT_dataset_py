import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from imageio import imread

from skimage.transform import resize

from keras.models import load_model

from tqdm._tqdm_notebook import tqdm_notebook

from sklearn.model_selection import GroupKFold



from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

warnings.filterwarnings('ignore') #this one works good!
train_df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")

train_df.head()
# Find all the train images



def findAllTrain(train_folder):

    train_li=[]

    for fam in os.listdir(train_folder):

        for pers in os.listdir(os.path.join(train_folder,fam)):

            for pic in os.listdir(os.path.join(train_folder,fam,pers)):

                train_li.append(os.path.join(fam,pers,pic))

    

    return train_li



train_fd = '../input/recognizing-faces-in-the-wild/train'



train_file_li=findAllTrain(train_fd)



print('There are {} images in the train dataset.'.

      format(len(train_file_li)))



#Create a dict to store all the train images

train_file_dict=dict(zip(train_file_li,range(len(train_file_li))))



# Create a DataFrame to store all the train images

train_file_df = pd.DataFrame()

train_file_df['image_fp']=train_file_li

train_file_df.sample(5)
# Find all the image pairs with kinship



train_fd = '../input/recognizing-faces-in-the-wild/train'



index_p1_li=[]

index_p2_li=[]



for idx, row in tqdm_notebook(train_df.iterrows(), total=len(train_df)):

    if os.path.isdir(os.path.join(train_fd,row['p1'])) and os.path.isdir(os.path.join(train_fd,row['p2'])): # some folders do not exist !!

        for p1_pic in os.listdir(os.path.join(train_fd,row['p1'])):

            for p2_pic in os.listdir(os.path.join(train_fd,row['p2'])):

                index_f1=train_file_dict[os.path.join(row['p1'].split('/')[0],row['p1'].split('/')[1],p1_pic)]

                index_f2=train_file_dict[os.path.join(row['p2'].split('/')[0],row['p2'].split('/')[1],p2_pic)]

                if index_f1<index_f2: # force the image pairs to have the same order of persons

                    index_p1_li.append(index_f1)

                    index_p2_li.append(index_f2)

                else:

                    index_p1_li.append(index_f2)

                    index_p2_li.append(index_f1)

                    

train_pairs_kinship=pd.DataFrame()

train_pairs_kinship['p1']=index_p1_li

train_pairs_kinship['p2']=index_p2_li



index_p1_li=[]

index_p2_li=[]



print('Total image pairs with kinship: {}'.format(len(train_pairs_kinship)))

train_pairs_kinship.sample(5)
# make image pairs of the same person

# for example: for this person "F0002\MID1", there are 10 images in the folder, so it can make 10*9/2=45 pairs.



def make_pair_same_person(source,pre_path):

    res_p1_li = []

    res_p2_li = []

    for p1 in range(len(source)):

        for p2 in range(p1+1,len(source)):

            index_f1=train_file_dict[os.path.join(pre_path,source[p1])]

            index_f2=train_file_dict[os.path.join(pre_path,source[p2])]

            if index_f1<index_f2: # force the image pairs to have the same order of persons

                res_p1_li.append(index_f1)

                res_p2_li.append(index_f2)

            else:

                res_p1_li.append(index_f2)

                res_p2_li.append(index_f1)

            

    return (res_p1_li,res_p2_li)



index_p1_li = []

index_p2_li = []

for fam in os.listdir(train_fd):

    for pers in os.listdir(os.path.join(train_fd,fam)):

        res_temp = make_pair_same_person([pic for pic in os.listdir(os.path.join(train_fd,fam,pers))],os.path.join(fam,pers))

        index_p1_li.extend(res_temp[0])

        index_p2_li.extend(res_temp[1])



train_pairs_same=pd.DataFrame()

train_pairs_same['p1']=index_p1_li

train_pairs_same['p2']=index_p2_li



index_p1_li = []

index_p2_li = []



print('Total image pairs of same person: {}'.format(len(train_pairs_same)))



train_pairs_same.sample(5)
fig, ax = plt.subplots(figsize=(12,2))

y2show=[len(train_pairs_kinship),len(train_pairs_same)]

plt.barh(range(2),y2show,0.35)

plt.title('Image pair number')

plt.yticks(range(2), ('With kinship', 'From same person'),)

plt.box(on=None)

plt.xticks([], [])

for i, v in enumerate(y2show):

    ax.text(v+1000, i-0.05, str(v), color='blue', fontweight='bold')
train_pairs_kinship=pd.concat([train_pairs_kinship,train_pairs_same],ignore_index=True) # Combine them together

train_pairs_same=None # to free RAM

print('Total POSITIVE image pairs: {}'.format(len(train_pairs_kinship)))
# Get all the possible image pairs



index_p1_li = []

index_p2_li = []



for p1 in tqdm_notebook(range(len(train_file_li))):    

    for p2 in range(p1+1,len(train_file_li)):

        index_p1_li.append(p1)

        index_p2_li.append(p2)



train_pairs_all=pd.DataFrame()

train_pairs_all['p1']=index_p1_li

index_p1_li = []

train_pairs_all['p2']=index_p2_li

index_p2_li = []



print('Total image pairs: {}'.format(len(train_pairs_all)))



train_pairs_all.sample(5)
# Add a col "is_related": 1 if POS, 0 if NEG



kin_index=np.arange(len(train_pairs_all))[train_pairs_all.merge(train_pairs_kinship, on=['p1','p2'],how='left', indicator=True)['_merge']=='both']

train_pairs_all['is_related']=0

train_pairs_all.loc[kin_index,'is_related']=1

kin_index=None # to free RAM

train_pairs_kinship=None # to free RAM
fig, ax = plt.subplots(figsize=(12,2))

y2show=[train_pairs_all.query('is_related == 0').shape[0],train_pairs_all.query('is_related == 1').shape[0]]

plt.barh(range(2),y2show,0.35)

plt.title('Image pair number')

plt.yticks(range(2), ('No Kinship (NEG)','With kinship (POS)'))

plt.box(on=None)

plt.xticks([], [])

for i, v in enumerate(y2show):

    ax.text(v+1000, i-0.05, str(v), color='blue', fontweight='bold')
print("The number of negative samples is {:.0f} times of positive samples!".

      format(train_pairs_all.query('is_related == 0').shape[0]/train_pairs_all.query('is_related == 1').shape[0]))
# Get family ID for each image



train_file_df['fam']=-1

train_file_df['fam']=train_file_df['image_fp'].apply(lambda x: int(x[1:5]))

train_file_df.reset_index(inplace=True)

train_file_df.head()
print('There are {} families in the train set.'.format(len(train_file_df.fam.unique())))
# Get family ID for each POSimage pair (use p1 only)

train_pairs_kinship = train_pairs_all.query('is_related == 1')

train_pairs_kinship=train_pairs_kinship.merge(train_file_df[['index','fam']], left_on='p1',right_on='index',how='left').drop(columns=['index'])

train_pairs_kinship.sample(5)
fig, ax = plt.subplots(figsize=(12,5))

sns.countplot(x='fam',data=train_pairs_kinship,

              order=train_pairs_kinship.fam.value_counts().iloc[:20].index)

plt.title('Top 20 families (image pair with kinship | POS samples)')

ax.text(12, 30000, 'Average POS samples per family is {:.0f}'.format(len(train_pairs_kinship)/len(train_pairs_kinship.fam.unique())),fontsize=12)

for i, v in enumerate(train_pairs_kinship['fam'].value_counts()[:20]):

    ax.text(i-0.4, v+500, str(v),color='gray')

plt.box(on=None)

plt.yticks([]);
print('Family 601 contains {:.0f}% of image pair of all the POS samples!'.format(train_pairs_kinship['fam'].value_counts().tolist()[0]/len(train_pairs_kinship)*100))

print('Family 9 contains {:.0f}% of image pair of all the POS samples.'.format(train_pairs_kinship['fam'].value_counts().tolist()[1]/len(train_pairs_kinship)*100))
limit_number = 3000



index_li = train_pairs_kinship['fam'].value_counts()[lambda x:x<=limit_number].index

train_fam_lim_df = train_pairs_kinship[train_pairs_kinship['fam'].isin(index_li)]



for i in train_pairs_kinship['fam'].value_counts()[lambda x:x>limit_number].index:

    df_temp = train_pairs_kinship.query('fam == {}'.format(i)).sample(limit_number,replace=False,random_state=2019)

    train_fam_lim_df = pd.concat([train_fam_lim_df, df_temp])

    

train_fam_lim_df=train_fam_lim_df.reset_index() # Reset index for GroupKFold method



print('Number of POS samples in the selected dataset: {}'.format(len(train_fam_lim_df)))
fig, ax = plt.subplots(figsize=(12,5))

sns.countplot(x='fam',data=train_fam_lim_df,order=train_fam_lim_df.fam.value_counts().iloc[:20].index)

plt.title('Top 20 families (image pair with kinship | POS samples)')

ax.text(12, 2500, 'Average POS samples per family is {:.0f}'.format(len(train_pairs_kinship)/len(train_fam_lim_df.fam.unique())),fontsize=12)

for i, v in enumerate(train_fam_lim_df['fam'].value_counts()[:20]):

    ax.text(i-0.4, v+100, str(v),color='gray')

plt.box(on=None)

plt.yticks([]);
gkf = GroupKFold(n_splits=6) # Group 6 as test set, Group0-5 as CV folders.



train_fam=train_fam_lim_df['fam']



fam_group=np.ones(max(train_fam_lim_df['fam'])+1)*(-1)

fam_group=fam_group.astype(int)



for idx,( _, test_index) in enumerate(gkf.split(X=train_fam,groups=train_fam)):

    print("Group {}: {}".format(idx,np.unique(train_fam[test_index])))

    fam_group[np.unique(train_fam[test_index])]=idx

    print('-'*85)
# Get group ID for each image



train_file_df['group']=train_file_df['fam'].apply(

    lambda x: fam_group[x])



train_file_df.sample(5)
# Get group ID for each image pair



tqdm_notebook.pandas()

group_li=train_file_df['group'].tolist()



train_pairs_all['group1']=train_pairs_all['p1'].progress_apply(lambda x: group_li[x])

train_pairs_all['group2']=train_pairs_all['p2'].progress_apply(lambda x: group_li[x])

tmp_li = (train_pairs_all['group1']==train_pairs_all['group2'])*(train_pairs_all['group1']+1)-1

train_pairs_all.drop(columns=['group1','group2'],inplace=True)

train_pairs_all['group']=tmp_li

tmp_li=None # to free RAM

train_pairs_all.sample(5)
fig, ax = plt.subplots(figsize=(12,5))

sns.countplot(y='group',data=train_pairs_all,orient='v')

plt.title('Number of image pair in each group')

ax.text(30000000,1, '"-1" means no group is assigned.',fontsize=16)

ax.text(25000000,2,

        '{:.0f}% of image pairs have no group assigned.'.format(train_pairs_all.query('group == -1').shape[0]/len(train_pairs_all)*100),

        fontsize=16)

plt.box(on=None)

plt.xticks([]);
# Drop group==-1

train_pairs_all = train_pairs_all[train_pairs_all['group']!=-1]

# Shuffle

train_pairs_all = train_pairs_all.sample(frac=1,random_state=2019)
train_dataset_df=train_fam_lim_df

train_dataset_df['group']=train_dataset_df['fam'].apply(lambda x: fam_group[x])

train_dataset_df.drop(columns=['index','fam'],inplace=True)

train_dataset_df.head()
group_num = train_fam_lim_df.groupby('group')['is_related'].count().tolist()

portion=1 # get equal number of NEG / POS

df_temp = pd.concat(

    [t.head(int(group_num[g]*portion)) for g, t in train_pairs_all.query('is_related == 0').groupby('group', sort=False, as_index=False)],

    ignore_index=True)



train_dataset_df=pd.concat([train_dataset_df,df_temp],ignore_index=True)

train_dataset_df.shape
# to free RAM

train_pairs_all=None

train_pairs_kinship=None
fig, ax = plt.subplots(figsize=(12,5))

sns.countplot(y='group',data=train_dataset_df,orient='v',hue='is_related')

plt.title('Number of image pair in each group')

plt.box(on=None)

plt.xticks([]);
train_dataset_df.head()
model_path = '../input/facenet-keras/facenet_keras.h5'

model_facenet_keras = load_model(model_path)
def prewhiten(x):

    if x.ndim == 4:

        axis = (1, 2, 3)

        size = x[0].size

    elif x.ndim == 3:

        axis = (0, 1, 2)

        size = x.size

    else:

        raise ValueError('Dimension should be 3 or 4')



    mean = np.mean(x, axis=axis, keepdims=True)

    std = np.std(x, axis=axis, keepdims=True)

    std_adj = np.maximum(std, 1.0/np.sqrt(size))

    y = (x - mean) / std_adj

    return y



def l2_normalize(x, axis=-1, epsilon=1e-10):

    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

    return output



def load_and_align_images(filepaths, margin,image_size = 160):

    

    aligned_images = []

    for filepath in filepaths:

        img = imread(filepath)

        aligned = resize(img, (image_size, image_size), mode='reflect')

        aligned_images.append(aligned)

            

    return np.array(aligned_images)
def calc_embs(filepaths, margin=10, batch_size=512):

    pd = []

    for start in tqdm_notebook(range(0, len(filepaths), batch_size)):

        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size], margin))

        pd.append(model_facenet_keras.predict_on_batch(aligned_images))

    embs = l2_normalize(np.concatenate(pd))



    return embs
# Calculate embs for train images



train_embs = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/train", f) for f in train_file_df['image_fp']])

train_file_df=pd.concat([train_file_df, pd.DataFrame(train_embs,columns=['fe'+str(i) for i in range(128)])],axis=1)

train_file_df.head()
# Use absolute distance as final features



p1_df = train_dataset_df.merge(train_file_df, left_on='p1',right_on='index',how='left').iloc[:,8:]

p2_df = train_dataset_df.merge(train_file_df, left_on='p2',right_on='index',how='left').iloc[:,8:]



train_dataset_df = pd.concat([train_dataset_df, abs(p1_df-p2_df)],axis=1)

p1_df=None

p2_df=None

train_dataset_df.head()
# shuffle the dataset

train_dataset_df=train_dataset_df.sample(frac=1,random_state=2019).reset_index(drop=True)

train_dataset_df.head()
X=train_dataset_df.iloc[:,4:]

y=train_dataset_df.iloc[:,2]

X.shape, y.shape
X_train=X[train_dataset_df['group']!=5]

X_test=X[train_dataset_df['group']==5]

y_train=train_dataset_df['is_related'][train_dataset_df['group']!=5]

y_test=train_dataset_df['is_related'][train_dataset_df['group']==5]



y_train_group=train_dataset_df['group'][train_dataset_df['group']!=5]



X_train.shape,X_test.shape,y_train.shape,y_test.shape,y_train_group.shape
# group kfolder

group_kfold = GroupKFold(n_splits=5)
# this is a check of GroupKFold result



for train_index, test_index in group_kfold.split(X_train, y_train, y_train_group):

    #print("TRAIN:", train_index, "TEST:", test_index)

    print(np.unique(y_train_group.as_matrix()[train_index]))

    print(np.unique(y_train_group.as_matrix()[test_index]))

    print('-'*20)
model=LogisticRegression(random_state=2019)

res=cross_validate(model,X_train,y_train,cv=group_kfold,n_jobs=1,groups=y_train_group,scoring=('accuracy', 'roc_auc'))

print("Mean ROC_AUC score: {:.4f} (std: {:.4f})".format(res['test_roc_auc'].mean(),res['test_roc_auc'].std()))
# test on Test Set

model.fit(X_train,y_train)

print("ROC_AUC socre on test set: {:.3f}".format(roc_auc_score(y_test,model.predict_proba(X_test)[:,1])))
# Calculate embs for test images

test_images = os.listdir("../input/recognizing-faces-in-the-wild/test/")

test_embs = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/test/", f) for f in test_images])
img2idx = dict()

for idx, img in enumerate(test_images):

    img2idx[img] = idx
test_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")

test_df.head()
test_np = []

for idx, row in tqdm_notebook(test_df.iterrows(), total=len(test_df)):

    imgs = [test_embs[img2idx[img]] for img in row.img_pair.split("-")]

    test_np.append(abs(imgs[0]-imgs[1]))

test_np = np.array(test_np)
# Predict

model.fit(X,y)

probs = model.predict_proba(test_np)[:,1]



sub_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")

sub_df.is_related = probs
sub_df.hist();
# use this function to show some image pairs.

def show2pic(fd,paire):

    plt.figure(figsize=(7,10))

    plt.subplot(121)

    plt.imshow(imread(os.path.join(fd,paire.split('-')[0])))

    plt.axis('off')

    plt.title(paire.split('-')[0])

    plt.subplot(122)

    plt.imshow(imread(os.path.join(fd,paire.split('-')[1])))

    plt.axis('off')

    plt.title(paire.split('-')[1])
sub_df.sort_values('is_related',ascending=False).head(10)
# here is an example of the top 5th result.



show2pic('../input/recognizing-faces-in-the-wild/test/',sub_df.loc[4636,'img_pair'])
# export result to csv file

sub_df.to_csv("submission.csv", index=False)