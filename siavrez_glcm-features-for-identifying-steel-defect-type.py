%matplotlib inline

import matplotlib.pyplot as plt

import plotly.express as px

from plotly.offline import iplot

import plotly.graph_objects as go

import random

import os

import numpy as np

import pandas as pd 

import cv2

import os

from matplotlib import style

import seaborn as sns

from skimage import img_as_float, img_as_uint, img_as_int

from skimage.feature import greycomatrix, greycoprops

import xgboost as xgb

from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

train['defect'] = train['EncodedPixels'].notnull()

train['ClassId'] = train['ImageId_ClassId'].str[-1:]

train['ImageId'] = train['ImageId_ClassId'].str[:-2]

train = train[['ImageId','ClassId','defect','EncodedPixels']]

train['EncodedPixels']=train['EncodedPixels'].fillna(0)

train.head()
def mask2rle(img):

    tmp = np.rot90( np.flipud( img ), k=3 )

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(startpos)+' '+str(endpos-startpos+1) )

    return " ".join(rle)



def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )



def rle2mask_eda(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
columns = 2

rows = 10

fig = plt.figure(figsize=(20,columns*rows+2))

for i in range(1,columns*rows+1):

    fn = train['ImageId'].iloc[i]

    c = train['ClassId'].iloc[i]

    fig.add_subplot(rows, columns, i).set_title(fn+"  ClassId="+c)

    img = cv2.imread( '../input/train_images/'+fn )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = rle2mask_eda(train.loc[train['EncodedPixels']!=0,'EncodedPixels'].iloc[i])

    img[mask==1,0] = 255

    plt.imshow(img)

plt.show()
columns = 2

rows = 10

fig = plt.figure(figsize=(20,columns*rows+2))

for i in range(1,columns*rows+1):

    fn = train['ImageId'].iloc[i]

    c = train['ClassId'].iloc[i]

    fig.add_subplot(rows, columns, i).set_title(fn+"  ClassId="+c)

    img = cv2.imread( '../input/train_images/'+fn )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = rle2mask_eda(train.loc[train['EncodedPixels']!=0,'EncodedPixels'].iloc[i])

    img[mask==1] = 255

    plt.imshow(img)
train_path =  '../input/train_images/'

test_path =  '../input/test_images/'
def GLCM(img_list, img_path):

    

    glcm_data = np.zeros((len(img_list), 80))

    for i, fn in tqdm(enumerate(img_list), total=len(img_list)):

        glcm_ = np.zeros(80)

        img = cv2.imread( img_path+fn )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image = img_as_uint(img)

        image =  image.astype(np.uint8)

        glcm = greycomatrix(image, [1, 2, 3, 4], [0, np.pi/8,np.pi/4, 3*np.pi/8], 256, symmetric=True, normed=True)

        glcm_[:16]   = greycoprops(glcm, 'contrast').ravel()

        glcm_[16:32] = greycoprops(glcm, 'dissimilarity').ravel()

        glcm_[32:48] = greycoprops(glcm, 'homogeneity').ravel()

        glcm_[48:64] = greycoprops(glcm, 'energy').ravel()

        glcm_[64:80] = greycoprops(glcm, 'ASM').ravel()

        glcm_data[i,:] = glcm_

    

    return pd.DataFrame(glcm_data)
images = list(train.ImageId.unique())

defected_steal_images = list(train[train.defect]['ImageId'].unique())

non_defected_steal_images = list(set(images)-set(defected_steal_images))
print(f'Number of unique images       :{len(train.ImageId.unique())}')

print(f'Number of non defected images :{len(defected_steal_images)}')

print(f'Number of defected images     :{len(non_defected_steal_images)}')
glcm_data_normal = GLCM(non_defected_steal_images, train_path)
defected = train[train.defect]

eachG = defected.groupby(['ClassId'])

type1 = eachG.get_group('1')

type2 = eachG.get_group('2')

type3 = eachG.get_group('3')

type4 = eachG.get_group('4')

type1 = type1[type1.defect].reset_index(drop=True)

type2 = type2[type2.defect].reset_index(drop=True)

type3 = type3[type3.defect].reset_index(drop=True)

type4 = type4[type4.defect].reset_index(drop=True)
type1_defected_img = list(type1.ImageId.unique())

type2_defected_img = list(type2.ImageId.unique())

type3_defected_img = list(type3.ImageId.unique())

type4_defected_img = list(type4.ImageId.unique())
glcm_data_type_1 = GLCM(type1_defected_img, train_path)
glcm_data_type_2 = GLCM(type2_defected_img, train_path)
glcm_data_type_3 = GLCM(type3_defected_img, train_path)
glcm_data_type_4 = GLCM(type4_defected_img, train_path)
glcm_data_type_1['classId'] = 1

glcm_data_type_2['classId'] = 2

glcm_data_type_3['classId'] = 3

glcm_data_type_4['classId'] = 4

glcm_data_normal['classId'] = 0

glcm_df = pd.concat([glcm_data_normal, glcm_data_type_1, glcm_data_type_2, 

                     glcm_data_type_3, glcm_data_type_4]).sample(frac=1, 

                     random_state=1365).reset_index(drop=True)

cols = glcm_df.iloc[:,:-1].columns

sc = StandardScaler()

glcm_df.iloc[:,:-1] = pd.DataFrame(sc.fit_transform(glcm_df.iloc[:,:-1]), columns=cols)

glcm_df.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(range(1,81)), y=glcm_data_normal.iloc[:,:-1].values.mean(axis=0),

                    mode='lines+markers',

                    name='Normal Images'))

fig.add_trace(go.Scatter(x=list(range(1,81)), y=glcm_data_type_1.iloc[:,:-1].values.mean(axis=0),

                    mode='lines+markers',

                    name='Type 1 Defect'))

fig.add_trace(go.Scatter(x=list(range(1,81)), y=glcm_data_type_2.iloc[:,:-1].values.mean(axis=0),

                    mode='lines+markers',

                    name='Type 2 Defect'))

fig.add_trace(go.Scatter(x=list(range(1,81)), y=glcm_data_type_3.iloc[:,:-1].values.mean(axis=0),

                    mode='lines+markers',

                    name='Type 3 Defect'))

fig.add_trace(go.Scatter(x=list(range(1,81)), y=glcm_data_type_4.iloc[:,:-1].values.mean(axis=0),

                    mode='lines+markers',

                    name='Type 4 Defect'))

fig.update_layout(title='GLCM Features For Steal Surface',

                   xaxis_title='Average of GLCM Features For Normal and Defected Steel')



fig.show()
parameters = {'n_estimators': [200,400], 

              'max_depth': [15, 25],

              'learning_rate' : [0.05], 

              'subsample' : [0.8]

             }
y_train = glcm_df.pop('classId')

X_train = glcm_df



clf = xgb.XGBClassifier(objective='multi:softmax')

grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, n_jobs=-1 , verbose = 0)

grid_search.fit(X_train , y_train)

print("Best score: %0.5f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters=grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print("Best Score")
glcm_data_normal['classId'] = 0

glcm_data_type_1['classId'] = 1

glcm_data_type_2['classId'] = 1

glcm_data_type_3['classId'] = 1

glcm_data_type_4['classId'] = 1
glcm_df = pd.concat([glcm_data_normal, glcm_data_type_1, glcm_data_type_2, glcm_data_type_3, glcm_data_type_4]).sample(frac=1, random_state=1365).reset_index(drop=True)

glcm_df.head()
y_train = glcm_df.pop('classId')

X_train = glcm_df



clf = xgb.XGBClassifier()

grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, n_jobs=-1 , verbose = 0)
grid_search.fit(X_train , y_train)

print("Best score: %0.5f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters=grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print("Best Score")