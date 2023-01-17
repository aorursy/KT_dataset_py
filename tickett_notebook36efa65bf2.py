# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

lstFilesDCM_train = dict()

lstFilesDCM_test = dict()

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if ".dcm" in filename.lower(): 

            print(os.path.join(dirname, filename))

            if 'train' in dirname:

                

                #lstFilesDCM.append(os.path.join('/kaggle/input',dirname,filename))

                lstFilesDCM_train.setdefault(dirname.split('/')[-1],[]).append(os.path.join('/kaggle/input',dirname,filename))

            else:

                lstFilesDCM_test.setdefault(dirname.split('/')[-1],[]).append(os.path.join('/kaggle/input',dirname,filename))

                



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pydicom import dcmread

from pydicom.data import get_testdata_files

import matplotlib.pyplot as plt

import random

import cv2



from skimage import measure

from skimage import morphology

from sklearn.cluster import KMeans

from pydicom.pixel_data_handlers.util import apply_color_lut

editmode=False
def smokeprocess(thiscat):

    smocat=['Currently smokes', 'Ex-smoker', 'Never smoked']

    rtoh=list(np.zeros(len(smocat),dtype=np.int64))

    rtoh[smocat.index(thiscat)]=1

    return list(rtoh)

def sexprocess(thiscat):

    smocat=['Female', 'Male']

    rtoh=list(np.zeros(len(smocat),dtype=np.int64))

    rtoh[smocat.index(thiscat)]=1

    return list(rtoh)



def fillna(train_df,col,typ):

    if typ=='cat':

        return train_df.groupby(col).count().idxmax()[0]

    else:

        return train_df[col].median()

from sklearn.preprocessing import RobustScaler       



train_df=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

print(train_df.columns)

miss=0

X=[]

Xi=[]

Y=[]

Xr=[]

sexna=fillna(train_df,'Sex','cat')

agena=fillna(train_df,'Age','')

bool_series = pd.isnull(train_df['Sex'])

train_df[bool_series]=sexna

bool_series = pd.isnull(train_df['Age'])

train_df[bool_series]=agena

idc=0

use_data=0

for dcmfk in list(lstFilesDCM_train.keys()):

    idc=idc+1

    if idc>20 and editmode:

        break

    pt=True

    for dcmf in lstFilesDCM_train[dcmfk]:

        #print(dcmf.split('/')[-1].split('.dcm')[0])

        ds = dcmread(dcmf)





        #print(dcmf.split('/')[-1].split('.dcm')[0])

        #print(train_df[(train_df['Patient']==dcmfk) & (train_df['Weeks']==dcmf.split('/')[-1].split('.dcm')[0])])

        try:

            """

            if ds.pixel_array.shape!=(512,512):

                res = cv2.resize(xx, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

            else:

                res = ds.pixel_array

            """

            image_2d = ds.pixel_array.astype(float)

            mean = np.mean(image_2d)

            std = np.std(image_2d)

            image_2d = image_2d - mean

            image_2d = image_2d / std

            image_2d = image_2d-image_2d.min()

            

            #image_2d = cv2.resize(image_2d, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

            #image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            image_2d_scaled = np.clip(image_2d / np.quantile(image_2d,0.99),0,1) #* 255.0

            a0=image_2d_scaled.shape[0]

            a1=image_2d_scaled.shape[1]

            kmeans = KMeans(n_clusters=2).fit(image_2d_scaled[int(a0/2)-100:int(a0/2)+100:,int(a1/2)-100:int(a1/2)+100].reshape(-1,1))

            centers = sorted(kmeans.cluster_centers_.flatten())

            threshold = np.mean(centers)

            image_2d_scaled = np.where(image_2d_scaled < threshold, 1.0, 0.0)            



            eroded = morphology.erosion(image_2d_scaled, np.ones([2, 2]))

            dilation = morphology.dilation(image_2d_scaled, np.ones([4, 4]))     



            



            labels = measure.label(dilation)  # Different labels are displayed in different colors

            label_vals = np.unique(labels)

            regions = measure.regionprops(labels)            

            good_labels = []

            for prop in regions:

                B = prop.bbox

                if B[2] - B[0] < image_2d_scaled.shape[0] / 10 * 9 and B[3] - B[1] <  image_2d_scaled.shape[1] / 10 * 9 and B[0] >  image_2d_scaled.shape[0] / 5 and B[2] <  image_2d_scaled.shape[1] / 5 * 4:

                    good_labels.append(prop.label)

            mask = np.ndarray([image_2d_scaled.shape[0], image_2d_scaled.shape[1]], dtype=np.int8)

            mask[:] = 0

            



            #

            #  After just the lungs are left, we do another large dilation

            #  in order to fill in and out the lung mask

            #

            for N in good_labels:

                mask = mask + np.where(labels == N, 1, 0)

            mask = morphology.dilation(mask, np.ones([10, 10]))            

            image_2d_scaled=image_2d_scaled*mask

            res = cv2.resize(image_2d_scaled, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

            """

            if pt:

                plt.figure()

                plt.pcolor(res)

                pt=False

            """

            #res = cv2.resize(ds.pixel_array, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

            #res=RobustScaler().fit_transform(res)

            meta=train_df[(train_df['Patient']==dcmfk) & (train_df['Weeks']==int(dcmf.split('/')[-1].split('.dcm')[0]))]

            succ=0

            try:

                

                Y.append(train_df[(train_df['Patient']==dcmfk) & (train_df['Weeks']==int(dcmf.split('/')[-1].split('.dcm')[0]))].FVC.values[0])

                X.append(res)

                Xi.append(sexprocess(meta['Sex'].values[0])+[meta['Age'].values[0]]+smokeprocess(meta['SmokingStatus'].values[0]))

                Xr.append([dcmfk,int(dcmf.split('/')[-1].split('.dcm')[0])])

                use_data=use_data+1



                for i0 in range(0): 

                    img_flip=random.choice([0,-1])

                    res= cv2.flip(res, img_flip)

                    res= cv2.resize(res, dsize=(236, 236), interpolation=cv2.INTER_CUBIC)

                    for i in range(1): 



                        shift=random.randint(1,12)

                        shift1=random.randint(1,12)

                        resn=res[shift:224+shift,shift1:224+shift1]

                        Y.append(train_df[(train_df['Patient']==dcmfk) & (train_df['Weeks']==int(dcmf.split('/')[-1].split('.dcm')[0]))].FVC.values[0])

                        X.append(resn)

                        Xi.append(sexprocess(meta['Sex'].values[0])+[meta['Age'].values[0]]+smokeprocess(meta['SmokingStatus'].values[0]))

                print('ID',idc,'USEDATA',use_data,'times3')

            except:



                miss=miss+1

        except:

            1==1





        # 列出所有後設資料（metadata）

        # print(ds)

        #print(ds.PatientName)

        # 以 matplotlib 繪製影像

        #plt.imshow(ds.pixel_array)

        #plt.show()
plt.figure()





plt.pcolor(ds.pixel_array.astype(float))



"""

X

Xi

Y

Xr

"""


test_df=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print(test_df.columns)

miss=0

X_test=[]

Xi_test=[]

Xr_test=[]

Y_test=[]



X_test_no=[]

Xi_test_no=[]

Xr_test_no=[]



bool_series = pd.isnull(test_df['Sex'])

test_df[bool_series]=sexna

bool_series = pd.isnull(test_df['Age'])

test_df[bool_series]=agena

idc=0

use_data=0

for dcmfk in list(lstFilesDCM_test.keys()):

    idc=idc+1

    if idc>10 and editmode:

        break

    for dcmf in lstFilesDCM_test[dcmfk]:

        ds = dcmread(dcmf)



        try:

            image_2d = ds.pixel_array.astype(float)

            mean = np.mean(image_2d)

            std = np.std(image_2d)

            image_2d = image_2d - mean

            image_2d = image_2d / std

            image_2d = image_2d-image_2d.min()

            

            #image_2d = cv2.resize(image_2d, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

            #image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            image_2d_scaled = np.clip(image_2d / np.quantile(image_2d,0.99),0,1) #* 255.0

            a0=image_2d_scaled.shape[0]

            a1=image_2d_scaled.shape[1]

            kmeans = KMeans(n_clusters=2).fit(image_2d_scaled[int(a0/2)-100:int(a0/2)+100:,int(a1/2)-100:int(a1/2)+100].reshape(-1,1))

            centers = sorted(kmeans.cluster_centers_.flatten())

            threshold = np.mean(centers)

            image_2d_scaled = np.where(image_2d_scaled < threshold, 1.0, 0.0)            



            eroded = morphology.erosion(image_2d_scaled, np.ones([2, 2]))

            dilation = morphology.dilation(image_2d_scaled, np.ones([4, 4]))     



            



            labels = measure.label(dilation)  # Different labels are displayed in different colors

            label_vals = np.unique(labels)

            regions = measure.regionprops(labels)            

            good_labels = []

            for prop in regions:

                B = prop.bbox

                if B[2] - B[0] < image_2d_scaled.shape[0] / 10 * 9 and B[3] - B[1] <  image_2d_scaled.shape[1] / 10 * 9 and B[0] >  image_2d_scaled.shape[0] / 5 and B[2] <  image_2d_scaled.shape[1] / 5 * 4:

                    good_labels.append(prop.label)

            mask = np.ndarray([image_2d_scaled.shape[0], image_2d_scaled.shape[1]], dtype=np.int8)

            mask[:] = 0

            



            #

            #  After just the lungs are left, we do another large dilation

            #  in order to fill in and out the lung mask

            #

            for N in good_labels:

                mask = mask + np.where(labels == N, 1, 0)

            mask = morphology.dilation(mask, np.ones([10, 10]))            

            image_2d_scaled=image_2d_scaled*mask

            res = cv2.resize(image_2d_scaled, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

            #res = cv2.resize(ds.pixel_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

            #res=RobustScaler().fit_transform(res)

            meta=test_df[(test_df['Patient']==dcmfk) & (test_df['Weeks']==int(dcmf.split('/')[-1].split('.dcm')[0]))]

            succ=0

            try:

                Y_test.append(test_df[(test_df['Patient']==dcmfk) & (test_df['Weeks']==int(dcmf.split('/')[-1].split('.dcm')[0]))].FVC.values[0])

                X_test.append(res)

                Xi_test.append(sexprocess(meta['Sex'].values[0])+[meta['Age'].values[0]]+smokeprocess(meta['SmokingStatus'].values[0]))

                Xr_test.append([dcmfk,int(dcmf.split('/')[-1].split('.dcm')[0])])

            except:

                miss=miss+1

                meta=pd.DataFrame(test_df[(test_df['Patient']==dcmfk)].iloc[0,:]).T

                X_test_no.append(res)

                Xi_test_no.append(sexprocess(meta['Sex'].values[0])+[meta['Age'].values[0]]+smokeprocess(meta['SmokingStatus'].values[0]))

                Xr_test_no.append([dcmfk,int(dcmf.split('/')[-1].split('.dcm')[0])])

        except:

            1==1
"""

rgb_batch_train = np.repeat(np.array(X)[..., np.newaxis], 3, -1)

rgb_batch_test = np.repeat(np.array(X_test)[..., np.newaxis], 3, -1)

rgb_batch_val = np.repeat(np.array(X_test_no)[..., np.newaxis], 3, -1)

print(rgb_batch_train.shape,rgb_batch_test.shape,rgb_batch_val.shape)



rgb_batch_lab=np.append(rgb_batch_train,rgb_batch_test,axis=0)

rgb_batch=np.append(rgb_batch_lab,rgb_batch_val,axis=0)

rgb_batch.shape

"""

from sklearn.preprocessing import MinMaxScaler

rgb_batch_Y=np.append(np.array(Y).reshape(-1,1),np.array(Y_test).reshape(-1,1),axis=0)

scaler = MinMaxScaler()

scaler.fit(rgb_batch_Y)

print(scaler.data_max_)

Yt=scaler.transform(rgb_batch_Y)

Yt.shape

plt.hist(Yt)

plt.hist(rgb_batch_Y)

Yt=rgb_batch_Y



"""

bins = np.linspace(min(np.unique(np.round(Y))), max(np.unique(np.round(Y))), 200)

digitized = np.digitize(Y, bins)

from sklearn.preprocessing import LabelBinarizer

y = LabelBinarizer().fit_transform(digitized)

"""
rgb_batch_lab=np.append(X,X_test,axis=0)

rgb_batch=np.append(rgb_batch_lab,X_test_no,axis=0)

rgb_batch=rgb_batch.reshape(-1,224,224,1)

rgb_batch.shape
import os

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use("ggplot")

%matplotlib inline



from tqdm import tqdm_notebook, tnrange

from itertools import chain

from skimage.io import imread, imshow, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split



import tensorflow as tf



from keras.models import Model, load_model

from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout

from keras.layers.core import Lambda, RepeatVector, Reshape

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D

from keras.layers.merge import concatenate, add

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    # first layer

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",

               padding="same")(input_tensor)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    # second layer

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",

               padding="same")(x)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # contracting path

    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    p1 = MaxPooling2D((2, 2)) (c1)

    p1 = Dropout(dropout*0.5)(p1)



    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    p2 = MaxPooling2D((2, 2)) (c2)

    p2 = Dropout(dropout)(p2)



    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    p3 = MaxPooling2D((2, 2)) (c3)

    p3 = Dropout(dropout)(p3)



    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    p4 = Dropout(dropout)(p4)

    

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    

    # expansive path

    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)

    u6 = concatenate([u6, c4])

    u6 = Dropout(dropout)(u6)

    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)



    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)

    u7 = concatenate([u7, c3])

    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)



    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    u8 = Dropout(dropout)(u8)

    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)



    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    u9 = Dropout(dropout)(u9)

    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model
input_img = Input((224, 224, 1), name='img')

model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)



model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])

model.summary()

callbacks = [

    EarlyStopping(patience=10, verbose=1),

    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),

    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)

]
results = model.fit(rgb_batch, rgb_batch, batch_size=32, epochs=50, callbacks=callbacks)
import keras

layer_model = keras.Model(inputs=model.input,outputs=model.output)



pres=layer_model.predict(res.reshape(1,224,224,1))

plt.figure()

plt.pcolor(res)

plt.colorbar()

plt.figure()

plt.pcolor(pres.reshape(224,224))

plt.colorbar()
import keras

intermediate_layer_model = keras.Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_3').output)

intermediate_output = intermediate_layer_model.predict(rgb_batch)



block4_pool_features=intermediate_output.reshape(intermediate_output.shape[0],-1)
"""

import numpy as np

import tensorflow as tf

from keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Concatenate, Input

from keras import Model

from keras import optimizers

base_model = ResNet50(weights='imagenet', include_top=False, input_shape= (224, 224,3),pooling='max')

block4_pool_features = base_model.predict(np.array(rgb_batch).reshape(-1,224, 224,3))

block4_pool_features.shape

"""

from sklearn.manifold import TSNE



X_embedded = TSNE(n_components=3,perplexity=30).fit_transform(block4_pool_features)

X_embedded.shape

#plt.scatter(X_embedded[:1241,0],X_embedded[:1241,1],c='b')

#plt.scatter(X_embedded[1241:,0],X_embedded[1241:,1],c='g')



Xi=np.array(Xi)

Xi_test=np.array(Xi_test)

Xi_test_no=np.array(Xi_test_no)

Xi_temp=np.append(Xi,Xi_test,axis=0)

Xi_em2=np.append(Xi_temp,Xi_test_no,axis=0)

Xi_em2[:,2]=Xi_em2[:,2]/100

Xi_em2.shape

Xx=pd.concat([pd.DataFrame(X_embedded),pd.DataFrame(Xi_em2.reshape(-1,6))],axis=1)

Xx.shape



Xr=np.array(Xr)

Xr_test=np.array(Xr_test)

Xr_test_no=np.array(Xr_test_no)

Xr_temp=np.append(Xr,Xr_test,axis=0)

Xr_em2=np.append(Xr_temp,Xr_test_no,axis=0)

Xr_em2.shape

plt.hist(Xr_em2[:,1].astype(float))

#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

#block4_pool_features = model.predict(np.array(rgb_batch).reshape(-1,224, 224,3))
datapkg_pidw=dict()

for l in range(len(Yt)):

    pid=Xr_em2[l][0]

    week=float(Xr_em2[l][1])

    try:

        if datapkg_pidw[pid]>week:

            datapkg_pidw.update({pid:week})

    except:

        datapkg_pidw.update({pid:week})



            

datapkg_test_pidw=dict()



for it in range(len(Yt),len(X_embedded))   :

    pid=Xr_em2[it][0]

    week=float(Xr_em2[it][1])

    try:

        if datapkg_test_pidw[pid]>week:

            datapkg_test_pidw.update({pid:week})

    except:

        datapkg_test_pidw.update({pid:week})
datapkg=dict()

for l in range(len(Yt)):

    pid=Xr_em2[l][0]

    week=Xr_em2[l][1]

    fw=datapkg_pidw[pid]

    datapkg.setdefault(pid,[]).append([list(X_embedded[l])+list(Xi_em2[l])+[(float(week)-fw)/500,Yt[l][0]]])

datapkg_test=dict()



for it in range(len(Yt),len(X_embedded))   :

    pid=Xr_em2[it][0]

    week=Xr_em2[it][1]

    fw=datapkg_test_pidw[pid]

    datapkg_test.setdefault(pid,[]).append([list(X_embedded[it])+list(Xi_em2[it])+[(float(week)-fw)/500]])

    print(week,fw,(float(week)-fw)/500)
import numpy as np

from matplotlib import pyplot as plt



from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def tgp(Xx, Yt):

    kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))

    gp = GaussianProcessRegressor(kernel=kernel)



    # Fit to data using Maximum Likelihood Estimation of the parameters

    gp.fit(Xx, Yt)

    ppp=gp.predict(Xx)

    return gp,ppp

gpd=dict()

for pid in datapkg.keys():

    iii=np.array(datapkg[pid]).reshape(-1,11)

    ix=iii[:,:10]

    iy=iii[:,10]



    igp,ppp=tgp(ix, iy)

    if np.mean(100*(abs(ppp-iy)/iy))<=0.5:



        gpd.setdefault(pid,[]).append(igp)

    else:

        print(pid,'out')



pdd=dict()

for ppid in  datapkg_test.keys():

    iii2=np.array(datapkg_test[ppid]).reshape(-1,10)

    for ie in iii2:

        fw=datapkg_test_pidw[ppid]

        pdk=ppid+'_-'+str(int(ie[-1]*500+fw))

        allpd=[]

        for mid in gpd.keys():

            gp=gpd[mid][0]

            y_pred, sigma = gp.predict([ie], return_std=True)

            y_pred_t11, sigma_t11 = gp.predict([np.array(list(ie[:3]*1.1)+list(ie[3:]))], return_std=True)

            y_pred_t09, sigma_t09 = gp.predict([np.array(list(ie[:3]*0.9)+list(ie[3:]))], return_std=True)

            if not y_pred_t11!=y_pred or y_pred_t09!=y_pred:

                allpd.append([y_pred[0],sigma[0]])

        

        pdd.setdefault(pdk,[]).append(allpd)

pdd.keys()

pdd['ID00426637202313170790466_-402']

import pickle



file = open('/kaggle/working/op.pkl', 'wb')

pickle.dump(pdd, file)

file.close()



finaloutput=[]

for ik in np.sort(list(pdd.keys())):



    pmean=np.array(np.array(pdd[ik])[0])[:,0]

    

    

    _f=np.array(np.array(pdd[ik])[0])[pmean>=1000,:]



    bins = range(0, max(Yt)[0]+3000,3000)

    digitized = np.digitize(_f[:,0], bins)

    counts = np.bincount(digitized)



    ll=np.array(bins)[np.argmax(counts)]-3000

    ul=np.array(bins)[np.argmax(counts)]

    finaloutput.append([ik]+list(np.mean(_f[(_f[:,0]>=ll) & (_f[:,0]<=ul),:],axis=0)))

    """

    plt.figure()

    plt.hist(_f[:,0])

    """

opdf=pd.DataFrame(finaloutput)    

opdf.columns=['Patient_Week','FVC','Confidence']

opdf.to_csv('/kaggle/working/submission.csv', index=False)

opdf.to_csv('submission.csv', index=False)

"""



# Make the prediction on the meshed x-axis (ask for MSE as well)

y_pred, sigma = gp.predict(Xx, return_std=True)

plt.plot(Yt)

plt.plot(y_pred)

plt.ylim([0,1])

scaler.inverse_transform(y_pred)

"""

























"""

# Plot the function, the prediction and the 95% confidence interval based on

# the MSE

plt.figure()

#plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')

plt.plot(X, yt, 'r.', markersize=10, label='Observations')

plt.plot(Xx, y_pred, 'b-', label='Prediction')

plt.fill(np.concatenate([x, x[::-1]]),

         np.concatenate([y_pred - 1.9600 * sigma,

                        (y_pred + 1.9600 * sigma)[::-1]]),

         alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('$x$')

plt.ylabel('$f(x)$')

plt.ylim(-10, 20)

plt.legend(loc='upper left')

"""
"""



import tensorflow as tf

from keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Concatenate, Input

from keras import Model

from keras import optimizers

base_model = ResNet50(weights='imagenet', include_top=False, input_shape= (224, 224,3))

#base_model.layers.pop()

#base_model.outputs = []

#x = base_model.output

#base_model.summary()

x = base_model.layers[-2].output

x = GlobalAveragePooling2D()(x)

###x = Dropout(0.7)(x)

prepredictions = Dense(128, activation= 'relu')(x)

inp2 = Input(shape=(6,))

conatenated = Concatenate(axis=1)([prepredictions, inp2])

predictions = Dense(154, activation= 'softmax')(conatenated)

model = Model(inputs = [ base_model.input, inp2], outputs = predictions)

model.summary()

Adam=optimizers.Adam

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max')

model.compile(optimizer=Adam(lr=5e-4, decay=5e-4 / 40) ,loss='categorical_crossentropy', metrics=['categorical_crossentropy'])

history = model.fit([np.array(rgb_batch).reshape(-1,224, 224,3),np.array(Xi).reshape(-1,6)], np.array(y), batch_size=16, epochs=40,validation_split=0.1, callbacks=[callback])

"""
iii2[:,-1]

[0]
"""

rgb_batch_test = np.repeat(np.array(X_test)[..., np.newaxis], 3, -1)



Yp=model.predict([np.array(rgb_batch_test).reshape(-1,224, 224,3),np.array(Xi_test).reshape(-1,6)],batch_size=16)

Yp_trans=np.argmax(Yp,axis=1)



digitized_test = np.digitize(Y_test, bins)

print(digitized_test,Yp_trans)

"""