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
plt.figure()

plt.pcolor(res)

plt.colorbar()

"""



sum(labs==0)

sum(labs==1)

plt.pcolor(labs.reshape(224,224))

"""
from sklearn.preprocessing import MinMaxScaler

rgb_batch_Y=np.append(np.array(Y).reshape(-1,1),np.array(Y_test).reshape(-1,1),axis=0)



Yt=rgb_batch_Y



rgb_batch_lab=np.append(X,X_test,axis=0)

rgb_batch=np.append(rgb_batch_lab,X_test_no,axis=0)

rgb_batch=rgb_batch.reshape(-1,224,224)

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
#X meta

Xi=np.array(Xi)

Xi_test=np.array(Xi_test)

Xi_test_no=np.array(Xi_test_no)

Xi_temp=np.append(Xi,Xi_test,axis=0)

Xi_em2=np.append(Xi_temp,Xi_test_no,axis=0)

Xi_em2[:,2]=Xi_em2[:,2]/100

Xi_em2.shape



#X rec

Xr=np.array(Xr)

Xr_test=np.array(Xr_test)

Xr_test_no=np.array(Xr_test_no)

Xr_temp=np.append(Xr,Xr_test,axis=0)

Xr_em2=np.append(Xr_temp,Xr_test_no,axis=0)

Xr_em2.shape

plt.hist(Xr_em2[:,1].astype(float))

""" 

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



for it in range(len(Yt),len(Xr_em2))   :

    pid=Xr_em2[it][0]

    week=float(Xr_em2[it][1])

    try:

        if datapkg_test_pidw[pid]>week:

            datapkg_test_pidw.update({pid:week})

    except:

        datapkg_test_pidw.update({pid:week})

"""         

        

datapkg_all_pidw=dict()



for it in range(len(Xr_em2))   :

    pid=Xr_em2[it][0]

    week=float(Xr_em2[it][1])

    try:

        if datapkg_all_pidw[pid]>week:

            datapkg_all_pidw.update({pid:week})

    except:

        datapkg_all_pidw.update({pid:week})

        
""" 

datapkg=dict()

for l in range(len(Yt)):

    pid=Xr_em2[l][0]

    week=Xr_em2[l][1]

    fw=datapkg_pidw[pid]

    datapkg.setdefault(pid,[]).append([list(X_embedded[l])+list(Xi_em2[l])+[(float(week)-fw)/500,Yt[l][0]]])

datapkg_test=dict()



for it in range(len(Yt),len(Xr_em2))   :

    pid=Xr_em2[it][0]

    week=Xr_em2[it][1]

    fw=datapkg_test_pidw[pid]

    datapkg_test.setdefault(pid,[]).append([list(X_embedded[it])+list(Xi_em2[it])+[(float(week)-fw)/500]])

    print(week,fw,(float(week)-fw)/500)

""" 



datapkg_all=dict()

datapkg_all_test=dict()



for l in range(len(Xr_em2)):



    pid=Xr_em2[l][0]

    week=Xr_em2[l][1]

    fw=datapkg_all_pidw[pid]

    res=rgb_batch[l]

    

    kmeans = KMeans(n_clusters=2, random_state=0).fit(res.reshape(-1,1))

    labs=kmeans.labels_

    if len(np.unique(labs))>1:



        if labs[np.where(res.reshape(-1)==max(res.reshape(-1)))][0]!=1:

            labs=labs*(-1)+1



    try:



        datapkg_all.setdefault(pid,[]).append(list([np.sum(labs)])+list(Xi_em2[l])+[(float(week)-fw),Yt[l][0]])

    except:

        datapkg_all_test.setdefault(pid,[]).append(list([np.sum(labs)])+list(Xi_em2[l])+[(float(week)-fw)])



        

tc_all=[]

tc_all_d=dict()

for eapid in datapkg_all.keys():

    a=np.array(datapkg_all[eapid])

    sa=a[a[:,-2].argsort()]

    m1=np.diff(sa,axis=0)

    sa2=sa[:-1,:]

    sa2[:,0]=m1[:,0]

    sa2[:,-1]=m1[:,-1]

    sa2[:,-2]=m1[:,-2]

    m2=sa-sa[0]

    sa[:,0]=m2[:,0]

    sa[:,-1]=m2[:,-1]

    sa[:,-2]=m2[:,-2]

    tc=np.concatenate([sa,sa2])

    if tc_all==[]:

        tc_all=tc

    else:

        tc_all=np.concatenate([tc_all,tc])

    tc_all_d.setdefault(eapid,[]).append(tc)



        

def gettestdata(datapkg_all_test,datapkg_all):

    tc_all_test_inver_d=dict()

    tc_all_test_d=dict()

    tc_all_test=[]   

    tc_all_test_inver=[]     

    for eapid in datapkg_all_test.keys():

        a=np.array(datapkg_all[eapid])

        sa=a[a[:,-2].argsort()]

        tc_test=datapkg_all_test[eapid]

        tc_testd=datapkg_all_test[eapid]-sa[0][:-1]

        np.array(tc_test)[:,0]=np.array(tc_testd)[:,0]

        np.array(tc_test)[:,-1]=np.array(tc_testd)[:,-1]

        tc_test_inver=[]

        for i in range(len(tc_test)):

            tc_test_inver.append([eapid]+list(sa[0]))

        if tc_all_test==[]:

            tc_all_test=tc_test

            tc_all_test_inver=tc_test_inver

        else:

            tc_all_test=np.concatenate([tc_all_test,tc_test])    

            tc_all_test_inver=np.concatenate([tc_all_test_inver,tc_test_inver])

        tc_all_test_d.setdefault(eapid,[]).append(tc_test)

        tc_all_test_inver_d.setdefault(eapid,[]).append(tc_test_inver)

    return tc_all_test,tc_all_test_inver,tc_all_test_d,tc_all_test_inver_d

tc_all_test,tc_all_test_inver,tc_all_test_d,tc_all_test_inver_d=gettestdata(datapkg_all_test,datapkg_all)

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

for pid in tc_all_d.keys():

    iii=np.array(tc_all_d[pid]).reshape(-1,9)

    ix=iii[:,:8]

    iy=iii[:,8]



    igp,ppp=tgp(ix, iy)

    gpd.setdefault(pid,[]).append(igp)



pdd=dict()

for ppid in  tc_all_test_d.keys():

    iii2=np.array(tc_all_test_d[ppid]).reshape(-1,8)

    wb=float(np.array(tc_all_test_inver_d[ppid])[0][0][-2])

    yb=float(np.array(tc_all_test_inver_d[ppid])[0][0][-1])

    for ie in iii2:

        fw=datapkg_all_pidw[ppid]

        pdk=ppid+'_-'+str(int(ie[-1]+fw+wb))

        allpd=[]

        for mid in gpd.keys():

            gp=gpd[mid][0]

            y_pred, sigma = gp.predict([ie], return_std=True)

            y_pred_t11, sigma_t11 = gp.predict([np.array(list(ie[:3]*1.1)+list(ie[3:]))], return_std=True)

            y_pred_t09, sigma_t09 = gp.predict([np.array(list(ie[:3]*0.9)+list(ie[3:]))], return_std=True)

            if not y_pred_t11!=y_pred or y_pred_t09!=y_pred:

                allpd.append([y_pred[0]+yb,sigma[0]])

        

        pdd.setdefault(pdk,[]).append(allpd)



        



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

tdf=train_df[['Patient','Weeks','FVC']]

tdf['Patient_Week']=tdf['Patient']+'_'+tdf['Weeks'].astype(str)

tdf['Confidence']=0

opdf=pd.DataFrame(finaloutput)    

opdf.columns=['Patient_Week','FVC','Confidence']

opdf=pd.concat([tdf[['Patient_Week','FVC','Confidence']],opdf])



opdf.to_csv('/kaggle/working/submission.csv', index=False)

opdf.to_csv('submission.csv', index=False)