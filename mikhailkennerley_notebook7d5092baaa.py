import matplotlib.pyplot as plt

import os

import numpy as np

from scipy.ndimage import zoom

import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,LSTM,TimeDistributed, Input,BatchNormalization, Activation, Conv3D,MaxPooling3D, Dropout, concatenate

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger

from tensorflow.math import abs, sqrt, minimum, maximum, log, multiply, divide_no_nan, add

from tensorflow.keras.backend import greater, zeros_like, variable, constant, int_shape, mean, shape, ones_like

from tensorflow.keras.regularizers import l2

from keras.utils import to_categorical

import tensorflow as tf

import h5py

import cv2

import pandas as pd

import skimage

from sklearn import preprocessing

from pydicom import dcmread

#from lungmask import mask

#import SimpleITK as sitk

import sys

from tensorflow_addons.losses import PinballLoss

from tensorflow.keras.losses import MSE

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import keras

from keras.utils.vis_utils import model_to_dot

from sklearn.linear_model import LinearRegression

from tensorflow import boolean_mask, where

from tensorflow_addons.losses import pinball_loss

from sklearn.model_selection import train_test_split

setup=0
loc='/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

loctest='/kaggle/input/osic-pulmonary-fibrosis-progression/test/'



dirs=os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/')

dirstest=os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/test/')

setup=0
def cropImg(img):

    

    if not (sum(img[0])==sum(img[round(len(img)/2)])).all():

        r_min=0

        r_max=len(img)

        c_min=0

        c_max=len(img[0])

        cropImg=[r_min,r_max,c_min,c_max]

        return cropImg

    r_min, r_max = None, None

    c_min, c_max = None, None

    

    for row in range(len(img)):

        if not (img[row,:]==img[0,0]).all() and r_min is None:

            r_min=row

        if (img[row,:]==img[0,0]).all() and r_min is not None and r_max is None:

            r_max=row

                

    flipImg=np.rot90(img)

    for col in range(len(flipImg)):

        if not (flipImg[col,:]==flipImg[0,0]).all() and c_min is None:

            c_min=col

        if (flipImg[col,:]==flipImg[0,0]).all() and c_min is not None and c_max is None:

            c_max=col

    cropImg=[r_min,r_max,c_min,c_max]

    return cropImg



def resizeImg(img,x3d,y3d,z3d):

    reX = x3d/len(img)

    reY = y3d/len(img[0])

    reZ = z3d/len(img[0][0])

    reImg=zoom(img,(reX,reY,reZ))

    return reImg





def LLL_metric(y_true, y_pred,return_values = False):

    sigma = y_pred[:, 2] - y_pred[:, 0]

    sd_clipped = maximum(sigma,70)

    delta= minimum(abs(y_pred[:,1] - y_true[:,0]), 1000)

    metric=- sqrt(2.0) * delta / sd_clipped - log(sqrt(2.0) * sd_clipped)

        #print('\n')

        #print(metric)

    return metric

        

def get_patient_data(ID):

    return table[table['Patient'] == ID]



def plot_patients(patient_list, age, sex, smoke):

    patient_length = len(patient_list)

    cols = 8



    fig, ax = plt.subplots(patient_length//cols+1, cols, sharey = True, sharex = 'none', figsize=(30,15))

    for num, patient in enumerate(patient_list):

        ax[num//cols,num%cols].plot(get_patient_data(patient)['Weeks'], get_patient_data(patient)['FVC'])

        if smoke[num] == 'Ex-smoker':

            smoke_str = 'EX'

        elif smoke[num] == 'Never smoked':

            smoke_str = 'NA'

        elif smoke[num] == 'Currently smokes':

            smoke_str = 'SM'

        ax[num//cols,num%cols].set_title(str(age[num])+'_'+sex[num]+'_'+smoke_str)



    plt.show()



def get_patients(lower, upper):

    patient_list = table['Patient'].unique()[lower:upper]



    age = []

    sex = []

    smoke = []



    for patient in patient_list:

        subset = table[table['Patient'] == patient].reset_index()

        age.append(subset['Age'][1])

        sex.append(subset['Sex'][1])

        smoke.append(subset['SmokingStatus'][1])



    return patient_list, age,sex,smoke



# Loss function

def LLL_metric2(y_true, y_pred):

  zeros = zeros_like(y_true)

  bool_mask = greater(y_true, [0])

  y_pred = where(bool_mask, y_pred, y_true)



  diff = abs(y_pred - y_true)

  sigma = constant(value = 200, shape = [146])



  delta = minimum(diff, constant(value = 1000, shape = [146]))

  delta = diff

  sqrt2 = constant(value = 1.414, shape = [146])

  

  loss = - divide_no_nan(sqrt2 * delta, sigma) - log(where(bool_mask, sqrt2 * sigma, ones_like(y_true)))

  avg_loss = mean(boolean_mask(loss, bool_mask), axis = 0)

  

  return avg_loss



def LLL_metric_neg(y_true, y_pred):

  zeros = zeros_like(y_true)

  bool_mask = greater(y_true, [0])

  y_pred = where(bool_mask, y_pred, y_true)



  diff = abs(y_pred - y_true)

  sigma = constant(value = 70, shape = [146])



  delta = minimum(diff, constant(value = 1000, shape = [146]))

  delta = diff

  sqrt2 = constant(value = 1.414, shape = [146])

  

  loss = - divide_no_nan(sqrt2 * delta, sigma) - log(where(bool_mask, sqrt2 * sigma, ones_like(y_true)))

  avg_loss = mean(boolean_mask(loss, bool_mask), axis = 0)

  

  return -(avg_loss)



def pinball2(y_true, y_pred):

  zeros = zeros_like(y_true)

  bool_mask = greater(y_true, [0])

  y_pred = where(bool_mask, y_pred, y_true)

  #tf.print(y_pred, summarize = 20)



  return pinball_loss(y_true, y_pred)



def MSE2(y_true, y_pred):

  zeros = zeros_like(y_true)

  bool_mask = greater(y_true, [0])

  y_pred = where(bool_mask, y_pred, y_true)

  #tf.print(y_pred, summarize = 20)



  return MSE(y_true, y_pred)



def createLine(outPred):

    click=0

    first=0

    last=0

    for i in range(len(outPred)):

        if outPred[i]<=1001 and click==0:

            first=i+1

        elif outPred[i]<=1001 and click==1 and i>20:

            last=i+1

            break

        if not (outPred[i]<=1001):

            click=1



    y=np.array(outPred[first:last])

    x=np.array(list(range(first-12,last-12)))

    xmean=np.mean(x)

    ymean=np.mean(y)



    xycov = (x - xmean) * (y - ymean)

    xvar = (x - xmean)**2

    beta = xycov.sum() / xvar.sum()

    alpha = ymean - (beta * xmean)

    line=[i*beta+alpha for i in np.array(list(range(-12,134)))]

    return line
my_data_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv', delimiter=',')
setup=1
d1 = os.listdir(loctest+dirstest[2])

d = sorted(os.listdir(loctest+dirstest[2]), key=lambda v:int(v.split('.')[0]))

d = {i+1:dcm for i,dcm in enumerate(d)}

center = len(d)//2

loctest+dirstest[0]

table = my_data_test

table.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

table = table.reset_index(drop = True)

patient_list, age, sex, smoke = get_patients(50, 70)

#plot_patients(patient_list,age,sex,smoke)

df = pd.DataFrame(columns = ['Patient','1st Read','1st Wk','Coef', 'Age','Sex','Smoker'])

for num, patient in enumerate(patient_list):

    patient_data = get_patient_data(patient).reset_index()

    x = patient_data['Weeks'].values.reshape(-1,1)

    y = patient_data['FVC'].values

    lin_reg = LinearRegression().fit(x,y)



    df = df.append({'Patient':patient,

                    '1st Read':patient_data['FVC'][0],

                    '1st Wk':patient_data['Weeks'][0],

                    'Coef':lin_reg.coef_,

                    'Age':age[num],

                    'Sex':sex[num],

                    'Smoker':smoke[num]}, ignore_index = True)

    

print(table.shape)

table.head()
# One-hot encode Sex and SmokingStatus

onehot = OneHotEncoder(sparse = False)

trf = onehot.fit_transform(table[['Sex','SmokingStatus']])



cat_list = []



for item in onehot.categories_:

  temp = item.tolist()

  cat_list.extend(temp)



trf_df = pd.DataFrame(trf, columns = cat_list)



merge = pd.concat([table, trf_df], axis = 1).drop(['Sex','SmokingStatus'], axis = 1)



# Standardize FVC, Age, and Percent

#norm_FVC = StandardScaler().fit(merge.FVC.values.reshape(-1,1))

#norm_age = StandardScaler().fit(merge.Age.values.reshape(-1,1))

#norm_percent = StandardScaler().fit(merge.Percent.values.reshape(-1,1))



#merge.FVC = pd.DataFrame(norm_FVC.transform(merge.FVC.values.reshape(-1,1)))

#merge.Age = pd.DataFrame(norm_age.transform(merge.Age.values.reshape(-1,1)))

#merge.Percent = pd.DataFrame(norm_percent.transform(merge.Percent.values.reshape(-1,1)))



final_df = pd.merge(merge.groupby('Patient')['Weeks'].apply(list).to_frame(),

                    merge.groupby('Patient')['FVC'].apply(list).to_frame(),

                    on = 'Patient')



final_df = pd.merge(final_df, merge.groupby('Patient')['Percent'].apply(list).to_frame(), on = 'Patient')

final_df = pd.merge(final_df, merge.groupby('Patient')['Age'].first().to_frame(), on = 'Patient')

try:

    final_df = pd.merge(final_df, merge.groupby('Patient')['Male'].first().to_frame(), on = 'Patient')

except:

    final_df['Male']=0



try:

    final_df = pd.merge(final_df, merge.groupby('Patient')['Female'].first().to_frame(), on = 'Patient')

except:

    final_df['Female']=0

try:

    final_df = pd.merge(final_df, merge.groupby('Patient')['Currently smokes'].first().to_frame(), on = 'Patient')

except:

    final_df['Currently smokes']=0

try:

    final_df = pd.merge(final_df, merge.groupby('Patient')['Ex-smoker'].first().to_frame(), on = 'Patient')

except:

    final_df['Ex-smoker']=0

try:

    final_df = pd.merge(final_df, merge.groupby('Patient')['Never smoked'].first().to_frame(), on = 'Patient')

except:

    final_df['Never smoked']=0

final_df = final_df.reset_index()



final_df.head()
if setup==1:

    xImg=[]

    time1=1

    for folder in final_df.Patient:

        print(time1)

        time1=time1+1

        folderdir=loctest+folder

        images=sorted(os.listdir(loctest+folder), key=lambda v:int(v.split('.')[0]))

        try:

            imagecrop=folderdir+"/"+images[round(len(images)/2)]

            ds1 = dcmread(imagecrop)

            imgCrop= cropImg(ds1.pixel_array)

            imgC = ds1.pixel_array[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3]]



        except:

            imgCrop=[0,48,0,48]

        

        imglist=[]

        for img in images:

            try:

                ds = dcmread(folderdir+"/"+img)

                imgC = ds.pixel_array[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3]]

            except:

                imgC=np.zeros([np.abs(imgCrop[1]-imgCrop[0]),np.abs(imgCrop[3]-imgCrop[4])])

            imglist.append(imgC)

        

        imgCR=resizeImg(imglist,48,48,48)

        xImg.append(imgCR)

            

    xImg=np.array(xImg)

    

# Setup dataset



feature_num = 9



x = np.zeros([final_df.shape[0], feature_num])

for index, row in final_df.iterrows():

  x[index, 0] = row['Weeks'][0] + 12

  x[index, 1] = row['FVC'][0]

  x[index, 2] = row['Percent'][0]

  x[index, 3] = row['Age']

  x[index, 4] = row['Male']

  x[index, 5] = row['Female']

  x[index, 6] = row['Currently smokes']

  x[index, 7] = row['Ex-smoker']

  x[index, 8] = row['Never smoked']



y = np.zeros((final_df.shape[0], 146))



for index, row in final_df.iterrows():

  y[index, :][np.array(row['Weeks']) + 12] = row['FVC']
from pickle import load



scaler = load(open('/kaggle/input/convmodel/scaler.pkl', 'rb'))

x_scaled=x.copy()

x_scaled[:,:4]=scaler.transform(x_scaled[:,:4])

x_scaled
xt=[]

for k in range(len(xImg)):

    a1=[]

    for j in range(len(xImg[k])):

        rImg=xImg[k][j]/32768*255

        aa=np.dstack((rImg,rImg,rImg))

        a1.append(aa)

    xt.append(a1)

xImg=np.array(xt)

xImg.shape

xImg
model3=tf.keras.models.load_model('/kaggle/input/convmodel/ConvRegResBLK.hdf5',compile=False)
submission=pd.DataFrame()

for i in range(len(y)):

    outPred=np.array(model3.predict([np.array([xImg[i]]),np.array([x_scaled[i]])])[0])

    patientList=[]

    confidenceList=np.ones(len(outPred))*200

    line=createLine(outPred)

    outPred2=[outPred[i] if np.abs(outPred[i]-line[i])<300 else line[i] for i in range(len(outPred))]

    

    for j in range(len(outPred2)):

        patientList.append(final_df.Patient[i]+'_'+str(j-12))

    data=[]

    for j in range(len(outPred2)):

        data.append([patientList[j],outPred2[j],200])

        

    newDF=pd.DataFrame(data,columns=['Patient_Week','FVC','Confidence'])

    if i==0:

        submission=newDF

        print(0)

    else:

        submission=submission.append(newDF)

        print(1)
submission.head()
submission.to_csv('submission.csv',index=False)
outPred
np.mean(outPred)


plt.plot([i*beta+alpha for i in np.array(list(range(-12,134)))])

plt.plot(outPred)
line=[i*beta+alpha for i in np.array(list(range(-12,134)))]
xImg