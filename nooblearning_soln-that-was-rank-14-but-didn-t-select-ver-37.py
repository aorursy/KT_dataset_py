import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom

import os

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor

from skimage import morphology

from skimage import measure

from skimage.transform import resize

import tensorflow as tf

from sklearn.cluster import KMeans

import matplotlib.patches as patches

import tensorflow.keras.backend as k
train_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_csv
###########################  BASE WEEK RECORDED #################################

base_week=train_csv.groupby('Patient')['Weeks'].min()

base_week_list=[]

for i in range(len(train_csv)):

    base_week_list.append(base_week[train_csv.iloc[i,0]])

train_csv['base_week']=base_week_list





###########################  COUNT FROM BASE WEEK RECORDED  #################################

base_week=train_csv.groupby('Patient')['Weeks'].min()

count_from_base_week=[]

for i in range(len(train_csv)):

    count_from_base_week.append(train_csv.iloc[i,1]-base_week[train_csv.iloc[i,0]])

train_csv['count_from_base_week']=count_from_base_week



###########################  CONFIDENCE(FOR MODEL)  #################################

confidence=np.zeros(train_csv.shape[0])

train_csv['confidence']=confidence



###########################  BASE FVC  #################################

base_fvc_dict={}

for id in train_csv['Patient'].unique():

    base_fvc_dict[id]=np.array(train_csv[(train_csv['Patient']==id) & (train_csv['Weeks']==base_week[id])]['FVC'])[0]

base_fvc=[]

for i in range(len(train_csv)):

    base_fvc.append(base_fvc_dict[train_csv.iloc[i,0]])

train_csv['base_fvc']=base_fvc



###########################  BASE FEV1  #################################

base_fev1_dict={}

for id in train_csv['Patient'].unique():

    A=train_csv[train_csv['Patient']==id]["base_fvc"].unique()[0]

    B=train_csv[train_csv['Patient']==id]["Age"].unique()[0]

    if train_csv[train_csv['Patient']==id]["Sex"].unique()[0]=='Male':

        base_fev1_dict[id]=0.77*A+0.32+0.0069*B

    else:

        base_fev1_dict[id]=0.77*A+0.28+0.0052*B

        

base_fev1=[]

for i in range(len(train_csv)):

    base_fev1.append(base_fev1_dict[train_csv.iloc[i,0]])

train_csv['base_fev1']=base_fev1



########################## BASE WEEK PERCENT RECORDED ############################### ONLY USE IF NOT USING PERCENT COLUMN ######

base_week_percent_dict={}

for id in train_csv['Patient'].unique():

    base_week_percent_dict[id]=np.array(train_csv[(train_csv['Patient']==id) & (train_csv['Weeks']==base_week[id])]['Percent'])[0]

    

base_week_percent=[]

for i in range(len(train_csv)):

    base_week_percent.append(base_week_percent_dict[train_csv.iloc[i,0]])

train_csv['base_week_percent']=base_week_percent



######################## BASE FEV1/FVC ####################

train_csv['base fev1/base fvc']=train_csv['base_fev1']/train_csv['base_fvc']



####################### BASE HEIGHT ########################

train_csv['base_height']=(train_csv['base_fvc']+9030)/77.0



###################### BASE WEIGHT #######################



base_weight_dict={}

for id in train_csv['Patient'].unique():

    FVC=train_csv[train_csv['Patient']==id]["base_fvc"].unique()[0]

    A=train_csv[train_csv['Patient']==id]["Age"].unique()[0]

    H=train_csv[train_csv['Patient']==id]["base_height"].unique()[0]

    if train_csv[train_csv['Patient']==id]["Sex"].unique()[0]=='Male':

        base_weight_dict[id]=(FVC+5458-49*H+8*A)/12.0

    else:

        base_weight_dict[id]=(FVC+3863-37*H+6*A)/14.0

base_weight=[]

for i in range(len(train_csv)):

    base_weight.append(base_weight_dict[train_csv.iloc[i,0]])

train_csv['base_weight']=base_weight



###################### BASE BMI ########################

train_csv['base_bmi']=train_csv['base_weight']/((train_csv['base_height']/100)**2)

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()#sex

train_csv.iloc[:,5]=lb.fit_transform(train_csv.iloc[:,5])

lb2=LabelEncoder()#ss

train_csv.iloc[:,6]=lb2.fit_transform(train_csv.iloc[:,6])
from sklearn.preprocessing import OneHotEncoder

#smoking status

oh1=OneHotEncoder(handle_unknown='ignore')

smoke_cat=pd.DataFrame(oh1.fit_transform(train_csv[['SmokingStatus']]).toarray(),columns=['smoking cat 0','smoking cat 1','smoking cat 2'])

train_csv=pd.concat([train_csv,smoke_cat],axis=1)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

train_scaled=pd.DataFrame(sc.fit_transform(train_csv[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])

train_scaled['Sex']=train_csv['Sex']

train_scaled['smoking cat 0']=train_csv['smoking cat 0']

train_scaled['smoking cat 1']=train_csv['smoking cat 1']
train_scaled
train_csv
sub=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test_week=[]

patient_id=[]

for i in range(len(sub)):

    test_week.append(int(sub.iloc[i,0].split('_')[-1]))

    patient_id.append(sub.iloc[i,0].split('_')[0])

sub['Patient']=patient_id

sub['Weeks']=test_week

sub.drop(['FVC','Confidence'],axis=1,inplace=True)



########## BASE FVC ##########

base_fvc=test_csv.groupby('Patient')['FVC'].min()

fvc=[]

for i in range(len(sub)):

    fvc.append(base_fvc[sub.iloc[i,1]])

sub['base_fvc']=fvc



######### BASE FEV1 ##########

base_fev1_dict_test={}

for id in sub['Patient'].unique():

    A=sub[sub['Patient']==id]["base_fvc"].unique()[0]

    B=test_csv[test_csv['Patient']==id]["Age"].unique()[0]

    if test_csv[test_csv['Patient']==id]["Sex"].unique()[0]=='Male':

        base_fev1_dict_test[id]=0.77*A+0.32+0.0069*B

    else:

        base_fev1_dict_test[id]=0.77*A+0.28+0.0052*B

        

base_fev1_test=[]

for i in range(len(sub)):

    base_fev1_test.append(base_fev1_dict_test[sub.iloc[i,1]])

sub['base_fev1']=base_fev1_test



###############################



sub['base_height']=(sub['base_fvc']+9030)/77.0



##############################





base_weight_dict_test={}

for id in sub['Patient'].unique():

    FVC=sub[sub['Patient']==id]["base_fvc"].unique()[0]

    A=test_csv[test_csv['Patient']==id]["Age"].unique()[0]

    H=sub[sub['Patient']==id]["base_height"].unique()[0]

    if test_csv[test_csv['Patient']==id]["Sex"].unique()[0]=='Male':

        base_weight_dict_test[id]=(FVC+5458-49*H+8*A)/12.0

    else:

        base_weight_dict_test[id]=(FVC+3863-37*H+6*A)/14.0

base_weight_test=[]

for i in range(len(sub)):

    base_weight_test.append(base_weight_dict_test[sub.iloc[i,1]])

sub['base_weight']=base_weight_test



##############################



test_csv.iloc[:,5]=lb.transform(test_csv.iloc[:,5])

test_csv.iloc[:,6]=lb2.transform(test_csv.iloc[:,6])



##############################

percent_dict={}

for id in test_csv['Patient'].unique():

    percent_dict[id]=float(test_csv[test_csv['Patient']==id]['Percent'])

    

sex_dict={}

for id in test_csv['Patient'].unique():

    sex_dict[id]=int(test_csv[test_csv['Patient']==id]['Sex'])



age_dict={}

for id in test_csv['Patient'].unique():

    age_dict[id]=int(test_csv[test_csv['Patient']==id]['Age'])

    

ss_dict={}

for id in test_csv['Patient'].unique():

    ss_dict[id]=int(test_csv[test_csv['Patient']==id]['SmokingStatus'])



percent=[]

sex=[]

age=[]

ss=[]

for i in range(len(sub)):

    percent.append(percent_dict[sub.iloc[i,1]])

    sex.append(sex_dict[sub.iloc[i,1]])

    age.append(age_dict[sub.iloc[i,1]])

    ss.append(ss_dict[sub.iloc[i,1]])    

sub['base_week_percent']=percent

sub['Age']=age

sub['Sex']=sex

sub['SmokingStatus']=ss



###############################



base_week_test=test_csv.groupby('Patient')['Weeks'].min()

count_from_base_week_test=[]

base_week=[]

for i in range(len(sub)):

    count_from_base_week_test.append(sub.iloc[i,2]-base_week_test[sub.iloc[i,1]])

    base_week.append(base_week_test[sub.iloc[i,1]])

sub['count_from_base_week']=count_from_base_week_test

sub['base_week']=base_week



###############################



sub['base fev1/base fvc']=sub['base_fev1']/sub['base_fvc']



###############################

sub['base_bmi']=sub['base_weight']/((sub['base_height']/100.0)**2)
smoke_cat_test=pd.DataFrame(oh1.transform(sub[['SmokingStatus']]).toarray(),columns=['smoking cat 0','smoking cat 1','smoking cat 2'])

sub=pd.concat([sub,smoke_cat_test],axis=1)
sub_scaled=pd.DataFrame(sc.transform(sub[['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi']]),columns=['Weeks','Age','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi'])

sub_scaled['Sex']=sub['Sex']

sub_scaled['smoking cat 0']=sub['smoking cat 0']

sub_scaled['smoking cat 1']=sub['smoking cat 1']
sub_scaled
#x=np.array(train_csv[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])

#x=np.array(train_csv[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])

#x=np.array(train_scaled[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])

x=np.array(train_scaled[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])

y=np.array(train_csv[['FVC','confidence']])



from sklearn.model_selection import train_test_split

xtrain,xvalid,ytrain,yvalid=train_test_split(x,y,test_size=0.2)
def metric(actual_fvc, predicted_fvc, confidence, return_values = False):

    """

    Calculates the modified Laplace Log Likelihood score for this competition.

    """

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)



    if return_values:

        return metric

    else:

        return np.mean(metric)

    

    

def model_loss(ytrue,ypred):   # this loss penalises both prediction and confidence value

    eps=1.0

    

    fvc_pred=ypred[:,0]

    sigmas=ypred[:,1]+eps     # so as to avoid log(0) . these are predicted connfidences

    

    ans=tf.math.log(sigmas)

    ans=ans+((ytrue[:,0]-fvc_pred)**2)/(2*sigmas**2)

    

    return tf.reduce_mean(ans)
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")



def score(y_true, y_pred):

    

    sigma = y_pred[:,1]

    fvc_pred = y_pred[:,0]

    



    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:,0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return tf.reduce_mean(metric)



def huber_loss(y_true, y_pred):

    

    error = y_true[:,0] - y_pred[:,0]

    is_small_error = tf.abs(error) <= 1000.0

    quad_loss = tf.square(error) / 2

    linear_loss = 1000*tf.abs(error) - tf.square(C2)*0.5

    return tf.reduce_mean(tf.where(is_small_error, quad_loss, linear_loss))





def custom_loss(y_true, y_pred):

    return 0.01*huber_loss(y_true, y_pred) + score(y_true, y_pred)
lr_scheduler=tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,monitor='val_loss',mode='min',patience=150,verbose=0)

class best_weights(tf.keras.callbacks.Callback):

    def __init__(self):

        self.metric_op=-30.0

        self.weights_op=None

        self.epoch_op=-1

    def on_epoch_end(self,epoch,logs={}):

        if logs['val_metric']>=self.metric_op:

            self.metric_op=logs['val_metric']

            self.epoch_op=epoch

            self.weights_op=self.model.get_weights()

    def on_train_end(self,logs={}):

        self.model.set_weights(self.weights_op)

        print('BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}'.format(self.epoch_op+1,self.metric_op))

        

        



class metrics_call(tf.keras.callbacks.Callback):

    def __init__(self,mertic,xtrain,ytrain,xvalid,yvalid):

        self.metric=metric

        self.xtrain=xtrain

        self.ytrain=ytrain

        self.xvalid=xvalid

        self.yvalid=yvalid

        

    def on_epoch_end(self,epoch,logs={}):

        train_preds=self.model.predict(self.xtrain)

        val_preds=self.model.predict(self.xvalid)

        #print('\r  metric on train set: ',self.metric(self.ytrain[:,0],train_preds[:,0],train_preds[:,1]),end='')

        logs['val_metric']=self.metric(self.yvalid[:,0],val_preds[:,0],val_preds[:,1])

        #print('  metric on valid set: ',self.metric(self.yvalid[:,0],val_preds[:,0],val_preds[:,1]))

        

        



def run_model(xtrain,ytrain,xvalid,yvalid,epoch=50):

    input=tf.keras.layers.Input(shape=xtrain.shape[1:])

    noisy=tf.keras.layers.GaussianNoise(0.6)(input)

    

    d1=tf.keras.layers.Dense(128,activation='relu')(noisy)

    d2=tf.keras.layers.Dense(128,activation='relu')(d1)

    d3=tf.keras.layers.Dense(128,activation='relu')(d2)

    mean_out1=tf.keras.layers.Dense(1)(d3)

    std_den1=tf.keras.layers.Dense(1)(d3)

    

    d4=tf.keras.layers.Dense(128,activation='relu')(noisy)

    d5=tf.keras.layers.Dense(128,activation='relu')(d4)

    d6=tf.keras.layers.Dense(128,activation='relu')(d5)

    mean_out2=tf.keras.layers.Dense(1)(d6)

    std_den2=tf.keras.layers.Dense(1)(d6)

    

    d7=tf.keras.layers.Dense(128,activation='relu')(noisy)

    d8=tf.keras.layers.Dense(128,activation='relu')(d7)

    d9=tf.keras.layers.Dense(128,activation='relu')(d8)

    mean_out3=tf.keras.layers.Dense(1)(d9)

    std_den3=tf.keras.layers.Dense(1)(d9)

    

    mean_combine=tf.keras.layers.Concatenate()([mean_out1,mean_out2,mean_out3])

    std_combine=tf.keras.layers.Concatenate()([std_den1,std_den2,std_den3])

    mean_final=tf.keras.layers.Dense(1)(mean_combine)

    std_final_den=tf.keras.layers.Dense(1)(std_combine)

    std_final=tf.keras.layers.Lambda(lambda x: tf.abs(x))(std_final_den)

    output=tf.keras.layers.Concatenate()([mean_final,std_final])

    model=tf.keras.models.Model(inputs=input,outputs=output)

    

    model.compile(loss=lambda ytrue,ypred: custom_loss(ytrue,ypred),optimizer=tf.keras.optimizers.Adam(lr=0.0001))

    print(model.summary())

    history=model.fit(xtrain,ytrain,epochs=epoch,batch_size=256,validation_data=(xvalid,yvalid),verbose=0,callbacks=[metrics_call(metric,xtrain,ytrain,xvalid,yvalid),best_weights()])

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.ylim(-10,10)

    plt.grid(True)



    return model
model=run_model(xtrain,ytrain,xvalid,yvalid,epoch=1000)
#xtest=np.array(sub[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])

#xtest=np.array(sub[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])

xtest=np.array(sub_scaled[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1']])

#xtest=np.array(sub_scaled[['Weeks','Age','Sex','base_week','count_from_base_week','base_fvc','base_week_percent','base_height','smoking cat 0','smoking cat 1']])



yans=model.predict(xtest)



sub['FVC']=yans[:,0]

sub['Confidence']=yans[:,1]

sub.drop(['Patient','Weeks','Age','Sex','SmokingStatus','base_week','count_from_base_week','base_fvc','base_fev1','base_week_percent','base fev1/base fvc','base_height','base_weight','base_bmi','smoking cat 0','smoking cat 1','smoking cat 2'],axis=1,inplace=True)
sub
sub.to_csv('submission.csv',index=False)