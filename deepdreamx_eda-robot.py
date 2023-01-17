#Importing Libraries

import warnings

warnings.simplefilter("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy as sc

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm_notebook

from sklearn.metrics import f1_score

import xgboost

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix

import os

from bayes_opt import BayesianOptimization

print(os.listdir("../input"))
#First i will be reading the train dataset

train_x=pd.read_csv("../input/X_train.csv")

train_y=pd.read_csv("../input/y_train.csv")

train_x.shape,train_y.shape
#Top rows

train_x.head()
train_y.head()
#We have around 3810 Series in total in Train data

len(train_x.series_id.unique())
#We have now verified each contunuos series (corresponding to robot) contains 128 values

pd.value_counts(train_x.series_id).unique()
plt.figure(figsize=(10,4))

sns.distplot(train_x.orientation_X)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.orientation_Y)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.orientation_Z)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.orientation_W)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.orientation_X.values,label='X')

sns.distplot(train_x.orientation_Y.values,label='Y')

sns.distplot(train_x.orientation_Z.values,label='Z')

sns.distplot(train_x.orientation_W.values,label='W')

plt.legend()

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.angular_velocity_X)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.angular_velocity_Y)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.angular_velocity_Z)

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.angular_velocity_X.values,label='X')

sns.distplot(train_x.angular_velocity_Y.values,label='Y')

sns.distplot(train_x.angular_velocity_Z.values,label='Z')

plt.legend()

plt.show()
plt.figure(figsize=(10,4))

sns.distplot(train_x.linear_acceleration_X.values,label='X')

sns.distplot(train_x.linear_acceleration_Y.values,label='Y')

sns.distplot(train_x.linear_acceleration_Z.values,label='Z')

plt.legend()

plt.show()
#First joining the train_x and train_y (i.e labels to the dataset)

df=pd.DataFrame.merge(train_x,train_y.loc[:,['series_id','surface']],on='series_id')

df.shape
#Top rows

df.head()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "orientation_X")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "orientation_X")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "orientation_Y")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "orientation_Y")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "orientation_Z")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "orientation_Z")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "orientation_W")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "orientation_W")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "angular_velocity_X")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "angular_velocity_X")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "angular_velocity_Y")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "angular_velocity_Y")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "angular_velocity_Z")
g = sns.FacetGrid(df, hue="surface", height=7,aspect=2)

g = g.map(sns.kdeplot, "angular_velocity_Z")

g=g.add_legend()
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "linear_acceleration_X")
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "linear_acceleration_Y")
g = sns.FacetGrid(df, col="surface",col_wrap=3)

g = g.map(sns.distplot, "linear_acceleration_Z")
plt.figure(figsize=(10,5))

sns.scatterplot(x='orientation_X',y='orientation_Y',hue='surface',data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.scatterplot(x='orientation_X',y='angular_velocity_X',hue='surface',data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(10,5))

sns.scatterplot(x='orientation_X',y='angular_velocity_Y',hue='surface',data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(10,5))

sns.scatterplot(x='angular_velocity_X',y='linear_acceleration_X',hue='surface',data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#Firstly checking if classes are balanced or not

plt.figure(figsize=(24,8))

sns.countplot(x='surface',data=df)

plt.show()
del df['row_id']
df.head()
def CPT5(x):

    den = len(x)*np.exp(np.std(x))

    return sum(np.exp(x))/den



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    xn_i1 = x[0:len(x)-2]  # xn-1

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))
#Now converting time series to ML Task

X=pd.DataFrame()

Y=[]

total_series=train_y.series_id.max()+1

c=list(df.columns)

c.remove('series_id')

c.remove('surface')

c.remove('measurement_number')



for s in tqdm_notebook(range(total_series)):

    mask=df.series_id==s

    Y.append((df.loc[mask,'surface']).values[0])

    

    X.loc[s,'orx_mean']=df.loc[mask,'orientation_X'].mean()

    X.loc[s,'orx_std']=df.loc[mask,'orientation_X'].std()

    X.loc[s,'orx_min']=df.loc[mask,'orientation_X'].min()

    X.loc[s,'orx_max']=df.loc[mask,'orientation_X'].max()

    X.loc[s,'orx_diff_mean']=df.loc[mask,'orientation_X'].diff().mean()

    X.loc[s,'orx_diff_min']=df.loc[mask,'orientation_X'].diff().min()

    X.loc[s,'orx_diff_max']=df.loc[mask,'orientation_X'].diff().max()

    X.loc[s,'orx_kur']=df.loc[mask,'orientation_X'].skew()

    X.loc[s,'orx_skew']=df.loc[mask,'orientation_X'].kurtosis()

    

    #Some Extra

    X.loc[s,'orx_mean']=df.loc[mask,'orientation_X'].abs().mean()

    X.loc[s,'orx_CPT5']=CPT5(df.loc[mask,'orientation_X'].values)

    X.loc[s,'orx_SSC']=SSC(df.loc[mask,'orientation_X'].values)

    X.loc[s,'orx_wavelength']=wave_length(df.loc[mask,'orientation_X'].values)

    X.loc[s,'orx_normentropy']=norm_entropy(df.loc[mask,'orientation_X'].values)

    X.loc[s,'orx_SRAV']=SRAV(df.loc[mask,'orientation_X'].values)

    X.loc[s,'orx_zerocrossing']=zero_crossing(df.loc[mask,'orientation_X'].values)



    X.loc[s,'ory_mean']=df.loc[mask,'orientation_Y'].mean()

    X.loc[s,'ory_std']=df.loc[mask,'orientation_Y'].std()

    X.loc[s,'ory_min']=df.loc[mask,'orientation_Y'].min()

    X.loc[s,'ory_max']=df.loc[mask,'orientation_Y'].max()

    X.loc[s,'ory_diff_mean']=df.loc[mask,'orientation_Y'].diff().mean()

    X.loc[s,'ory_diff_min']=df.loc[mask,'orientation_Y'].diff().min()

    X.loc[s,'ory_diff_max']=df.loc[mask,'orientation_Y'].diff().max()

    X.loc[s,'ory_kur']=df.loc[mask,'orientation_Y'].skew()

    X.loc[s,'ory_skew']=df.loc[mask,'orientation_Y'].kurtosis()

    

     #Some Extra

    X.loc[s,'ory_mean']=df.loc[mask,'orientation_Y'].abs().mean()

    X.loc[s,'ory_CPT5']=CPT5(df.loc[mask,'orientation_Y'].values)

    X.loc[s,'ory_SSC']=SSC(df.loc[mask,'orientation_Y'].values)

    X.loc[s,'ory_wavelength']=wave_length(df.loc[mask,'orientation_Y'].values)

    X.loc[s,'ory_normentropy']=norm_entropy(df.loc[mask,'orientation_Y'].values)

    X.loc[s,'ory_SRAV']=SRAV(df.loc[mask,'orientation_Y'].values)

    X.loc[s,'ory_zerocrossing']=zero_crossing(df.loc[mask,'orientation_Y'].values)

    

    X.loc[s,'orz_mean']=df.loc[mask,'orientation_Z'].mean()

    X.loc[s,'orz_std']=df.loc[mask,'orientation_Z'].std()

    X.loc[s,'orz_min']=df.loc[mask,'orientation_Z'].min()

    X.loc[s,'orz_max']=df.loc[mask,'orientation_Z'].max()

    X.loc[s,'orz_diff_mean']=df.loc[mask,'orientation_Z'].diff().mean()

    X.loc[s,'orz_diff_min']=df.loc[mask,'orientation_Z'].diff().min()

    X.loc[s,'orz_diff_max']=df.loc[mask,'orientation_Z'].diff().max()

    X.loc[s,'orz_kur']=df.loc[mask,'orientation_Z'].skew()

    X.loc[s,'orz_skew']=df.loc[mask,'orientation_Z'].kurtosis()

    

     #Some Extra

    X.loc[s,'orz_mean']=df.loc[mask,'orientation_Z'].abs().mean()

    X.loc[s,'orz_CPT5']=CPT5(df.loc[mask,'orientation_Z'].values)

    X.loc[s,'orz_SSC']=SSC(df.loc[mask,'orientation_Z'].values)

    X.loc[s,'orz_wavelength']=wave_length(df.loc[mask,'orientation_Z'].values)

    X.loc[s,'orz_normentropy']=norm_entropy(df.loc[mask,'orientation_Z'].values)

    X.loc[s,'orz_SRAV']=SRAV(df.loc[mask,'orientation_Z'].values)

    X.loc[s,'orz_zerocrossing']=zero_crossing(df.loc[mask,'orientation_Z'].values)

    

    X.loc[s,'orw_mean']=df.loc[mask,'orientation_W'].mean()

    X.loc[s,'orw_std']=df.loc[mask,'orientation_W'].std()

    X.loc[s,'orw_min']=df.loc[mask,'orientation_W'].min()

    X.loc[s,'orw_max']=df.loc[mask,'orientation_W'].max()

    X.loc[s,'orw_diff_mean']=df.loc[mask,'orientation_W'].diff().mean()

    X.loc[s,'orw_diff_min']=df.loc[mask,'orientation_W'].diff().min()

    X.loc[s,'orw_diff_max']=df.loc[mask,'orientation_W'].diff().max()

    X.loc[s,'orw_kur']=df.loc[mask,'orientation_W'].skew()

    X.loc[s,'orw_skew']=df.loc[mask,'orientation_W'].kurtosis()

    

     #Some Extra

    X.loc[s,'orw_mean']=df.loc[mask,'orientation_W'].abs().mean()

    X.loc[s,'orw_CPT5']=CPT5(df.loc[mask,'orientation_W'].values)

    X.loc[s,'orw_SSC']=SSC(df.loc[mask,'orientation_W'].values)

    X.loc[s,'orw_wavelength']=wave_length(df.loc[mask,'orientation_W'].values)

    X.loc[s,'orw_normentropy']=norm_entropy(df.loc[mask,'orientation_W'].values)

    X.loc[s,'orw_SRAV']=SRAV(df.loc[mask,'orientation_W'].values)

    X.loc[s,'orw_Werocrossing']=zero_crossing(df.loc[mask,'orientation_W'].values)

    

    X.loc[s,'angx_mean']=df.loc[mask,'angular_velocity_X'].mean()

    X.loc[s,'angx_std']=df.loc[mask,'angular_velocity_X'].std()

    X.loc[s,'angx_min']=df.loc[mask,'angular_velocity_X'].min()

    X.loc[s,'angx_max']=df.loc[mask,'angular_velocity_X'].max()

    X.loc[s,'angx_diff_mean']=df.loc[mask,'angular_velocity_X'].diff().mean()

    X.loc[s,'angx_diff_min']=df.loc[mask,'angular_velocity_X'].diff().min()

    X.loc[s,'angx_diff_max']=df.loc[mask,'angular_velocity_X'].diff().max()

    X.loc[s,'angx_kur']=df.loc[mask,'angular_velocity_X'].skew()

    X.loc[s,'angx_skew']=df.loc[mask,'angular_velocity_X'].kurtosis()

    

    #Some Extra

    X.loc[s,'angx_mean']=df.loc[mask,'angular_velocity_X'].abs().mean()

    X.loc[s,'angx_CPT5']=CPT5(df.loc[mask,'angular_velocity_X'].values)

    X.loc[s,'angx_SSC']=SSC(df.loc[mask,'angular_velocity_X'].values)

    X.loc[s,'angx_wavelength']=wave_length(df.loc[mask,'angular_velocity_X'].values)

    X.loc[s,'angx_normentropy']=norm_entropy(df.loc[mask,'angular_velocity_X'].values)

    X.loc[s,'angx_SRAV']=SRAV(df.loc[mask,'angular_velocity_X'].values)

    X.loc[s,'angx_Werocrossing']=zero_crossing(df.loc[mask,'angular_velocity_X'].values)

    

    X.loc[s,'angy_mean']=df.loc[mask,'angular_velocity_Y'].mean()

    X.loc[s,'angy_std']=df.loc[mask,'angular_velocity_Y'].std()

    X.loc[s,'angy_min']=df.loc[mask,'angular_velocity_Y'].min()

    X.loc[s,'angy_max']=df.loc[mask,'angular_velocity_Y'].max()

    X.loc[s,'angy_diff_mean']=df.loc[mask,'angular_velocity_Y'].diff().mean()

    X.loc[s,'angy_diff_min']=df.loc[mask,'angular_velocity_Y'].diff().min()

    X.loc[s,'angy_diff_max']=df.loc[mask,'angular_velocity_Y'].diff().max()

    X.loc[s,'angy_kur']=df.loc[mask,'angular_velocity_Y'].skew()

    X.loc[s,'angy_skew']=df.loc[mask,'angular_velocity_Y'].kurtosis()

    

    #Some Extra

    X.loc[s,'angy_mean']=df.loc[mask,'angular_velocity_Y'].abs().mean()

    X.loc[s,'angy_CPT5']=CPT5(df.loc[mask,'angular_velocity_Y'].values)

    X.loc[s,'angy_SSC']=SSC(df.loc[mask,'angular_velocity_Y'].values)

    X.loc[s,'angy_wavelength']=wave_length(df.loc[mask,'angular_velocity_Y'].values)

    X.loc[s,'angy_normentropy']=norm_entropy(df.loc[mask,'angular_velocity_Y'].values)

    X.loc[s,'angy_SRAV']=SRAV(df.loc[mask,'angular_velocity_Y'].values)

    X.loc[s,'angy_Werocrossing']=zero_crossing(df.loc[mask,'angular_velocity_Y'].values)

    

    X.loc[s,'angz_mean']=df.loc[mask,'angular_velocity_Z'].mean()

    X.loc[s,'angz_std']=df.loc[mask,'angular_velocity_Z'].std()

    X.loc[s,'angz_min']=df.loc[mask,'angular_velocity_Z'].min()

    X.loc[s,'angz_max']=df.loc[mask,'angular_velocity_Z'].max()

    X.loc[s,'angz_diff_mean']=df.loc[mask,'angular_velocity_Z'].diff().mean()

    X.loc[s,'angz_diff_min']=df.loc[mask,'angular_velocity_Z'].diff().min()

    X.loc[s,'angz_diff_max']=df.loc[mask,'angular_velocity_Z'].diff().max()

    X.loc[s,'angz_kur']=df.loc[mask,'angular_velocity_Z'].skew()

    X.loc[s,'angz_skew']=df.loc[mask,'angular_velocity_Z'].kurtosis()

    

    #Some Extra

    X.loc[s,'angz_mean']=df.loc[mask,'angular_velocity_Z'].abs().mean()

    X.loc[s,'angz_CPT5']=CPT5(df.loc[mask,'angular_velocity_Z'].values)

    X.loc[s,'angz_SSC']=SSC(df.loc[mask,'angular_velocity_Z'].values)

    X.loc[s,'angz_wavelength']=wave_length(df.loc[mask,'angular_velocity_Z'].values)

    X.loc[s,'angz_normentropy']=norm_entropy(df.loc[mask,'angular_velocity_Z'].values)

    X.loc[s,'angz_SRAV']=SRAV(df.loc[mask,'angular_velocity_Z'].values)

    X.loc[s,'angz_Werocrossing']=zero_crossing(df.loc[mask,'angular_velocity_Z'].values)

    

    X.loc[s,'linx_mean']=df.loc[mask,'linear_acceleration_X'].mean()

    X.loc[s,'linx_std']=df.loc[mask,'linear_acceleration_X'].std()

    X.loc[s,'linx_min']=df.loc[mask,'linear_acceleration_X'].min()

    X.loc[s,'linx_max']=df.loc[mask,'linear_acceleration_X'].max()

    X.loc[s,'linx_diff_mean']=df.loc[mask,'linear_acceleration_X'].diff().mean()

    X.loc[s,'linx_diff_min']=df.loc[mask,'linear_acceleration_X'].diff().min()

    X.loc[s,'linx_diff_max']=df.loc[mask,'linear_acceleration_X'].diff().max()

    X.loc[s,'linx_kur']=df.loc[mask,'linear_acceleration_X'].skew()

    X.loc[s,'linx_skew']=df.loc[mask,'linear_acceleration_X'].kurtosis()

    

    #Some Extra

    X.loc[s,'linx_mean']=df.loc[mask,'linear_acceleration_X'].abs().mean()

    X.loc[s,'linx_CPT5']=CPT5(df.loc[mask,'linear_acceleration_X'].values)

    X.loc[s,'linx_SSC']=SSC(df.loc[mask,'linear_acceleration_X'].values)

    X.loc[s,'linx_wavelength']=wave_length(df.loc[mask,'linear_acceleration_X'].values)

    X.loc[s,'linx_normentropy']=norm_entropy(df.loc[mask,'linear_acceleration_X'].values)

    X.loc[s,'linx_SRAV']=SRAV(df.loc[mask,'linear_acceleration_X'].values)

    X.loc[s,'linx_Werocrossing']=zero_crossing(df.loc[mask,'linear_acceleration_X'].values)

    

    X.loc[s,'liny_mean']=df.loc[mask,'linear_acceleration_Y'].mean()

    X.loc[s,'liny_std']=df.loc[mask,'linear_acceleration_Y'].std()

    X.loc[s,'liny_min']=df.loc[mask,'linear_acceleration_Y'].min()

    X.loc[s,'liny_max']=df.loc[mask,'linear_acceleration_Y'].max()

    X.loc[s,'liny_diff_mean']=df.loc[mask,'linear_acceleration_Y'].diff().mean()

    X.loc[s,'liny_diff_min']=df.loc[mask,'linear_acceleration_Y'].diff().min()

    X.loc[s,'liny_diff_max']=df.loc[mask,'linear_acceleration_Y'].diff().max()

    X.loc[s,'liny_kur']=df.loc[mask,'linear_acceleration_Y'].skew()

    X.loc[s,'liny_skew']=df.loc[mask,'linear_acceleration_Y'].kurtosis()

    

    #Some Extra

    X.loc[s,'liny_mean']=df.loc[mask,'linear_acceleration_Y'].abs().mean()

    X.loc[s,'liny_CPT5']=CPT5(df.loc[mask,'linear_acceleration_Y'].values)

    X.loc[s,'liny_SSC']=SSC(df.loc[mask,'linear_acceleration_Y'].values)

    X.loc[s,'liny_wavelength']=wave_length(df.loc[mask,'linear_acceleration_Y'].values)

    X.loc[s,'liny_normentropy']=norm_entropy(df.loc[mask,'linear_acceleration_Y'].values)

    X.loc[s,'liny_SRAV']=SRAV(df.loc[mask,'linear_acceleration_Y'].values)

    X.loc[s,'liny_Werocrossing']=zero_crossing(df.loc[mask,'linear_acceleration_Y'].values)

    

    X.loc[s,'linz_mean']=df.loc[mask,'linear_acceleration_Z'].mean()

    X.loc[s,'linz_std']=df.loc[mask,'linear_acceleration_Z'].std()

    X.loc[s,'linz_min']=df.loc[mask,'linear_acceleration_Z'].min()

    X.loc[s,'linz_max']=df.loc[mask,'linear_acceleration_Z'].max()

    X.loc[s,'linz_diff_mean']=df.loc[mask,'linear_acceleration_Z'].diff().mean()

    X.loc[s,'linz_diff_min']=df.loc[mask,'linear_acceleration_Z'].diff().min()

    X.loc[s,'linz_diff_max']=df.loc[mask,'linear_acceleration_Z'].diff().max()

    X.loc[s,'linz_kur']=df.loc[mask,'linear_acceleration_Z'].skew()

    X.loc[s,'linz_skew']=df.loc[mask,'linear_acceleration_Z'].kurtosis()

    

    #Some Extra

    X.loc[s,'linz_mean']=df.loc[mask,'linear_acceleration_Z'].abs().mean()

    X.loc[s,'linz_CPT5']=CPT5(df.loc[mask,'linear_acceleration_Z'].values)

    X.loc[s,'linz_SSC']=SSC(df.loc[mask,'linear_acceleration_Z'].values)

    X.loc[s,'linz_wavelength']=wave_length(df.loc[mask,'linear_acceleration_Z'].values)

    X.loc[s,'linz_normentropy']=norm_entropy(df.loc[mask,'linear_acceleration_Z'].values)

    X.loc[s,'linz_SRAV']=SRAV(df.loc[mask,'linear_acceleration_Z'].values)

    X.loc[s,'linz_Werocrossing']=zero_crossing(df.loc[mask,'linear_acceleration_Z'].values)

Y=np.array(Y)
X.shape,Y.shape
X.head()
#Predict on Test Set Start Initilization
X_final=pd.read_csv("../input/X_test.csv")

X_final.shape
X_final.head()
#Now converting time series to ML Task

X_predict=pd.DataFrame()

total_series=X_final.series_id.max()+1

c=list(X_final.columns)

c.remove('row_id')

c.remove('series_id')

c.remove('measurement_number')



for s in tqdm_notebook(range(total_series)):

    mask=X_final.series_id==s

    

    X_predict.loc[s,'orx_mean']=X_final.loc[mask,'orientation_X'].mean()

    X_predict.loc[s,'orx_std']=X_final.loc[mask,'orientation_X'].std()

    X_predict.loc[s,'orx_min']=X_final.loc[mask,'orientation_X'].min()

    X_predict.loc[s,'orx_max']=X_final.loc[mask,'orientation_X'].max()

    X_predict.loc[s,'orx_diff_mean']=X_final.loc[mask,'orientation_X'].diff().mean()

    X_predict.loc[s,'orx_diff_min']=X_final.loc[mask,'orientation_X'].diff().min()

    X_predict.loc[s,'orx_diff_max']=X_final.loc[mask,'orientation_X'].diff().max()

    X_predict.loc[s,'orx_kur']=X_final.loc[mask,'orientation_X'].skew()

    X_predict.loc[s,'orx_skew']=X_final.loc[mask,'orientation_X'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'orx_mean']=X_final.loc[mask,'orientation_X'].abs().mean()

    X_predict.loc[s,'orx_CPT5']=CPT5(X_final.loc[mask,'orientation_X'].values)

    X_predict.loc[s,'orx_SSC']=SSC(X_final.loc[mask,'orientation_X'].values)

    X_predict.loc[s,'orx_wavelength']=wave_length(X_final.loc[mask,'orientation_X'].values)

    X_predict.loc[s,'orx_normentropy']=norm_entropy(X_final.loc[mask,'orientation_X'].values)

    X_predict.loc[s,'orx_SRAV']=SRAV(X_final.loc[mask,'orientation_X'].values)

    X_predict.loc[s,'orx_zerocrossing']=zero_crossing(X_final.loc[mask,'orientation_X'].values)



    X_predict.loc[s,'ory_mean']=X_final.loc[mask,'orientation_Y'].mean()

    X_predict.loc[s,'ory_std']=X_final.loc[mask,'orientation_Y'].std()

    X_predict.loc[s,'ory_min']=X_final.loc[mask,'orientation_Y'].min()

    X_predict.loc[s,'ory_max']=X_final.loc[mask,'orientation_Y'].max()

    X_predict.loc[s,'ory_diff_mean']=X_final.loc[mask,'orientation_Y'].diff().mean()

    X_predict.loc[s,'ory_diff_min']=X_final.loc[mask,'orientation_Y'].diff().min()

    X_predict.loc[s,'ory_diff_max']=X_final.loc[mask,'orientation_Y'].diff().max()

    X_predict.loc[s,'ory_kur']=X_final.loc[mask,'orientation_Y'].skew()

    X_predict.loc[s,'ory_skew']=X_final.loc[mask,'orientation_Y'].kurtosis()

    

     #Some Extra

    X_predict.loc[s,'ory_mean']=X_final.loc[mask,'orientation_Y'].abs().mean()

    X_predict.loc[s,'ory_CPT5']=CPT5(X_final.loc[mask,'orientation_Y'].values)

    X_predict.loc[s,'ory_SSC']=SSC(X_final.loc[mask,'orientation_Y'].values)

    X_predict.loc[s,'ory_wavelength']=wave_length(X_final.loc[mask,'orientation_Y'].values)

    X_predict.loc[s,'ory_normentropy']=norm_entropy(X_final.loc[mask,'orientation_Y'].values)

    X_predict.loc[s,'ory_SRAV']=SRAV(X_final.loc[mask,'orientation_Y'].values)

    X_predict.loc[s,'ory_zerocrossing']=zero_crossing(X_final.loc[mask,'orientation_Y'].values)

    

    X_predict.loc[s,'orz_mean']=X_final.loc[mask,'orientation_Z'].mean()

    X_predict.loc[s,'orz_std']=X_final.loc[mask,'orientation_Z'].std()

    X_predict.loc[s,'orz_min']=X_final.loc[mask,'orientation_Z'].min()

    X_predict.loc[s,'orz_max']=X_final.loc[mask,'orientation_Z'].max()

    X_predict.loc[s,'orz_diff_mean']=X_final.loc[mask,'orientation_Z'].diff().mean()

    X_predict.loc[s,'orz_diff_min']=X_final.loc[mask,'orientation_Z'].diff().min()

    X_predict.loc[s,'orz_diff_max']=X_final.loc[mask,'orientation_Z'].diff().max()

    X_predict.loc[s,'orz_kur']=X_final.loc[mask,'orientation_Z'].skew()

    X_predict.loc[s,'orz_skew']=X_final.loc[mask,'orientation_Z'].kurtosis()

    

     #Some Extra

    X_predict.loc[s,'orz_mean']=X_final.loc[mask,'orientation_Z'].abs().mean()

    X_predict.loc[s,'orz_CPT5']=CPT5(X_final.loc[mask,'orientation_Z'].values)

    X_predict.loc[s,'orz_SSC']=SSC(X_final.loc[mask,'orientation_Z'].values)

    X_predict.loc[s,'orz_wavelength']=wave_length(X_final.loc[mask,'orientation_Z'].values)

    X_predict.loc[s,'orz_normentropy']=norm_entropy(X_final.loc[mask,'orientation_Z'].values)

    X_predict.loc[s,'orz_SRAV']=SRAV(X_final.loc[mask,'orientation_Z'].values)

    X_predict.loc[s,'orz_zerocrossing']=zero_crossing(X_final.loc[mask,'orientation_Z'].values)

    

    X_predict.loc[s,'orw_mean']=X_final.loc[mask,'orientation_W'].mean()

    X_predict.loc[s,'orw_std']=X_final.loc[mask,'orientation_W'].std()

    X_predict.loc[s,'orw_min']=X_final.loc[mask,'orientation_W'].min()

    X_predict.loc[s,'orw_max']=X_final.loc[mask,'orientation_W'].max()

    X_predict.loc[s,'orw_diff_mean']=X_final.loc[mask,'orientation_W'].diff().mean()

    X_predict.loc[s,'orw_diff_min']=X_final.loc[mask,'orientation_W'].diff().min()

    X_predict.loc[s,'orw_diff_max']=X_final.loc[mask,'orientation_W'].diff().max()

    X_predict.loc[s,'orw_kur']=X_final.loc[mask,'orientation_W'].skew()

    X_predict.loc[s,'orw_skew']=X_final.loc[mask,'orientation_W'].kurtosis()

    

     #Some Extra

    X_predict.loc[s,'orw_mean']=X_final.loc[mask,'orientation_W'].abs().mean()

    X_predict.loc[s,'orw_CPT5']=CPT5(X_final.loc[mask,'orientation_W'].values)

    X_predict.loc[s,'orw_SSC']=SSC(X_final.loc[mask,'orientation_W'].values)

    X_predict.loc[s,'orw_wavelength']=wave_length(X_final.loc[mask,'orientation_W'].values)

    X_predict.loc[s,'orw_normentropy']=norm_entropy(X_final.loc[mask,'orientation_W'].values)

    X_predict.loc[s,'orw_SRAV']=SRAV(X_final.loc[mask,'orientation_W'].values)

    X_predict.loc[s,'orw_Werocrossing']=zero_crossing(X_final.loc[mask,'orientation_W'].values)

    

    X_predict.loc[s,'angx_mean']=X_final.loc[mask,'angular_velocity_X'].mean()

    X_predict.loc[s,'angx_std']=X_final.loc[mask,'angular_velocity_X'].std()

    X_predict.loc[s,'angx_min']=X_final.loc[mask,'angular_velocity_X'].min()

    X_predict.loc[s,'angx_max']=X_final.loc[mask,'angular_velocity_X'].max()

    X_predict.loc[s,'angx_diff_mean']=X_final.loc[mask,'angular_velocity_X'].diff().mean()

    X_predict.loc[s,'angx_diff_min']=X_final.loc[mask,'angular_velocity_X'].diff().min()

    X_predict.loc[s,'angx_diff_max']=X_final.loc[mask,'angular_velocity_X'].diff().max()

    X_predict.loc[s,'angx_kur']=X_final.loc[mask,'angular_velocity_X'].skew()

    X_predict.loc[s,'angx_skew']=X_final.loc[mask,'angular_velocity_X'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'angx_mean']=X_final.loc[mask,'angular_velocity_X'].abs().mean()

    X_predict.loc[s,'angx_CPT5']=CPT5(X_final.loc[mask,'angular_velocity_X'].values)

    X_predict.loc[s,'angx_SSC']=SSC(X_final.loc[mask,'angular_velocity_X'].values)

    X_predict.loc[s,'angx_wavelength']=wave_length(X_final.loc[mask,'angular_velocity_X'].values)

    X_predict.loc[s,'angx_normentropy']=norm_entropy(X_final.loc[mask,'angular_velocity_X'].values)

    X_predict.loc[s,'angx_SRAV']=SRAV(X_final.loc[mask,'angular_velocity_X'].values)

    X_predict.loc[s,'angx_Werocrossing']=zero_crossing(X_final.loc[mask,'angular_velocity_X'].values)

    

    X_predict.loc[s,'angy_mean']=X_final.loc[mask,'angular_velocity_Y'].mean()

    X_predict.loc[s,'angy_std']=X_final.loc[mask,'angular_velocity_Y'].std()

    X_predict.loc[s,'angy_min']=X_final.loc[mask,'angular_velocity_Y'].min()

    X_predict.loc[s,'angy_max']=X_final.loc[mask,'angular_velocity_Y'].max()

    X_predict.loc[s,'angy_diff_mean']=X_final.loc[mask,'angular_velocity_Y'].diff().mean()

    X_predict.loc[s,'angy_diff_min']=X_final.loc[mask,'angular_velocity_Y'].diff().min()

    X_predict.loc[s,'angy_diff_max']=X_final.loc[mask,'angular_velocity_Y'].diff().max()

    X_predict.loc[s,'angy_kur']=X_final.loc[mask,'angular_velocity_Y'].skew()

    X_predict.loc[s,'angy_skew']=X_final.loc[mask,'angular_velocity_Y'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'angy_mean']=X_final.loc[mask,'angular_velocity_Y'].abs().mean()

    X_predict.loc[s,'angy_CPT5']=CPT5(X_final.loc[mask,'angular_velocity_Y'].values)

    X_predict.loc[s,'angy_SSC']=SSC(X_final.loc[mask,'angular_velocity_Y'].values)

    X_predict.loc[s,'angy_wavelength']=wave_length(X_final.loc[mask,'angular_velocity_Y'].values)

    X_predict.loc[s,'angy_normentropy']=norm_entropy(X_final.loc[mask,'angular_velocity_Y'].values)

    X_predict.loc[s,'angy_SRAV']=SRAV(X_final.loc[mask,'angular_velocity_Y'].values)

    X_predict.loc[s,'angy_Werocrossing']=zero_crossing(X_final.loc[mask,'angular_velocity_Y'].values)

    

    X_predict.loc[s,'angz_mean']=X_final.loc[mask,'angular_velocity_Z'].mean()

    X_predict.loc[s,'angz_std']=X_final.loc[mask,'angular_velocity_Z'].std()

    X_predict.loc[s,'angz_min']=X_final.loc[mask,'angular_velocity_Z'].min()

    X_predict.loc[s,'angz_max']=X_final.loc[mask,'angular_velocity_Z'].max()

    X_predict.loc[s,'angz_diff_mean']=X_final.loc[mask,'angular_velocity_Z'].diff().mean()

    X_predict.loc[s,'angz_diff_min']=X_final.loc[mask,'angular_velocity_Z'].diff().min()

    X_predict.loc[s,'angz_diff_max']=X_final.loc[mask,'angular_velocity_Z'].diff().max()

    X_predict.loc[s,'angz_kur']=X_final.loc[mask,'angular_velocity_Z'].skew()

    X_predict.loc[s,'angz_skew']=X_final.loc[mask,'angular_velocity_Z'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'angz_mean']=X_final.loc[mask,'angular_velocity_Z'].abs().mean()

    X_predict.loc[s,'angz_CPT5']=CPT5(X_final.loc[mask,'angular_velocity_Z'].values)

    X_predict.loc[s,'angz_SSC']=SSC(X_final.loc[mask,'angular_velocity_Z'].values)

    X_predict.loc[s,'angz_wavelength']=wave_length(X_final.loc[mask,'angular_velocity_Z'].values)

    X_predict.loc[s,'angz_normentropy']=norm_entropy(X_final.loc[mask,'angular_velocity_Z'].values)

    X_predict.loc[s,'angz_SRAV']=SRAV(X_final.loc[mask,'angular_velocity_Z'].values)

    X_predict.loc[s,'angz_Werocrossing']=zero_crossing(X_final.loc[mask,'angular_velocity_Z'].values)

    

    X_predict.loc[s,'linx_mean']=X_final.loc[mask,'linear_acceleration_X'].mean()

    X_predict.loc[s,'linx_std']=X_final.loc[mask,'linear_acceleration_X'].std()

    X_predict.loc[s,'linx_min']=X_final.loc[mask,'linear_acceleration_X'].min()

    X_predict.loc[s,'linx_max']=X_final.loc[mask,'linear_acceleration_X'].max()

    X_predict.loc[s,'linx_diff_mean']=X_final.loc[mask,'linear_acceleration_X'].diff().mean()

    X_predict.loc[s,'linx_diff_min']=X_final.loc[mask,'linear_acceleration_X'].diff().min()

    X_predict.loc[s,'linx_diff_max']=X_final.loc[mask,'linear_acceleration_X'].diff().max()

    X_predict.loc[s,'linx_kur']=X_final.loc[mask,'linear_acceleration_X'].skew()

    X_predict.loc[s,'linx_skew']=X_final.loc[mask,'linear_acceleration_X'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'linx_mean']=X_final.loc[mask,'linear_acceleration_X'].abs().mean()

    X_predict.loc[s,'linx_CPT5']=CPT5(X_final.loc[mask,'linear_acceleration_X'].values)

    X_predict.loc[s,'linx_SSC']=SSC(X_final.loc[mask,'linear_acceleration_X'].values)

    X_predict.loc[s,'linx_wavelength']=wave_length(X_final.loc[mask,'linear_acceleration_X'].values)

    X_predict.loc[s,'linx_normentropy']=norm_entropy(X_final.loc[mask,'linear_acceleration_X'].values)

    X_predict.loc[s,'linx_SRAV']=SRAV(X_final.loc[mask,'linear_acceleration_X'].values)

    X_predict.loc[s,'linx_Werocrossing']=zero_crossing(X_final.loc[mask,'linear_acceleration_X'].values)

    

    X_predict.loc[s,'liny_mean']=X_final.loc[mask,'linear_acceleration_Y'].mean()

    X_predict.loc[s,'liny_std']=X_final.loc[mask,'linear_acceleration_Y'].std()

    X_predict.loc[s,'liny_min']=X_final.loc[mask,'linear_acceleration_Y'].min()

    X_predict.loc[s,'liny_max']=X_final.loc[mask,'linear_acceleration_Y'].max()

    X_predict.loc[s,'liny_diff_mean']=X_final.loc[mask,'linear_acceleration_Y'].diff().mean()

    X_predict.loc[s,'liny_diff_min']=X_final.loc[mask,'linear_acceleration_Y'].diff().min()

    X_predict.loc[s,'liny_diff_max']=X_final.loc[mask,'linear_acceleration_Y'].diff().max()

    X_predict.loc[s,'liny_kur']=X_final.loc[mask,'linear_acceleration_Y'].skew()

    X_predict.loc[s,'liny_skew']=X_final.loc[mask,'linear_acceleration_Y'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'liny_mean']=X_final.loc[mask,'linear_acceleration_Y'].abs().mean()

    X_predict.loc[s,'liny_CPT5']=CPT5(X_final.loc[mask,'linear_acceleration_Y'].values)

    X_predict.loc[s,'liny_SSC']=SSC(X_final.loc[mask,'linear_acceleration_Y'].values)

    X_predict.loc[s,'liny_wavelength']=wave_length(X_final.loc[mask,'linear_acceleration_Y'].values)

    X_predict.loc[s,'liny_normentropy']=norm_entropy(X_final.loc[mask,'linear_acceleration_Y'].values)

    X_predict.loc[s,'liny_SRAV']=SRAV(X_final.loc[mask,'linear_acceleration_Y'].values)

    X_predict.loc[s,'liny_Werocrossing']=zero_crossing(X_final.loc[mask,'linear_acceleration_Y'].values)

    

    X_predict.loc[s,'linz_mean']=X_final.loc[mask,'linear_acceleration_Z'].mean()

    X_predict.loc[s,'linz_std']=X_final.loc[mask,'linear_acceleration_Z'].std()

    X_predict.loc[s,'linz_min']=X_final.loc[mask,'linear_acceleration_Z'].min()

    X_predict.loc[s,'linz_max']=X_final.loc[mask,'linear_acceleration_Z'].max()

    X_predict.loc[s,'linz_diff_mean']=X_final.loc[mask,'linear_acceleration_Z'].diff().mean()

    X_predict.loc[s,'linz_diff_min']=X_final.loc[mask,'linear_acceleration_Z'].diff().min()

    X_predict.loc[s,'linz_diff_max']=X_final.loc[mask,'linear_acceleration_Z'].diff().max()

    X_predict.loc[s,'linz_kur']=X_final.loc[mask,'linear_acceleration_Z'].skew()

    X_predict.loc[s,'linz_skew']=X_final.loc[mask,'linear_acceleration_Z'].kurtosis()

    

    #Some Extra

    X_predict.loc[s,'linz_mean']=X_final.loc[mask,'linear_acceleration_Z'].abs().mean()

    X_predict.loc[s,'linz_CPT5']=CPT5(X_final.loc[mask,'linear_acceleration_Z'].values)

    X_predict.loc[s,'linz_SSC']=SSC(X_final.loc[mask,'linear_acceleration_Z'].values)

    X_predict.loc[s,'linz_wavelength']=wave_length(X_final.loc[mask,'linear_acceleration_Z'].values)

    X_predict.loc[s,'linz_normentropy']=norm_entropy(X_final.loc[mask,'linear_acceleration_Z'].values)

    X_predict.loc[s,'linz_SRAV']=SRAV(X_final.loc[mask,'linear_acceleration_Z'].values)

    X_predict.loc[s,'linz_Werocrossing']=zero_crossing(X_final.loc[mask,'linear_acceleration_Z'].values)

X_predict.shape
dfx=pd.DataFrame()

dfx['series_id']=np.arange(total_series)
#Test Initilization End
#Bayesian Tuning Starts
# def lgbm_evaluate(**params):

#     warnings.simplefilter('ignore')

    

#     params['num_leaves'] = int(params['num_leaves'])

#     params['max_depth'] = int(params['max_depth'])

        

#     clf = lgb.LGBMClassifier(**params, n_estimators=5000, nthread=-1)



#     #Xgboost 5 Fold Cross Validation

#     #Cross Validation

#     ulabels=list(train_y.surface.unique())

#     trainALL=[]

#     testALL=[]

#     sss = StratifiedShuffleSplit(n_splits=5,test_size=0.30,random_state=60)

#     for train_index, test_index in sss.split(X, Y):

#         X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]

#         Y_train, Y_test = Y[train_index], Y[test_index]



#         clf=lgb.LGBMClassifier(**params,objective='multiclass',is_unbalance=True,

#                                learning_rate=0.05,n_estimators=500,num_class=9)

#         clf.fit(X_train, Y_train,

#             eval_set=[(X_train, Y_train), (X_test, Y_test)],

#             early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error

#             verbose=False,eval_metric='multi_error')

#         testR=f1_score(Y_test,clf.predict(X_test),average='weighted',labels=ulabels)

#         testALL.append(testR)

#     return np.mean(testALL)
# params = {'colsample_bytree': (0.6, 1),

#       'num_leaves': (8, 50), 

#       'subsample': (0.6, 1), 

#       'max_depth': (3, 25), 

#       'reg_alpha': (.05, 15.0), 

#       'reg_lambda': (.05, 15.0), 

#       'min_split_gain': (.001, .03),

#       'min_child_weight': (12, 80)}



# #bo = BayesianOptimization(lgbm_evaluate, params)

# bo.maximize(init_points=5, n_iter=20)
#Bayesian Tuning Ends
#Xgboost 5 Fold Cross Validation

#Cross Validation

ulabels=list(train_y.surface.unique())

trainALL=[]

testALL=[]

sss = StratifiedShuffleSplit(n_splits=5,test_size=0.30,random_state=60)

k=1

for train_index, test_index in sss.split(X, Y):

    print("FOLD :",k)

    X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]

    Y_train, Y_test = Y[train_index], Y[test_index]

    

    clf=lgb.LGBMClassifier(objective='multiclass',is_unbalance=True,max_depth=9,

                           learning_rate=0.05,n_estimators=500,num_leaves=25)

    clf.fit(X_train, Y_train,

        eval_set=[(X_train, Y_train), (X_test, Y_test)],

        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error

        verbose=False,eval_metric='multi_error')

    trainR=f1_score(Y_train,clf.predict(X_train),average='weighted',labels=ulabels)

    testR=f1_score(Y_test,clf.predict(X_test),average='weighted',labels=ulabels)

    print("F1 Score Train: ",trainR)

    print("F1 Score Test : ",testR)

    print("****************")

    trainALL.append(trainR)

    testALL.append(testR)

    

    #Feature Importance

#     plt.figure(figsize=(15,35))

#     ax=plt.axes()

#     lgb.plot_importance(clf, height=0.5,ax=ax)

#     plt.show()



    #Test Predict

    dfx['P'+str(k)]=clf.predict(X_predict)

    k+=1
print("Train Score: ",np.mean(trainALL))

print("Test Score:  ",np.mean(testALL))
dfx.head()
dfx['surface']=dfx.loc[:,['P1','P2','P3','P4','P5']].mode(axis=1)[0]
del dfx['P1']

del dfx['P2']

del dfx['P3']

del dfx['P4']

del dfx['P5']
print(dfx.shape)

display(dfx.head())
dfx.to_csv("sub1.csv",index=False)