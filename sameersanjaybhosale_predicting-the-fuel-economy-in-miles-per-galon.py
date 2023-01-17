# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
raw_data=pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

raw_data.describe(include='all')
raw_data.isnull().sum()
raw_data_mod=raw_data.drop(['car name'],axis=1)
raw_data_mod[raw_data_mod.horsepower.values=='?']
model_data=raw_data_mod.copy()

model_data['horsepower']=model_data.horsepower.replace({'?':None})

model_data.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



y=model_data['mpg']

x=model_data.drop(['mpg'],axis=1)

X_train,X_valid,y_train,y_valid=train_test_split(x,y,random_state=1)

my_imp=SimpleImputer()

my_imp_1=SimpleImputer()

imputed_X_train=pd.DataFrame(my_imp.fit_transform(X_train))

imputed_X_valid=pd.DataFrame(my_imp.transform(X_valid))

imputed_mod_data=pd.DataFrame(my_imp_1.fit_transform(model_data))





imputed_X_train.columns=X_train.columns

imputed_mod_data.columns=model_data.columns

imputed_X_valid.columns=X_valid.columns

plt.subplots_adjust(right=2.0,wspace=0.7,hspace=0.8)

plt.subplot(2,4,1)

mpg=sns.distplot(imputed_mod_data['mpg'])

plt.subplot(2,4,2)

cyn=sns.distplot(imputed_mod_data['cylinders'])

plt.subplot(2,4,3)

dis=sns.distplot(imputed_mod_data['displacement'])

plt.subplot(2,4,4)

wgt=sns.distplot(imputed_mod_data['weight'])

plt.subplot(2,4,5)

acc=sns.distplot(imputed_mod_data['acceleration'])

plt.subplot(2,4,6)

myr=sns.distplot(imputed_mod_data['model year'])

plt.subplot(2,4,7)

org=sns.distplot(imputed_mod_data['origin'])

plt.subplot(2,4,8)

hrs=sns.distplot(imputed_mod_data['horsepower'])



plt.show()
imp_X_tr=imputed_X_train.drop(['displacement'],axis=1)

imp_X_ts=imputed_X_valid.drop(['displacement'],axis=1)

imp_mod_data=imputed_mod_data.drop(['displacement'],axis=1)
f,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,sharey=True,figsize=(8,25))



ax1.scatter(np.log(imputed_mod_data['displacement']),imputed_mod_data['mpg'])

ax1.set_title('displacement and mpg')

ax2.scatter(np.log(imputed_mod_data['horsepower']),imputed_mod_data['mpg'])

ax2.set_title('horsepower and mpg')

ax3.scatter(1/(model_data['acceleration']),imputed_mod_data['mpg'])

ax3.set_title('acceleration and mpg')

ax4.scatter(np.log(imputed_mod_data['weight']),imputed_mod_data['mpg'])

ax4.set_title('weight and mpg')



plt.show()

imp_X_tr.horsepower=np.log(imp_X_tr.horsepower)

imp_X_tr.weight=np.log(imp_X_tr.weight)

imp_X_ts.horsepower=np.log(imp_X_ts.horsepower)

imp_X_ts.weight=np.log(imp_X_ts.weight)



imp_X_tr.acceleration=1/(imp_X_tr.acceleration)

imp_X_ts.acceleration=1/(imp_X_ts.acceleration)



imp_mod_data.horsepower=np.log(imp_mod_data.horsepower)

imp_mod_data.weight=np.log(imp_mod_data.weight)



imp_mod_data.acceleration=1/(imp_mod_data.acceleration)

from sklearn.preprocessing import LabelEncoder



label_encoder=LabelEncoder()



label_X_train=imp_X_tr.copy()

label_X_valid=imp_X_ts.copy()



label_X_train['cylinders']=label_encoder.fit_transform(label_X_train['cylinders'])

label_X_valid['cylinders']=label_encoder.transform(label_X_valid['cylinders'])

imp_mod_data['cylinders']=label_encoder.fit_transform(imp_mod_data['cylinders'])



from sklearn.preprocessing import OneHotEncoder



oh_enc=OneHotEncoder(handle_unknown='ignore',sparse=False)



oh_X_train=label_X_train.copy()

oh_X_valid=label_X_valid.copy()



oh_X_tr=pd.DataFrame(oh_enc.fit_transform(np.array(oh_X_train['origin']).reshape([-1,1])))

oh_X_va=pd.DataFrame(oh_enc.transform(np.array(oh_X_valid['origin']).reshape([-1,1])))

imp_mod_data_tr=pd.DataFrame(oh_enc.fit_transform(np.array(imp_mod_data['origin']).reshape([-1,1])))

oh_X_tr.index=oh_X_train.index

oh_X_va.index=oh_X_valid.index

imp_mod_data_tr.index=imp_mod_data.index



oh_X_train.drop(['origin'],axis=1,inplace=True)

oh_X_valid.drop(['origin'],axis=1,inplace=True)

imp_mod_data.drop(['origin'],axis=1,inplace=True)



#dropping the '0' column produced by one-hot encoding as it was adding up to multi-collinearity.

oh_X_tr.drop([0],axis=1,inplace=True)

oh_X_va.drop([0],axis=1,inplace=True)

imp_mod_data_tr.drop([0],axis=1,inplace=True)





oh_X_train=pd.concat([oh_X_train,oh_X_tr],axis=1)

oh_X_valid=pd.concat([oh_X_valid,oh_X_va],axis=1)

imp_mod_data=pd.concat([imp_mod_data,imp_mod_data_tr],axis=1)





imp_mod_data
oh_X_train
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler_1=StandardScaler()

#scaler.var_=np.ones((1,4))

#scaler.fit(oh_X_train[['displacement','horsepower','weight','model year']])

scaler.fit(oh_X_train[['acceleration']])

scaler_1.fit(imp_mod_data[['acceleration']])



oh_x_tr_sca=scaler.transform(oh_X_train[['acceleration']])

oh_x_vl_sca=scaler.transform(oh_X_valid[['acceleration']])

imp_mod_data_sca=scaler_1.transform(imp_mod_data[['acceleration']])



oh_x_tr_sca=pd.DataFrame(oh_x_tr_sca)

oh_x_vl_sca=pd.DataFrame(oh_x_vl_sca)

imp_mod_data_sca=pd.DataFrame(imp_mod_data_sca)



oh_x_tr_sca.columns=oh_X_train[['acceleration']].columns

oh_x_vl_sca.columns=oh_X_valid[['acceleration']].columns

imp_mod_data_sca.columns=imp_mod_data[['acceleration']].columns
oh_tr_r=oh_X_train.drop(['acceleration'],axis=1)

oh_tv_r=oh_X_valid.drop(['acceleration'],axis=1)

mod_data_r=imp_mod_data.drop(['acceleration'],axis=1)



oh_X_train=pd.concat([oh_tr_r,pd.DataFrame(oh_x_tr_sca)],axis=1)

oh_X_valid=pd.concat([oh_tv_r,pd.DataFrame(oh_x_vl_sca)],axis=1)

imp_mod_data=pd.concat([mod_data_r,pd.DataFrame(imp_mod_data_sca)],axis=1)
for col in oh_X_train[['horsepower','model year','weight']].columns:

    oh_X_train[col]=oh_X_train[col]-oh_X_train.mean(axis=0)[col]

    oh_X_valid[col]=oh_X_valid[col]-oh_X_valid.mean(axis=0)[col]

    imp_mod_data[col]=imp_mod_data[col]-imp_mod_data.mean(axis=0)[col]
y_tr_log=pd.DataFrame(np.log(y_train))

y_tr_log
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(oh_X_train,y_tr_log)
from sklearn.model_selection import cross_val_score



scores = -1 * cross_val_score(reg,oh_X_train, y_tr_log,

                              cv=5,

                              scoring='neg_mean_absolute_error')

print("Average MAE score (across experiments):{}".format(scores.mean()))





reg.score(oh_X_train,y_tr_log)
plt.scatter(np.exp(reg.predict(oh_X_valid)),y_valid)
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = oh_X_train

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif["features"] = variables.columns
vif