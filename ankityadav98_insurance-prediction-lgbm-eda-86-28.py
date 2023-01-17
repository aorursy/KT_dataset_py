import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import RandomOverSampler

from collections import Counter

from sklearn.model_selection import ShuffleSplit, cross_val_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_orig= pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test_orig= pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')

subm= pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
train_orig.head()
test_orig.head()
train_orig.isna().sum()
test_orig.isna().sum()
for col in test_orig.columns:

    print(f"{col}")

    print(f"Train:{train_orig[col].nunique()}\nTest:{test_orig[col].nunique()}")

    print("===============================")
train_orig.info()
sns.countplot(x='Response',data=train_orig);
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

plt.hist(train_orig['Region_Code'],bins=30,label='Train');

plt.legend()

plt.subplot(1,2,2)

plt.hist(test_orig['Region_Code'],bins=30,label='Test');

plt.legend();
sns.distplot(train_orig['Age']);
sns.countplot(x='Gender',data=train_orig,hue='Response');
sns.countplot(x='Driving_License',data=train_orig,hue='Gender');
sns.countplot(x='Driving_License',data=train_orig,hue='Response');
sns.countplot(x='Previously_Insured',data=train_orig,hue='Response');
sns.distplot(train_orig['Policy_Sales_Channel']);
sns.distplot(train_orig['Vintage']);
data= pd.concat([train_orig,test_orig],axis=0,sort=False)
data.info()
data.nunique()
plt.figure(figsize=(16,4))

plt.subplot(1,3,1)

plt.hist(train_orig['Annual_Premium'],bins=30,label='Train');

plt.legend()

plt.subplot(1,3,2)

plt.hist(test_orig['Annual_Premium'],bins=30,label='Test');

plt.legend();

plt.subplot(1,3,3)

plt.hist(data['Annual_Premium'],bins=30,label='Data');

plt.legend();
def outliers(df, variable, distance):



    # Let's calculate the boundaries outside which the outliers are for skewed distributions



    # distance passed as an argument, gives us the option to estimate 1.5 times or 3 times the IQR to calculate

    # the boundaries.



    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)



    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)

    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)



    return upper_boundary, lower_boundary



upper_limit, lower_limit = outliers(data, 'Annual_Premium', 1.5)

upper_limit, lower_limit
data['Annual_Premium']= np.where(data['Annual_Premium'] > upper_limit, upper_limit,

                       np.where(data['Annual_Premium'] < lower_limit, lower_limit, data['Annual_Premium']))
plt.hist(data['Annual_Premium'],bins=30);
data['Vehicle_Age'].value_counts()
data.groupby(['Gender','Driving_License','Response']).size()
data['Driving_License']= data['Driving_License'].astype(str)

data['DL_Gender']= data['Driving_License']+'_'+ data['Gender']

data['Driving_License']= data['Driving_License'].astype(int)
data.groupby(['Vehicle_Age','Vehicle_Damage']).size()
data['Vehicle_Age_and_Damage']= data['Vehicle_Age']+'_'+data['Vehicle_Damage']
data.groupby(['Vehicle_Age','Previously_Insured']).size()
data['Previously_Insured']= data['Previously_Insured'].astype(str)

data['Previously_Insured_Vehicle_Age']= data['Vehicle_Age']+'_'+data['Previously_Insured']

data['Previously_Insured']= data['Previously_Insured'].astype(int)
data.head()
data['Vintage']= data['Vintage'].apply(lambda x: x/365)
gender_map= {'Male':0,'Female':1}

vehicle_age_map= {'< 1 Year':0,'1-2 Year':1,'> 2 Years':2}

vehicle_damage_map= {'Yes':1,'No':0}



data['Gender']= data['Gender'].map(gender_map)

data['Vehicle_Age']= data['Vehicle_Age'].map(vehicle_age_map)

data['Vehicle_Damage']= data['Vehicle_Damage'].map(vehicle_damage_map)
sns.distplot(data['Region_Code']);
plt.figure(figsize=(14,6))

sns.countplot(x='Region_Code',data=data);

plt.xticks(rotation=90);
data.dtypes
cat_col= [col for col in data.columns if data[col].dtypes=='object']

cat_col
for col in cat_col:

    dummies= pd.get_dummies(data[col])

    data=pd.concat([data,dummies],axis=1)

    data.drop(columns=[col],inplace=True)
data.drop(columns=['id','Response','Driving_License','0_Female'],inplace=True)
data.head().T
train_new= data[:len(train_orig)]

test_new= data[len(train_orig):]
train_os=RandomOverSampler(random_state=101)

y=train_orig['Response']



X_os,y_os=train_os.fit_sample(train_new,y)
print('Original dataset shape {}'.format(Counter(y)))

print('Resampled dataset shape {}'.format(Counter(y_os))) 
#y= train_orig['Response']



X_train, X_val, y_train, y_val = train_test_split(X_os, y_os, test_size=0.3, random_state=101)
check= pd.concat([train_new,pd.DataFrame(data=train_orig['Response'],columns=['Response'])],axis=1,sort=False)
plt.figure(figsize=(16,10))

sns.heatmap(check.corr(),cbar=False,cmap='inferno',annot=True);
check.corr()["Response"].sort_values()
model= LGBMClassifier(boosting_type='gbdt',objective='binary',random_state=101)
#model_tuning.best_estimator_
model=LGBMClassifier(colsample_bytree=0.5, learning_rate=0.03,

                     n_estimators=600, objective='binary', reg_alpha=0.1,

                     random_state=101,reg_lambda=0.8)



model.fit(X_train,y_train)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

scores=cross_val_score(model, X_val, y_val, cv=cv,scoring='roc_auc')

scores.mean()
val_pred= model.predict_proba(X_val)[:,1]
val_pred
print(roc_auc_score(y_val,val_pred))
pred= model.predict_proba(test_new)[:,1]
subm.head()
subm['Response']= pred
subm.to_csv('submission.csv',index=False)