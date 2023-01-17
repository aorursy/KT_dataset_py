import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,roc_auc_score

%matplotlib inline



train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

#test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')



print("First five rows of training dataset are:")

train.head()
train.info()
print(f"Train set has {train.shape[0]} rows and {train.shape[1]} columns.")

print(f"Missing values are present in data: {train.isnull().sum().any()}")
train_eda = train.copy()

cols=['Driving_License','Previously_Insured','Response']

for col in cols:

    train_eda[col] = train_eda[col].map({0:'No',1:'Yes'})
sns.countplot(train_eda['Response'],palette='rocket')

plt.title("Target variable Distribution in data");
sns.countplot(train_eda['Gender'],palette='summer')

plt.title("Gender Distribution in data");
print("Age distribution according to Response")

facetgrid = sns.FacetGrid(train_eda,hue="Response",aspect = 4)

facetgrid.map(sns.kdeplot,"Age",shade = True)

facetgrid.set(xlim = (0,train_eda["Age"].max()))

facetgrid.add_legend();
pd.crosstab(train_eda['Response'], train_eda['Driving_License'])
train_eda['Region_Code'].value_counts().plot(kind='barh',cmap='Accent',figsize=(12,10));
pd.crosstab(train_eda['Response'], train_eda['Previously_Insured'])
pd.crosstab(train_eda['Response'], train_eda['Previously_Insured']).plot(kind='bar');
plt.rcParams['figure.figsize']=(6,8)

color = ['yellowgreen','gold',"lightskyblue"]

train_eda['Vehicle_Age'].value_counts().plot.pie(y="Vehicle_Age",colors=color,explode=(0.02,0,0.3),startangle=50,shadow=True,autopct="%0.1f%%")

plt.axis('on');
sns.countplot(train_eda['Vehicle_Age'],hue=train_eda['Response'],palette='autumn');
pd.crosstab(train_eda['Response'], train_eda['Vehicle_Damage']).plot(kind='bar');
print("Annual Premium distribution according to Response")

facetgrid = sns.FacetGrid(train_eda,hue="Response",aspect = 4)

facetgrid.map(sns.kdeplot,"Annual_Premium",shade = True)

facetgrid.set(xlim = (0,train_eda["Annual_Premium"].max()))

facetgrid.add_legend();
print("Policy_Sales_Channel distribution according to Response")

facetgrid = sns.FacetGrid(train_eda,hue="Response",aspect = 4)

facetgrid.map(sns.kdeplot,"Policy_Sales_Channel",shade = True)

facetgrid.set(xlim = (0,train_eda["Policy_Sales_Channel"].max()))

facetgrid.add_legend();
print("Vintage feature according to Response")

facetgrid = sns.FacetGrid(train_eda,hue="Response",aspect = 4)

facetgrid.map(sns.kdeplot,"Vintage",shade = True)

facetgrid.set(xlim = (0,train_eda["Vintage"].max()))

facetgrid.add_legend();
print("Correlation matrix-")

plt.rcParams['figure.figsize']=(8,6)

sns.heatmap(train.corr(),cmap='Spectral');
train.corr()[:-1]['Response'].sort_values().round(2)
#creating a checkpoint

df4model = train.copy()

#dropping Vintage column as suggested by EDA

df4model.drop(['id','Vintage'],axis=1,inplace=True)

#checking target variable

df4model.Response.value_counts()
X_train, X_test, y_train, y_test = train_test_split(df4model.drop(['Response'], axis = 1), 

                                                    df4model['Response'], test_size = 0.2)
print(f"Target variable disribution in train set: \n{y_train.value_counts()}\n\nand in test set: \n{y_test.value_counts()}")
#combining train features and target

df = pd.concat([X_train,y_train],axis=1)



from sklearn.utils import resample,shuffle

df_majority = df[df['Response']==0]

df_minority = df[df['Response']==1]

df_minority_upsampled = resample(df_minority,replace=True,n_samples=y_train.value_counts()[0],random_state = 123)

balanced_df = pd.concat([df_minority_upsampled,df_majority])

balanced_df = shuffle(balanced_df)

balanced_df.Response.value_counts()
from sklearn.preprocessing import OrdinalEncoder

encoder= OrdinalEncoder()

cat_cols=['Gender','Vehicle_Damage']

balanced_df[cat_cols] = encoder.fit_transform(balanced_df[cat_cols])

X_test[cat_cols] = encoder.transform(X_test[cat_cols])



dummy = pd.get_dummies(balanced_df['Vehicle_Age'],drop_first=True)

features = pd.concat([dummy,balanced_df],axis=1)

features.drop('Vehicle_Age',axis=1,inplace=True)



features.head()
#to get uniform output

features = features.astype('float64')

X_train = features.drop('Response',axis=1)

y_train = features['Response']



#creating dummies in test set

dummy1 = pd.get_dummies(X_test['Vehicle_Age'],drop_first=True)

X_test = pd.concat([dummy1,X_test],axis=1)

X_test.drop('Vehicle_Age',axis=1,inplace=True)
logisticRegression = LogisticRegression(max_iter = 10000)

logisticRegression.fit(X_train, y_train)

predictions = logisticRegression.predict(X_test)

print(f"Accuracy score is {100*accuracy_score(y_test,predictions).round(2)}\nROC-AUC score is {100*roc_auc_score(y_test,predictions).round(2)}")
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(f"Accuracy score is {100*accuracy_score(y_test,rfc_pred).round(2)}\nROC-AUC score is {100*roc_auc_score(y_test,rfc_pred).round(2)}")
rfc_preds = rfc.predict_proba(X_test)

print("AUC score after taking probabilities predictions and not classes predictions is")

roc_auc_score(y_test, rfc_preds[:,1], average = 'weighted')
X_train.columns= ['less than 1 Year','greater than 2 Years', 'Gender', 'Age','Driving_License',

                  'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel']

X_test.columns= ['less than 1 Year','greater than 2 Years', 'Gender', 'Age','Driving_License',

                  'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel']



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print(f"Accuracy score is {100*accuracy_score(y_test,xgb_pred).round(2)}\nROC-AUC score is {100*roc_auc_score(y_test,xgb_pred).round(2)}")
xgb_preds = xgb.predict_proba(X_test)

roc_auc_score(y_test, xgb_preds[:,1], average = 'weighted')