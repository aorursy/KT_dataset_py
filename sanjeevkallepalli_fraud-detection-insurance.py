import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 500)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename) 

        print(path)
data = pd.read_csv(path)

print(data.shape)

data.head()
#classe imbalance

data['fraud_reported'].value_counts()
#replace Y:1,N:0

data['fraud_reported'] = pd.Series(map(lambda x: dict(Y=1, N=0)[x],

              data['fraud_reported'].values.tolist()), data.index)
#bar plot of target

import matplotlib.pyplot as plt

%matplotlib inline

data['fraud_reported'].value_counts().plot(kind = 'bar', figsize = (4,3))
def levels(df):

    return (pd.DataFrame({'dtype':df.dtypes, 

                         'levels':df.nunique(), 

                         'uni_values':[df[x].unique() for x in df.columns],

                         'null_values':df.isna().sum(),

                         'unique':df.nunique()}))

levels(data)
data.drop('_c39',axis=1,inplace=True)
data.replace('?','oth',inplace=True)
import seaborn as sns



sns.boxplot(data.fraud_reported,data.months_as_customer,orient='v')
data.groupby('fraud_reported').agg({'months_as_customer':'std'})
df = data.sort_values('months_as_customer').reset_index(drop=True)

fig_dims = (20, 10)

fig, ax =plt.subplots(4,1,figsize=fig_dims)

sns.countplot(df.months_as_customer.loc[0:199],orient='v',hue=df['fraud_reported'], ax=ax[0])

sns.countplot(df.months_as_customer.loc[200:399],orient='v',hue=df['fraud_reported'], ax=ax[1])

sns.countplot(df.months_as_customer.loc[400:699],orient='v',hue=df['fraud_reported'], ax=ax[2])

sns.countplot(df.months_as_customer.loc[700:],orient='v',hue=df['fraud_reported'], ax=ax[3])

plt.xticks(rotation=90)

fig.show()
data['months_as_customer'].max()
bins = [-1, 60, 120, 180, 240, 300, 360, 420, 480]

data['year_bin'] = pd.cut(data['months_as_customer'], bins)
plt.figure(figsize=(10,3))

sns.countplot(data.year_bin,orient='v',hue=data['fraud_reported'])
data.drop('months_as_customer',axis=1,inplace=True)
sns.boxplot(data.fraud_reported,data.policy_number,orient='v')
data.groupby('fraud_reported').agg({'policy_number':'std'})
data.drop('policy_number',axis=1,inplace=True)
sns.countplot(data.policy_state,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(10,5))

sns.countplot(data.umbrella_limit,orient='v',hue=data['fraud_reported'])
data[['incident_city','incident_location','fraud_reported']].head(10)
plt.figure(figsize=(10,5))

sns.countplot(data.incident_city,orient='v',hue=data['fraud_reported'])
sns.countplot(data.incident_state,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(10,5))

sns.countplot(data.incident_type,orient='v',hue=data['fraud_reported'])
data['pin']= data["incident_location"].str.slice(0,4,1)
data.drop('incident_location',axis=1,inplace=True)
data['pin']
data.pin.nunique()
df = data[['pin','fraud_reported']][data['pin'].isin(data['pin'].value_counts()[data['pin'].value_counts()>1].index)]
plt.figure(figsize=(25,5))

sns.countplot(df.pin,orient='v',hue=df['fraud_reported'])
df = data[['pin','fraud_reported']][data['pin'].isin(data['pin'].value_counts()[data['pin'].value_counts()==1].index)]

plt.figure(figsize=(25,3))

sns.countplot(df.pin.loc[:50],orient='v',hue=df['fraud_reported'])
plt.figure(figsize=(25,3))

sns.countplot(df.pin.loc[51:100],orient='v',hue=df['fraud_reported'])
data['pin'].max()
sns.boxplot(data.fraud_reported,data.pin.astype('int'),orient='v')
bins = [0,2000,4000,6000,8000,10000]

data['pin_bin'] = pd.cut(data['pin'].astype('int'), bins)

data.drop('pin',axis=1,inplace=True)
plt.figure(figsize=(20,3))

sns.countplot(data.pin_bin,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(20,3))

sns.countplot(data.insured_occupation,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(20,3))

sns.countplot(data.insured_education_level,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(20,3))

sns.countplot(data.insured_hobbies,orient='v',hue=data['fraud_reported'])

plt.xticks(rotation=30)
df = data.groupby('insured_hobbies')['fraud_reported'].value_counts().unstack().reset_index()

df['ratio_1/0'] = df[1]/df[0]

df
df.sort_values('ratio_1/0')
data = data.merge(df[['insured_hobbies','ratio_1/0']], on = 'insured_hobbies', how='left')
data['insured_hobbies'][data['ratio_1/0']<=0.15] = 'h1'

data['insured_hobbies'][(data['ratio_1/0']>0.15)&(data['ratio_1/0']<=0.25)] = 'h2'

data['insured_hobbies'][(data['ratio_1/0']>0.25)&(data['ratio_1/0']<=0.35)] = 'h3'

data['insured_hobbies'][(data['ratio_1/0']>0.35)&(data['ratio_1/0']<=0.45)] = 'h4'

data['insured_hobbies'][(data['ratio_1/0']>0.45)&(data['ratio_1/0']<=3)] = 'h5'

data['insured_hobbies'][(data['ratio_1/0']>3)] = 'h6'
plt.figure(figsize=(20,3))

sns.countplot(data.insured_hobbies,orient='v',hue=data['fraud_reported'])

plt.xticks(rotation=30)
plt.figure(figsize=(20,3))

sns.countplot(data.age,orient='v',hue=data['fraud_reported'])
sns.boxplot(data.fraud_reported,data.age,orient='v')
bins = [18, 22, 35, 50, 65]

data['age_bin'] = pd.cut(data['age'], bins)
data.drop('age',axis=1,inplace=True)
plt.figure(figsize=(10,3))

sns.countplot(data.age_bin,orient='v',hue=data['fraud_reported'])
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])

data['incident_date'] = pd.to_datetime(data['incident_date'])
data['policy_age'] = data['incident_date'] - data['policy_bind_date']

data['policy_age'] = data['policy_age'].astype('str')

data['policy_age'] = data['policy_age'].str.slice(0,-5,1)
sns.boxplot(data.fraud_reported,data.policy_age.astype('int')/365,orient='v')
data['policy_age'].astype('int').max()/365
bins = [-1, 5, 10, 15, 20, 26]

data['policy_age_bin'] = pd.cut(data['policy_age'].astype('int')/365, bins)

data.drop('policy_age',axis=1,inplace=True)
plt.figure(figsize=(10,3))

sns.countplot(data.policy_age_bin,orient='v',hue=data['fraud_reported'])
data['age_at_incident'] = data['incident_date'].astype('str').str.slice(0,4,1).astype('int') - data['auto_year']
data.drop(['policy_bind_date','incident_date','auto_year'],axis=1,inplace=True)
plt.figure(figsize=(10,3))

sns.countplot(data.age_at_incident,orient='v',hue=data['fraud_reported'])
sns.boxplot(data.fraud_reported,data.insured_zip/1000,orient='v')
bins = [42.5, 45, 50, 65]

data['insured_zip_bin'] = pd.cut(data['insured_zip']/10000, bins)
plt.figure(figsize=(10,3))

sns.countplot(data.insured_zip_bin,orient='v',hue=data['fraud_reported'])
data.drop('insured_zip',axis=1,inplace=True)
plt.figure(figsize=(10,3))

sns.countplot(data.incident_hour_of_the_day,orient='v',hue=data['fraud_reported'])
data['incident_hour_of_the_day'][data['incident_hour_of_the_day']==0]=24

bins = [0, 6, 9, 13, 17, 25]

data['incident_hour_bin'] = pd.cut(data['incident_hour_of_the_day'], bins)

data.drop('incident_hour_of_the_day',axis=1,inplace=True)
plt.figure(figsize=(10,3))

sns.countplot(data.incident_hour_bin,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(15,3))

sns.countplot(data.auto_make,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(25,3))

sns.countplot(data.auto_model,orient='v',hue=data['fraud_reported'])

plt.xticks(rotation=90)
data['auto_make_model']=data['auto_make']+'_'+data['auto_model']

plt.figure(figsize=(25,3))

sns.countplot(data.auto_make_model,orient='v',hue=data['fraud_reported'])

plt.xticks(rotation=90)
data.drop(['auto_make','auto_model'],axis=1,inplace=True)
sns.boxplot(data.fraud_reported,data.policy_annual_premium,orient='v')
bins = [0, 500, 1000, 1500, 2500]

data['premium_bin'] = pd.cut(data['policy_annual_premium'], bins)

data.drop('policy_annual_premium',axis=1,inplace=True)
plt.figure(figsize=(25,3))

sns.countplot(data.premium_bin,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(25,3))

sns.countplot(data.police_report_available,orient='v',hue=data['fraud_reported'])
plt.figure(figsize=(25,3))

sns.countplot(data.witnesses,orient='v',hue=data['fraud_reported'])
# Compute the correlation matrix

corr = data.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, vmin=-.9, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#total_claim_amount is caused by summation of injury_claim, property_claim, vehicle_claim. As it does occur before, we shall drop.

data.drop('total_claim_amount',axis=1,inplace=True)
data.columns
x = data.copy().drop("fraud_reported",axis=1)

y = data["fraud_reported"]
num_cols = ['capital-gains','capital-loss','injury_claim', 'property_claim', 'vehicle_claim','ratio_1/0']

cat_cols = x.columns.difference(num_cols)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 200)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale.fit(x_train[num_cols])

x_train[num_cols] = scale.transform(x_train[num_cols])

x_test[num_cols] = scale.transform(x_test[num_cols])
from sklearn.preprocessing import LabelEncoder

cols = ['umbrella_limit','insured_occupation','age_at_incident','auto_make_model']

import bisect

for col in cols:

    le = LabelEncoder()

    x_train[col] = le.fit_transform(x_train[col])

    x_test[col] = x_test[col].map(lambda s: 'other' if s not in le.classes_ else s)

    le_classes = le.classes_.tolist()

    #bisect.insort_left(le_classes, 'other')

    le.classes_ = le_classes

    x_test[col] = le.transform(x_test[col])
cat_cols = x_train[cat_cols].columns.difference(cols)

cat_cols
x_train = pd.get_dummies(x_train,columns=cat_cols,drop_first=False,)

x_test = pd.get_dummies(x_test,columns=cat_cols,drop_first=False,)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
x_train.columns.difference(x_test.columns)
x_test.columns.difference(x_train.columns)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=200,max_iter=2000)

lr.fit(x_train,y_train)

train_pred_lr = lr.predict(x_train)

test_pred_lr = lr.predict(x_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, f1_score

cm = confusion_matrix(y_test, test_pred_lr)

print("Accuracy on train is:",accuracy_score(y_train,train_pred_lr))

print("Accuracy on test is:",accuracy_score(y_test,test_pred_lr))
from sklearn.metrics import classification_report

# making a classification report

print("===============Classification report for test===============")

cr = classification_report(y_test,  test_pred_lr)

print(cr)



# making a confusion matrix

cm = confusion_matrix(y_test, test_pred_lr)

sns.heatmap(cm, annot = True, cmap = 'copper',fmt='g')

plt.show()
print("===============Classification report for train===============")

cr = classification_report(y_train,  train_pred_lr)

print(cr)



# making a confusion matrix

cm = confusion_matrix(y_train, train_pred_lr)

sns.heatmap(cm, annot = True, cmap = 'copper',fmt='g')

plt.show()
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc

dtc.fit(x_train, y_train)

train_pred_dtc = dtc.predict(x_train)

test_pred_dtc = dtc.predict(x_test)
print("Accuracy on train is:",accuracy_score(y_train,train_pred_dtc))

print("Accuracy on test is:",accuracy_score(y_test,test_pred_dtc))
dtc
confusion_matrix(y_test, test_pred_dtc)
from sklearn.model_selection import GridSearchCV

parameters={'max_depth':range(2,20,3)}

dt_grid = GridSearchCV(DecisionTreeClassifier(),param_grid=parameters,n_jobs=-1,cv=10)

dt_grid.fit(x_train,y_train)

print(dt_grid.best_score_)

print(dt_grid.best_params_)

train_pred_dt_grid = dt_grid.predict(x_train)

test_pred_dt_grid = dt_grid.predict(x_test)
print("Accuracy on train is:",accuracy_score(y_train,train_pred_dt_grid))

print("Accuracy on test is:",accuracy_score(y_test,test_pred_dt_grid))
%%time

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc



parameters={'n_estimators':[100,300],

           'max_features':range(10,71,20),

           'max_depth':[2, 5,9,13],

           'bootstrap':[True,False]}

rf = GridSearchCV(rfc,param_grid=parameters,n_jobs=-1,cv=10,scoring='accuracy')

rf.fit(x_train,y_train)
rf.best_params_
rfgrid = rf.best_estimator_

rfgrid.fit(x_train,y_train)
train_pred_rfgrid = rfgrid.predict(x_train)

test_pred_rfgrid = rfgrid.predict(x_test)
print("Accuracy on train is:",accuracy_score(y_train,train_pred_rfgrid))

print("Accuracy on test is:",accuracy_score(y_test,test_pred_rfgrid))
# making a classification report

cr = classification_report(y_test,  test_pred_rfgrid)

print(cr)



# making a confusion matrix

cm = confusion_matrix(y_test, test_pred_rfgrid)

sns.heatmap(cm, annot = True, cmap = 'copper',fmt='g')

plt.show()
cr = classification_report(y_train,  train_pred_rfgrid)

print(cr)



# making a confusion matrix

cm = confusion_matrix(y_train, train_pred_rfgrid)

sns.heatmap(cm, annot = True, cmap = 'copper',fmt='g')

plt.show()