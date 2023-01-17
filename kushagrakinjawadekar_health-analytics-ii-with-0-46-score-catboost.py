import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from lightgbm import LGBMClassifier

import xgboost as xb

import lightgbm as lbm

from catboost import Pool, CatBoostClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
train = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')

test = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')
train.head()
train.info()
train.describe()
train.isnull().sum()
test.isnull().sum()
train['kind'] = 'train'

test['kind']  = 'test'

data = pd.concat([train,test],axis=0,sort=False)
data.head()
data.isnull().sum()
sns.set(rc = {'figure.figsize':(18,8)})

sns.countplot('City_Code_Patient',data=data)
data['City_Code_Patient'].unique()
sns.countplot('Bed Grade',data=data)
ds = data.groupby(['Hospital_type_code', 'kind'])['patientid'].count().reset_index()

ds.columns = ['hospital', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='hospital', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Cases hospital type distribution', 

    width=900,

    height=600

)

fig.show()
sns.countplot('Ward_Facility_Code',data=data)
sns.countplot('Ward_Type',data=data)
value = data['Ward_Type'].unique()

per= []

for i in value:

    per.append((data[data['Ward_Type']==i]['Ward_Type'].count())*100/(len(data['Ward_Type'])))
ds = data[data['kind']=='train']

fig = px.pie(

    ds, 

    names='Ward_Type', 

    title='Ward type pie chart for train set', 

    width=900,

    height=600

)

fig.show()
sns.set(rc = {'figure.figsize':(12,8)})

sns.countplot('Department',data=data)
ds = data[data['kind']=='train']

fig = px.pie(

    ds, 

    names='Available Extra Rooms in Hospital', 

    title='Availablity of extra room', 

    width=900,

    height=600

)

fig.show()
#Dealing with null values
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['Department'] = label_encoder.fit_transform(data['Department'])

print('Department :{}' .format(data['Department'].unique()))



data['Ward_Type'] = label_encoder.fit_transform(data['Ward_Type'])

print('Ward_Type : {}'.format(data['Ward_Type'].unique()))



data['Ward_Facility_Code'] = label_encoder.fit_transform(data['Ward_Facility_Code'])

print('Ward_Facility_Code :{}'.format(data['Ward_Facility_Code'].unique()))



data['Hospital_type_code'] = label_encoder.fit_transform(data['Hospital_type_code'])

print('Hospital_type_code: {}'.format(data['Hospital_type_code'].unique()))



data['Hospital_region_code'] = label_encoder.fit_transform(data['Hospital_region_code'])

print('Hospital_region_code: {}'.format(data['Hospital_region_code'].unique()))

      

data['Type of Admission'] = label_encoder.fit_transform(data['Type of Admission'])

print('Type of Admission : {}'.format(data['Type of Admission'].unique()))

      

data['Severity of Illness'] = label_encoder.fit_transform(data['Severity of Illness'])

print('Severity of Illness: {}'.format(data['Severity of Illness'].unique()))

      

data['Age'] = label_encoder.fit_transform(data['Age'])

print('Age : {}'.format(data['Age'].unique()))
data['Bed Grade'].fillna(2.0,inplace=True)

data['City_Code_Patient'].fillna(8.0,inplace=True)
data.isnull().sum()
data.loc[data['Stay'] == '0-10', 'Stay'] = 0

data.loc[data['Stay'] == '11-20', 'Stay'] = 1

data.loc[data['Stay'] == '21-30', 'Stay'] = 2

data.loc[data['Stay'] == '31-40', 'Stay'] = 3

data.loc[data['Stay'] == '41-50', 'Stay'] = 4

data.loc[data['Stay'] == '51-60', 'Stay'] = 5

data.loc[data['Stay'] == '61-70', 'Stay'] = 6

data.loc[data['Stay'] == '71-80', 'Stay'] = 7

data.loc[data['Stay'] == '81-90', 'Stay'] = 8

data.loc[data['Stay'] == '91-100', 'Stay'] = 9

data.loc[data['Stay'] == 'More than 100 Days', 'Stay'] = 10
data.head()
train = data[data['kind'] == 'train']

test = data[data['kind'] == 'test']



train.drop(['kind'], axis=1, inplace=True)

test.drop(['kind','Stay'], axis=1, inplace=True)
train.head()
test.head()
X = train[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

        'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit']]

y = train['Stay']

X_predict = test[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

        'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit']]

y=y.astype('int')
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X = std.fit_transform(X)

X_predict = std.transform(X_predict)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle = True)
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(solver='sag',max_iter =1000)

model1.fit(X_train,y_train)

model1.score(X_test,y_test)
train_dataset = Pool(data=X_train, label=y_train)

eval_dataset = Pool(data=X_test, label=y_test)

model = CatBoostClassifier(iterations=750,

                           learning_rate=0.08,

                           depth=7,

                           loss_function='MultiClass',

                           eval_metric='Accuracy')



model.fit(train_dataset)
model.get_best_score()
eval_pred = model.predict(eval_dataset)
cm = confusion_matrix(y_test, eval_pred)

cm
test_dataset = Pool(X_predict)
y_pred = model.predict(test_dataset)
y_pred
pd.DataFrame(y_pred).to_csv("prediction.csv")
output = pd.DataFrame(test['case_id'].values,columns=['case_id'])
output
len(y_pred)
len(X_predict)
prediction = pd.read_csv('./prediction.csv',names=['Stay'],header=0)
prediction.loc[prediction['Stay'] == 0, 'Stay'] = '0-10'

prediction.loc[prediction['Stay'] ==1 , 'Stay'] = '11-20'

prediction.loc[prediction['Stay'] ==2 , 'Stay'] = '21-30'

prediction.loc[prediction['Stay'] ==3 , 'Stay'] = '31-40'

prediction.loc[prediction['Stay'] ==4 , 'Stay'] = '41-50'

prediction.loc[prediction['Stay'] ==5 , 'Stay'] = '51-60'

prediction.loc[prediction['Stay'] ==6 , 'Stay'] = '61-70'

prediction.loc[prediction['Stay'] ==7 , 'Stay'] = '71-80'

prediction.loc[prediction['Stay'] ==8 , 'Stay'] = '81-90'

prediction.loc[prediction['Stay'] ==9 , 'Stay'] = '91-100'

prediction.loc[prediction['Stay'] ==10, 'Stay'] = 'More than 100 Days'
output = pd.DataFrame(test['case_id'].values,columns=['case_id'])
output['Stay'] = prediction
output.head()
output.to_csv('Healthcare_Submission.csv',index=False)