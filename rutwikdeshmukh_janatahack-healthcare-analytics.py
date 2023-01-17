import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")
sns.set()
df = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')
test_data = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')
df.shape
df.isnull().sum()
df['City_Code_Patient'] = df['City_Code_Patient'].fillna(8.0)
test_data['City_Code_Patient'] = test_data['City_Code_Patient'].fillna(8.0)
df['Bed Grade'] = df['Bed Grade'].fillna(2.0)
test_data['Bed Grade'] = test_data['Bed Grade'].fillna(2.0)
data = df.groupby('Hospital_code')['patientid'].count().reset_index()
data.columns = ['Hospital','Count']

fig = px.bar(
    data,
    x='Hospital',
    y='Count',
    width=900,
    height=600,
    title='Patient count per Hospital'
)

fig.show()
data = df.groupby(['Available Extra Rooms in Hospital'])['patientid'].count().reset_index()
data.columns = ['Available Extra Rooms in Hospital','Count']

fig = px.bar(
    data,
    x='Available Extra Rooms in Hospital',
    y='Count',
    width=900,
    height=600,
    title='Count of Available Extra Rooms in Hospital'
)

fig.show()
data = df.groupby(['Department'])['patientid'].count().reset_index()
data.columns = ['Department','Count']

fig = px.bar(
    data,
    x='Department',
    y='Count',
    width=600,
    height=600,
    title='Cases per Department'
)

fig.show()
data = df.groupby(['Stay'])['patientid'].count().reset_index()
data.columns = ['Stay','Count']

fig = px.bar(
    data,
    x='Stay',
    y='Count',
    width=900,
    height=600,
    title='Stay'
)

fig.show()
data = df.groupby(['Age'])['patientid'].count().reset_index()
data.columns = ['Age','Patient Count']

fig = px.bar(
    data,
    x='Age',
    y='Patient Count',
    width=900,
    height=600,
    title='Age-wise distribution of patients'
)

fig.show()
data = df.groupby(['Visitors with Patient'])['patientid'].count().reset_index()
data.columns = ['Visitors with Patient','Patient Count']

fig = px.bar(
    data,
    x='Visitors with Patient',
    y='Patient Count',
    width=900,
    height=600,
    title='Visitors with Patient'
)

fig.show()
data = df.groupby(['Type of Admission'])['patientid'].count().reset_index()
data.columns = ['Type of Admission','Count']

fig = px.bar(
    data,
    x='Type of Admission',
    y='Count',
    width=500,
    height=500,
    title='Type of Admissions'
)

fig.show()
#Separating Categorical and Numerical Data

cols_to_label=[]
for i in df.columns:
    if df[i].dtypes == 'O':
        cols_to_label.append(i)

cols_to_label
df['Bill_per_patient'] = df.groupby('patientid')['Admission_Deposit'].transform('sum')

test_data['Bill_per_patient'] = test_data.groupby('patientid')['Admission_Deposit'].transform('sum')
cols_to_label = ['Hospital_code','City_Code_Hospital','Hospital_type_code','Available Extra Rooms in Hospital','Bed Grade','City_Code_Patient','Visitors with Patient','Hospital_region_code','Department','Ward_Type','Ward_Facility_Code','Type of Admission','Severity of Illness','Age','Stay']
cols_to_label2 = ['Hospital_code','City_Code_Hospital','Hospital_type_code','Available Extra Rooms in Hospital','Bed Grade','City_Code_Patient','Visitors with Patient','Hospital_region_code','Department','Ward_Type','Ward_Facility_Code','Type of Admission','Severity of Illness','Age']
#Encoding the dataset

df[cols_to_label] = df[cols_to_label].apply(LabelEncoder().fit_transform)
test_data[cols_to_label2] = test_data[cols_to_label2].apply(LabelEncoder().fit_transform)
df['Stay'].value_counts()
df.head()
#Scaling the dataset

s_scaler = StandardScaler()

data = s_scaler.fit_transform(df.drop('Stay',axis=1))
data2 = s_scaler.fit_transform(test_data)
#Plotting Correlation Heatmap

plt.subplots(figsize=(10,7))
sns.heatmap(df.corr(),cmap='coolwarm_r')
data = pd.DataFrame(data)
data.columns = ['case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',
       'Hospital_region_code', 'Available Extra Rooms in Hospital',
       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',
       'patientid', 'City_Code_Patient', 'Type of Admission',
       'Severity of Illness', 'Visitors with Patient', 'Age',
       'Admission_Deposit','Bill_per_patient']
data = data.drop(['case_id'], axis=1)
data2 = pd.DataFrame(data2)
data2.columns = ['case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',
       'Hospital_region_code', 'Available Extra Rooms in Hospital',
       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',
       'patientid', 'City_Code_Patient', 'Type of Admission',
       'Severity of Illness', 'Visitors with Patient', 'Age',
       'Admission_Deposit','Bill_per_patient']
data2 = data2.drop(['case_id'], axis=1)
X = data
y = df['Stay']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
X_train.shape, y_train.shape
gnb = GaussianNB()
gnb.fit(X_train, y_train)
score = gnb.score(X_train, y_train)
print(f'The Train accuracy of the GaussianNB model is : {score}')
predictions = gnb.predict(X_test)
acc = accuracy_score(predictions, y_test)
print(f'The Test accuracy of the GaussianNB model is : {acc}')
lgb_cl = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=5000, importance_type='gain', objective='multiclass', num_boost_round=100,
                            num_leaves=300, max_depth=5, 
                            max_bin=60, bagging_faction=0.9, feature_fraction=0.9, subsample_freq=2, scale_pos_weight=2.5, 
                            random_state=1994, n_jobs=-1, silent=False)

lgb_cl.fit(X_train, y_train)
preds = lgb_cl.predict(X_test)
atrain = round(lgb_cl.score(X_train, y_train)*100,2)
acc = round(accuracy_score(preds, y_test)*100,2)
print(f'The Train accuracy of the LGBMClassifier model is: {atrain}%')
print(f'The Test accuracy of the LGBMClassifier model is: {acc}%')
final_predictions = lgb_cl.predict(data2)
final_predictions = pd.DataFrame(final_predictions)
final_predictions[0] = final_predictions[0].map({0:'0-10',1:'11-20',2:'21-30',3:'31-40',4:'41-50',5:'51-60',6:'61-70',7:'71-80',8:'81-90',9:'91-100',10:'More than 100 Days'})
final_preds = pd.DataFrame(columns=['case_id','Stay'])
final_preds['case_id'] = test_data['case_id']
final_preds['Stay'] = final_predictions[0]
final_preds
final_submission_data = final_preds.to_csv('submissions.csv', index=False)