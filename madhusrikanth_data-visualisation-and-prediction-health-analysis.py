import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import cufflinks as cf

train = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')
train.head()
train.info()
#rows : 318438
#columns : 18
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#There are null values in city_code_patient column and its better to drop that column to get accurate results.
train['Ward_Type'].unique()
def change(ch):
    if(ch=='R'):
        return 0
    elif(ch=='S'):
        return 1
    elif(ch=='Q'):
        return 2
    elif(ch=='P'):
        return 3
    elif(ch=='T'):
        return 4
    elif(ch=='U'):
        return 5
train['Ward_Type']=train['Ward_Type'].apply(change)
#Only numerical data will be accpeted by machine learning model
#Drop the columns having null value
train.drop('City_Code_Patient',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#now no null values
train.groupby('patientid')['Bed Grade'].mean().sort_values(ascending=False).head()
train.groupby('Bed Grade')['Bed Grade'].count().sort_values(ascending=False).head()
sns.countplot(train['Type of Admission'],hue=train['Department'],palette='rainbow')
train['Age'].hist(bins=20,figsize=(10,4))
def change1(ch):
    if(ch=='Extreme'):
        return 0
    elif(ch=='Minor'):
        return 1
    elif(ch=='Moderate'):
        return 2
train['Severity of Illness'] = train['Severity of Illness'].apply(change1)
sns.countplot(train['Severity of Illness'],hue=train['Department'],palette='rainbow')
train['Admission_Deposit'].hist(bins=100,figsize=(10,4))
train.groupby('Severity of Illness')['Admission_Deposit'].mean().sort_values(ascending=False)
train['Stay'].hist(bins=20,figsize=(10,4))
train.drop(['Ward_Facility_Code','Hospital_region_code','Hospital_type_code'],axis=1,inplace=True)
train.drop('Age',inplace=True,axis=1)
def change3(ch):
    if ch=='radiotherapy':
        return 0
    elif ch== 'anesthesia':
        return 1
    elif ch=='gynecology':
        return 2
    elif ch== 'TB & Chest disease':
        return 3
    elif ch== 'surgery':
        return 4
train['Department'] = train['Department'].apply(change3)
def change4(ch):
    if ch=='Emergency':
        return 0
    elif ch=='Trauma':
        return 1
    elif ch =='Urgent':
        return 2
train['Type of Admission'] = train['Type of Admission'].apply(change4)
#Data type of stay is converted to numbers[0-9]
train['Stay']=train.Stay.astype("category").cat.codes
#number is assinged(ascending)
train['Stay'].unique()
#Totally ten different categories
from sklearn.model_selection import train_test_split
train.drop(['City_Code_Hospital','Visitors with Patient'],inplace=True,axis=1)
train.drop('Bed Grade',inplace=True,axis=1)
#Train data and test data(to predict number of stay)
X = train.drop('Stay',axis=1,inplace=False)
Y = train['Stay']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=2)
kn.fit(X_train,Y_train)
print(kn.score(X_train,Y_train)*100)
test = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')
test
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#city_code_patient contains null values so we have to remove it
test.drop(['Hospital_type_code','City_Code_Hospital','Hospital_region_code','Ward_Facility_Code','Bed Grade','City_Code_Patient','Visitors with Patient','Age'],inplace=True,axis=1)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['Ward_Type']=test['Ward_Type'].apply(change)
test['Severity of Illness'] = test['Severity of Illness'].apply(change1)
test['Department'] = test['Department'].apply(change3)
test['Type of Admission'] = test['Type of Admission'].apply(change4)
test
#All columns are numeric and it is ready to be fed into algorithm.
#Prediction of values for the stay
predict1 = kn.predict(test)
test['predict'] = predict1
key_value={
0:'0-10',
1:'11-20',
2:'21-30',
3:'31-40',
4:'41-50',
5:'51-60',
6:'61-70',
7:'71-80',
8:'81-90',
8:'91-100'
}
test['value'] = test.predict.replace(key_value)
values_arr = np.array(test['case_id'])
predict = np.array(test['value'])
#Final DataFrame
df = pd.DataFrame(data=[values_arr,predict],index=['case_id','Stay'])
df
