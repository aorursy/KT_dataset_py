# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.columns.values
data['Cover_Type'].value_counts()
import matplotlib.pyplot as plt



plt.scatter(data['Cover_Type'], data['Elevation'])
groups = data.groupby('Cover_Type')
groups.boxplot(column='Elevation')
groups.boxplot(column='Aspect')
plt.scatter(data['Cover_Type'], data['Aspect'])
plt.scatter(data['Cover_Type'], data['Slope'])
plt.scatter(data['Cover_Type'], data['Horizontal_Distance_To_Hydrology'])
data.corr()['Cover_Type']
plt.boxplot(data['Soil_Type5'])
sub = pd.read_csv('../input/sampleSubmission.csv')
from sklearn.preprocessing import LabelEncoder

import random

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train['Type']='Train' #Create a flag for Train and Test Data set

test['Type']='Test'

fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set
fullData.columns # This will show all the column names

fullData.head(10) # Show first 10 records of dataframe

fullData.describe() #You can look at summary of numerical fields by using describe() function
data.columns
ID_col = ['Id']

target_col = ["Cover_Type"]

cat_cols = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))

other_col=['Type'] #Test and Train Data set identifier
fullData.isnull().any()
num_cat_cols = num_cols+cat_cols
for var in num_cat_cols:

    if fullData[var].isnull().any()==True:

        fullData[var+'_NA']=fullData[var].isnull()*1
fullData['Cover_Type'] = fullData['Cover_Type'].fillna(value = -9999)
#create label encoders for categorical features

for var in cat_cols:

     number = LabelEncoder()

     fullData[var] = number.fit_transform(fullData[var].astype('str'))



#Target variable is also a categorical so convert it

fullData["Cover_Type"] = number.fit_transform(fullData["Cover_Type"].astype('str'))



train=fullData[fullData['Type']=='Train']

test=fullData[fullData['Type']=='Test']



train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75

Train, Validate = train[train['is_train']==True], train[train['is_train']==False]
features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))
x_train = Train[list(features)].values

y_train = Train["Cover_Type"].values

x_validate = Validate[list(features)].values

y_validate = Validate["Cover_Type"].values

x_test=test[list(features)].values
random.seed(100)

rf = RandomForestClassifier(n_estimators=1000)

rf.fit(x_train, y_train)
status = rf.predict_proba(x_validate)

fpr, tpr, _ = roc_curve(y_validate, status[:,1])

roc_auc = auc(fpr, tpr)

print(roc_auc)



final_status = rf.predict_proba(x_test)

test["Cover_Type"]=final_status[:,1]
data.load_iris()
from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

y_pred = gnb.fit(data[['Elevation', 'Slope']], data['Cover_Type']).predict(data[['Elevation', 'Slope']])
print("Number of mislabeled points out of a total %d points : %d" % (data[['Elevation', 'Slope']].shape[0],(data['Cover_Type'] != y_pred).sum()))
plt.scatter(y_pred, data['Cover_Type'])
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data['Cover_Type'], y_pred)
plt.imshow(cm)

plt.colorbar(label='Number of Rows')

plt.ylabel('True category')

plt.xlabel('Predicted category')
test = pd.read_csv('../input/test.csv')



from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(data[['Elevation', 'Slope']], data['Cover_Type'])

train_pred = model.predict(data[['Elevation', 'Slope']])

test_pred = model.predict(test[['Elevation', 'Slope']])





print("Train %d points : %d" % (data[['Elevation', 'Slope']].shape[0],(data['Cover_Type'] != train_pred).sum()))

print("Test %d points : %d" % (test[['Elevation', 'Slope']].shape[0],(test['Cover_Type'] != test_pred).sum()))
test.columns
test