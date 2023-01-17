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





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/av-janatahack-machine-learning-in-agriculture/train_yaOffsB.csv")

test = pd.read_csv("../input/av-janatahack-machine-learning-in-agriculture/test_pFkWwen.csv")

train.head()
train.shape
def get_summary(df):

  summary = pd.DataFrame(index = df.columns)

  summary['data_types'] = df.dtypes 

  summary['null_values'] = df.isnull().sum()

  summary['unique_values'] = df.nunique()

  return summary
summary = get_summary(train)

summary
# Dependent and independent variable

# Train data

x = train.drop(['Crop_Damage','ID'], axis = 1) # independent variable

y = train['Crop_Damage']  # dependent variable



# Test Data

x_test = test # independent variable







original = train
# plt.figure()

# x['Estimated_Insects_Count'].plot.hist(bins = 10)

# x['Estimated_Insects_Count'].plot.bar()

plt.figure(1)

plt.subplot(121)

sns.distplot((x['Estimated_Insects_Count']))



plt.subplot(122)

x['Estimated_Insects_Count'].plot.box(figsize=(16,5))

plt.show()


# Right Skewness can be removed by using square root transformation

plt.figure(1)

plt.subplot(121)

sns.distplot((np.sqrt(x['Estimated_Insects_Count'])))



plt.subplot(122)

np.sqrt(x['Estimated_Insects_Count']).plot.box(figsize=(16,5))

plt.show()
plt.figure()

x['Number_Doses_Week'].plot.hist()

# x['Estimated_Insects_Count'].plot.bar()



plt.figure(1)

plt.subplot(121)

sns.distplot(x['Number_Doses_Week'])



plt.subplot(122)

x['Number_Doses_Week'].plot.box(figsize=(16,5))

plt.show()
plt.figure(1)

plt.subplot(121)

sns.distplot((np.sqrt(x['Number_Doses_Week'])))



plt.subplot(122)

np.sqrt(x['Number_Doses_Week']).plot.box(figsize=(16,5))

plt.show()
# plt.figure()

# x['Number_Weeks_Used'].plot.hist()



plt.figure(1)

plt.subplot(121)

sns.distplot(x['Number_Weeks_Used'])



plt.subplot(122)

x['Number_Weeks_Used'].plot.box(figsize=(16,5))

plt.show()
# plt.figure()

# x['Number_Weeks_Quit'].plot.hist()







plt.figure(1)

plt.subplot(121)

sns.distplot(x['Number_Weeks_Quit'])



plt.subplot(122)

x['Number_Weeks_Quit'].plot.box(figsize=(16,5))

plt.show()


plt.figure(1)

plt.subplot(121)

sns.distplot(np.power((x['Number_Weeks_Quit']+0.1)*100,1/2))



plt.subplot(122)

np.log((x['Number_Weeks_Quit']+0.1)*100).plot.box(figsize=(16,5))

plt.show()
# x['Estimated_Insects_Count'] = np.sqrt(x['Estimated_Insects_Count'])

# x['Number_Doses_Week'] = np.sqrt(x['Number_Doses_Week'])

# x['Number_Weeks_Quit'] = np.log((x['Number_Weeks_Quit']+0.1)*100)
summary
x['Number_Weeks_Used'] = x['Number_Weeks_Used'].fillna(x['Number_Weeks_Used'].mode()[0]) 
get_summary(x)
test['Number_Weeks_Used'] = test['Number_Weeks_Used'].fillna(x['Number_Weeks_Used'].mode()[0]) 
get_summary(test)
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer([('standard',StandardScaler(),[0,4,5,6])], remainder = 'passthrough')

columns = x.columns

new_indices = [0,4,5,6,1,2,3,7]

new_columns = [columns[index] for index in new_indices]

x_scaled = pd.DataFrame(ct.fit_transform(x),columns = new_columns)
test_copy = test.drop(['ID'], axis = 1)
# Feature Scaling the test data

test_scaled = pd.DataFrame(ct.transform(test_copy),columns = new_columns)
# # Getting the dummy variable

# train_objs_num = len(x_scaled)

# dataset = pd.concat(objs=[x_scaled, test], axis=0)

# dataset_preprocessed = pd.get_dummies(dataset,columns = ['Crop_Type','Soil_Type','Pesticide_Use_Category','Season'])

# train_preprocessed = dataset_preprocessed[:train_objs_num]

# test_preprocessed = dataset_preprocessed[train_objs_num:]
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_score(model,x_train, x_test, y_train, y_test):

    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    return accuracy

    
# from sklearn.model_selection import KFold

# kf = KFold(n_splits = 6)

# score_l = []

# score_svm = []

# score_rf  = []

# for train_index, test_index in kf.split(x):

#     x_train, x_test, y_train, y_test = x.iloc[train_index],x.iloc[test_index],y.iloc[train_index],y.iloc[test_index]

#     score_l.append(get_score(LogisticRegression(max_iter = 1000),x_train, x_test, y_train, y_test))

#     score_svm.append(get_score(SVC(),x_train, x_test, y_train, y_test))

#     score_rf.append(get_score(RandomForestClassifier(),x_train, x_test, y_train, y_test))

    

    

        

        
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(x_scaled,y)



test_scaled
y_predict = gbc.predict(test_scaled)
print(y_predict)

predict = pd.DataFrame()

predict['ID'] = x_test['ID']

predict['Crop_Damage'] = y_predict

predict
predict.to_csv("Submission_7.csv", index = False)