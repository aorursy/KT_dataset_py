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
# importing required librery



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.pandas.set_option('display.max_columns',None)
data = pd.read_csv("/kaggle/input/summeranalytics2020/train.csv")

data2 = pd.read_csv("/kaggle/input/summeranalytics2020/test.csv")

data_y = data.Attrition

data.head()
data.shape
data2.shape
data.info()
data2.info()
data.describe()
import pandas_profiling

data.profile_report()
data = data.drop(['Id','Attrition','EmployeeNumber','Behaviour'],axis=1)

data2 = data2.drop(['Id','EmployeeNumber','Behaviour'],axis=1)
catagory_col = [col for col in data.columns if data[col].dtype=='object']

catagory_col
for column in data[catagory_col]:

    print(str(column) + str(' : ') + str(data[column].unique()))

    print(data[column].value_counts())

    print('____________________________________________________')

    print('')
# import librery for lable encoding



from sklearn.preprocessing import LabelEncoder



l_encoder = LabelEncoder()



for column in catagory_col:

    data[column+'_n'] = l_encoder.fit_transform(data[column])

    data2[column+'_n'] = l_encoder.fit_transform(data2[column])

    data = data.drop([column],axis=1)

    data2 = data2.drop([column],axis=1)
"""



temp = pd.get_dummies(data[catagory_col])

data = pd.concat([data,temp],axis=1)

data = data.drop(catagory_col,axis=1)



temp2 = pd.get_dummies(data2[catagory_col])

data2 = pd.concat([data2,temp2],axis=1)

data2 = data2.drop(catagory_col,axis=1)
data.head()
data.shape
# make set ofcolumn required standerdizing



# runonly if we want to modeling on nontree based model



""" 



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



data_arr = scaler.fit_transform(data)

data2_arr = scaler.fit_transform(data2)



data = pd.DataFrame(data_arr,columns = data.columns)

data2 = pd.DataFrame(data2_arr,columns = data2.columns)
data.head()
data.shape
plt.subplots(figsize=(10,4))

sns.boxplot(x=data.MonthlyIncome)

plt.figure()

plt.subplots(figsize=(10,4))

sns.boxplot(x=data2.MonthlyIncome)
sns.distplot(data.Age,bins=100)

sns.distplot(data2.Age,bins=100)
sns.distplot(data.MonthlyIncome,bins=100)

sns.distplot(data2.MonthlyIncome,bins=100)
sns.distplot(data.NumCompaniesWorked,bins=50)

sns.distplot(data2.NumCompaniesWorked,bins=50)
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

feature = ExtraTreesClassifier()

feature.fit(data,data_y)



score = feature.feature_importances_

score
feat_score = pd.Series(score, index=data.columns).sort_values(ascending = False)



plt.figure(figsize=(10,20))

feat_score.plot(kind='barh')

plt.show()
top_feature = list(feat_score.index[0:29])

top_feature
feat_score
data_imp = data[top_feature]

data2_imp = data2[top_feature]



print(data_imp.shape)

print(data2_imp.shape)
top_corr_feat = []

for i in data.columns:

    if(abs(data_y.corr(data[i]))>=0.01):

        top_corr_feat.append(i)

        

print(top_corr_feat)
data_imp = data[top_corr_feat]

data2_imp = data2[top_corr_feat]



print(data_imp.shape)

print(data2_imp.shape)
# train test split



from sklearn.model_selection import train_test_split as tts



x_train,x_test,y_train,y_test = tts(data_imp,data_y,test_size=0.3,random_state=4)
x_train.shape
x_test.shape
from sklearn.ensemble import RandomForestClassifier as RFS
model = RFS(random_state=24,n_estimators=250,max_depth=20)



#model.fit(x_train,y_train)



model.fit(data_imp,data_y)
y_predict_1 = model.predict(x_test)

y_predict_2 = model.predict(data_imp)
from sklearn.metrics import roc_auc_score



roc_auc_score(y_test,y_predict_1)
roc_auc_score(data_y,y_predict_2)
y_prob = model.predict_proba(data2_imp)

y_prob = list(y_prob[:,1])

data2_predict = model.predict(data2_imp)

data2_predict
Id = np.arange(1,len(y_prob)+1)



Id = list(Id)
ans = pd.DataFrame(list(zip(Id,y_prob)),columns=['Id','Attrition'])

ans
ans.to_csv('answer4.csv',index=False)