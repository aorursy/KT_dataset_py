# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)
print(df_test.shape)
## as we have many column, set below properties, else all column will not be shown when you will describe
pd.set_option('display.max_columns', None) 
df_train.describe()
df_train.info()
## check if any column is null
df_train.isnull().sum().any()
df_train.columns
df_train.Cover_Type.value_counts()
## we can see 7 cover types are there and count is also same
data_corr = df_train.corr()
size=10
threshold=0.5
#create a dataframe with only 'size' features
data=df_train.iloc[:,:size] 

#get the names of all the columns
cols=data.columns 
corr_list=[]
for i in range(0,size):
    for j in range(i+1, size):
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j])
#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
## check the skewness of the data
df_train.skew()
col_list=df_train.columns
col_list = [col for col in col_list if not 'Soil' in col]
plt.figure(figsize=(12,8))
sns.heatmap(df_train[col_list].corr(),annot=True)
sns.pairplot(df_train[col_list],hue='Cover_Type')
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
X = df_train.drop(['Id','Cover_Type'],axis=1)
y = df_train['Cover_Type']
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
models = []
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('GB',GaussianNB()))
models.append(('RFC',RandomForestClassifier()))

results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold = model_selection.KFold(n_splits=20,random_state=12345)
    cv_results = model_selection.cross_val_score(model,X, y, cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)
rf = RandomForestClassifier()
train_X = df_train.drop(['Id','Cover_Type'],axis=1)
train_Y = df_train['Cover_Type']
test_x = df_test.drop('Id',axis=1)
from sklearn.model_selection import train_test_split
##Split the training set into training and validation sets
X_train,X_test,Y_train,Y_test = train_test_split(train_X,train_Y,test_size=0.2)
rf.fit(X_train,Y_train)
y_predict = rf.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(Y_test,y_predict))
print(accuracy_score(Y_test,y_predict))

test_id = pd.DataFrame(df_test.Id)
test_id.head()
test_id['Cover_Type']=rf.predict(df_test.drop('Id',axis=1))
test_id.head()
data_to_submit = pd.DataFrame({
    'Id':test_id['Id'],
    'Cover_Type':test_id['Cover_Type']
})
data_to_submit.head()
data_to_submit.to_csv('Forest_Cover_Type.submit.csv', index = False)

