# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows',None)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_raw = pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')

# drop columns won't be used.
data = data_raw.drop(['Survived','Cabin','Name','Ticket'],axis=1)


# Any results you write to the current directory are saved as output.
# Preprocess Missing data and object data.

# Impute Embarked using fillna with most frequent value(S).
data.Embarked.fillna('S',inplace=True)

# Encode Sex and Embarked
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder(handle_unknown='ignore')
Embarked_enc=pd.DataFrame(one_hot_enc.fit_transform(data[['Embarked']]).toarray(),
                         index=data.index, 
                         columns = one_hot_enc.get_feature_names(['Embarked']))
data_enc=pd.concat([Embarked_enc,data],axis=1).drop(['Embarked'],axis=1)


data_enc['Sex'] = np.where(data_enc.Sex=='female',1,0)

# Extract honorifics from names that contain the infomation in terms of passengers's Age.
list = []
for i in range(1,len(data_raw)+1):
    list.append(data_raw.Name.loc[i].split(",")[1].split()[0])
    
honorifics=pd.DataFrame(data = list , columns=['honorifics'], index = data_raw.index)
honorifics.honorifics=np.where(honorifics.honorifics.isin(['Mr.','Miss.','Mrs.','Master.','Dr.']),
                    honorifics.honorifics, 'Rare')

# This concat is for checking whether extracting was done well.
name=pd.concat([honorifics,data_raw.Name],axis=1)

# Encode it and concatenate it to data_enc
_name=pd.DataFrame(one_hot_enc.fit_transform(honorifics[['honorifics']]).toarray(),
            index = honorifics.index, columns = one_hot_enc.get_feature_names(['']))
data_enc_name = pd.concat([data_enc,_name],axis=1)

# Split into two data set.
train = data_enc_name[data.Age.isnull()==False]
test = data_enc_name[data.Age.isnull()==True]
plt.figure(figsize=(14,10))
relevant=['Pclass','SibSp','Parch','_Master.','_Miss.','_Mrs.','_Miss.','_Rare']
sns.heatmap(train.corr(),annot=True,center=0,cmap= 'coolwarm')
train_relevant=train[relevant]

# Split and train model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_relevant, train.Age, 
                                                    test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=1000, random_state=0)
rfr.fit(x_train, y_train)

# Get predication array to inpect how accurate the model is.
pred=rfr.predict(x_test)
MAE_rfr=np.abs(pred-y_test).mean()

# To inspect, Categorize pred and y_test date.
pred_cat=pd.cut(x=pred, bins=[0,18,30, pred.max()], labels=[2,0,1])
y_test_cat=pd.cut(x=y_test, bins=[0,18,30, pred.max()], labels=[2,0,1])

# Inspect how accurate the model is.
result_rfr = np.where(pred_cat==y_test_cat,1,0)
accuracy_rfr = result_rfr.sum()/len(result_rfr)
print("The accuracy from Random Forest Regression :",accuracy_rfr)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)

# To use classifier, Categorize data first.
y_train_cat = pd.cut(x = y_train,bins=[0,18,30, y_train.max()], labels=[2,0,1])
y_test_cat = pd.cut(x = y_test,bins=[0,18,30, y_test.max()], labels=[2,0,1])

rfc.fit(x_train , y_train_cat)
pred = rfc.predict(x_test)

result_rfc = np.where(pred==y_test_cat, 1, 0)
accuracy_rfc = result_rfc.sum()/len(result_rfc)
print("The accuracy from Random Forest Classifier :",accuracy_rfc)