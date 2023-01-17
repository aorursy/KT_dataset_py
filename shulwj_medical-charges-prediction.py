#Basic libs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Ml libs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#input data
train_data=pd.read_csv('../input/insurance.csv')
#floar to int
train_data['bmi']=train_data['bmi'].astype(int)
train_data['charges']=train_data['charges'].astype(int)
# Binarization Processing of smoker
train_data['smoker']=train_data['smoker'].map({'yes':1,'no':0})
train_data['region']=train_data['region'].map({'southwest':0,'northwest':1,'southeast':2,'northeast':3})
train_data['sex']=train_data['sex'].map({'female':0,'male':1})
train_data.head(10)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#To see the dependencies of each features, we can see from the following figure that in addition to smoker and charges correlation is bigger, other features correlation is not big, which means each is useful;
train_data.describe()
#other people's way to see the features relationships 
sns.pairplot(train_data, size=2)
plt.show()
train_data[['age','charges']].groupby(['age'],as_index=False).mean()
#The high age,the high charges
train_data[['sex','charges']].groupby(['sex'],as_index=False).mean()
#men have high charges
train_data[['smoker','charges']].groupby(['smoker'],as_index=False).mean()
#smoker has high charges
train_data[['region','charges']].groupby(['region'],as_index=False).mean()
#We can see that the region has little influnce of charges;
train_data[['bmi','charges']].groupby(['bmi'],as_index=False).mean()
train_data[['children','charges']].groupby(['children'],as_index=False).mean()
#2 and 3 chilren have high charges
#split the data into 80% for training and 20 for testing
new_data=train_data
sep=int(0.8*len(new_data))
train=new_data[:sep]
test=new_data[sep:]
train_data=train.iloc[:,:6]
train_target=train.iloc[:,6:]
test_data=test.iloc[:,:6]
test_target=test.iloc[:,6:]
train_data.tail()
test_data.head()
train_data.info()
print('--'*20)
test_data.info()
test_target.head()
#LogistRegression
medical=LogisticRegression().fit(train_data,train_target)
result=medical.predict(test_data)
print(result)
test_target.as_matrix(['charges']).T
