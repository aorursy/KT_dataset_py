# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head(5)
data.info()
data['salary'].fillna(0,inplace = True)
def plot(data,x,y):
    plt.Figure(figsize =(10,10))
    sns.boxplot(x = data[x],y= data[y])
    g = sns.FacetGrid(data, row = y)
    g = g.map(plt.hist,x)
    plt.show()
plot(data,"salary","gender")
sns.countplot(data['status'],hue=data['gender'])
plot(data,"salary","ssc_b")
sns.countplot(data['status'],hue=data['ssc_b'])
plot(data,"salary","hsc_b")
sns.countplot(data['status'],hue=data['hsc_b'])
from scipy.stats import pearsonr
corr, _ = pearsonr(data['ssc_p'], data['hsc_p'])
print('Pearsons correlation: %.3f' % corr)
sns.regplot(x='ssc_p',y='hsc_p',data = data)
sns.jointplot(x="degree_p",y="salary", data=data ,kind= "regplot")
corr, _ = pearsonr(data['salary'], data['degree_p'])
print(corr)
sns.pairplot(data)
plt.figure(figsize =(10,10))
sns.heatmap(data.corr())
list1 =[]
cor = data.corr()
for i in cor.columns:
    for j in cor.columns :
        if abs(cor[i][j])>0.5 and i!=j:
            list1.append(i)
            list1.append(j)
print(set(list1))
            
sns.catplot(x="status", y="ssc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="hsc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="degree_p", data=data,kind="swarm",hue='gender')
sns.violinplot(x="degree_t", y="salary", data=data)
sns.stripplot(x="degree_t", y="salary", data=data,hue='status')
columns_needed =['gender','ssc_p','ssc_b','hsc_b','hsc_p','degree_p','degree_t']
data_x = data[columns_needed]
data_x.info()
def cat_to_num(data_x,col):
    dummy = pd.get_dummies(data_x[col])
    del dummy[dummy.columns[-1]]#To avoid dummy variable trap
    data_x= pd.concat([data_x,dummy],axis =1)
    return data_x
for i in data_x.columns:
    if data_x[i].dtype ==object:
        print(i)
        data_x =cat_to_num(data_x,i)
data_x.drop(['gender','ssc_b','hsc_b','degree_t'],inplace =True,axis =1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['status'] = le.fit_transform(data['status'])
y = data['status']
x = data_x
x.info()
y.value_counts()
from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier as DTC
model = DTC()
model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,model.predict(X_test)))
model.predict(X_test)
ax1 = sns.distplot(y_test,hist=True,kde =False,color ="r",label ="Actual Value")
sns.distplot(model.predict(X_test),color ="b",hist = True,kde =False, label = "Preicted Value",ax =ax1)
!pip install pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
plt.figure(figsize=(100,70))
Image(graph.create_png())