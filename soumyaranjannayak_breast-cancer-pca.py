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
data=pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
print(data.head())

print(data.tail())
sns.countplot(data['diagnosis'])

plt.show()
print(data.describe())
print(data.columns)
print(data.info())
sns.pairplot(data,hue='diagnosis')

plt.show()
sns.heatmap(data.corr(),annot=True)

plt.plot()
x=data.iloc[:,:-1]

y=data.iloc[:,-1]
from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()

pd.DataFrame(scalar.fit_transform(x))
from sklearn.decomposition import PCA

pca=PCA(n_components=1)

pca.fit(x)

data_new=pd.DataFrame(pca.transform(x))

data_new.columns=['PC1']

data_new.index=data.index

print(pca.explained_variance_ratio_)
from sklearn.linear_model import  LogisticRegression

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.tree import  DecisionTreeClassifier
models=[('random_forest',RandomForestClassifier()),('DecisionTreeClassifier',DecisionTreeClassifier()),

       ('AdaBoost',AdaBoostClassifier())]
from sklearn.model_selection import cross_val_score

def classification(name,model):

    score=cross_val_score(model,x,y,cv=5)

    print(name,' : ',score,'\naverage accuracy',round(score.mean()*100,2),'%')
for name,model in models:

    classification(name,model)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0) 
model=RandomForestClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

print('confusion_matrix\n',confusion_matrix(y_test,y_pred))

print('\n\nclassification_report\n',classification_report(y_test,y_pred))