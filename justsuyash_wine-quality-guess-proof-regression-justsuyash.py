# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



#import the necessary modelling algos.

from sklearn.decomposition import PCA



#classifiaction.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#preprocessing

from sklearn.preprocessing import StandardScaler



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification
data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data['quality'].unique()
sp = data['quality'].value_counts()

sp = pd.DataFrame(sp)

sp.T
sns.barplot(x = sp.index, y=sp['quality'])

plt.xlabel("Quality Score")

plt.ylabel("Count")
plt.figure(figsize = (16,7))

sns.set(font_scale=1.2)

sns.heatmap(data.corr(), annot=True, linewidths=0.5, cmap='YlGnBu')
data_cleaned = data
data_cleaned.isnull().sum()
data_cleaned.describe()
data_try  = data_cleaned
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

data_try['category'] = pd.cut(data_try['quality'], bins = bins, labels = group_names)
data_try['category'].value_counts()
from sklearn.model_selection import train_test_split
data_cleaned.columns
x1 = data_try[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']]



y1 = data_try['category']
x_train_dummy,x_test_dummy,y_train_dummy,y_test_dummy = train_test_split(x1,y1,test_size = 0.3, random_state=42)
y_test_dummy.count()
y_dummy_predict = []

y_dummy_predict = ['bad']*y_test_dummy.count()
accuracy_score(y_dummy_predict,y_test_dummy)
print(classification_report(y_test_dummy, y_dummy_predict))
data_cleaned.info()
quality =  data_cleaned['quality'].values



category_balanced = []



for num in quality:

    if num<=5:

        category_balanced.append('bad')

    elif num>=6:

        category_balanced.append('good')  
category_balanced  = pd.DataFrame(data=category_balanced,columns=['category_balanced'])
category_balanced.isnull().sum()
data_cleaned = pd.concat([data_cleaned,category_balanced],axis=1)
data_cleaned = data_cleaned.dropna()
data_cleaned['category_balanced'].value_counts()
x = data_cleaned[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']]

y = data_cleaned['category_balanced']
scl = StandardScaler()
x = scl.fit_transform(x)
pca = PCA()
x_pca = pca.fit_transform(x)
plt.figure(figsize=(10,10))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
pca_new = PCA(n_components=8)
x_pca_8 = pca_new.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=420)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
models=[LogisticRegression(),SVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),

        DecisionTreeClassifier(),GradientBoostingClassifier()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',

             'GradientBoostingClassifier','GaussianNB']



acc=[]

d={}



for model in range(len(models)):

    clf=models[model]

    clf.fit(x_train,y_train)

    pred=clf.predict(x_test)

    acc.append(accuracy_score(pred,y_test))

     

d={'Modelling Algo':model_names,'Accuracy':acc}

d
print(classification_report(y_test,y_dummy_predict ))