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



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc

import psutil



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.decomposition import PCA, KernelPCA



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,

                             make_scorer,classification_report,roc_auc_score,roc_curve,

                             average_precision_score,precision_recall_curve)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier



from imblearn.pipeline import make_pipeline

from imblearn.under_sampling import OneSidedSelection

from imblearn.under_sampling import NearMiss

from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.under_sampling import TomekLinks

from imblearn.under_sampling import RandomUnderSampler



pd.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings("ignore")



RANDOM_SEED = 101



import collections
import pandas as pd

task_InputData = pd.read_csv("../input/task_InputData.csv")
task_InputData.head() #To have a glance at data*/ 
task_InputData['label'].value_counts().plot.bar() #Target variable is 'label' and it is categorical*/
task_InputData.info() #It defines the column as in no of non null columns and also give its type
task_InputData.shape #It tells no of rows,columns of the data
task_InputData.describe() 
task_InputData.columns #Can check the columns by it
sns.distplot(task_InputData['age']) #People are of age in b/w 10 to 77
sns.distplot(task_InputData['zip code'])
sns.kdeplot(task_InputData['earnings'], shade=True) #People are of varying earnings ranging 0 to 155000
task_InputData['lifestyle'].value_counts().plot.bar() #All type of lifestyle are equal in frequency. So might not cause change.
task_InputData['family status'].value_counts().plot.bar() #All type of family status are equal in frequency. So might not cause change.
task_InputData['car'].value_counts().plot.bar()
task_InputData['sports'].value_counts().plot.bar() ##All type of sports are equal in frequency. So might not cause change.
task_InputData['Living area'].value_counts().plot.bar() #The frequency of urban people are more than rural.
sns.boxplot(x='label',

            y='age',

            data=task_InputData)

plt.xlabel('Reaction')

plt.ylabel('Age')

plt.title('Distribution of Age with respect to label',fontsize=10)

#Mostly the people not responding mean age is approx 37 while mean age of responding people is around 60 

#i.e old people are likely to respond to mail
sns.boxplot(x='label',

            y='earnings',

            data=task_InputData)

plt.xlabel('Reaction')

plt.ylabel('Earnings')

plt.title('Distribution of Earnings with respect to label',fontsize=10)

#95% approx people who responding are earning high while not responding people are those esrning less
num_cols=['age','earnings']
sns.pairplot(task_InputData[num_cols]) #We cannot differentiate by only age and earnings cause its evenly distributed.
cat_cols=['lifestyle', 'family status', 'car',

       'sports', 'Living area']
for num_col in num_cols:

    fig = plt.figure(figsize = (30,10))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxplot(y = task_InputData[num_col],

                    x = task_InputData[cat_col], 

                    data = task_InputData, 

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('{} with respect to {}'.format(num_col,cat_col), fontsize = 20)

        j = j + 1

        #When each numerical column is being analysed with each of the categorical column 
for num_col in num_cols:

    fig = plt.figure(figsize = (30,10))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxplot(y = task_InputData[num_col],

                    x = task_InputData[cat_col], 

                    data = task_InputData,

                    hue= task_InputData['label'],

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('{} with respect to {}'.format(num_col,cat_col), fontsize = 20)

        j = j + 1

        #Comparing by keeping target column as hue shows that the age and the earnings is an important feature varying label.
task_InputData['sports'].replace(np.nan,'other',inplace=True) #To remove null values with other
t_data, check_data = train_test_split(task_InputData, test_size = 0.2, random_state=20,stratify=task_InputData['label'])

#Dividing train as well as test data. So that train could be used for building model and test for predicting the output.
t_data.shape
check_data.shape
t_data.head()
check_data.head()
train_data, test_data = train_test_split(t_data, test_size = 0.2, random_state=20)
x_train = train_data.drop(['name','zip code','label'],axis=1)

y_train = train_data['label']
x_test = test_data.drop(['name','zip code','label'],axis=1)

y_test = test_data['label']
x_train = pd.get_dummies(x_train, prefix_sep='_', drop_first=True)

x_train.head()
x_test = pd.get_dummies(x_test, prefix_sep='_', drop_first=True)

x_test.head()
check_data = check_data.drop(['name','zip code','label'],axis=1)
x_check = pd.get_dummies(check_data, prefix_sep='_', drop_first=True)

# X head

x_check.head()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
accuracy_score(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=10, random_state=101, min_samples_leaf=15)

dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)
accuracy_score(y_test,y_pred)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='modified_huber', shuffle = True, random_state = 101)

sgd.fit(x_train, y_train)

y_pred= sgd.predict(x_test)
accuracy_score(y_test,y_pred)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(x_train, y_train)

y_pred= knn.predict(x_test)
accuracy_score(y_test,y_pred)
