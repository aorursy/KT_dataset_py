# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import figure

import os

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.formula.api as sm

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import SMOTENC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')
data.head()
#Checking for null values

data.isnull().sum()
data['left'].value_counts()
def count_target_plot(data,target):

    plt.figure(figsize=(8,8))

    ax=sns.countplot(data=data,x=data[target],order=data[target].value_counts().index)

    plt.xlabel('Target Variable- Salary')

    plt.ylabel('Distribution of target variable')

    plt.title('Distribution of Salary')

    total = len(data)

    for p in ax.patches:

            ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))
count_target_plot(data,'left')
sns.distplot(data['satisfaction_level'],color='pink')
sns.boxplot(x='satisfaction_level',data=data,color='pink')
sns.distplot(data['last_evaluation'],color='blue')
sns.boxplot(x='last_evaluation',data=data,color='blue')
sns.distplot(data['average_montly_hours'],color='red')
sns.boxplot(x='average_montly_hours',data=data,color='red')
## Checking for correlation

subset_data=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident',

'promotion_last_5years']]

data_corr=subset_data.corr()
sns.heatmap(data_corr,annot=True)
# Determine the relationship between satisfaction level and working hours on the exit of employees.



ax=sns.scatterplot(x='satisfaction_level',y='average_montly_hours',data=data[data['left']==1])
pairplot_data=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident',

'promotion_last_5years','left']]



sns.pairplot(pairplot_data[pairplot_data['left']==1])



# Find the effect of satisfaction level and the average monthly hours with department and salary level

# on exit of  employees.

plt.figure(figsize=(9,9))

sns.relplot(x="satisfaction_level",

                y="average_montly_hours",

                col="Department",

                hue="salary",

                kind="scatter",

                height=10,

                aspect=0.3,

                data=data[data['left']==1])
plt.figure(figsize=(9,9))

sns.relplot(x="last_evaluation",

                y="average_montly_hours",

                col="Department",

                hue="salary",

                kind="scatter",

                height=10,

                aspect=0.3,

                data=data[data['left']==1])
#Build a simple Machine Learning model using KNN, Decision Trees and Logistic Regression to predict the exit of employees



## Changing categorical variables to cat codes

category_data_list=['Department','salary']

int_data_list=['number_project','time_spend_company','Work_accident','promotion_last_5years','satisfaction_level','last_evaluation','average_montly_hours']

## Converting variables to cat codes

for column in category_data_list:

    data[column]=data[column].astype('category')



data[category_data_list] = data[category_data_list].apply(lambda column: column.cat.codes)



total_list=category_data_list+int_data_list

train_data=data[total_list]

label_data=data[['left']]
train_data.head()
label_data.head()
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

for train_index, test_index in splitter.split(train_data, label_data):

        features_train, features_test = train_data.iloc[train_index], train_data.iloc[test_index]

        labels_train, labels_test = label_data.iloc[train_index], label_data.iloc[test_index]
smote = SMOTE(sampling_strategy='minority', random_state=47)

os_values, os_labels = smote.fit_sample(features_train, labels_train)



features_train = pd.DataFrame(os_values)

labels_train = pd.DataFrame(os_labels)



print("Dimensions of Oversampled dataset is :", os_values.shape)
count_target_plot(labels_train,'left')
## Building baseline models



DT = DecisionTreeClassifier()

y_pred_DT = DT.fit(features_train, labels_train.values.ravel()).predict(features_test)



logr = LogisticRegression()

y_pred_logr = logr.fit(features_train, labels_train.values.ravel()).predict(features_test)



knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

y_pred_knn=knn.fit(features_train, labels_train.values.ravel()).predict(features_test)
print("DT:", accuracy_score(labels_test, y_pred_DT))

print("Logr:", accuracy_score(labels_test, y_pred_logr))

print("KNN:", accuracy_score(labels_test, y_pred_knn))
print("DT:", classification_report(labels_test, y_pred_DT))

print("Logr:", classification_report(labels_test, y_pred_logr))

print("KNN:", classification_report(labels_test, y_pred_knn))