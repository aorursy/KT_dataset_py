import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
h_df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv') 

h_df.head(10)
h_df.shape
h_df.columns
h_df.info()
h_df.describe([0.25,0.50,0.75])
h_df.isnull().sum()
h_df['age'].sort_values(ascending = False)
bins = [32,48,64,80,96]

labels = ['32-48', '48-64', '64-80', '80-96']

h_df['agegroup']=pd.cut(h_df['age'], bins, labels = labels)
h_df['agegroup'].value_counts()
def plot_Outlier(var_list):

    plt.figure(figsize=(20, 15))

    for var in var_list:

        plt.subplot(4,4,var_list.index(var)+1)

        ax=sns.boxplot(x = h_df[var])   

    plt.show()
plot_Outlier(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',

       'DEATH_EVENT'])
Q1 = h_df['creatinine_phosphokinase'].quantile(0.25)

Q3 = h_df['creatinine_phosphokinase'].quantile(0.75)

IQR = Q3 - Q1

h_df=h_df.loc[(h_df['creatinine_phosphokinase'] >= Q1 - 1.5*IQR) & (h_df['creatinine_phosphokinase'] <= Q3 + 1.5*IQR)]

h_df.shape
Q1 = h_df['platelets'].quantile(0.25)

Q3 = h_df['platelets'].quantile(0.75)

IQR = Q3 - Q1

h_df=h_df.loc[(h_df['platelets'] >= Q1 - 1.5*IQR) & (h_df['platelets'] <= Q3 + 1.5*IQR)]

h_df.shape
Q1 = h_df['serum_creatinine'].quantile(0.25)

Q3 = h_df['serum_creatinine'].quantile(0.75)

IQR = Q3 - Q1

h_df=h_df.loc[(h_df['serum_creatinine'] >= Q1 - 1.5*IQR) & (h_df['serum_creatinine'] <= Q3 + 1.5*IQR)]

h_df.shape
Q1 = h_df['serum_sodium'].quantile(0.25)

Q3 = h_df['serum_sodium'].quantile(0.75)

IQR = Q3 - Q1

h_df=h_df.loc[(h_df['serum_sodium'] >= Q1 - 1.5*IQR) & (h_df['serum_sodium'] <= Q3 + 1.5*IQR)]

h_df.shape
Q1 = h_df['ejection_fraction'].quantile(0.25)

Q3 = h_df['ejection_fraction'].quantile(0.75)

IQR = Q3 - Q1

h_df=h_df.loc[(h_df['ejection_fraction'] >= Q1 - 1.5*IQR) & (h_df['ejection_fraction'] <= Q3 + 1.5*IQR)]

h_df.shape
plot_Outlier(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',

       'DEATH_EVENT'])
h_df.columns
plt.figure(figsize=(10, 5))

sns.countplot('agegroup', data=h_df)
plt.figure(figsize=(10, 5))

sns.countplot('anaemia', data=h_df, hue = 'agegroup')
plt.figure(figsize=(10, 5))

sns.countplot('agegroup', data=h_df, hue = 'anaemia')
plt.figure(figsize=(10, 5))

sns.countplot('agegroup', data=h_df, hue = 'DEATH_EVENT')
plt.figure(figsize=(10, 5))

sns.countplot('anaemia', data=h_df, hue = 'DEATH_EVENT')
plt.figure(figsize=(10, 5))

sns.countplot('high_blood_pressure', data=h_df, hue = 'DEATH_EVENT')
plt.figure(figsize=(15, 10))

sns.heatmap(h_df.corr(),annot = True)
corr = h_df.corr()

corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
plt.figure(figsize=(10, 5))

sns.barplot(y ='ejection_fraction', data=h_df, x = 'DEATH_EVENT')
def bar_plot(var_list):

    plt.figure(figsize=(20, 20))

    for var in var_list:

        plt.subplot(2,3,var_list.index(var)+1)

        ax=sns.barplot(y = h_df[var], data = h_df, x = 'DEATH_EVENT')   

    plt.show()
bar_plot(['age','ejection_fraction', 'serum_creatinine', 'time', 'serum_sodium'])
corr_df=h_df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]
x = corr_df

y = h_df['DEATH_EVENT']



## Train and test data split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.preprocessing import MinMaxScaler





scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn import metrics

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred)))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)

#s3=accuracy_score(y_test,p3)

print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred)))
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred)))
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=30, criterion='entropy')

dt_clf.fit(x_train, y_train)

y_pred = dt_clf.predict(x_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))

print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, y_pred)))
pd.concat([pd.DataFrame(x.columns, columns = ['variable']),

           pd.DataFrame(rfc.feature_importances_, columns = ['importance'])],

          axis = 1).sort_values(by = 'importance', ascending = False)