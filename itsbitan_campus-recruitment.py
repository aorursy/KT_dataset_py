# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.info()
df.describe()
df.profile_report(title='Campus Placement Data - Report', progress_bar=False)
df.drop('sl_no', axis=1, inplace= True)
sns.countplot("gender", hue="status", data=df)

plt.show()
#This plot ignores NaN values for salary, igoring students who are not placed

sns.kdeplot(df.salary[ df.gender=="M"])

sns.kdeplot(df.salary[ df.gender=="F"])

plt.legend(["Male", "Female"])

plt.xlabel("Salary (100k)")

plt.show()
#At first we check our target variable

sns.set_style("whitegrid")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(df['salary'],color="b")

sns.despine(trim=True, left=True)

plt.xlabel('salary')

plt.ylabel('Frequency')

plt.title('Distribution of salary')

print('Skewness: %f', df['salary'].skew())

print("Kurtosis: %f" % df['salary'].kurt())

plt.show()
sns.countplot(x = df['ssc_b'])

plt.title('count of ssc_b')

plt.xlabel('ssc_b')

plt.ylabel('count')

plt.show()
sns.countplot(x = df['hsc_b'])

plt.title('count of hsc_b')

plt.xlabel('hsc_b')

plt.ylabel('hsc_b')

plt.show()
sns.countplot(x = df['hsc_s'])

plt.title('count of hsc_s')

plt.xlabel('hsc_s')

plt.ylabel('hsc_s')

plt.show()
sns.countplot(x = df['degree_t'])

plt.title('count of degree_t')

plt.xlabel('degree_t')

plt.ylabel('degree_t')

plt.show()
sns.countplot(x = df['workex'])

plt.title('count of workex')

plt.xlabel('workex')

plt.ylabel('workex')

plt.show()
sns.countplot(x = df['specialisation'])

plt.title('count of specialisation')

plt.xlabel('specialisation')

plt.ylabel('specialisation')

plt.show()
sns.countplot(x = df['status'])

plt.title('count of status')

plt.xlabel('status')

plt.ylabel('status')

plt.show()
df_clf = df.copy()

df_reg = df.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ('gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status'):

  df_clf[col] = le.fit_transform(df_clf[col])
df_clf.head()
#we can drop the salary coloumn for our classification problem

df_clf.drop('salary', axis=1, inplace= True)
#Correlation and Heatmap

corr = df_clf.corr()

colormap = sns.diverging_palette(220, 10, as_cmap= True)

plt.figure(figsize =(12, 8))

sns.heatmap(corr,

            xticklabels = corr.columns.values,

            yticklabels = corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor = 'white')

plt.title('correlations between features')
#Pairplot

sns.pairplot(data = df_clf)

plt.title('pairplot of data')

plt.show()
df_clf.head()
x = df_clf.iloc[:, 0:12].values

print(x)
y = df_clf.iloc[:, 12:13].values

print(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='liblinear', random_state = 0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print(cm)

print(cr)
from sklearn.model_selection import cross_val_score

accuraices = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuraices.mean()*100))

print("Standerd Deviation: {:.2f} %".format(accuraices.std()*100))
#Let see the ROC curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)

print('AUC: %.3f' % auc)



fig = plt.figure(figsize=(15,15))

ax = plt.subplot2grid((3,2), (0,0))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.plot(fpr, tpr, marker='.')

plt.show()
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print(cm)

print(cr)
from sklearn.model_selection import cross_val_score

accuraices = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuraices.mean()*100))

print("Standerd Deviation: {:.2f} %".format(accuraices.std()*100))
#Dropping NaNs (in Salary)

df_reg.dropna(inplace=True)
df_reg.info()
#we can drop the status coloumn

df_reg.drop('status', axis=1, inplace= True)
df_reg = pd.get_dummies(df_reg).reset_index(drop=True)

df_reg.shape
#Correlaion & Heatmap

corr = df_reg.corr()

colormap = sns.diverging_palette(220, 10, as_cmap= True)

plt.figure(figsize =(12, 8))

sns.heatmap(corr,

            xticklabels = corr.columns.values,

            yticklabels = corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor = 'white')

plt.title('correlations between features')
#Lets look the correlation score

print (corr['salary'].sort_values(ascending=False), '\n')
#Seperating Depencent and Independent Vaiiables

y1 = df_reg["salary"].values

x1 = df_reg.drop("salary", axis=1)

x1 = df_reg.drop("ssc_b_Central",axis=1)

x1 = df_reg.drop("hsc_b_Others",axis=1)

x1 = df_reg.drop("hsc_s_Commerce",axis=1)

x1 = df_reg.drop("degree_t_Others",axis=1)

x1 = df_reg.drop("hsc_s_Arts",axis=1)

x1 = df_reg.drop("degree_p",axis=1)

x1 = df_reg.drop("workex_No",axis=1)

x1 = df_reg.drop("specialisation_Mkt&HR",axis=1)

x1 = df_reg.drop("gender_F",axis=1)

x1 = df_reg.drop("degree_t_Comm&Mgmt",axis=1)

column_names = x1.columns.values
#Removing outlayer from data

x1 = x1[y1 < 400000]

y1 = y1[y1 < 400000]
x1 = x1.iloc[:,:].values

print(x1)
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x1_train = sc.fit_transform(x1_train)

x1_test = sc.transform(x1_test)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators= 100, random_state=0)

regressor.fit(x1_train, y1_train) 
y1_pred = regressor.predict(x1_test)
print(y1_pred)
print(y1_test)
accuraices_train = regressor.score(x1_train, y1_train)

accuraices_test = regressor.score(x1_test, y1_test)

print(accuraices_train)

print(accuraices_test)
#Visualising the Acutal and predicted Result

plt.plot(y1_test, color = 'deeppink', label = 'Actual')

plt.plot(y1_pred, color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Salary')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()