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
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
#brief data information to recognize data type

"""

sl_no= serial number

gneder

ssc_p= Secondary educaion percentage - 10th grade

ssc_b= Board of education - Central/ Others

hsc_p= Higher secondary Educaion percentage-12th grade

hsc_b= Board of Education - Central/Others

hsc_s= Specialization in Higher Secondary Education- Commerce/Science

degree_p= Degree percentage

degree_t= Field of degree educaion(e.g. Marketing/Science)

workex= Work experience

etest_p= Employability test percentage

specialisation= Post Graduation(MBA)

mba_p= MBA percentage

status= Placed or not placed

Salary= How much

"""

data.info()
#data report

data.profile_report(title='Campus Placement Data - Report', progress_bar=False)
#Replace Nan with 0

data['salary'] = data['salary'].replace(np.nan, 0) 
#check

data.head(5)
data_fix = data

data_fix['status'].values[data_fix['status']=='Not Placed'] = 0 

data_fix['status'].values[data_fix['status']=='Placed'] = 1

data_fix.status = data_fix.status.astype('int')
plt.figure(figsize=(14,12))

data2 = data_fix.loc[:,data_fix.columns != 'Id']

sns.heatmap(data2.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)
#Gender

sns.countplot("gender", hue="status", data=data)

plt.show()
male = 39/139

female = 28/76

if male<female:

    print('male has more chance to get placed')

else:

        print('female has more chance to get placed')

#ssc_b

sns.countplot("ssc_b", hue="status", data=data)

plt.show()
#hsc_s

sns.countplot("hsc_s", hue="status", data=data)

plt.show()
#degree_t

sns.countplot("degree_t", hue="status", data=data)

plt.show()
#workexperience

sns.countplot("workex", hue="status", data=data)

plt.show()
#specialisation

sns.countplot("specialisation", hue="status", data=data)

plt.show()
fig,axes = plt.subplots(3,2, figsize=(20, 20))

sns.barplot(x='status', y='ssc_p', data=data2, ax=axes[0][0])

sns.barplot(x='status', y='hsc_p', data=data2, ax=axes[0][1])

sns.barplot(x='status', y='degree_p', data=data2, ax=axes[1][0])

sns.barplot(x='status', y='mba_p', data=data2, ax=axes[1][1])

fig.delaxes(ax = axes[2][0]) 
#specialisation count

sns.countplot(x='specialisation',data=data)

plt.show()
#Replace 0 with NAN

data['salary'] = data['salary'].replace(0, np.nan)
#gender

plt.figure(figsize =(20,6))

sns.boxplot("salary", "gender", data=data)

plt.show()
#ssc_b

plt.figure(figsize =(20,6))

sns.boxplot("salary", "ssc_b", data=data)

plt.show()

#ssc_p

sns.lineplot("ssc_p", "salary", hue="ssc_b", data=data)

plt.show()
#hsc_b

plt.figure(figsize =(20,6))

sns.boxplot("salary", "hsc_b", data=data)

plt.show()

#hsc_p

sns.lineplot("hsc_p", "salary", hue="hsc_b", data=data)

plt.show()
#degree_p

sns.lineplot("degree_p", "salary", hue="degree_t", data=data)

plt.show()

#degree_t

plt.figure(figsize =(20,6))

sns.boxplot("salary", "degree_t", data=data)

plt.show()
#workex

plt.figure(figsize =(20,6))

sns.boxplot("salary", "workex", data=data)

plt.show()
#etest_p

sns.lineplot("etest_p", "salary", data=data)

plt.show()
#specialisation

plt.figure(figsize =(20,6))

sns.boxplot("salary", "specialisation", data=data)

plt.show()
#mba_p

sns.lineplot("mba_p", "salary", data=data)

plt.show()
#Drop data

data.drop(['ssc_b','hsc_b','etest_p','sl_no'], axis=1, inplace=True)
#Check

data.head(5)
#Change float data

data["gender"] = data.gender.map({"M":0,"F":1})

data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})

data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})

data["workex"] = data.workex.map({"No":0, "Yes":1})

data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
#check

data.head(5)
#make dataset for BCP and Regression

data_placement = data.copy()

data_salary = data.copy()
#check

data_placement.head(5)
#check

data_salary.head(5)
#Library import

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

# Seperating Variables and Target

x = data_placement[['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex','specialisation', 'mba_p',]]

y = data_placement['status']
#Train Test Split(7:3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#decision tree classifier model

dtree = DecisionTreeClassifier(criterion='entropy')

dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)
#R ^2 

accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#Random Forest Model

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#Feature importance

rows = list(x.columns)

imp = pd.DataFrame(np.zeros(6*len(rows)).reshape(2*len(rows), 3))

imp.columns = ["Classifier", "Feature", "Importance"]

#Add Rows

for index in range(0, 2*len(rows), 2):

    imp.iloc[index] = ["DecisionTree", rows[index//2], (100*dtree.feature_importances_[index//2])]

    imp.iloc[index + 1] = ["RandomForest", rows[index//2], (100*random_forest.feature_importances_[index//2])]
plt.figure(figsize=(15,5))

sns.barplot("Feature", "Importance", hue="Classifier", data=imp)

plt.title("Computed Feature Importance")

plt.show()
#Remove hsc_s, degree_t, workex, specialisation

x = data_placement[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'mba_p',]]

y = data_placement['status']
#Train Test Split(7:3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#decision tree classifier model

dtree = DecisionTreeClassifier(criterion='entropy')

dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)
#R^2

accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#Random Forest Model

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
#R^2

accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#Feature importance

rows = list(x.columns)

imp = pd.DataFrame(np.zeros(6*len(rows)).reshape(2*len(rows), 3))

imp.columns = ["Classifier", "Feature", "Importance"]

#Add Rows

for index in range(0, 2*len(rows), 2):

    imp.iloc[index] = ["DecisionTree", rows[index//2], (100*dtree.feature_importances_[index//2])]

    imp.iloc[index + 1] = ["RandomForest", rows[index//2], (100*random_forest.feature_importances_[index//2])]
plt.figure(figsize=(15,5))

sns.barplot("Feature", "Importance", hue="Classifier", data=imp)

plt.title("Computed Feature Importance")

plt.show()
#library import

from sklearn.preprocessing import MinMaxScaler # for scaling Salary

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, r2_score
#drop salary NaN 

data_salary.dropna(inplace=True)

#drop Status

data_salary.drop("status", axis=1, inplace=True)
#check

data_salary.head(5)
#Seperating Depencent and Independent Vaiiables

y = data_salary["salary"] 

x = data_salary.drop("salary", axis=1)

column_names = x.columns.values
#scale salary 0-1

x_scaled = MinMaxScaler().fit_transform(x)
#PDF of Salary

sns.kdeplot(y)

plt.show()
# Salary outlier removal over 400,000 as most of boxplot marks outlier over 400,000

x_scaled = x_scaled[y < 400000]

y = y[y < 400000]
#check

sns.kdeplot(y)

plt.show()
#Converting to DF for as  column names gives readibility

x_scaled = pd.DataFrame(x_scaled, columns=column_names)

y = y.values



# We must add a constants 1s for intercept before doing Linear Regression with statsmodel

x_scaled = sm.add_constant(x_scaled)

x_scaled.head()

#Constants 1 added for intercept term
# All

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= ssc_p

x_scaled = x_scaled.drop('ssc_p', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= degree_p

x_scaled = x_scaled.drop('degree_p', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= specialisation

x_scaled = x_scaled.drop('specialisation', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= hsc_p

x_scaled = x_scaled.drop('hsc_p', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= workex

x_scaled = x_scaled.drop('workex', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= hsc_s

x_scaled = x_scaled.drop('hsc_s', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
# Drop biggest p value= degree_t

x_scaled = x_scaled.drop('degree_t', axis=1)

model = sm.OLS(y, x_scaled)

results = model.fit()

results.summary()
print('Thank you')