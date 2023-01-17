# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Wrangling:

import numpy as np

import pandas as pd



#Data Visualization:

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head(10)
df.shape
df.info()
plt.figure(figsize=(12,6))

plt.title('Searching for missing values:')

sns.heatmap(data=df.isnull(),cmap = 'coolwarm', cbar = False)
df.describe()
plt.style.use('ggplot')

fig, axis  = plt.subplots(nrows = 3, ncols = 2, figsize = (15,9))



ax0, ax1, ax2, ax3, ax4, ax5 = axis.flatten()



ax0.hist(df['age'])

ax0.set_xlabel('Age')



ax1.hist(df['creatinine_phosphokinase'])

ax1.set_xlabel('CPK Enzyme')



ax2.hist(df['platelets'], bins = 15)

ax2.set_xlabel('Platelets')



ax3.hist(df['serum_creatinine'])

ax3.set_xlabel('Serum Creatinine')



ax4.hist(df['serum_sodium'], bins = 15)

ax4.set_xlabel('Serum Sodium')



ax5.hist(df['ejection_fraction'])

ax5.set_xlabel('Ejection Fraction')



plt.tight_layout()
fig, ax  = plt.subplots(nrows = 3, ncols = 2, figsize = (12,6))

plt.tight_layout()

sns.countplot(df['anaemia'], ax=ax[0,0])

sns.countplot(df['diabetes'], ax=ax[0,1])

sns.countplot(df['high_blood_pressure'], ax=ax[1,0])

sns.countplot(df['sex'], ax=ax[1,1])

sns.countplot(df['smoking'], ax=ax[2,0])

fig.delaxes(ax[2,1])
x1 = (len(df[df['anaemia'] == 1]))/len(df['anaemia'])

x2 = (len(df[df['diabetes'] == 1]))/len(df['diabetes'])

x3 = (len(df[df['high_blood_pressure'] == 1]))/len(df['high_blood_pressure'])

x4 = (len(df[df['sex'] == 1]))/len(df['sex'])

x5 = (len(df[df['smoking'] == 1]))/len(df['smoking'])



data = {'Anaemia': x1, 'Diabetes': x2, 'High Blood Pressure': x3, 'Sex': x4, 

       'Smoking': x5}

categ_zeros = pd.DataFrame(data, index=[1])

categ_zeros
df[['sex', 'DEATH_EVENT']].groupby(['sex'], as_index = False).mean()
df[['smoking', 'DEATH_EVENT']].groupby(['smoking'], as_index = False).mean()
df[['anaemia', 'DEATH_EVENT']].groupby(['anaemia'], as_index = False).mean()
df[['diabetes', 'DEATH_EVENT']].groupby(['diabetes'], as_index = False).mean()
df[['high_blood_pressure', 'DEATH_EVENT']].groupby(['high_blood_pressure'], 

                                                   as_index = False).mean()
plt.figure(figsize = (12,6))

sns.heatmap(df[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time', 'DEATH_EVENT']].corr(), cmap = 'coolwarm', annot = True)
g = sns.FacetGrid(df, col = 'DEATH_EVENT')

g.map(plt.hist, 'age')
g = sns.FacetGrid(df, col = 'DEATH_EVENT')

g.map(plt.hist, 'time')
g = sns.FacetGrid(df, col = 'DEATH_EVENT')

g.map(plt.hist, 'ejection_fraction')
g = sns.FacetGrid(df, col = 'DEATH_EVENT')

g.map(plt.hist, 'serum_creatinine')
g = sns.FacetGrid(df, col = 'DEATH_EVENT', row = 'sex')

g.map(plt.hist, 'age')
print("Before", df.shape)

df = df.drop(['platelets', 'creatinine_phosphokinase', 'serum_sodium', 'sex', 'smoking'], axis = 1)

print("After", df.shape)
df['UnderCon'] = 0

df.loc[((df['anaemia'] == 1) | (df['diabetes'] == 1) | df['high_blood_pressure'] == 1), 'UnderCon'] = 1

df.drop(['anaemia', 'diabetes', 'high_blood_pressure'], axis = 1, inplace = True)

df.head()
df.shape
df[['DEATH_EVENT', 'UnderCon']].groupby('UnderCon', as_index = False).mean()
df = df.drop('UnderCon', axis = 1)
df.head()
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12,6))

plt.tight_layout()



ax0,ax1,ax2,ax3 = ax.flatten()



ax0.boxplot(df['serum_creatinine'])

ax0.set_title('Serum Creatinine')



ax1.boxplot(df['ejection_fraction'])

ax1.set_title('Ejection Fraction')



ax2.boxplot(df['age'])

ax2.set_title('Age')



ax3.boxplot(df['time'])

ax3.set_title('time')
serum_cmode = df['serum_creatinine'].median()

df.loc[(df['serum_creatinine'] > 4), 'serum_creatinine'] = serum_cmode

df.head(10)
df['AgeBand'] = pd.cut(df['age'],5)

df[['AgeBand', 'DEATH_EVENT']].groupby('AgeBand', as_index = False).mean()
#Mapping:



df.loc[df['age'] <= 51, 'age'] = 0

df.loc[(df['age'] > 51) & (df['age'] <= 62), 'age'] = 1

df.loc[(df['age'] > 62) & (df['age'] <= 73), 'age'] = 2

df.loc[(df['age'] > 73) & (df['age'] <= 84), 'age'] = 3

df.loc[(df['age'] > 84) & (df['age'] <= 95), 'age'] = 4



df.drop('AgeBand', axis = 1, inplace = True)
df['TimeBand'] = pd.cut(df['time'],5)

df[['TimeBand', 'DEATH_EVENT']].groupby('TimeBand', as_index = False).mean()
#Mapping:



df.loc[df['time'] <= 60.2, 'time'] = 0

df.loc[(df['time'] > 60.2) & (df['time'] <= 116.4), 'time'] = 1

df.loc[(df['time'] > 116.4) & (df['time'] <= 172.6), 'time'] = 2

df.loc[(df['time'] > 172.6) & (df['time'] <= 228.8), 'time'] = 3

df.loc[(df['time'] > 228.8) & (df['time'] <= 285), 'time'] = 4



df.drop('TimeBand', axis = 1, inplace = True)
df['SCBand'] = pd.cut(df['serum_creatinine'],5)

df[['SCBand', 'DEATH_EVENT']].groupby('SCBand', as_index = False).mean()
#Mapping:



df.loc[df['serum_creatinine'] <= 1.2, 'serum_creatinine'] = 0

df.loc[(df['serum_creatinine'] > 1.2) & (df['serum_creatinine'] <= 1.9), 'serum_creatinine'] = 1

df.loc[(df['serum_creatinine'] > 1.9) & (df['serum_creatinine'] <= 2.6), 'serum_creatinine'] = 2

df.loc[(df['serum_creatinine'] > 2.6) & (df['serum_creatinine'] <= 3.3), 'serum_creatinine'] = 3

df.loc[(df['serum_creatinine'] > 3.3) & (df['serum_creatinine'] <= 4), 'serum_creatinine'] = 4



df.drop('SCBand', axis = 1, inplace = True)
df['EJBand'] = pd.cut(df['ejection_fraction'],5)

df[['EJBand', 'DEATH_EVENT']].groupby('EJBand', as_index = False).mean()
#Mapping:



df.loc[df['ejection_fraction'] <= 27.2, 'ejection_fraction'] = 0

df.loc[(df['ejection_fraction'] > 27.2) & (df['ejection_fraction'] <= 40.4), 'ejection_fraction'] = 1

df.loc[(df['ejection_fraction'] > 40.4) & (df['ejection_fraction'] <= 53.6), 'ejection_fraction'] = 2

df.loc[(df['ejection_fraction'] > 53.6) & (df['ejection_fraction'] <= 66.8), 'ejection_fraction'] = 3

df.loc[(df['ejection_fraction'] > 66.8) & (df['ejection_fraction'] <= 80), 'ejection_fraction'] = 4



df.drop('EJBand', axis = 1, inplace = True)
#Converting age and serum creatinine to integers:

df['age'] = df['age'].astype(int)

df['serum_creatinine'] = df['serum_creatinine'].astype(int)

df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
train, test = train_test_split(df, test_size = 0.2, random_state = 42)

print(train.head())

print('_'*40)

print(test.head())
X_train = train.drop('DEATH_EVENT', axis = 1)

Y_train = train['DEATH_EVENT']

X_test = test.drop('DEATH_EVENT', axis=1).copy()

Y_test = test['DEATH_EVENT']



X_train.shape,Y_train.shape,X_test.shape, Y_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)



acc_log_train = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_log_train,'%')

print('-'*25)

print('Test Score:',acc_log_test,'%')
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)



acc_svc_train = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_svc_train,'%')

print('-'*25)

print('Test Score:',acc_svc_test,'%')

error_rate = []

for i in range(1,40):

 knn = KNeighborsClassifier(n_neighbors=i)

 knn.fit(X_train,Y_train)

 pred_i = knn.predict(X_test)

 error_rate.append(np.mean(pred_i != Y_test))



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 

         marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
knn = KNeighborsClassifier(n_neighbors = error_rate.index(min(error_rate)))

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)



acc_knn_train = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_knn_train,'%')

print('-'*25)

print('Test Score:',acc_knn_test,'%')
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)



acc_gaussian_train = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_gaussian_train,'%')

print('-'*25)

print('Test Score:',acc_gaussian_test,'%')
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)



acc_perceptron_train = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_perceptron_train,'%')

print('-'*25)

print('Test Score:',acc_perceptron_test,'%')
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)



acc_linear_svc_train = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_linear_svc_train,'%')

print('-'*25)

print('Test Score:',acc_linear_svc_test,'%')
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



acc_sgd_train = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_sgd_train,'%')

print('-'*25)

print('Test Score:',acc_sgd_test,'%')
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)



acc_decision_tree_train = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_decision_tree_train,'%')

print('-'*25)

print('Test Score:',acc_decision_tree_test,'%')
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)



acc_random_forest_train = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest_test = round(accuracy_score(Y_pred, Y_test) * 100,2)



print('Training Score:',acc_random_forest_train,'%')

print('-'*25)

print('Test Score:',acc_random_forest_test,'%')
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc_train, acc_knn_train, acc_log_train, 

              acc_random_forest_train, acc_gaussian_train, acc_perceptron_train, 

              acc_sgd_train, acc_linear_svc_train, acc_decision_tree_train]})

models.sort_values(by='Score', ascending=False)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc_test, acc_knn_test, acc_log_test, 

              acc_random_forest_test, acc_gaussian_test, acc_perceptron_test, 

              acc_sgd_test, acc_linear_svc_test, acc_decision_tree_test]})

models.sort_values(by='Score', ascending=False)