# Basic

import numpy as np 

import pandas as pd 



# Plotting

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Scaling

from sklearn.preprocessing import StandardScaler



# train test split

from sklearn.model_selection import train_test_split



# Import models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Evaluation metrics

from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



# Feature Selection

from sklearn.feature_selection import SelectKBest, chi2



# Cross validation

from sklearn.model_selection import cross_val_score
dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col = False)



dataset.head()
# Checking for missing values



dataset.isna().sum()
dataset.describe()
dataset.info()
cols = dataset.select_dtypes(include=['object']).columns



for col in cols:

    print('----'+str(col).upper()+'----')

    print(pd.crosstab(index = dataset[col], columns = dataset['status']))

    print()

sns.countplot(x = 'status', data = dataset, hue = 'gender')

plt.title('Gender vs Campus Placement')

plt.xlabel('Status')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x='gender', data = dataset)

plt.title('Gender')

plt.xlabel('Gender')

plt.ylabel('Number of students')

plt.show()
# Secondary percentage vs campus placement

ax = sns.violinplot(x = 'status', y = 'ssc_p', data = dataset)



medians = dataset.groupby(['status'])['ssc_p'].median().values

nobs = dataset['status'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Secondary Education percentage vs Campus Placement')

plt.show()
sns.countplot(x='ssc_b', data = dataset)

plt.title('Senior Secondary Board')

plt.xlabel('Board')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x = 'status', data = dataset, hue = 'ssc_b')

plt.title('Senior Secondary Board vs Campus Placement')

plt.xlabel('Status')

plt.ylabel('Number of students')

plt.show()
# Highesr Secondary percentage vs campus placement

ax = sns.violinplot(x = 'status', y = 'hsc_p', data = dataset)



medians = dataset.groupby(['status'])['hsc_p'].median().values

nobs = dataset['status'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Higher Secondary percentage vs Campus Placement')

plt.show()
sns.countplot(x='hsc_b', data = dataset)

plt.title('Higher Secondary Board')

plt.xlabel('Board')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x = 'status', data = dataset, hue = 'hsc_b')

plt.title('Higher Secondary Board vs Campus Placement')

plt.xlabel('Status')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x='hsc_s', data = dataset)

plt.title('Subject in HSC Board')

plt.xlabel('Subjects')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x = 'hsc_s', data = dataset, hue = 'status')

plt.title('Subject in HSC vs Campus Placement')

plt.xlabel('hsc_s')

plt.ylabel('Number of students')

plt.show()
# Degree percentage vs campus placement

ax = sns.violinplot(x = 'status', y = 'degree_p', data = dataset)



medians = dataset.groupby(['status'])['degree_p'].median().values

nobs = dataset['status'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Degree percentage vs Campus Placement')

plt.show()
sns.countplot(x='degree_t', data = dataset)

plt.title('Graduation Degree')

plt.xlabel('Graduation Degree')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x = 'degree_t', data = dataset, hue = 'status')

plt.title('Graduation Degree vs Campus Placement')

plt.xlabel('Graduation Degree')

plt.ylabel('Number of students')

plt.show()
sns.countplot(x='workex', data = dataset)

plt.title('Work Experience')

plt.xlabel('Work Experience')

plt.ylabel('Number of students')

plt.show()
# Work experience vs candidate placement

sns.countplot(x = 'workex', data = dataset, hue = 'status')

plt.title('Work Experience vs Campus Placement')

plt.xlabel('Work Experience')

plt.ylabel('Number of students')

plt.show()
# Employability test percentage vs campus placement

ax = sns.violinplot(x = 'status', y = 'etest_p', data = dataset)



medians = dataset.groupby(['status'])['etest_p'].median().values

nobs = dataset['status'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Employability test percentage vs Campus Placement')

plt.show()
sns.countplot(x='specialisation', data = dataset)

plt.title('MBA Specialisation')

plt.xlabel('MBA Specialisation')

plt.ylabel('Number of students')

plt.show()
# Degree specialization vs candidate placement

sns.countplot(x = 'specialisation', data = dataset, hue = 'status')

plt.title('Degree Specialisation vs Campus Placement')

plt.xlabel('Specialisation')

plt.ylabel('Number of students')

plt.show()
# MBA percentage vs campus placement

ax = sns.violinplot(x = 'status', y = 'mba_p', data = dataset)



medians = dataset.groupby(['status'])['mba_p'].median().values

nobs = dataset['status'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('MBA percentage vs Campus Placement')

plt.show()
# Dropping the serial number column

dataset.drop('sl_no', axis=1, inplace = True)

dataset.head(3)
# Dropping ssc_b and hsc_b

dataset.drop('ssc_b', axis=1, inplace = True)

dataset.drop('hsc_b', axis=1, inplace = True)



print(dataset.shape)

dataset.head(3)

# Gender: F coded as 0 and M as 1

dummy_variable_1 = pd.get_dummies(dataset['gender'])

dummy_variable_1.rename(columns={'M':'Gender'}, inplace=True)



# drop original column 

dataset.drop("gender", axis = 1, inplace=True)



# merge data frame "dataset" and "dummy_variable_1: Gender column" 

df = pd.concat([dummy_variable_1['Gender'], dataset], axis=1)



df.head(1)
# Higher Secondary Specialisation: Science: 10 and Commerce: 01 and Arts: 00

dummy = pd.get_dummies(df['hsc_s'])

dummy.rename(columns={'Science': 'HS_Sci', 'Commerce': 'HS_Comm'}, inplace=True)

dummy = pd.concat([dummy['HS_Sci'], dummy['HS_Comm']], axis=1)

dummy.head()



# drop original

df.drop('hsc_s', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:3], dummy, df.iloc[:, 3:]], axis=1)



df.head(1)
# Undergrad specialisation: Sci&Tech: 10 and Comm&Mgmt: 01 and Others: 00

dummy = pd.get_dummies(df['degree_t'])

dummy.rename(columns={'Sci&Tech': 'UG_Sci', 'Comm&Mgmt': 'UG_Comm'}, inplace=True)

dummy = pd.concat([dummy['UG_Sci'], dummy['UG_Comm']], axis=1)

dummy.head()



# drop original

df.drop('degree_t', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:6], dummy, df.iloc[:, 6:]], axis=1)



df.head()
# Work experience: Yes as 1 nd No as 0

dummy = pd.get_dummies(df['workex'])

dummy.rename(columns={'Yes': 'workex'}, inplace=True)

# dummy.head()



# drop original

df.drop('workex', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:8], dummy['workex'], df.iloc[:, 8:]], axis=1)



df.head(1)
# Specialisation: Mkt&Fin as 1 and Mkt&HR as 0

dummy = pd.get_dummies(df['specialisation'])

dummy.rename(columns={'Mkt&Fin': 'specialisation'}, inplace=True)

# dummy.head()



# drop original data

df.drop('specialisation', axis=1, inplace=True)



# merge data

df= pd.concat([df.iloc[:, 0:10], dummy['specialisation'], df.iloc[:, 10:]], axis=1)



df.head(2)
# Change Placed into dummy variables: Placed as 1 and Not Placed as 0

dummy = pd.get_dummies(df['status'])

dummy.rename(columns={'Placed': 'status'}, inplace=True)

# dummy.head()



# drop original

df.drop('status', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:12], dummy['status'], df.iloc[:, 12:]], axis=1)



df.head()
# spliting the dataset to get the independent and dependent variables separately

X = df.iloc[:, :-2].values

y = df.iloc[:, -2].values



print('X_shape {}'.format(X.shape))

print('y_shape {}'.format(y.shape))
# Splitting



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print('Shape of training set: {} and test set: {}'.format(X_train.shape, X_test.shape))
# Feature Scaling



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



print('Scaled Successfully')
classifiers = [LogisticRegression(), KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p=2), SVC(kernel = 'linear'), SVC(kernel = 'rbf'), DecisionTreeClassifier(criterion='entropy'), RandomForestClassifier(n_estimators = 10, criterion = 'entropy')]



for classifier in classifiers:

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    

    # print classifier name

    print(str(type(classifier)).split('.')[-1][:-2])

    

    # Accuracy Score

    print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))

    

    # jaccard Score

    print('Jaccard Score: {}'.format(jaccard_score(y_test, y_pred)))

    

    # F1 score

    print('F1 Score: {}'.format(f1_score(y_test, y_pred)))

    

    # Log Loss

    print('Log Loss: {}'.format(log_loss(y_test, y_pred)))

    

    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, lw = 2, cbar=False)

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.title('Confusion Matrix: {}'.format(str(type(classifier)).split('.')[-1][:-2]))

    plt.show()
X = df.iloc[:, :-2].values

y = df.iloc[:, -2].values



# Finding out the significance of each feature in the dataset 

kb = SelectKBest(chi2, k = 'all')

X_new = kb.fit_transform(X, y)



print(kb.pvalues_)
X = df.iloc[:, :-2].values

y = df.iloc[:, -2].values



acc=[]

logloss=[]

f1=[]

jaccard=[]



for k in range(1, 13):

    

    # Selecting features

    kb = SelectKBest(chi2, k = k)

    X_new = kb.fit_transform(X, y)

    

    # Split dataset

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)

    

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)

    X_test = sc.fit_transform(X_test)

    

    #classifier

    classifier = LogisticRegression(random_state=0)

    

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    

    acc.append(accuracy_score(y_test, y_pred))

    jaccard.append(jaccard_score(y_test, y_pred))

    f1.append(f1_score(y_test, y_pred))

    logloss.append(log_loss(y_test,y_pred))

    



    
plt.figure()

plt.plot(acc)

plt.title('Accuracy Score: LR')



plt.figure()

plt.plot(jaccard)

plt.title('Jaccard Score: LR')



plt.figure()

plt.plot(f1)

plt.title('F1 Score: LR')



plt.figure()

plt.plot(logloss)

plt.title('Log Loss Score: LR')
X = df.iloc[:, :-2].values

y = df.iloc[:, -2].values



# Selecting 7 best features

kb = SelectKBest(chi2, k = 6)

X_new = kb.fit_transform(X, y)



# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)



# Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

    

#classifier

classifier = LogisticRegression(random_state=0)

    

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

    

print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))

print('Jaccard Score: {}'.format(jaccard_score(y_test, y_pred)))

print('F1 Score: {}'.format(f1_score(y_test, y_pred)))

print('Log Loss: {}'.format(log_loss(y_test,y_pred)))
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies
print('Evaluation of our model performance: {}'.format(accuracies.mean()))

print('Variance check: {}'.format(accuracies.std()))