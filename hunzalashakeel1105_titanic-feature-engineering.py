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
# %matplotlib inline
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import string
import math
import sys
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_copy = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test_copy = pd.read_csv('/kaggle/input/titanic/train.csv')
# check the types of dataframe
titanic.dtypes
target = titanic['Survived']
target.unique()
# check if any null value in target
target.isnull().sum()
# plot target values
target.value_counts().plot.pie(autopct='%1.2f%%')
titanic.info()
print('----------------------------------------------')
titanic_test.info()
# check how many unique values each feature have
for column in titanic.columns:
    print(column, len(titanic[column].unique()))
print('Train Data Embarked', titanic['Embarked'].isnull().sum())
print('Train Data Fare', titanic['Fare'].isnull().sum())

print('test Data embarked', titanic_test['Embarked'].isnull().sum())
print('test Data Fare', titanic_test['Fare'].isnull().sum())


# because missing values are less we replace embarked with most recurrent value and fare with median
titanic['Embarked'] = titanic['Embarked'].fillna("S")
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic['Age'].describe()
# create feature where missing age is imputed with mean of age values that are not missing
titanic['Age_mean'] =np.where(titanic.Age.isnull(), titanic['Age'].mean(), titanic['Age'])
titanic_test['Age_mean'] =np.where(titanic_test.Age.isnull(), titanic_test['Age'].mean(), titanic_test['Age'])
# drop PasssengerId because it has no impact in survival rate
titanic = titanic.drop('PassengerId', axis=1)
titanic_test = titanic_test.drop('PassengerId', axis=1)
titanic.head()

def ticket_sep(tickets):
    ticket_type = []
    for i in range(len(tickets)):
        one = tickets.iloc[i].split(" ")[0]
        if(one.isdecimal()):
            ticket_type.append('NO')
        else:
            ticket_type.append(''.join(e for e in one if e.isalnum()))
    return ticket_type
    
titanic['ticket_type'] = ticket_sep(titanic['Ticket'])
titanic_test['ticket_type'] = ticket_sep(titanic_test['Ticket'])
print(titanic_test['ticket_type'].value_counts())
# put all tickets which are less than 15 into OTHER_T
train_values = titanic['ticket_type'].value_counts()
test_values = titanic_test['ticket_type'].value_counts()

for i,t in enumerate(titanic['ticket_type']):
    if (train_values[t] < 15):
        titanic['ticket_type'][i] = 'OTHER_T'
        
for i,t in enumerate(titanic_test['ticket_type']):
    if (t not in titanic['ticket_type'].unique()):
        titanic_test['ticket_type'][i] = 'OTHER_T'
print(titanic['ticket_type'].unique())
print(titanic_test['ticket_type'].unique())
sns.barplot(x = 'ticket_type', y = 'Survived', data = titanic)
# where ticket_type is 'SOTONOQ' convert it to 'A5'
titanic["ticket_type"] = np.where(titanic["ticket_type"]=='SOTONOQ', 'A5', titanic["ticket_type"])
titanic_test['ticket_type'] = np.where(titanic_test['ticket_type'] == 'SOTONOQ', 'A5', titanic_test['ticket_type'])
sns.barplot(x='ticket_type', y='Survived', data=titanic)
# drop ticket from dataframe
titanic = titanic.drop('Ticket', axis = 1)
titanic_test = titanic_test.drop('Ticket', axis = 1)
print('null in training set', titanic['Cabin'].isnull().sum())
print('null in testing set', titanic_test['Cabin'].isnull().sum())
# create function that takes cabin type from cabin and if value is missing copy NaN
def sep_cabin(cabin):
    cabin_type = []
    
    for c in range(len(cabin)):
        if cabin.isnull()[c] == True:
            cabin_type.append('NaN')
        else:
            cabin_type.append(cabin[c][:1])
    return cabin_type
            
titanic['cabin_type'] = sep_cabin(titanic['Cabin'])
titanic_test['cabin_type'] = sep_cabin(titanic_test['Cabin'])
titanic.head()
sns.barplot(x='cabin_type', y='Survived', data=titanic)
# put all cabin which are less than 15 into OTHER_C
train_values = titanic['cabin_type'].value_counts()
test_values = titanic_test['cabin_type'].value_counts()

for i,t in enumerate(titanic['cabin_type']):
    if (train_values[t] < 15):
        titanic['cabin_type'][i] = 'OTHER_C'
        
for i,t in enumerate(titanic_test['cabin_type']):
    if (t not in titanic['cabin_type'].unique()):
        titanic_test['cabin_type'][i] = 'OTHER_C'
sns.barplot(x='cabin_type', y='Survived', data=titanic)
# drop cabin from dataset
titanic = titanic.drop('Cabin', axis = 1)
titanic_test = titanic_test.drop('Cabin', axis = 1)
titanic['Name']
# Create function that take name and separates it into title, family name and deletes all puntuation from name column:
def sep_name(data):
    families=[]
    titles = []
    new_name = []
    #for each row in dataset:
    for i in range(len(data)):
        name = data.iloc[i]
        # extract name inside brakets into name_bracket:
        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(",")[0]
        title = name_no_bracket.split(",")[1].strip().split(" ")[0]
        
        #remove punctuations accept brackets:
        for c in string.punctuation:
            name = name.replace(c,"").strip()
            family = family.replace(c,"").strip()
            title = title.replace(c,"").strip()
            
        families.append(family)
        titles.append(title)
        new_name.append(name)
            
    return [families, titles, new_name]   
# apply name_sep on train and test set:
titanic['family'], titanic['title'], titanic['Name']  = sep_name(titanic.Name)
titanic_test['family'], titanic_test['title'], titanic_test['Name'] = sep_name(titanic_test.Name)

titanic.head()
g = sns.barplot(x='title', y='Survived', data=titanic)
g.figure.set_figwidth(15)
# put all title which are less than 15 into OTHER
train_values = titanic['title'].value_counts()
test_values = titanic_test['title'].value_counts()

for i,t in enumerate(titanic['title']):
    if (train_values[t] < 15):
        titanic['title'][i] = 'OTHER'
        
for i,t in enumerate(titanic_test['title']):
    if (t not in titanic['title'].unique()):
        titanic_test['title'][i] = 'OTHER'
g = sns.barplot(x='title', y='Survived', data=titanic)
# amount of overlapping family names in train and test set:
len([x for x in titanic.family.unique() if x in titanic_test.family.unique()])
# amount of non overlapping names in train and test set
len([x for x in titanic.family.unique() if x not in titanic_test.family.unique()])
# amount of non overlapping with train set unique family names in test set:
len([x for x in titanic_test.family.unique() if x not in titanic.family.unique()])
#create a list with all overlapping families
overlap = [x for x in titanic.family.unique() if x in titanic_test.family.unique()]
# introduce new column to data called family_size:
titanic['family_size'] = titanic.SibSp + titanic.Parch +1
titanic_test['family_size'] = titanic_test.SibSp + titanic_test.Parch +1

# calculate survival rate for each family in train_set:
rate_family = titanic.groupby('family')['Survived', 'family','family_size'].median()
rate_family.head()
# if family size is more than 1 and family name is in overlap list 
overlap_family ={}
for i in range(len(rate_family)):
    if rate_family.index[i] in overlap and  rate_family.iloc[i,1] > 1:
        overlap_family[rate_family.index[i]] = rate_family.iloc[i,0]
mean_survival_rate = np.mean(titanic.Survived)
family_survival_rate = []
family_survival_rate_NA = []

for i in range(len(titanic)):
    if titanic.family[i] in overlap_family:
        family_survival_rate.append(overlap_family[titanic.family[i]])
        family_survival_rate_NA.append(1)
    else:
        family_survival_rate.append(mean_survival_rate)
        family_survival_rate_NA.append(0)
        
titanic['family_survival_rate']= family_survival_rate
titanic['family_survival_rate_NA']= family_survival_rate_NA
# repeat the same for test set:
mean_survival_rate = np.mean(titanic.Survived)
family_survival_rate = []
family_survival_rate_NA = []

for i in range(len(titanic_test)):
    if titanic_test.family[i] in overlap_family:
        family_survival_rate.append(overlap_family[titanic_test.family[i]])
        family_survival_rate_NA.append(1)
    else:
        family_survival_rate.append(mean_survival_rate)
        family_survival_rate_NA.append(0)
titanic_test['family_survival_rate']= family_survival_rate
titanic_test['family_survival_rate_NA']= family_survival_rate_NA
# drop name and family from dataset:
titanic = titanic.drop(['Name', 'family'], axis=1)
titanic_test = titanic_test.drop(['Name', 'family'], axis=1)

titanic.head()
sns.boxplot(titanic.Age)
sns.boxplot(titanic.Fare)
sns.boxplot(titanic.Age_mean)
print('skew for fare', titanic['Fare'].skew())
print('skew for age mean', titanic['Age_mean'].skew())
# use IQR to handle skewed data of Fare
quar_range = titanic.Fare.quantile(0.75) - titanic.Fare.quantile(0.25)
upper_bound = titanic.Fare.quantile(0.75) + 3*quar_range
titanic.loc[titanic.Fare > upper_bound, 'Fare'] = upper_bound
titanic_test.loc[titanic_test.Fare > upper_bound, 'Fare'] = upper_bound
max(titanic.Fare)
# use IQR to handle age_mean
quar_range = titanic.Age_mean.quantile(0.75) - titanic.Age_mean.quantile(0.25)
upper_bound = titanic.Age_mean.quantile(0.75) + 3*quar_range
titanic.loc[titanic.Age_mean > upper_bound, 'Age_mean'] = upper_bound
titanic_test.loc[titanic_test.Age_mean > upper_bound, 'Age_mean'] = upper_bound
max(titanic.Age_mean)
# use IQR to handle age 
quar_range = titanic.Age.quantile(0.75) - titanic.Age.quantile(0.25)
upper_bound = titanic.Age.quantile(0.75) + 3*quar_range
titanic.loc[titanic.Age > upper_bound, 'Age'] = upper_bound
titanic_test.loc[titanic_test.Age > upper_bound, 'Age'] = upper_bound
max(titanic.Age)
# check all columns values in training set are in test set also
columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

for c in columns:
    print(c)
    print(titanic[c].unique())
    print(titanic_test[c].unique())
sns.barplot(x='SibSp', y = 'Survived', data=titanic)

sns.barplot(x='Parch', y = 'Survived', data=titanic)

sns.barplot(x='family_size', y = 'Survived', data=titanic)
# combine train and test set
data = pd.concat([titanic.drop('Survived', axis = 1), titanic_test], axis = 0, sort=False)
data.head()
# encode variables onto numberic labels
le = LabelEncoder()
columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']

for c in columns:
    le.fit(data[c])
    data[c] = le.transform(data[c])
data.head()
# drop columns that have information about age or are strongly correlated with other features
data = data.drop(['Age_mean'], axis =1)
sns.pairplot(data, 
             x_vars=['Pclass', 'Sex','Fare','Embarked','ticket_type','cabin_type','title', 'family_survival_rate'],
            y_vars='Age')
colormap = plt.cm.RdBu
plt.figure(figsize=(14,8))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.corr(), linewidth=0.1, vmax=1.0, square=False, linecolor='black', cmap=colormap, annot=True)
x_train_age = data.dropna().drop(['Age'], axis=1)
y_train_age = data.dropna()['Age']
x_test_age = data[pd.isnull(data.Age)].drop(['Age'], axis=1)
model_lin = make_pipeline(StandardScaler(), KernelRidge())
kfold = model_selection.KFold(n_splits=10, random_state=4, shuffle=True)
parameters = {'kernelridge__gamma' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'kernelridge__kernel': ['rbf', 'linear'],
               'kernelridge__alpha' :[0.001, 0.01, 0.1, 1, 10, 100, 1000],
             }
search_lin = GridSearchCV(model_lin, parameters, n_jobs=-1, cv=kfold, scoring='r2', verbose=1)
search_lin.fit(x_train_age, y_train_age)
print('Best paramter are:', search_lin.best_params_)
print("Best accuracy achieved:",search_lin.cv_results_['mean_test_score'].mean())
y_test_age = search_lin.predict(x_test_age)
data.loc[data['Age'].isnull(), 'Age'] = y_test_age
titanic.shape
# seperate train and test data from data variable
idx = titanic.shape[0]
titanic['Age'] = data.iloc[:idx].Age
titanic_test['Age'] = data.iloc[idx:].Age
titanic['Age'].isnull().sum()
le = LabelEncoder()
titanic_train_LE = titanic.copy()
titanic_test_LE = titanic_test.copy()

columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']

for c in columns:
    le.fit(titanic_train_LE[c])
    titanic_train_LE[c] = le.transform(titanic_train_LE[c])    
    titanic_test_LE[c] = le.transform(titanic_test_LE[c])
titanic_train_LE.head()
plt.figure(figsize=(17,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(titanic_train_LE.corr(),linewidths=0.1,vmax=1.0, 
            square=True,linecolor='white',cmap=colormap, annot=True)
drop_col = ['Age_mean', 'SibSp', 'Parch']
titanic_train_LE = titanic_train_LE.drop(drop_col, axis=1)
titanic_test_LE = titanic_test_LE.drop(drop_col, axis=1)
drop_col = ['Age_mean', 'SibSp', 'Parch']
X_train_ohe = titanic.drop(drop_col, axis = 1)
X_test_ohe = titanic_test.drop(drop_col, axis = 1)

columns = ['cabin_type', 'title',  'Sex', 'Embarked', 'ticket_type', 'Pclass']

for c in columns:
    X_train_ohe = pd.concat([X_train_ohe, pd.get_dummies(X_train_ohe[c], drop_first=True)], axis=1)
#     X_test_ohe = pd.concat([X_train_ohe, pd.get_dummies(X_test_ohe[c], drop_first=True)], axis = 1)
    X_test_ohe = pd.concat([X_test_ohe, pd.get_dummies(X_test_ohe[c], drop_first = True)], axis =1)
X_train_ohe = X_train_ohe.drop(columns, axis=1)
X_test_ohe = X_test_ohe.drop(columns, axis=1)
X_test_ohe.head()
sns.clustermap(X_train_ohe.corr(),linewidths=0.1,vmax=1.0, 
            square=True,linecolor='white',cmap=colormap)
X_train_lab = titanic.drop(drop_col, axis=1)
X_test_lab = titanic_test.drop(drop_col, axis=1)
le = LabelEncoder()
columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']

for col in columns:
    le.fit(titanic[col])
    X_train_lab[col] = le.transform(X_train_lab[col])
    X_test_lab[col] = le.transform(X_test_lab[col])
    
X_test_lab.head()
sns.clustermap(X_train_lab.corr(),linewidths=0.1,vmax=1.0, 
            square=True,linecolor='white',cmap=colormap)
X_train_mean = titanic.drop(drop_col, axis=1)
X_test_mean = titanic_test.drop(drop_col, axis=1)
columns = ['cabin_type', 'title',  'Sex', 'Embarked', 'ticket_type']

for col in columns:
    ordered_labels = X_train_mean.groupby([col])['Survived'].mean().to_dict()
    X_train_mean[col] = X_train_mean[col].map(ordered_labels)
    X_test_mean[col] = X_test_mean[col].map(ordered_labels)
X_train_mean.head()
sns.clustermap(X_train_mean.corr(),linewidths=0.1,vmax=1.0, 
            square=True,linecolor='white',cmap=colormap)
X_train_freq = titanic.drop(drop_col, axis=1)
X_test_freq = titanic_test.drop(drop_col, axis=1)
columns = ['cabin_type', 'title',  'Sex', 'Embarked', 'ticket_type']

for col in columns:
    ordered_labels = X_train_freq[col].value_counts().to_dict()
    X_train_freq[col] = X_train_freq[col].map(ordered_labels)
    X_test_freq[col] = X_test_freq[col].map(ordered_labels)
X_train_freq.head()
sns.clustermap(X_train_freq.corr(),linewidths=0.1,vmax=1.0, 
            square=True,linecolor='white',cmap=colormap)
#  function split into x and y
def split_x_y(dataset):
    X = dataset.drop(columns= ['Survived'])
    Y = dataset['Survived']
    return X,Y
kfold = StratifiedKFold(n_splits=5)
X_ohe, Y_ohe = split_x_y(X_train_ohe)
X_lab, Y_lab = split_x_y(X_train_lab)
X_mean, Y_mean  = split_x_y(X_train_mean)
X_freq, Y_freq  = split_x_y(X_train_freq)
random_state = 4
classifiers = []

classifiers.append(('SVC', make_pipeline(StandardScaler(),SVC(random_state=random_state))))
classifiers.append(('DecisionTree', DecisionTreeClassifier(random_state=random_state)))
classifiers.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1)))
classifiers.append(('RandomForest', RandomForestClassifier(random_state=random_state)))
classifiers.append(('GradientBoost', GradientBoostingClassifier(random_state=random_state)))
classifiers.append(('MPL', make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state))))
classifiers.append(('KNN',make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=7))))

results = []
names = []
means = []
stds = []
for name, classifier in classifiers:
    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle=True)
    cv_result = model_selection.cross_val_score(classifier, X_ohe, y=Y_ohe, cv=kfold, scoring='accuracy')
    results.append(cv_result)
    means.append(cv_result.mean())
    stds.append(cv_result.std())
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_result.mean(), cv_result.std())
    print(msg)
g=sns.barplot(x=names, y=means)
g.figure.set_figwidth(15)
def random_forest(X, Y, X_test):
    parameters = {'max_depth': [2, 4, 5, 10],
                 'n_estimators': [200, 500, 1000, 2000],
                 'min_samples_split': [3, 4, 5]}
    
    kfold = model_selection.KFold(n_splits=3, random_state=4, shuffle=True)
    modelRFC = RandomForestClassifier(random_state=4, n_jobs=-1)
    searchRFC = GridSearchCV(modelRFC, parameters, n_jobs=-1, cv=kfold, scoring='accuracy', verbose=1)
    searchRFC.fit(X, Y)
    predict = searchRFC.predict(X_test)
    
    print('Best Parameters of RFC are:', searchRFC.best_params_)
    print('Best accuracy achieved of RFC', searchRFC.best_score_)
    
    return searchRFC.best_params_, modelRFC, searchRFC, predict
paramsRFCohe,modelRFCohe,searchRFCohe, predictRFCohe = random_forest(X_ohe, Y_ohe, X_test_ohe)
paramsRFClab,modelRFClab,searchRFClab, predictRFClab = random_forest(X_lab, Y_lab, X_test_lab)
paramsRFCmean,modelRFCmean,searchRFCmean, predictRFCmean = random_forest(X_mean, Y_mean, X_test_mean)
paramsRFCfreq,modelRFCfrq,searchRFCfreq, predictRFCfreq = random_forest(X_freq, Y_freq, X_test_freq)


