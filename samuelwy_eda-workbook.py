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
import seaborn as sns

print('Seaborn version : {}'.format(sns.__version__))

import matplotlib.pyplot as plt

plt.style.use('ggplot')

print('Pandas version : {}'.format(pd.__version__))

print('Numpy version : {}'.format(np.__version__))

print('Matplotlib version : {}'.format(None))

plt.style.use('ggplot')

sns.set_palette('Pastel2')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



print('Train data import sucessfully. : df_train')

print('Test data import sucessfully. : df_test')



train = df_train.copy()

test = df_test.copy()



print('Copy of Train data created sucessfully. : train')

print('Copy of Test data created sucessfully. : test')
print('Train data : {}'.format(train.shape))

print('Test data : {}'.format(test.shape))
print(train.info(),'\n')

print(train.describe(),'\n')
print(test.info(),'\n')

print(test.describe(),'\n')
train
test
train_na = train.isnull().sum()

test_na = test.isnull().sum()



print('Missing value\'s count for each column from train and test data :')

print(pd.concat([train_na, test_na], axis = 1, keys=['Train', 'Test']))
print('Unique values for Categorical features :','\n')



print('Train')

for col in train.drop(['PassengerId', 'Name','Age', 'Ticket', 'Fare', 'Cabin'],axis = 1):

    print(col + ' : {}'.format(train[col].unique()))

print('\n')    

print('Test')

for col in test.drop(['PassengerId', 'Name','Age', 'Ticket', 'Fare', 'Cabin'],axis = 1):

    print(col + ' : {}'.format(test[col].unique()))
from pandas_profiling import ProfileReport
ProfileReport(train, title="Pandas Profiling Report")
train.Survived.unique()
def univariate_countplot(data, x,xlabel,ylabel = 'No. of Passengers',title = '',fontsize = 14, legend = '',label = '',ax = None):

    '''Plot countplot for a single variable.'''

    sns.countplot(data = data, x = x,ax = ax)

    plt.title(title, fontsize = fontsize)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)



def univariate_count(on, norm = False):

    '''Print the Pmf for one variable.'''

    print('Proportion of {} :'.format(on))

    print(train[on].value_counts(normalize = norm ).sort_index())

    

def bivariate_countplot(data, x,hue,xlabel,ylabel,title,legend = '',label = '',ax = ''):

    '''Plot countplot for a single variable with hue as second variable.'''

    sns.countplot(data = train, x = x, hue = hue, ax=ax)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title, fontsize = 14)

    if legend == True:

        plt.legend(loc = 'best',labels = label, title = hue)



    

def bivariate_groupby(by,on,normalize):

    '''Print the Pmf for two variable.'''

    print('Proportion of {} by {} :'.format(by,on))

    print(train.groupby(by)[on].value_counts(normalize = normalize).sort_index())
print('Survival rate :')

print(train.Survived.value_counts(normalize=True))
fig,ax = plt.subplots(figsize = (5,5))



univariate_countplot(train,'Survived','Survived',ylabel = 'No. of Passengers',title = 'Survived',fontsize = 14, legend = '',label = '',ax = ax)

plt.xticks([0,1], ['No','Yes'])

plt.yticks(np.arange(0,600,50))
univariate_count('Pclass', True)
fig,ax = plt.subplots()

univariate_countplot(train,'Pclass','Pclass', 'No. of passengers', 'Ticket Class',ax=ax)
bivariate_groupby('Pclass','Survived', True)
bivariate_groupby(['Pclass','Sex'],'Survived', True)
fig = plt.figure(figsize = (10,6))



ax1 = fig.add_subplot(121)

bivariate_countplot(train,'Pclass', 'Survived','Class', 'No. of passengers','Class by Survival',True, ['No','Yes'],ax1)



ax2 = fig.add_subplot(122)

bivariate_countplot(train,'Pclass', 'Sex','Class', 'No. of passengers','Class by Sex',ax=ax2)
g = sns.catplot(data = train, x = 'Pclass', hue = 'Survived',col = 'Sex', kind = 'count')
train['Title'] = train.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())



def title_type(row):

    if row in ['Don', 'Mme',

       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',

       'Jonkheer','Dona','Dr','Rev']:

        # label as rare for titles that are low in counts

        return 'Rare'

    elif row == 'Miss':

        return 'Ms'

    else:

        return row

    

train['Title'] = train.Title.apply(title_type)
print('Proportion of Title :')

print(train.Title.value_counts(normalize = True).sort_values(ascending = False))
fig,ax = plt.subplots()

univariate_countplot(train, 'Title','Title of Passengers', title = 'Title of Passenger',ax=ax )

plt.xticks(rotation = 90)
fig =plt.figure(figsize = (20,15))

ax1 = fig.add_subplot(221)

train.groupby('Title')['Pclass'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax1)

ax1.set(title = 'Proportion of Ticket class by Title')



ax2 = fig.add_subplot(222)

train.groupby('Title')['Embarked'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax2)

ax2.set(title = 'Proportion of destination by Title')



ax3 = fig.add_subplot(223)

sns.boxplot(data = train, x = 'Title', y = 'Fare',ax=ax3)

ax3.set(title = 'Fare of ticket by Title')



ax4 = fig.add_subplot(224)

train.groupby('Title')['Survived'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax4)

ax4.set(title = 'Proportion of Survival by Title')

plt.legend(title = 'Survived', labels = ['No','Yes'])



plt.xticks(rotation = 90)
univariate_count('Sex',True)
bivariate_groupby('Sex', 'Survived', True)
fig = plt.figure(figsize = (10,5))

ax1 = fig.add_subplot(121)

univariate_countplot(train,'Sex',xlabel = 'Sex', title = 'Sex of Passengers', ax=ax1)



ax2 = fig.add_subplot(122)

bivariate_countplot(train,'Sex','Survived','Sex', 'No.of passengers', 'Survival by Sex', True, ['No','Yes'],ax=ax2)
univariate_count('Age',True)
train.Age.describe()
# explore min and max age

train[(train.Age == train.Age.max()) | (train.Age == train.Age.min())]
fig = plt.figure(figsize = (15,8))

ax1 = fig.add_subplot(221)

sns.distplot(train['Age'],ax=ax1)

ax1.set(title = 'Age\'s PDF')



ax2 = fig.add_subplot(222)

sns.kdeplot(train.Age, cumulative = True, shade = True,ax=ax2)

ax2.set(title = 'Age\'s CDF',xlabel = 'Age')

ax2.axhline(0.5, color = 'b', label = 'median')

plt.legend()



ax3 = fig.add_subplot(223)

from scipy.stats import norm

x = np.arange(0,80)

y = norm(train.Age.mean(), train.Age.std()).pdf(x)

ax3.plot(x,y, label = 'theorical')

sns.kdeplot(train['Age'],ax=ax3, label = 'sample')

plt.legend(loc = 'upper left')



ax4 = fig.add_subplot(224)

y = norm(train.Age.mean(), train.Age.std()).cdf(x)

ax4.plot(x,y,label = 'theorical')

sns.kdeplot(train.Age, cumulative = True, shade = True,ax=ax4, label = 'sample')

plt.legend()
# Children , adult and senior citizens

def age_diff(row):

    if row < 18:

        return 'Child'

    elif (row < 60) & (row >=18):

        return 'Adult'

    else:

        return 'Elderly'



train['Age_cat'] = train.Age.apply(age_diff)
univariate_count('Age_cat',True)
fig,ax = plt.subplots()

univariate_countplot(train,'Age_cat','Age',title = 'Age category', ax=ax)
fig = plt.figure(figsize = (10,12))

ax1 = fig.add_subplot(221)

sns.boxplot(data = train, x = 'Sex', y = 'Age',ax=ax1)

sns.stripplot(data = train, x = 'Sex', y = 'Age',ax=ax1,size = 2)

ax1.set(title = 'Age by Sex')



ax2 = fig.add_subplot(222)

sns.regplot(data = train, x = 'Age', y = 'Fare',ax=ax2,scatter_kws = {'alpha':0.5, 's' : 5})

ax2.set(title = 'Age vs Fare')



ax3 = fig.add_subplot(223)

sns.boxplot(data = train, x = 'Survived', y = 'Age',ax=ax3)

sns.stripplot(data = train, x = 'Survived', y = 'Age',ax=ax3,size = 2)

ax3.set(title = 'Age by Survived')

plt.xticks([0,1], ['No','Yes'])



ax4 = fig.add_subplot(224)

sns.boxplot(data = train, x = 'Pclass', y = 'Age',ax=ax4)

sns.stripplot(data = train, x = 'Pclass', y = 'Age',ax=ax4,size = 2)

ax4.set(title = 'Age by class')

sns.catplot(data = train, x = 'Sex', y = 'Age', col = 'Survived', kind = 'box')
univariate_count('SibSp', True)
fig = plt.figure(figsize = (15,8))

ax1 = fig.add_subplot(131)



sns.countplot(data = train, x = 'SibSp',ax=ax1)

ax1.set(title='Siblings and Spouse', ylabel = 'No. of passengers')



ax2 = fig.add_subplot(132)

train.groupby('SibSp')['Survived'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax2)

ax2.set(title = 'Siblings and Spouse by Survival')

plt.xticks(rotation = False)

plt.legend(loc = 'upper right', title = 'Survived', labels = ['No','Yes'])



ax3 = fig.add_subplot(133)

train.groupby('SibSp')['Sex'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax3)

ax3.set(title = 'Siblings and Spouse by Sex')

plt.xticks(rotation = False)

plt.legend(loc = 'upper right', title = 'Sex')
univariate_count('Parch', True)
fig = plt.figure(figsize = (15,8))

ax1 = fig.add_subplot(131)



sns.countplot(data = train, x = 'Parch',ax=ax1)

ax1.set(title='Parent and Children', ylabel = 'No. of passengers')



ax2 = fig.add_subplot(132)

train.groupby('Parch')['Survived'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax2)

ax2.set(title = 'Parent and Children by Survival')

plt.xticks(rotation = False)

plt.legend(loc = 'upper right', title = 'Survived', labels = ['No','Yes'])



ax3 = fig.add_subplot(133)

train.groupby('Parch')['Sex'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax3)

ax3.set(title = 'Parent and Children by Sex')

plt.xticks(rotation = False)

plt.legend(loc = 'upper right', title = 'Sex')
train['Family'] = train.SibSp + train.Parch + 1

train['Family_type'] = pd.cut(train.Family, [0,1,4,7,11], labels = ['Single', 'Small', 'Medium', 'Large'])
univariate_count('Family_type', True)
fig,ax = plt.subplots()

univariate_countplot(train,'Family_type','Family',title = 'Type of family size', ax=ax)
train.Fare.describe()
train[(train.Fare == train.Fare.min()) |(train.Fare == train.Fare.max())]
fig = plt.figure(figsize = (15,8))

ax1 = fig.add_subplot(221)

sns.distplot(train['Fare'],ax=ax1)

ax1.set(title = 'Fare\'s PDF')



ax2 = fig.add_subplot(222)

sns.kdeplot(train.Fare, cumulative = True, shade = True,ax=ax2)

ax2.set(title = 'Fare\'s CDF',xlabel = 'Age')

plt.legend()

fig = plt.figure(figsize = (12,8))

ax1 = fig.add_subplot(121)

sns.boxplot(data = train, x = 'Survived', y = 'Fare',ax=ax1)

sns.stripplot(data = train, x = 'Survived', y = 'Fare',ax=ax1,size = 2)

ax1.set(title = 'Fare by Survived')

plt.xticks([0,1], labels = ['No','Yes'])



ax2 = fig.add_subplot(122)

sns.boxplot(data = train, x = 'Pclass', y = 'Fare',ax=ax2)

sns.stripplot(data = train, x = 'Pclass', y = 'Fare',ax=ax2,size = 2)

ax2.set(title = 'Fare by Ticket Class')

fig,ax = plt.subplots()



sns.kdeplot(train.query('Pclass == 1').Fare, shade = True,ax=ax, label = '1')

sns.kdeplot(train.query('Pclass == 2').Fare, shade = True,ax=ax, label = '2')

sns.kdeplot(train.query('Pclass == 3').Fare, shade = True,ax=ax, label = '3')

ax.set(xlabel = 'Fare', title = 'Fare by Ticket Class')

plt.legend(title = 'Pclass', labels = [1,2,3])
train['Cabin_floor'] = train.Cabin.apply(lambda x: list(str(x))[0])

train['Cabin_floor'] = train.Cabin_floor.replace('n', np.nan)
univariate_count('Cabin_floor', True)
fig = plt.figure(figsize = (15,8))



ax1 = fig.add_subplot(131)

sns.countplot(data = train, x = 'Cabin_floor', order = ['A','B','C','D','E','F','T'],ax=ax1)

ax1.set(title = 'Cabin floor', xlabel = 'Cabin floor', ylabel = 'no. of passengers')



ax2 = fig.add_subplot(132)

train.groupby('Cabin_floor')['Survived'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax2)

ax2.set(title = 'Cabin floor by Survival')

plt.xticks(rotation = False)

plt.legend(loc = 'upper right', title = 'Survived')



ax3 = fig.add_subplot(133)

train.groupby('Cabin_floor')['Pclass'].value_counts(normalize = True).unstack().plot(kind = 'bar', stacked = True,ax=ax3)

ax3.set(title = 'Cabin floor by Ticket class')

plt.xticks(rotation = False)

plt.legend(loc = 'upper right', title = 'Pclass')
univariate_count('Embarked', True)
fig = plt.figure(figsize = (10,6))



ax1 = fig.add_subplot(121)

sns.countplot(data = train, x = 'Embarked',ax=ax1)

ax1.set(xlabel = 'Embarked', ylabel = 'No. of passengers', title = 'Embarked')



ax2 = fig.add_subplot(122)

sns.countplot(data = train, x = 'Embarked',hue = 'Survived',ax=ax2)

ax1.set(xlabel = 'Embarked', ylabel = 'No. of passengers', title = 'Embarked by Survived')

plt.legend(title = 'Survived', labels = ['No', 'Yes'])
fig,ax = plt.subplots(figsize = (8,8))

sns.heatmap(train.corr(),annot = True,ax=ax)
import missingno as msno

msno.bar(train)
msno.matrix(train)
train.Age.fillna(train.Age.median(), inplace = True)

train.Cabin_floor.fillna(train.Cabin_floor.mode().values[0], inplace= True)



train.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis = 1, inplace = True)



train.isna().sum()
train.info()
fig = plt.figure(figsize = (10,8))

ax1 = fig.add_subplot(121)

sns.boxplot(data = train, y = 'Age',ax=ax1)

ax1.set( title = 'Age')



ax2 = fig.add_subplot(122)

sns.boxplot(data = train, y = 'Fare',ax=ax2)

ax2.set( title = 'Fare')
# Age, Fare

# use IQR approach

Q1_age = train.Age.quantile(0.25)

Q3_age = train.Age.quantile(0.75)

IQR_age = Q3_age - Q1_age



train = train[~((train.Age < (Q1_age - 1.5 * IQR_age)) | (train.Age > (Q3_age + 1.5 * IQR_age )))]



Q1_fare = train.Fare.quantile(0.25)

Q3_fare= train.Fare.quantile(0.75)

IQR_fare = Q3_fare - Q1_fare



train = train[~((train.Fare < (Q1_fare - 1.5 * IQR_fare)) | (train.Fare > (Q3_fare + 1.5 * IQR_fare )))]
fig = plt.figure(figsize = (10,8))

ax1 = fig.add_subplot(121)

sns.boxplot(data = train, y = 'Age',ax=ax1)

ax1.set( title = 'Age')



ax2 = fig.add_subplot(122)

sns.boxplot(data = train, y = 'Fare',ax=ax2)

ax2.set( title = 'Fare')
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()



for col in train.select_dtypes('object').columns:

    train[col] = encoder.fit_transform(train[col])
y = train['Survived']

X = train.drop(['Survived','SibSp','Parch','Age','Family','Cabin_floor'],axis = 1)



print('Feature\'s shape : {}'.format(X.shape))

print('Target\'s shape : {}'.format(y.shape))
X.Family_type = encoder.fit_transform(X.Family_type)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_auc_score



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score



from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

import xgboost as xgb



seed = 225



model_result = dict()
steps = [('Scaler', StandardScaler()), ('KNN', KNN())]

pipeline = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



params = {'KNN__n_neighbors': np.arange(1,20)}

knn_cv = GridSearchCV(pipeline,params, cv = 5, verbose = 1, n_jobs = -1 )



knn_cv.fit(X_train,y_train)



print('Best params : {}'.format(knn_cv.best_params_))

print('Best score : {:.2f}'.format(knn_cv.best_score_))



y_pred_train = knn_cv.predict(X_train)

y_pred_test = knn_cv.predict(X_test)



print('KNN\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('KNN\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('KNN\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['KNN'] = accuracy_score(y_test,y_pred_test)
steps = [('Scaler', StandardScaler()), ('LR', LogisticRegression(random_state = seed))]

pipeline = Pipeline(steps)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



params = {'LR__C': [0.001,0.01,0.1,1,10,100,1000]}



lr_cv = GridSearchCV(pipeline,params, cv = 5, verbose = 1, n_jobs = -1 )



lr_cv.fit(X_train,y_train)



print('Best params : {}'.format(lr_cv.best_params_))

print('Best score : {:.2f}'.format(lr_cv.best_score_))



y_pred_train = lr_cv.predict(X_train)

y_pred_test = lr_cv.predict(X_test)



print('LR\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('LR\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('LR\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['LR'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



rf = RandomForestClassifier(random_state = seed)

params = {'n_estimators' : [200,300,400], 'max_depth':[10,12,14], 'max_features':['auto','sqrt','log2']}



rf_cv = GridSearchCV(rf,params, cv = 3, verbose = 1, n_jobs = -1 )



rf_cv.fit(X_train,y_train)



print('Best params : {}'.format(rf_cv.best_params_))

print('Best score : {:.2f}'.format(rf_cv.best_score_))



y_pred_train = rf_cv.predict(X_train)

y_pred_test = rf_cv.predict(X_test)



print('RF\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('RF\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('RF\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['RF'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



steps = [('Scaler', StandardScaler()), ('SVC', SVC(random_state = seed))]

pipeline = Pipeline(steps)





parameters = {'SVC__C':[0.1, 1, 10,100, 1000], 'SVC__gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1,1,10]}

searcher = GridSearchCV(pipeline, parameters, cv = 5, n_jobs = -1, verbose = 1)



searcher.fit(X_train,y_train)



print('Best params : {}'.format(searcher.best_params_))

print('Best score : {:.2f}'.format(searcher.best_score_))



y_pred_train = searcher.predict(X_train)

y_pred_test = searcher.predict(X_test)



print('SVC\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('SVC\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('SVC\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['SVC'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



steps = [('Scaler', StandardScaler()), ('LinearSVC', LinearSVC(random_state = seed))]

pipeline = Pipeline(steps)





parameters = {'LinearSVC__C':[0.1, 1, 10,100, 1000], 'LinearSVC__penalty':['l1','l2'], 'LinearSVC__loss' : ['hinge', 'squared_hinge']}

searcher = GridSearchCV(pipeline, parameters, cv = 5, n_jobs = -1, verbose = 1)



searcher.fit(X_train,y_train)



print('Best params : {}'.format(searcher.best_params_))

print('Best score : {:.2f}'.format(searcher.best_score_))



y_pred_train = searcher.predict(X_train)

y_pred_test = searcher.predict(X_test)



print('LinearSVC\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('LinearSVC\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('LinearSVC\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['LinearSVC'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



steps = [('Scaler', StandardScaler()), ('SGD', SGDClassifier(random_state = seed))]

pipeline = Pipeline(steps)





parameters = {'SGD__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'SGD__loss':['hinge', 'log'], 'SGD__penalty':['l1', 'l2']}

searcher = GridSearchCV(pipeline, parameters, cv = 5, n_jobs = -1, verbose = 1)



searcher.fit(X_train,y_train)



print('Best params : {}'.format(searcher.best_params_))

print('Best score : {:.2f}'.format(searcher.best_score_))



y_pred_train = searcher.predict(X_train)

y_pred_test = searcher.predict(X_test)



print('SGD\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('SGD\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('SGD\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['SGD'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



dt = DecisionTreeClassifier(max_depth = 4,random_state = seed)

ada = AdaBoostClassifier(dt,random_state = seed)



params = {'n_estimators' : [200,300,400], 'learning_rate' : [0.1,0.2,0.4,1]}



searcher = GridSearchCV(ada,params, cv = 5, verbose = 1, n_jobs = -1 )



searcher.fit(X_train,y_train)



print('Best params : {}'.format(searcher.best_params_))

print('Best score : {:.2f}'.format(searcher.best_score_))

y_pred_train = searcher.predict(X_train)

y_pred_test = searcher.predict(X_test)



print('Adaboost\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('Adaboost\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('Adaboost\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['Adaboost'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



gb = GradientBoostingClassifier(random_state = seed, subsample = 0.8)



params = {'learning_rate' : [0.1,0.2], 'n_estimators' : [200,300,400],'max_depth' : [2,3,4,6]}



searcher = GridSearchCV(gb,params, cv = 5, verbose = 1, n_jobs = -1 )



searcher.fit(X_train,y_train)



print('Best params : {}'.format(searcher.best_params_))

print('Best score : {:.2f}'.format(searcher.best_score_))



y_pred_train = searcher.predict(X_train)

y_pred_test = searcher.predict(X_test)



print('Gradient Boosting\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('Gradient Boosting\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('Gradient Boosting\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['Gradient Boosting'] = accuracy_score(y_test,y_pred_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



steps = [('Scaler', StandardScaler()), ('XG', xgb.XGBClassifier(random_state = seed))]

pipeline = Pipeline(steps)



params = {'XG__learning_rate' : [0.1,0.2,0.4], 'XG__gamma' : [0.0001,0.001,0.01,1,10],'XG__max_depth' : [2,3,4,6]}



searcher = GridSearchCV(pipeline,params, cv = 5, verbose = 1, n_jobs = -1 )



searcher.fit(X_train,y_train)



print('Best params : {}'.format(searcher.best_params_))

print('Best score : {:.2f}'.format(searcher.best_score_))



y_pred_train = searcher.predict(X_train)

y_pred_test = searcher.predict(X_test)



print('XGBoost\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))

print('XGBoost\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test,y_pred_test))

print('XGBoost\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test),[1,0]).plot()



model_result['XGBoost'] = accuracy_score(y_test,y_pred_test)
print(model_result)
# Voting Classifier

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)



lr_pipeline = Pipeline([('scale', StandardScaler()), ('LR',LogisticRegression(random_state=seed, C = 1))])



sgd = SGDClassifier(alpha = 0.01, loss = 'hinge', penalty = 'l2', random_state = seed)



gb = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 2, n_estimators = 200, random_state = seed, subsample = 0.8)



xg = xgb.XGBClassifier(random_state = seed, gamma = 0.01, learning_rate = 0.2, max_depth = 3)



classifiers = [('Logistic Regression', lr_pipeline), ('SGD', sgd), ('Gradient Boosting', gb), ('XGBoost',xg)]



for clf_name, clf in classifiers:    

 

    # Fit clf to the training set

    clf.fit(X_train, y_train)    

   

    # Predict y_pred

    y_pred = clf.predict(X_test)

    

    # Calculate accuracy

    accuracy = accuracy_score(y_test, y_pred) 

   

    # Evaluate clf's accuracy on the test set

    print('{:s} : {:.3f}'.format(clf_name, accuracy))



# Import VotingClassifier from sklearn.ensemble

from sklearn.ensemble import VotingClassifier



# Instantiate a VotingClassifier vc

vc = VotingClassifier(estimators=classifiers)     



# Fit vc to the training set

vc.fit(X_train, y_train)   



# Evaluate the test set predictions

y_pred = vc.predict(X_test)



# Calculate accuracy score

accuracy = accuracy_score(y_test, y_pred)

print('Voting Classifier: {:.3f}'.format(accuracy))