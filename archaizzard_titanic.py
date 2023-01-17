DATA_PATH = '../input'
import os

import warnings

import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from IPython import display





%matplotlib inline

warnings.filterwarnings('ignore')

mpl.style.use('ggplot')

sns.set_style('white')
titanic = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

data = titanic.copy(deep=True)

data.info()
data.head(10)
n_missing = data.isnull().sum()

percent_missing = (n_missing/data.isnull().count())

missing_summary = pd.concat([n_missing, percent_missing], axis=1)

missing_summary
from sklearn.impute import SimpleImputer





numerical_imputer   = SimpleImputer(strategy='median')

categorical_imputer = SimpleImputer(strategy='most_frequent')



data['Age'] = numerical_imputer.fit_transform(data['Age'].values.reshape(-1, 1))

data['Fare'] = numerical_imputer.fit_transform(data['Fare'].values.reshape(-1, 1))

data['Embarked'] = categorical_imputer.fit_transform(data['Embarked'].values.reshape(-1, 1))



data.drop(['Cabin'], axis=1, inplace=True)



n_missing = data.isnull().sum()

percent_missing = (n_missing/data.isnull().count())

missing_summary = pd.concat([n_missing, percent_missing], axis=1)

missing_summary
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



data['IsAlone'] = 1

data['IsAlone'].loc[data['FamilySize'] > 1] = 0



data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



data['FareBinned'] = pd.qcut(data['Fare'], 4)



data['AgeBinned'] = pd.qcut(data['Age'].astype(np.int64), 4)
stat_min = 10

title_names = (data['Title'].value_counts() < stat_min)

data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] else x)
dropping_cols = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Fare']

data.drop(dropping_cols, axis=1, inplace=True)
data.head()
from sklearn.preprocessing import LabelEncoder





label_encoder = LabelEncoder()

data['SexCode']      = label_encoder.fit_transform(data['Sex'])

data['EmbarkedCode'] = label_encoder.fit_transform(data['Embarked'])

data['TitleCode']    = label_encoder.fit_transform(data['Title'])

data['AgeCode']      = label_encoder.fit_transform(data['AgeBinned'])

data['FareCode']     = label_encoder.fit_transform(data['FareBinned'])



data.head()
target = ['Survived']

data_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'Age', 'FamilySize', 'IsAlone']

data_x_calc = ['SexCode','Pclass', 'EmbarkedCode', 'TitleCode', 'Age']

data_xy =  target + data_x



data_x_bin = ['SexCode','Pclass', 'EmbarkedCode', 'TitleCode', 'FamilySize', 'AgeCode', 'FareCode']

data_xy_bin = target + data_x_bin



data_dummy = pd.get_dummies(data[data_x])

data_x_dummy = data_dummy.columns.tolist()

data_xy_dummy = target + data_x_dummy



data_dummy.head()
from sklearn.model_selection import train_test_split





X_train, X_val, y_train, y_val = train_test_split(data[data_x_calc], data[target], random_state=0)

X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(data[data_x_bin], data[target], random_state=0)

X_train_dummy, X_val_dummy, y_train_dummy, y_val_dummy = train_test_split(data_dummy[data_x_dummy], data[target], random_state=0)



print("Data Shape: {}".format(data.shape))

print("Train Shape: {}".format(X_train.shape))

print("Val Shape: {}".format(X_val.shape))



X_train_bin.head()
plt.figure(figsize=(16, 12))



data['Fare'] = titanic['Fare']



plt.subplot(231)

plt.boxplot(x=data['Fare'], showmeans=True, meanline=True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data['Age'], showmeans=True, meanline=True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data['FamilySize'], showmeans=True, meanline=True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x=[data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], 

         stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x=[data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], 

         stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x=[data[data['Survived']==1]['FamilySize'], data[data['Survived']==0]['FamilySize']], 

         stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
fig, saxis = plt.subplots(2, 3,figsize=(16,12))



for sax in saxis[0]:

    sax.set_ylim(0, 1)

    

for sax in saxis[1]:

    sax.set_ylim(0, 1)



sns.barplot(x='Embarked', y='Survived', data=data, ax=saxis[0, 0])

sns.barplot(x='Pclass', y='Survived', order=[1, 2, 3], data=data, ax=saxis[0, 1])

sns.barplot(x='IsAlone', y='Survived', order=[1, 0], data=data, ax=saxis[0, 2])



sns.barplot(x='FareBinned', y='Survived', data=data, ax=saxis[1, 0])

sns.barplot(x='AgeBinned', y='Survived', data=data, ax=saxis[1, 1])

sns.barplot(x='FamilySize', y='Survived', data=data, ax=saxis[1, 2])
fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(14,12))



sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data, ax=axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data, split=True, ax=axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=data, ax=axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
fig, qaxis = plt.subplots(1,3,figsize=(14,12))



sns.barplot(x='Sex', y='Survived', hue='Embarked', data=data, ax=qaxis[0])

axis1.set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x='Sex', y='Survived', hue='Pclass', data=data, ax=qaxis[1])

axis1.set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x='Sex', y='Survived', hue='IsAlone', data=data, ax=qaxis[2])

axis1.set_title('Sex vs IsAlone Survival Comparison')
e = sns.FacetGrid(data, col = 'Embarked')

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')

e.add_legend()
a = sns.FacetGrid(data, hue='Survived', aspect=4)

a.map(sns.kdeplot, 'Age', shade=True)

a.set(xlim=(0 , data['Age'].max()))

a.add_legend()
h = sns.FacetGrid(data, row='Sex', col='Pclass', hue='Survived')

h.map(plt.hist, 'Age', alpha=.75)

h.add_legend()
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(data)
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import LogisticRegressionCV, PassiveAggressiveClassifier, RidgeClassifierCV, SGDClassifier, Perceptron

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier





MLA = [

    #Ensemble Methods

    AdaBoostClassifier(),

    BaggingClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    RandomForestClassifier(),



    #Gaussian Processes

    GaussianProcessClassifier(),

    

    #GLM

    LogisticRegressionCV(),

    PassiveAggressiveClassifier(),

    RidgeClassifierCV(),

    SGDClassifier(),

    Perceptron(),

    

    #Navies Bayes

    BernoulliNB(),

    GaussianNB(),

    

    #Nearest Neighbor

    KNeighborsClassifier(),

    

    #SVM

    SVC(probability=True),

    NuSVC(probability=True),

    LinearSVC(),

    

    #Trees    

    DecisionTreeClassifier(),

    ExtraTreeClassifier(),

    

    #Discriminant Analysis

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

]
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_validate





cv_split = StratifiedShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns=MLA_columns)



MLA_predict = data[target]



for row_index, alg in enumerate(MLA):

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    cv_results = cross_validate(alg, data[data_x_bin], data[target], cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    #MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    alg.fit(data[data_x_bin], data[target])

    MLA_predict[MLA_name] = alg.predict(data[data_x_bin])

    

MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

MLA_compare
sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='m')



plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

import random

from sklearn.metrics import accuracy_score





for index, row in data.iterrows():

    data.set_value(index, 'RandomPredict', 1 if random.random() > .5 else 0)

    

data['RandomScore'] = 0

data.loc[(data['Survived'] == data['RandomPredict']), 'RandomScore'] = 1

print('Coin Flip Model Accuracy: {:.2f}%'.format(data['RandomScore'].mean()*100))

print('Coin Flip Model Accuracy w/ Scikit-Learn: {:.2f}%'.format(accuracy_score(data['Survived'], data['RandomPredict'])*100))
pivot_female = data[data.Sex=='female'].groupby(['Sex','Pclass', 'Embarked','FareBinned'])['Survived'].mean()

print('Survival Decision Tree w/Female Node: \n',pivot_female)



pivot_male = data[data.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)
def mytree(df):

    

    #initialize table to store predictions

    Model = pd.DataFrame(data = {'Predict':[]})

    male_title = ['Master'] #survived titles



    for index, row in df.iterrows():



        #Question 1: Were you on the Titanic; majority died

        Model.loc[index, 'Predict'] = 0



        #Question 2: Are you female; majority survived

        if (df.loc[index, 'Sex'] == 'female'):

                  Model.loc[index, 'Predict'] = 1



        #Question 3A Female - Class and Question 4 Embarked gain minimum information



        #Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0       

        if ((df.loc[index, 'Sex'] == 'female') & 

            (df.loc[index, 'Pclass'] == 3) & 

            (df.loc[index, 'Embarked'] == 'S')  &

            (df.loc[index, 'Fare'] > 8)



           ):

                  Model.loc[index, 'Predict'] = 0



        #Question 3B Male: Title; set anything greater than .5 to 1 for majority survived

        if ((df.loc[index, 'Sex'] == 'male') &

            (df.loc[index, 'Title'] in male_title)

            ):

            Model.loc[index, 'Predict'] = 1

        

        

    return Model





#model data

Tree_Predict = mytree(data)

print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(accuracy_score(data['Survived'], Tree_Predict)*100))
import itertools

from  sklearn.metrics import confusion_matrix





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



    

# Compute confusion matrix

cnf_matrix = confusion_matrix(data['Survived'], Tree_Predict)

np.set_printoptions(precision=2)



class_names = ['Dead', 'Survived']

# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 

                      title='Normalized confusion matrix')
data.head()
X_train_dummy.head()
X_train_bin.head()
X_train.head()
X_train = pd.concat([pd.concat([X_train_bin, X_train_dummy], axis=1), pd.concat([X_val_bin, X_val_dummy], axis=1)])
try:

    X_train.drop(['SexCode', 'EmbarkedCode', 'TitleCode', 'Age', 'FamilySize', 'Pclass'], axis=1, inplace=True)

except KeyError:

    pass

X_train['FamilySize'] = pd.concat([X_train_bin['FamilySize'], X_val_bin['FamilySize']])

X_train['Pclass'] = pd.concat([X_train_bin['Pclass'], X_val_bin['Pclass']])

X_train.head()
X_train['IsAlone'].value_counts()
y_train = pd.concat([y_train_bin, y_val_bin], axis=0)

y_train.head()
y_train['Survived'].value_counts()
from sklearn.model_selection import GridSearchCV





param_grid = [

    {

        'kernel': ['rbf'],

        'nu': [i/10 for i in range(2, 8)],

        'gamma': [i/1000 for i in range(12, 20)],

    },

]

nusv_clf = NuSVC()

grid_search = GridSearchCV(nusv_clf, param_grid=param_grid, cv=4)

grid_search.fit(X_train, y_train)



best_nusv_clf = grid_search.best_estimator_

print(grid_search.best_params_)

print(grid_search.best_score_)
test_data = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

ids = test_data['PassengerId']
test_data.head()
n_missing = test_data.isnull().sum()

percent_missing = (n_missing/test_data.isnull().count())

missing_summary = pd.concat([n_missing, percent_missing], axis=1)

missing_summary
numerical_imputer   = SimpleImputer(strategy='median')

categorical_imputer = SimpleImputer(strategy='most_frequent')



test_data['Age']      = numerical_imputer.fit_transform(test_data['Age'].values.reshape(-1, 1))

test_data['Fare']     = numerical_imputer.fit_transform(test_data['Fare'].values.reshape(-1, 1))

test_data['Embarked'] = categorical_imputer.fit_transform(test_data['Embarked'].values.reshape(-1, 1))



test_data.drop(['Cabin'], axis=1, inplace=True)



n_missing = test_data.isnull().sum()

percent_missing = (n_missing/test_data.isnull().count())

missing_summary = pd.concat([n_missing, percent_missing], axis=1)

missing_summary
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1



test_data['IsAlone'] = 1

test_data['IsAlone'].loc[test_data['FamilySize'] > 1] = 0



test_data['Title'] = test_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



test_data['FareBinned'] = pd.qcut(test_data['Fare'], 4)



test_data['AgeBinned'] = pd.qcut(test_data['Age'].astype(np.int64), 4)
stat_min = 10

title_names = (test_data['Title'].value_counts() < stat_min)

test_data['Title'] = test_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] else x)
dropping_cols = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Fare']

test_data.drop(dropping_cols, axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder





label_encoder = LabelEncoder()

test_data['SexCode']      = label_encoder.fit_transform(test_data['Sex'])

test_data['EmbarkedCode'] = label_encoder.fit_transform(test_data['Embarked'])

test_data['TitleCode']    = label_encoder.fit_transform(test_data['Title'])

test_data['AgeCode']      = label_encoder.fit_transform(test_data['AgeBinned'])

test_data['FareCode']     = label_encoder.fit_transform(test_data['FareBinned'])



test_data.head()
target = ['Survived']

data_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'Age', 'FamilySize', 'IsAlone']

data_x_calc = ['SexCode','Pclass', 'EmbarkedCode', 'TitleCode', 'Age']

data_xy =  target + data_x



data_x_bin = ['SexCode','Pclass', 'EmbarkedCode', 'TitleCode', 'FamilySize', 'AgeCode', 'FareCode']

data_xy_bin = target + data_x_bin



data_dummy = pd.get_dummies(test_data[data_x])

data_x_dummy = data_dummy.columns.tolist()

data_xy_dummy = target + data_x_dummy



data_dummy.head()
test_data.describe()
X_train.head()
X_test = pd.concat([

    test_data['AgeCode'],

    test_data['FareCode'],

    test_data['IsAlone'],

    data_dummy['Sex_female'],

    data_dummy['Sex_male'],

    data_dummy['Embarked_C'],

    data_dummy['Embarked_Q'],

    data_dummy['Embarked_S'],

    data_dummy['Title_Master'],

    data_dummy['Title_Misc'],

    data_dummy['Title_Miss'],

    data_dummy['Title_Mr'],

    data_dummy['Title_Mrs'],

    test_data['FamilySize'],

    test_data['Pclass']

], axis=1)
X_test.head()
submit_data = pd.DataFrame()

submit_data['PassengerId'] = ids

submit_data.head()
submit_data['Survived'] = best_nusv_clf.predict(X_test)



print('Validation Data Distribution: \n', submit_data['Survived'].value_counts(normalize=True))
submit_data.head()
submit_data.to_csv('submission.csv', index=False)