# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
#import data from file: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_train = pd.read_csv('../input/titanic/train.csv')
data_test  = pd.read_csv('../input/titanic/test.csv')

#group together so they can be cleaned at the same time
data_all = [data_train, data_test]

#preview data
print (data_train.info())
#data_train.head()
#data_train.tail()
data_train.sample(10)

data_train.shape
#look for missing values
print('Train columns with null values:\n', data_train.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_test.isnull().sum())
print("-"*10)

#data_train.describe(include = 'all')
data_age_corr = data_train.corr().abs()['Age'].sort_values(kind="quicksort", ascending=False)
print(data_age_corr)
age_by_pclass_sex = data_train.groupby(['Sex', 'Pclass']).median()['Age']

print(age_by_pclass_sex)
#print(age_by_pclass_sex['female'][1])

def median_age(entry):
    if entry.isnull()['Age'] == False:
        return entry['Age']
    else:
        return age_by_pclass_sex[entry['Sex']][entry['Pclass']]

for dataset in data_all:
    dataset['Age'] = dataset.apply(lambda x: median_age(x), axis=1)
    
#check data1 and data_test have no missing values in the age feature
print(data_train.isnull().sum())
print("-"*10)
print(data_test.isnull().sum())
print(data_train.iloc[0].isnull()['Age'])
for dataset in data_all:
    age_bins = [0, 10, 17 ,25, 32, 40, 47, 60, 80]
    dataset['AgeBin'] = pd.cut(dataset['Age'], age_bins)
#create FamilySize feature
for dataset in data_all:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
fare_by_pclass_sex_family_size = data_train.groupby(['Sex', 'Pclass', 'FamilySize']).median()['Fare']

#print(fare_by_pclass_sex_family_size)
#print(fare_by_pclass_sex_family_size['female'][1][4])

def median_fare(entry):
    if entry.isnull()['Fare'] == False:
        return entry['Fare']
    else:
        return fare_by_pclass_sex_family_size[entry['Sex']][entry['Pclass']][entry['FamilySize']]

for dataset in data_all:
    dataset['Fare'] = dataset.apply(lambda x: median_fare(x), axis=1)
    
#check data1 and data_test have no missing values in the age feature
print(data_train.isnull().sum())
print("-"*10)
print(data_test.isnull().sum())
#cut into equal sized fare bins based on data_train - use .qcut()
data_train['FareBin'], fare_bins = pd.qcut(data_train['Fare'], 4,  retbins=True)
fare_bins[0] = -0.001
#use bin ranges from data_train to cut data_test so bin ranges are the same - use .cut()
data_test['FareBin'] = pd.cut(data_test['Fare'], fare_bins)
print("fare bins: ", fare_bins)
for dataset in data_all:
    dataset['Embarked'].fillna(data_train['Embarked'].mode()[0], inplace = True) #need the index as there can be multiple modes
for dataset in data_all: 
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]#expand=True means split elements are indexed, and indexes say which part to take
    
#print(data_train[data_train['Title'] == 'Mlle'])
    
for dataset in data_all:
    #clean up rare titles
    #dataset['Title'] = dataset.loc[(dataset.Title == 'Mlle'),'Title']='Mrs'
    #dataset['Title'] = dataset.loc[(dataset.Title == 'Mme'),'Title']='Miss'
    #dataset['Title'] = dataset.loc[(dataset.Title == 'Ms'),'Title']='Mrs'
    #dataset['Title'] = dataset.loc[(dataset.Title == 'Dona'),'Title']='Mrs'  #All these are first class so probably not worth it.
    stat_min = 5
    title_rare = (data_train['Title'].value_counts() < stat_min) #boolean of titles being rare based on numbers in data1
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if x not in title_rare or title_rare[x] else x)
    
print(data_train['Title'].value_counts())
print(data_test['Title'].value_counts())
for dataset in data_all:
    family_bins = [0, 1, 4, 7, 11]
    dataset['FamilyBin'] = pd.cut(dataset['FamilySize'], family_bins)
    dataset.drop(['FamilySize'], axis=1, inplace = True)
ticket_counts_train = data_train['Ticket'].value_counts()
ticket_counts_train_survived = data_train['Ticket'].drop(data_train[data_train['Survived'] == 0].index).value_counts()
ticket_counts_test = data_test['Ticket'].value_counts()
ticket_counts_total = pd.concat([ticket_counts_train, ticket_counts_test], axis=1).sum(axis=1)
ticket_counts = pd.DataFrame({'train':ticket_counts_train, 'test':ticket_counts_test, 'total':ticket_counts_total, 'train_survived':ticket_counts_train_survived})
ticket_counts.fillna(0, inplace=True)
print(ticket_counts)

def get_ticket_size_and_survival(ticket):
    ticket_size = ticket_counts['total'][ticket]
    ticket_survival_rate = 0
    ticket_survival_rate_na = 0
    if ticket_counts['train_survived'][ticket] != 0 and ticket_counts['train'][ticket] >= 2:
        ticket_survival_rate = ticket_counts['train_survived'][ticket]/ticket_counts['train'][ticket]
    if ticket_counts['train'][ticket] < 2:
        ticket_survival_rate_na = 1
    return [ticket_size, ticket_survival_rate, ticket_survival_rate_na]
    
for dataset in data_all:
    dataset['TicketSize'] = dataset['Ticket'].apply(lambda x: get_ticket_size_and_survival(x)[0])
    dataset['TicketSurvival'] = dataset['Ticket'].apply(lambda x: get_ticket_size_and_survival(x)[1])
    dataset['TicketSurvival_NA'] = dataset['Ticket'].apply(lambda x: get_ticket_size_and_survival(x)[2])
    

#data_train.head()
data_test.head()
#data_test['TicketSurvival_NA'].value_counts()
data_train['Cabin'].dropna().value_counts()
for dataset in data_all:
    dataset['Deck'] = dataset['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'Missing')
    
cabin_by_class = data_train.groupby(['Deck', 'Pclass', 'Sex']).count()['PassengerId']
cabin_by_class_survival = data_train.groupby(['Deck', 'Pclass', 'Sex']).mean()['Survived']
print(cabin_by_class)
print(cabin_by_class_survival)
for dataset in data_all:
    dataset['Deck'] = dataset['Deck'].replace(['T'], 'Missing')
    dataset['Deck'] = dataset['Deck'].replace(['D', 'E'], 'DE')
    dataset['Deck'] = dataset['Deck'].replace(['F', 'G'], 'FG')
    
    
cabin_by_class = data_train.groupby(['Deck', 'Pclass', 'Sex']).count()['PassengerId']
cabin_by_class_survival = data_train.groupby(['Deck', 'Pclass', 'Sex']).mean()['Survived']
#print(cabin_by_class)
#print(cabin_by_class_survival)
print(data_train.columns.values.tolist())
# could keep Age and Fare in and use more iterations
colums_to_drop = ['PassengerId', 'Name', 'Age', 'Fare', 'Ticket', 'Cabin']
columns_to_encode = ['Sex', 'Embarked', 'AgeBin', 'FareBin', 'Title', 'FamilyBin', 'Deck']
columns_to_one_hot = ['Sex', 'Embarked', 'Title', 'FamilyBin', 'Deck'] #probably don't need to one-hot encode family bin


for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_all['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if x not in title_rare or title_rare[x] else x)



for dataset in data_cleaner:
    dataset['Age'].fillna(data1['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(data1['Embarked'].mode()[0], inplace = True) #need the index as there can be multiple modes
    dataset['Fare'].fillna(data1['Fare'].median(), inplace=True)
    
    drop_columns = ['Cabin', 'Ticket']
    dataset.drop(drop_columns, axis=1, inplace = True)
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]#expand=True means split elements are indexed, and indexes say which part to take
    #clean up rare titles
    stat_min = 5
    title_rare = (data1['Title'].value_counts() < stat_min) #boolean of titles being rare based on numbers in data1
    #try using apply (apply function along axis of df) and lambda functions to avoid for loop - see https://towardsdatascience.com/apply-and-lambda-usage-in-pandas-b13a1ea037f7
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if x not in title_rare or title_rare[x] else x)

#have a look
print(data1['Title'].value_counts())
print(data_test['Title'].value_counts())
data1.info()
data_test.info()
data1.sample(10)
#cut into equal sized fare bins based on data1 - use .qcut()
data1['FareBin'], fare_bins = pd.qcut(data1['Fare'], 4,  retbins=True)
fare_bins[0] = -0.001
#use bin ranges from data1 to cut data_test so bin ranges are the same - use .cut()
data_test['FareBin'] = pd.cut(data_test['Fare'], fare_bins)
print("fare bins: ", fare_bins)

#cut into age bins
#need to make sure the age bins cover all the possible values
print("data1 age range: ", data1['Age'].min(), " - ", data1['Age'].max())
print("data_test age range: ", data_test['Age'].min(), "-", data_test['Age'].max())

age_bins = [0, 15, 30, 45, 60, 80]
data1['AgeBin'] = pd.cut(data1['Age'], age_bins)
data_test['AgeBin'] = pd.cut(data_test['Age'], age_bins)

#data1.head()
data_test.head()
#check for null values
data1.info()
data_test.info()
#print(data_test_cat[data_test_cat['FareBin'].isnull()])
#from notebook above
#we will use seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html

#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
#graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])

#how does embark port factor with class, sex, and survival compare
#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
e = sns.FacetGrid(data1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep', order = [1, 2, 3], hue_order = ['female', 'male'])
e.add_legend()
#plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
a.add_legend()
#correlation heatmap of dataset
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
    
#Catagorical variables need to be coded eg. Sex, AgeBin etc.
label = LabelEncoder()
data1_coded = data1.copy(deep = True)
data_test_coded = data_test.copy(deep = True)
data_cleaner_coded = [data1_coded, data_test_coded]
for dataset in data_cleaner_coded:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

correlation_heatmap(data1_coded)
data1.head()
for dataset in data_cleaner:
    age_bins = [0, 10, 17 ,25, 32, 40, 47, 60, 80]
    dataset['AgeBinNew'] = pd.cut(dataset['Age'], age_bins)
    family_bins = [0, 1, 4, 7, 11]
    dataset['FamilyBin'] = pd.cut(dataset['FamilySize'], family_bins)
    
data1.info()
data_test.info()

data1.head()
data1_dummy = pd.get_dummies(data1, columns=['Embarked', 'Title', 'AgeBinNew', 'FamilyBin'])
data1_dummy.info()
data1_dummy.head()
#correlation heatmap of dataset again
correlation_heatmap(data1_dummy)
data1.to_csv('titanic_data1_for_pivot1.csv')
def hand_tree_model(df):
    df['Prediction'] = 0
    df.loc[df['Sex'] == "female", "Prediction"] = 1
    df.loc[(df['Sex'] == "female") & (df['Pclass'] == 3) & (df['Embarked'] == 'S'), 'Prediction'] = 0
    #df.loc[(df['Sex'] == "female") & (df['Embarked'] == 'S') & (df['FareBin'] == '(-0.001, 7.91]'), 'Prediction'] = 1
    df.loc[(df['Title'] == "Master") & ((df['Pclass'] == 1) | (df['Pclass'] == 2)), 'Prediction'] = 1
    
    return df

hand_tree_predict = hand_tree_model(data1)
print('accuracy of hand_tree_model on test1: {:.1f}%' .format(metrics.accuracy_score(data1['Survived'], hand_tree_predict['Prediction'])*100))
hand_tree_predict.head()
#can't seem to get submission accepted - any ideas?
hand_tree_predict_test = hand_tree_model(data_test)
just_predictions = hand_tree_predict_test['Prediction'].to_numpy()
hand_tree_predict_submission = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived' : just_predictions})
#hand_tree_predict_submission['PassengerId'] = hand_tree_predict_test['PassengerId']
#hand_tree_predict_submission['Survived'] = hand_tree_predict_test['Prediction'].astype(int)
#hand_tree_predict_test = hand_tree_predict_test[['PassengerId', 'Prediction']].rename(columns={"Prediction": "Survived"})
hand_tree_predict_submission.head(20)
print(just_predictions)
hand_tree_predict_submission.info()
print(hand_tree_predict_submission)
#hand_tree_predict_submission.to_csv('hand_tree_predict_test.csv')
#converting to coded categories
data1_coded = data1.copy(deep = True)
data_test_coded = data_test.copy(deep = True)
data_cleaner_coded = [data1_coded, data_test_coded]
label = LabelEncoder()
for dataset in data_cleaner_coded:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

#y variable - target/outcome
Target_label = ['Survived']

#x variables suitable for logistic regression
data1_x_lables = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
#using age and fare bins rather than continuous values (['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'FamilySize', 'Age', 'Fare']), otherwise need many more iterations
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
algorithm = linear_model.LogisticRegressionCV() #CV - cross  - means it runs on train and test subsets and so can be used to get accuracy score.
cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
print('Average train score: {:.2f}' .format(cv_results['train_score'].mean()))
print('Average test score: {:.2f}' .format(cv_results['test_score'].mean()))
cv_results
#try using continuous values for age and fare. More iterations needed.
data1_x_lables = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'FamilySize', 'Age', 'Fare']
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0) # run model 10x with 60/30 split intentionally leaving out 10%
algorithm = linear_model.LogisticRegressionCV(max_iter = 500) #CV - cross  - means it runs on train and test subsets and so can be used to get accuracy score.
cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
print('Average train score: {:.2f}' .format(cv_results['train_score'].mean()))
print('Average test score: {:.2f}' .format(cv_results['test_score'].mean()))
cv_results
MLAs = [
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    ensemble.RandomForestClassifier(),
    ensemble.ExtraTreesClassifier()
]
data1_x_lables = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
for algorithm in MLAs:
    cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
    print(algorithm.__class__.__name__)
    print('Average train score: {:.3f}' .format(cv_results['train_score'].mean()))
    print('Average test score: {:.3f}' .format(cv_results['test_score'].mean()))
    print("-"*10)    
MLAs = [
    svm.SVC(),
    svm.LinearSVC(max_iter=5000), #warning message to increase max_iter. (Other two have no max_iter limit.)
    svm.NuSVC()
]
data1_x_lables = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
for algorithm in MLAs:
    cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
    print(algorithm.__class__.__name__)
    print('Average train score: {:.3f}' .format(cv_results['train_score'].mean()))
    print('Average test score: {:.3f}' .format(cv_results['test_score'].mean()))
    print("-"*10)    
MLAs = [
    neighbors.KNeighborsClassifier()
]
data1_x_lables = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
for algorithm in MLAs:
    cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
    print(algorithm.__class__.__name__)
    print('Average train score: {:.3f}' .format(cv_results['train_score'].mean()))
    print('Average test score: {:.3f}' .format(cv_results['test_score'].mean()))
    print("-"*10) 
MLAs = [
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    #ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    #ensemble.RandomForestClassifier(),
    XGBClassifier()
]
data1_x_lables = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
for algorithm in MLAs:
    cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
    print(algorithm.__class__.__name__)
    print('Average train score: {:.3f}' .format(cv_results['train_score'].mean()))
    print('Average test score: {:.3f}' .format(cv_results['test_score'].mean()))
    print("-"*10) 
MLAs = [
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    svm.SVC(),
        
    neighbors.KNeighborsClassifier(),
    
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    XGBClassifier()
]

MLA_predictions = data1[Target_label].copy()

for alg in MLAs:
    MLA_name = alg.__class__.__name__
    #cv_results = model_selection.cross_validate(algorithm, data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel(), cv  = cv_split, return_train_score=True)
    alg.fit(data1_coded[data1_x_lables], data1_coded[Target_label].values.ravel())
    MLA_predictions[MLA_name] = alg.predict(data1_coded[data1_x_lables])
MLAs = [
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    svm.SVC(),
        
    neighbors.KNeighborsClassifier(),
    
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    XGBClassifier()
]
    
column_names = ['Survived']
for alg in MLAs:
    column_names.append(alg.__class__.__name__)
MLA_predictions2 = pd.DataFrame(columns = column_names)

for i in range(9):
    index_range = np.arange(99*i,99*(i+1)).tolist()
    train_subset = data1_coded.drop(index_range)
    test_subset = data1_coded.loc[index_range , : ]
    
    MLA_predictions_sample = test_subset[Target_label].copy()
    for alg in MLAs:
        MLA_name = alg.__class__.__name__
        alg.fit(train_subset[data1_x_lables], train_subset[Target_label].values.ravel())
        MLA_predictions_sample[MLA_name] = alg.predict(test_subset[data1_x_lables])
    MLA_predictions2 = MLA_predictions2.append(MLA_predictions_sample)
MLA_predictions.tail()
MLA_predictions2.tail()
Survived_0 = MLA_predictions2[MLA_predictions2['Survived'] == 0]
Survived_1 = MLA_predictions2[MLA_predictions2['Survived'] == 1]
Survived_1.head()
def compare_predictions(MLA_predictions_to_use):
    MLA_compare_columns = ['MLA Name', 'correct', 'accuracy', 'precision', 'recall', '1 correct', '% 1 correct', '0 correct', '% 0 correct']
    MLA_compare = pd.DataFrame(columns = MLA_compare_columns)

    row_index = 0
    for alg in MLAs:
        
        Survived_1 = MLA_predictions_to_use[MLA_predictions_to_use['Survived'] == 1]
        Survived_0 = MLA_predictions_to_use[MLA_predictions_to_use['Survived'] == 0]
        
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        
        comparison = MLA_predictions_to_use.apply(lambda x: True if x['Survived'] == x[alg.__class__.__name__] else False , axis=1)
        count = len(comparison[comparison == True].index)
        MLA_compare.loc[row_index, 'correct'] = count
        MLA_compare.loc[row_index, 'accuracy'] = count/len(MLA_predictions_to_use.index)
        
        comparison1 = MLA_predictions_to_use.apply(lambda x: True if (x['Survived'] == 1) and (x[alg.__class__.__name__] == 1) else False , axis=1)
        count1 = len(comparison1[comparison1 == True].index)
        comparison2 = MLA_predictions_to_use.apply(lambda x: True if x[alg.__class__.__name__] == 1 else False , axis=1)
        count2 = len(comparison2[comparison2 == True].index)
        MLA_compare.loc[row_index, 'precision'] = count1/count2
        
        comparison = MLA_predictions_to_use.apply(lambda x: True if x['Survived'] == x[alg.__class__.__name__] else False , axis=1)
        count = len(comparison[comparison == True].index)
        MLA_compare.loc[row_index, 'recall'] = count1/len(Survived_1.index)
                        
        comparison = Survived_1.apply(lambda x: True if x['Survived'] == x[alg.__class__.__name__] else False , axis=1)
        count = len(comparison[comparison == True].index)
        MLA_compare.loc[row_index, '1 correct'] = count
        MLA_compare.loc[row_index, '% 1 correct'] = count/len(Survived_1.index)
        
        comparison = Survived_0.apply(lambda x: True if x['Survived'] == x[alg.__class__.__name__] else False , axis=1)
        count = len(comparison[comparison == True].index)
        MLA_compare.loc[row_index, '0 correct'] = count
        MLA_compare.loc[row_index, '% 0 correct'] = count/len(Survived_0.index)

        row_index+=1
    return MLA_compare    
MLA_compare2 = compare_predictions(MLA_predictions2)
MLA_compare2.sort_values(by = ['correct'], ascending = False, inplace = True)
MLA_compare2
MLA_compare = compare_predictions(MLA_predictions)
MLA_compare.sort_values(by = ['correct'], ascending = False, inplace = True)
MLA_compare
def combined_prediction1(individual_predictions):
    total = sum(individual_predictions)
    num_algs = len(individual_predictions)
    if (total > num_algs*0.8): #5 must agree
        return 1
    elif (total < num_algs*0.2):
        return 0
    else:
        return 'unknown'

MLA_predictions2['Combined'] = MLA_predictions2.apply(lambda x: combined_prediction1([x['SVC'], x['GradientBoostingClassifier'], x['XGBClassifier'], x['BaggingClassifier'], x['RandomForestClassifier'], x['AdaBoostClassifier']]) , axis=1)
def identify_randoms(survived, individual_predictions):
    total = sum(individual_predictions)
    num_algs = len(individual_predictions)
    if (survived==0 and num_algs==total) or (survived==1 and total==0):
        return True
    else:
        return False

MLA_predictions2['Random'] = MLA_predictions2.apply(lambda x: identify_randoms(x['Survived'], [x['SVC'], x['GradientBoostingClassifier'], x['XGBClassifier'], x['BaggingClassifier'], x['RandomForestClassifier'], x['AdaBoostClassifier']]) , axis=1)
MLA_predictions2.info()
MLA_predictions2.head()
incor_cmb_pred = MLA_predictions2[MLA_predictions2['Survived'] != MLA_predictions2['Combined']]
incor_cmb_pred.info()
incor_cmb_pred.head()
incor_1 = incor_cmb_pred[(incor_cmb_pred['Survived'] == 1) & (incor_cmb_pred['Combined'] == 0)].copy()
incor_0 = incor_cmb_pred[(incor_cmb_pred['Survived'] == 0) & (incor_cmb_pred['Combined'] == 1)].copy()
incor_1_unknown = incor_cmb_pred[(incor_cmb_pred['Survived'] == 1) & (incor_cmb_pred['Combined'] == 'unknown')].copy()
incor_0_unknown = incor_cmb_pred[(incor_cmb_pred['Survived'] == 0) & (incor_cmb_pred['Combined'] == 'unknown')].copy()

incor_1.info()
incor_0.info()
incor_1_unknown.info()
incor_0_unknown.info()
incor_0_unknown.head()
#remove the random entires that we can't really correctly predict
incor_subsets = [incor_1, incor_0, incor_1_unknown, incor_0_unknown]
def remove_randoms(incor_subsets):
    for subset in incor_subsets:
        indexNames = subset[subset['Random'] == True].index
        subset.drop(indexNames , inplace=True)

remove_randoms(incor_subsets)
incor_1.info()
incor_0.info()
incor_1_unknown.info()
incor_0_unknown.info()
incor_0
column_data = ['algorithm', incor_1_unknown, incor_0_unknown, incor_1, incor_0]

def get_incor_comparison(column_data):
    column_name = ['algorithm', 'incor_1_unknown', 'incor_0_unknown', 'incor_1', 'incor_0']
    MLA_names = ['SVC', 'GradientBoostingClassifier', 'XGBClassifier', 'BaggingClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
    incor_comparison = pd.DataFrame(columns = column_name)

    incor_comparison.loc[0, 'algorithm'] = 'num entries'
    for i in range(1, 5):
        incor_comparison.loc[0, column_name[i]] = len(column_data[i].index)
    row_index = 1
    for alg in MLA_names:
        incor_comparison.loc[row_index, 'algorithm'] = alg
        incor_comparison.loc[row_index, 'incor_1_unknown'] = column_data[1][alg].sum()
        incor_comparison.loc[row_index, 'incor_0_unknown'] = incor_comparison.loc[0, 'incor_0_unknown'] - column_data[2][alg].sum()
        incor_comparison.loc[row_index, 'incor_1'] = column_data[3][alg].sum()
        incor_comparison.loc[row_index, 'incor_0'] = incor_comparison.loc[0, 'incor_0'] - column_data[4][alg].sum()


        row_index += 1
        
    return incor_comparison
get_incor_comparison(column_data)
#Condition 2a: SVC==GB==AB==0 --> 0
condition2a_good = incor_0_unknown.apply(lambda x: True if (x['SVC'] == 0) and (x['GradientBoostingClassifier'] == 0) and (x['AdaBoostClassifier'] == 0) else False , axis=1)
count2a_good = len(condition2a_good[condition2a_good == True].index)
condition2a_bad = incor_1_unknown.apply(lambda x: True if (x['SVC'] == 0) and (x['GradientBoostingClassifier'] == 0) and (x['AdaBoostClassifier'] == 0) else False , axis=1)
count2a_bad = len(condition2a_bad[condition2a_bad == True].index)

print('improvement to correct predictions: ', count2a_good, ' --> but increase in wrong predictions: ', count2a_bad)
#Condition 2b: SVC==AB==0 --> 0
condition2b_good = incor_0_unknown.apply(lambda x: True if (x['SVC'] == 0) and (x['AdaBoostClassifier'] == 0) and ((x['GradientBoostingClassifier'] + x['XGBClassifier'] + x['BaggingClassifier'] + x['RandomForestClassifier']) >= 4) else False , axis=1)
count2b_good = len(condition2b_good[condition2b_good == True].index)
condition2b_bad = incor_1_unknown.apply(lambda x: True if (x['SVC'] == 0) and (x['AdaBoostClassifier'] == 0) and ((x['GradientBoostingClassifier'] + x['XGBClassifier'] + x['BaggingClassifier'] + x['RandomForestClassifier']) >= 4) else False , axis=1)
count2b_bad = len(condition2b_bad[condition2b_bad == True].index)

print('improvement to correct predictions: ', count2b_good, ' --> but increase in wrong predictions: ', count2b_bad)
def combined_prediction2(individual_predictions):
    MLA_names = ['SVC', 'GradientBoostingClassifier', 'XGBClassifier', 'BaggingClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
    total = sum(individual_predictions)
    num_algs = len(individual_predictions)
    if (total > num_algs*0.8):
        prediction = 1
    elif (total < num_algs*0.2):
        prediction = 0
    else:
        prediction = 'unknown'
        
    if prediction == 'unknown':
        if (individual_predictions[MLA_names.index('SVC')] + individual_predictions[MLA_names.index('GradientBoostingClassifier')] + individual_predictions[MLA_names.index('AdaBoostClassifier')]) == 0:
            prediction = 0
        #elif individual_predictions[MLA_names.index('SVC')] == 1 and individual_predictions[MLA_names.index('AdaBoostClassifier')] == 1:
            #prediction = 0
    return prediction

MLA_predictions2['Combined2'] = MLA_predictions2.apply(lambda x: combined_prediction2([x['SVC'], x['GradientBoostingClassifier'], x['XGBClassifier'], x['BaggingClassifier'], x['RandomForestClassifier'], x['AdaBoostClassifier']]) , axis=1)
incor_cmb_pred2 = MLA_predictions2[MLA_predictions2['Survived'] != MLA_predictions2['Combined2']]
incor_cmb_pred2.info()
incor_cmb_pred2.head()
incor_1_2 = incor_cmb_pred2[(incor_cmb_pred2['Survived'] == 1) & (incor_cmb_pred2['Combined2'] == 0)].copy()
incor_0_2 = incor_cmb_pred2[(incor_cmb_pred2['Survived'] == 0) & (incor_cmb_pred2['Combined2'] == 1)].copy()
incor_1_unknown_2 = incor_cmb_pred2[(incor_cmb_pred2['Survived'] == 1) & (incor_cmb_pred2['Combined2'] == 'unknown')].copy()
incor_0_unknown_2 = incor_cmb_pred2[(incor_cmb_pred2['Survived'] == 0) & (incor_cmb_pred2['Combined2'] == 'unknown')].copy()

incor_1_2.info()
incor_0_2.info()
incor_1_unknown_2.info()
incor_0_unknown_2.info()
incor_0_unknown_2.head()
incor_subsets = [incor_1_2, incor_0_2, incor_1_unknown_2, incor_0_unknown_2]
remove_randoms(incor_subsets)
incor_1_2.info()
incor_0_2.info()
incor_1_unknown_2.info()
incor_0_unknown_2.info()
column_data2 = ['algorithm', incor_1_unknown_2, incor_0_unknown_2, incor_1_2, incor_0_2]

incor_comparison2 = get_incor_comparison(column_data2)
incor_comparison2
incor_1_unknown_2
incor_0_unknown_2

