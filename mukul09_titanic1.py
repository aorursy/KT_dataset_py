import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import random
import time
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
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
from pandas.tools.plotting import scatter_matrix


%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
data_raw=pd.read_csv('../input/train.csv')

data_val=pd.read_csv('../input/test.csv')

data1=data_raw.copy(deep=True)

data_cleaner=[data1, data_val]
print(data_raw.info())
data_raw.sample(5)
print('Train columns with null values:\n', data1.isnull().sum())
print('Test columns with null values:\n', data_val.isnull().sum())

data_raw.describe(include='all')
for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    
drop_column=['PassengerId', 'Cabin','Ticket']
data1.drop(drop_column, axis=1, inplace=True)

print(data1.isnull().sum())
print('-'*10)
print(data_val.isnull().sum())
for dataset in data_cleaner:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
    dataset['IsAlone']=1
    dataset['IsAlone'].loc[dataset['FamilySize']>1]=0
    
    dataset['Title']=dataset['Name'].str.split(", ",expand=True)[1].str.split('.',expand=True)[0]
    
    dataset['FareBin']=pd.qcut(dataset['Fare'],4)
    dataset['AgeBin']=pd.cut(dataset['Age'].astype(int),5)
    
    #clean rare title names

stat_min=10
title_names = (data1['Title'].value_counts() < stat_min)
data1['Title']=data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x )
print(data1['Title'].value_counts())

data1.info()
print('#'*20)
data_val.info()
print('#'*20)
data1.sample(5)
# convert Formats
    
label=LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code']=label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code']=label.fit_transform(dataset['Embarked'])
    dataset['Title_Code']=label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code']=label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code']=label.fit_transform(dataset['FareBin'])
    
Target=['Survived']

data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']

data1_xy=Target + data1_x
print('Original X Y: ', data1_xy, '\n')

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_x_dummy, '\n')

data1_dummy.head()

#double check cleaned data
'''print('Train columns with null values:\n', data1.isnull().sum())
print('#'*20)
print(data1.info())
print('#'*20)

print('Test values with null values:\n', data_val.isnull().sum())
print('#'*20)
print(data_val.info())
print('#'*20)

data_raw.describe(include='all')'''
#Split training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(data1[data1_x_calc], data1[Target], random_state=0)

X_train_bin, X_test_bin, y_train_bin, y_test_bin=train_test_split(data1[data1_x_bin], data1[Target], random_state=0)

X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy=train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state=0)

print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(X_train.shape))
print("Test1 Shape: {}".format(X_test.shape))

X_train_bin.head()
# discrete variable correlation
for x in data1_x:
    if data1[x].dtype!='float64':
        print('Survival correlation by:',x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*20)
        
print(pd.crosstab(data1['Title'],Target[0]))
#visualiation
plt.figure(figsize=[16,12])


plt.subplot(231)
plt.boxplot(data1['Fare'], showmeans=True, meanline=True)
plt.ylabel('Fare($)')
plt.title('Fare Boxplot')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans=True, meanline=True)
plt.ylabel('Age(years)')
plt.title('Age Boxplot')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans=True, meanline=True)
plt.ylabel('Family Size (#)')
plt.title('Family Size Boxplot')


plt.subplot(234)
plt.hist(x=[data1[data1['Survived']==0]['Fare'],data1[data1['Survived']==1]['Fare']], stacked=True, color=['g','b'], label=['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived']==0]['Age'],data1[data1['Survived']==1]['Age']], stacked=True, color=['g','b'], label=['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived']==0]['FamilySize'],data1[data1['Survived']==1]['FamilySize']], stacked=True, color=['g','b'], label=['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

fig, saxis=plt.subplots(2,3, figsize=(20,15))

sns.barplot(x='Embarked', y='Survived', data=data1, ax=saxis[0,0])
sns.barplot(x='Pclass', y='Survived', data=data1, ax=saxis[0,1])
sns.barplot(x='IsAlone', y='Survived', data=data1, ax=saxis[0,2])

sns.pointplot(x='FareBin', y='Survived', data=data1, ax=saxis[1,0])
sns.pointplot(x='AgeBin', y='Survived', data=data1, ax=saxis[1,1])
sns.pointplot(x='FamilySize', y='Survived', data=data1, ax=saxis[1,2])
fig,(ax1,ax2,ax3)=plt.subplots(1,3, figsize=(15,10))

sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data1, ax=ax1)
ax1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data1, ax=ax2,split=True)
ax2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=data1, ax=ax3)
ax3.set_title('Pclass vs FamilySize Survival Comparison')

fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
qaxis[0].set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])
qaxis[1].set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])
qaxis[2].set_title('Sex vs IsAlone Survival Comparison')
fig,(axis1,axis2)=plt.subplots(1,2,figsize=(8,8))

sns.pointplot(x='FamilySize', y='Survived', hue='Sex', data=data1,ax=axis1)

sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data1,ax=axis2)

pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
pp.set(xticklabels=[])


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

correlation_heatmap(data1)
#Modeling
MLA=[
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    gaussian_process.GaussianProcessClassifier(),
    
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    neighbors.KNeighborsClassifier(),
    
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    XGBClassifier()   
    
]


cv_split=model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare=pd.DataFrame(columns=MLA_columns)

MLA_predict=data1[Target]

row_index=0
for alg in MLA:
    MLA_name=alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name']=MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters']=str(alg.get_params())
    
    cv_results=model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split)
    
    MLA_compare.loc[row_index, 'MLA Time']=cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean']=cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean']=cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD']=cv_results['test_score'].std()*3
    
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name]=alg.predict(data1[data1_x_bin])
    
    row_index+=1
    
    
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

MLA_compare

#MLA_predict

sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name',data=MLA_compare, color='m')

plt.title('MLA Accuracy Score \n')
plt.xlabel('Accuracy Score(%)')
plt.ylabel('Algorithm')
for index, row in data1.iterrows():
    if random.random() > 0.5:
        data1.set_value(index, 'Random_Predict',1)
    else:
        data1.set_value(index, 'Random_Predict', 0)

data1['Random_Score']=0

data1.loc[(data1['Survived'] == data1['Random_Predict']), 'Random_Score'] = 1
#data1.head()


print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(metrics.accuracy_score(data1['Survived'], data1['Random_Predict'])*100))
#group by
pivot_female=data1[data1.Sex=='female'].groupby(['Sex','Pclass','Embarked','FareBin'])['Survived'].mean()
print(pivot_female)

pivot_male=data1[data1['Sex']=='male'].groupby(['Sex','Pclass','Title'])['Survived'].mean()
print(pivot_male)
#model performance with cross-Validation
dtree=tree.DecisionTreeClassifier(random_state=0)

base_results=model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv=cv_split)

dtree.fit(data1[data1_x_bin], data1[Target])
print('Before DT Parameters:', dtree.get_params())
print('Before DT Training w/bin score mean:{:.2f}'.format(base_results['train_score'].mean()*100))
print('Before DT Testing w/bin score mean:{:.2f}'.format(base_results['test_score'].mean()*100))
print('Before DT Testing w/bin score std*3:{:.2f}'.format(base_results['test_score'].std()*100*3))
print('Before DT Test w/bin score min:{:.2f}'.format(base_results['test_score'].min()*100))
print('#'*20)


param_grid = {'criterion': ['gini','entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none
              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }
#print(list(model_selection.ParameterGrid(param_grid)))

tune_model=model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, cv=cv_split)

tune_model.fit(data1[data1_x_bin], data1[Target])

#print(tune_model.cv_results_.keys())
#print(tune_model.cv_results_['params'])

print('After DT Parameters:',tune_model.best_params_)
#print(tune_model.cv_results_['mean_train_score'])
print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
#print(tune_model.cv_results_['mean_test_score'])
print("AFTER DT Testing w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('After DT Testing w/bin score std*3:{:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))


#duplicates gridsearchcv
#tune_results = model_selection.cross_validate(tune_model, data1[data1_x_bin], data1[Target], cv  = cv_split)

#print('AFTER DT Parameters: ', tune_model.best_params_)
#print("AFTER DT Training w/bin set score mean: {:.2f}". format(tune_results['train_score'].mean()*100)) 
#print("AFTER DT Test w/bin set score mean: {:.2f}". format(tune_results['test_score'].mean()*100))
#print("AFTER DT Test w/bin set score min: {:.2f}". format(tune_results['test_score'].min()*100))
#print('-'*10)

#Tune mOdel with feature Selection
#sklearn has several opetions we will use recursive feature selection (RFE) with cross validation
print('Before DT RFE Training shape old:  ', data1[data1_x_bin].shape)
print('Before DT RFE Training Columns Old:  ', data1[data1_x_bin].shape)
print('Before DT Training w/bin score mean:{:.2f}'.format(base_results['train_score'].mean()*100))
print('Before DT Testing w/bin score mean:{:.2f}'.format(base_results['test_score'].mean()*100))
print('Before DT Testing w/bin score std*3:{:.2f}'.format(base_results['test_score'].std()*100*3))
print('#'*20)

#feature selection
dtree_rfe=feature_selection.RFECV(dtree, step=1, scoring='accuracy',cv=cv_split)
dtree_rfe.fit(data1[data1_x_bin], data1[Target])

X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
print('AFTER DT RFE Training Columns New: ', X_rfe)
rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv  = cv_split)

print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 
print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))
print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))
print('#'*20)


#tune rfe model
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
rfe_tune_model.fit(data1[X_rfe], data1[Target])

#print(rfe_tune_model.cv_results_.keys())
#print(rfe_tune_model.cv_results_['params'])
print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
#print(rfe_tune_model.cv_results_['mean_train_score'])
print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
#print(rfe_tune_model.cv_results_['mean_test_score'])
print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('#'*20)







#base model with XGB Classifier
xgb=XGBClassifier(random_state=0)
base_results=model_selection.cross_validate(xgb, data1[data1_x_bin], data1[Target], cv=cv_split)
xgb.fit(data1[data1_x_bin], data1[Target])

print('Before XGB parameters: ', xgb.get_params())
print('Before XGB Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('Before XGB Testing w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('Before XGB Testing w/bin score std*3: {:.2f}'.format(base_results['test_score'].std()*100*3))
print('#'*20)


param_grid={
        'learning_rate': [.01, .03, .05, .1, .25],
        'max_depth': [1,2,4,6,8,10], #default 2
        'n_estimators': [10,50,100,301],
        'seed': [0]
}

tune_model=model_selection.GridSearchCV(XGBClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split)
tune_model.fit(data1[data1_x_bin], data1[Target])

print('After XGB Parameters: ', tune_model.best_params_)
print('After XGB Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('After XGB Testing w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_].mean()*100))
print('After XGB Testing w/bin score std*3: {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('#'*20)









#correlation_heatmap(MLA_predict)


param_grid={
        'learning_rate': [.01, .03, .05, .1, .25],
        'max_depth': [1,2,4,6,8,10], #default 2
        'n_estimators': [10,50,100,301],
        'seed': [0]
}
submit_xgb=XGBClassifier()
submit_xgb=model_selection.GridSearchCV(XGBClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split)
submit_xgb.fit(data1[data1_x_bin], data1[Target])
data_val['Survived']=submit_xgb.predict(data_val[data1_x_bin])
#submit=data_val[['PassengerId', 'Survived']]
#submit.to_csv('../working/submit.csv', index=False)
#print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize=True))
#submit.sample(10)'

submission=pd.DataFrame({
    "PassengerId":data_val['PassengerId'],
    "Survived": data_val['Survived']
})
submission.to_csv('titanic.csv', index=False)




