# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# data process
import numpy as np 
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz 
%matplotlib inline

# modeling
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble
from xgboost import XGBClassifier

# system
import os
import sys

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Python version
print("Python version: {}". format(sys.version))

# Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory.
print(os.listdir("../input"))

# read data into pandas' data frame
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

# make a deep copy to keep original data
train_df = train_raw.copy(deep = True)
test_df = test_raw.copy(deep = True)
combine = [train_df, test_df]
# training data sample
train_df.head()

# test data sample
test_df.head()
# training data info
train_df.info()
# test data info
test_df.info()
# check missing value
print('Train columns with null values:\n', train_df.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', test_df.isnull().sum())
# complete missing values in train and test dataset
for dataset in combine:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
# delete columns not informative
drop_column = ['Cabin', 'Ticket']
for dataset in combine:  
    dataset.drop(drop_column, axis=1, inplace=True)
               
# create features for modeling
for dataset in combine:    

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


# rename the less frequent Titile to 'Misc'
#pd.crosstab(train_df['Title'], train_df['Sex'])
main_titles = (train_df['Title'].append(test_df['Title']).value_counts() >10)
for dataset in combine:
    dataset['Title'] = dataset['Title'].apply(lambda x: x if main_titles.loc[x] == True else 'Misc')
train_df['Title'].value_counts()
# convert objects to category
label = LabelEncoder()
for dataset in combine:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
# check surviving rate for dimensions
Target = ['Survived']
columns_to_check = ['Sex','Pclass', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
for x in columns_to_check:
    print('Survival Correlation by:', x)
    #print(train_df[[x, Target[0]]].groupby(x, as_index=False).mean())
    print(train_df[[x, Target[0]]].groupby(x, as_index=False).agg(['count', 'mean']))
    print('-'*10, '\n')
#graph individual features by survival
sns.set_context("paper", font_scale=1.6) 
fig, saxis = plt.subplots(2, 3,figsize=(10,8))
fig.tight_layout()

sns.barplot(x = 'Embarked', y = 'Survived', data=train_df, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train_df, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=train_df, ax = saxis[0,2])

ax1=sns.pointplot(x = 'FareBin', y = 'Survived',  data=train_df, ax = saxis[1,0])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 30)
ax2 = sns.pointplot(x = 'AgeBin', y = 'Survived',  data=train_df, ax = saxis[1,1])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 30)
sns.pointplot(x = 'FamilySize', y = 'Survived', data=train_df, ax = saxis[1,2])
#more side-by-side comparisons
fig, (maxis1, maxis2, maxis3, maxis4) = plt.subplots(1, 4, figsize=(16,8))
fig.tight_layout() # increase space between plots

#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)

#how does fair factor with sex & survival compare
sns.pointplot(x="FareBin", y="Survived", hue="Sex", data=train_df,
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
maxis2.set_xticklabels(maxis2.get_xticklabels(), rotation = 30)

#how does age factor with sex & survival compare
sns.pointplot(x="AgeBin", y="Survived", hue="Sex", data=train_df,
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis3)
maxis3.set_xticklabels(maxis3.get_xticklabels(), rotation = 30)

#how does family size factor with sex & survival compare
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=train_df,
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis4)
train_df.columns
# select features for modeling
data_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data_xy_bin = Target + data_x_bin

data = train_df[data_xy_bin]
X = train_df[data_x_bin]
y = train_df[Target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_pred = test_df[data_x_bin]


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(10, 8))
    sns.set(font_scale=1.5) 
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        #cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    #plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data)
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_train_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_test_log = round(logreg.score(X_test, y_test) * 100, 2)
print('logistic regression train accurary: ',acc_train_log)
print('logistic regression test accurary: ',acc_test_log)

#coefficients of each factors
coeff_df = pd.DataFrame(X_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()
svc.fit(X_train, y_train)
acc_train_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_test_svc = round(svc.score(X_test, y_test) * 100, 2)
print('Support Vector Machine train accurary: ',acc_train_svc)
print('Support Vector Machine test accurary: ',acc_test_svc)

# K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_train_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_test_knn = round(knn.score(X_test, y_test) * 100, 2)
print('K-Nearest Neighbors train accurary: ',acc_train_knn)
print('K-Nearest Neighbors test accurary: ',acc_test_knn)

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
acc_train_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_test_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
print('Decision Tree train accurary: ',acc_train_decision_tree)
print('Decision Tree test accurary: ',acc_test_decision_tree)

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Y_pred = random_forest.predict(X_test)
acc_train_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_test_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest train accurary: ',acc_train_random_forest)
print('Random Forest test accurary: ',acc_test_random_forest)
# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    #ensemble.AdaBoostClassifier(),
    #ensemble.BaggingClassifier(),
    #ensemble.ExtraTreesClassifier(),
    #ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    #gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    #linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    #linear_model.Perceptron(),
    
    #Navies Bayes
    #naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    #svm.NuSVC(probability=True),
    #svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    #tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    #discriminant_analysis.LinearDiscriminantAnalysis(),
    #discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = y

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, train_df[data_x_bin], train_df[Target],cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions
    alg.fit(train_df[data_x_bin], train_df[Target])
    MLA_predict[MLA_name] = alg.predict(train_df[data_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict
#base model
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, train_df[data_x_bin], train_df[Target], cv  = cv_split)
dtree.fit(train_df[data_x_bin], train_df[Target])

print('BEFORE DT Parameters: ', dtree.get_params())
print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
#Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
dot_data = tree.export_graphviz(dtree, out_file=None, 
                                feature_names = data_x_bin, class_names = True,
                                filled = True, rounded = True)
graph = graphviz.Source(dot_data) 
graph
#tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none
              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }

#choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
tune_model.fit(train_df[data_x_bin], train_df[Target])

#print(tune_model.cv_results_.keys())
#print(tune_model.cv_results_['params'])
print('AFTER DT Parameters: ', tune_model.best_params_)
#print(tune_model.cv_results_['mean_train_score'])
print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
#print(tune_model.cv_results_['mean_test_score'])
print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)

dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, random_state = 0)
dtree.fit(train_df[data_x_bin], train_df[Target])
dot_data = tree.export_graphviz(dtree, out_file=None, 
                                feature_names = data_x_bin, class_names = True,
                                filled = True, rounded = True)
graph = graphviz.Source(dot_data) 
graph
Y_pred = dtree.predict(X_pred)

# get result file
submission_tree = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": Y_pred
    })

submission_tree.head()

#save the submission file
submission_tree.to_csv('submission_tree_turned.csv', index=False)