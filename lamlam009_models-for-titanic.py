# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
print("data_train.shape: {}".format(data_train.shape))
print("data_test.shape: {}".format(data_test.shape))
X = data_train.copy(deep = True)
X_test = data_test.copy(deep = True)
X.dtypes
X.isnull().sum()
X.drop(["Survived","PassengerId"], axis = 1, inplace=True)
X_test.drop(["PassengerId"], axis = 1, inplace=True)
import re

def getTitle(s):
    m = re.search(r", (\w+).", s)
    return m.group(1)
X["Title"] = X["Name"].apply(getTitle)
X_test["Title"] = X_test["Name"].apply(getTitle)
def cleanTitle(df):
    df["Title"][df["Title"] == "Mlle"] = "Miss"
    df["Title"][df["Title"] == "Ms"] = "Miss"
    df["Title"][df["Title"] == "Mme"] = "Mrs"

    Rare_Title = (df["Title"] != "Mr") & (df["Title"] != "Miss") & (df["Title"] != "Mrs") & (df["Title"] != "Master")
    df["Title"][Rare_Title] = "Rare_Title"
    df.drop(['Name'], axis = 1, inplace = True)
    return df
X = cleanTitle(X)
X_test = cleanTitle(X_test)
X['Family_size'] = (X['SibSp'] + X['Parch']).astype('object')
X_test['Family_size'] = (X_test['SibSp'] + X_test['Parch']).astype('object')
X.drop(['SibSp','Parch'], axis = 1, inplace=True)
X_test.drop(['SibSp','Parch'], axis = 1, inplace=True)
X['Pclass'] = X['Pclass'].astype('object')
X_test['Pclass'] = X_test['Pclass'].astype('object')
X_num = X.select_dtypes(exclude=["object"])
X_test_num = X_test.select_dtypes(exclude=["object"])
columns = X_num.columns
X_num.head()
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline

# making pipeline for organization
my_pipeline = Pipeline([('imputer',Imputer()), ('StandardScaler', StandardScaler())])
X_num = my_pipeline.fit_transform(X_num)
X_test_num = my_pipeline.transform(X_test_num)
X_num = pd.DataFrame(X_num, columns = columns)
X_test_num = pd.DataFrame(X_test_num, columns = columns)
print("X_num.shape: {}".format(X_num.shape))
print("X_test_num.shape: {}".format(X_test_num.shape))
X_dum = X.select_dtypes(include=["object"])
X_test_dum = X_test.select_dtypes(include=["object"])
X_dum.head()
X_dum.drop(["Ticket", "Cabin"], axis = 1, inplace=True)
X_test_dum.drop(["Ticket", "Cabin"], axis = 1, inplace=True)
X_dum = pd.get_dummies(X_dum)
X_test_dum = pd.get_dummies(X_test_dum)
X_train = pd.concat([X_num, X_dum], axis=1)
X_test = pd.concat([X_test_num, X_test_dum], axis=1)
y_train = data_train["Survived"].copy(deep=True)
print("X_train.shape: {}".format(X_train.shape))
print("X_test.shape: {}".format(X_test.shape))
X_test.isnull().sum()
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn import model_selection
from sklearn import metrics
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]
cv_split = model_selection.ShuffleSplit(n_splits = 10, random_state = 0)
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = pd.DataFrame()#y_train
MLA_pred = pd.DataFrame()

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv = cv_split, scoring='accuracy')

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(X_train, y_train)
    MLA_predict[MLA_name] = alg.predict(X_train)
    MLA_pred[MLA_name] = alg.predict(X_test)
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
model = linear_model.RidgeClassifierCV()
model.fit(X_train, y_train)
from sklearn.ensemble import VotingClassifier
top5 = [('rc',linear_model.RidgeClassifierCV()), ('lda',discriminant_analysis.LinearDiscriminantAnalysis()),
        ('svc',svm.SVC(probability=True)), ('xgb', XGBClassifier()), ('lsvc',svm.LinearSVC())]
eclf = VotingClassifier(estimators= top5, voting='hard')
cv_results = model_selection.cross_validate(eclf, X_train, y_train, cv = cv_split, scoring='accuracy')
print(cv_results['train_score'].mean())
print(cv_results['test_score'].mean())
model = linear_model.RidgeClassifierCV()
model.fit(X_train,y_train)
predict = model.predict(X_test)
y_test = pd.read_csv("../input/gender_submission.csv")
my_submission = pd.DataFrame({'PassengerId': y_test.PassengerId, 'Survived': predict.astype(int)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
