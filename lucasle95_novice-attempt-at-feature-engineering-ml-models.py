# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
df_test = test_data
df.head()
df.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived','PassengerId'], axis = 1), df[['Survived']], test_size = 0.25, random_state = 99)

# PassengerId for validation set
df_test = df_test.drop('PassengerId', axis = 1)

print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())
import matplotlib.pyplot as plt
plt.hist(X_train['Age'], bins = 50, color = 'blue')
plt.show()
print(X_train[X_train["Age"].isnull()]['Sex'].value_counts())
print(df[df["Age"].isnull()]['Survived'].value_counts())
!pip install feature_engine
import feature_engine
from feature_engine import missing_data_imputers as mdi

imputer = mdi.ArbitraryNumberImputer(arbitrary_number=-1,
                                variables=['Age'])
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


## Age for validation set
df_test = imputer.transform(df_test)
plt.hist(X_train['SibSp'], bins = 50, color = 'blue')
plt.show()
print(X_train['SibSp'].isnull().any())
plt.hist(X_train['Parch'], bins = 50, color = 'blue')
plt.show()
print(X_train['Parch'].isnull().any())
plt.hist(X_train['Fare'], bins =50, color = 'blue')
plt.show()
print(X_train['Fare'].isnull().any())
pd.value_counts(X_train.Fare.sort_values())
imputer = mdi.ArbitraryNumberImputer(arbitrary_number=-1,
                                variables=['Fare'])
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
df_test = imputer.transform(df_test)

X_train['Pclass'].unique()
X_train['Name'].unique()
print("If the name consist of the title Mr:\n",df[df['Name'].str.contains("Mr")]['Survived'].value_counts())
print("If the name consist of the title Mrs:\n",df[df['Name'].str.contains("Mrs")]['Survived'].value_counts())
print("If the name consist of the title Master: \n",df[df['Name'].str.contains("Master")]['Survived'].value_counts())
print("If the name consist of the title Sir: \n",df[df['Name'].str.contains("Sir")]['Survived'].value_counts())
print("If the name consist of the title Capt: \n",df[df['Name'].str.contains("Capt")]['Survived'].value_counts())
print("If the name consist of the title Don: \n",df[df['Name'].str.contains("Don")]['Survived'].value_counts())
X_train = X_train.drop('Name', axis=1)
X_test = X_test.drop('Name', axis=1)

# Name for validation set
df_test = df_test.drop('Name', axis=1)
X_train['Sex'].unique()
from feature_engine import categorical_encoders as ce
ordinal = ce.OrdinalCategoricalEncoder(encoding_method='ordered',
                                  variables=['Sex'])
ordinal.fit(X_train, y_train)
X_train = ordinal.transform(X_train)
X_test = ordinal.transform(X_test)
df_test = ordinal.transform(df_test)
print(df[df['Ticket'].str.contains("PC")]['Survived'].value_counts()) #class 1
print(df[df['Ticket'].str.contains("SC")]['Survived'].value_counts()) #class 2
print(df[df['Ticket'].str.contains("SOTON")]['Survived'].value_counts()) #class 3
print(df[df['Ticket'].str.contains("C.A")]['Survived'].value_counts())  # class 2 and 3
print(df[df['Ticket'].str.contains("S.O")]['Survived'].value_counts()) #class 2 and 3
print(df[df['Ticket'].str.contains("W./C")]['Survived'].value_counts()) #class 2 and 3
X_train = X_train.drop('Ticket', axis=1)
X_test = X_test.drop('Ticket', axis=1)

# Ticket for validation set
df_test = df_test.drop('Ticket', axis=1)
from feature_engine.categorical_encoders import OrdinalCategoricalEncoder

# training set
X_train['cabin_num'] = X_train['Cabin'].str.extract('(\d+)') # captures numerical part
X_train['cabin_num'] = X_train['cabin_num'].astype('float')
X_train['cabin_cat'] = X_train['Cabin'].str[0] # captures the first letter

# testing set
X_test['cabin_num'] = X_test['Cabin'].str.extract('(\d+)') # captures numerical part
X_test['cabin_num'] = X_test['cabin_num'].astype('float')
X_test['cabin_cat'] = X_test['Cabin'].str[0] # captures the first letter

# validation set
df_test['cabin_num'] = df_test['Cabin'].str.extract('(\d+)') # captures numerical part
df_test['cabin_num'] = df_test['cabin_num'].astype('float')
df_test['cabin_cat'] = df_test['Cabin'].str[0] # captures the first letter

imputer_cabinNum = mdi.ArbitraryNumberImputer(arbitrary_number = -1, variables=['cabin_num'])
imputer_cabinCat = mdi.CategoricalVariableImputer(variables=['cabin_cat'])
imputer_cabinNum.fit(X_train)
imputer_cabinCat.fit(X_train)

X_train = imputer_cabinNum.transform(X_train)
X_train = imputer_cabinCat.transform(X_train)


X_test = imputer_cabinNum.transform(X_test)
X_test = imputer_cabinCat.transform(X_test)


df_test = imputer_cabinNum.transform(df_test)
df_test = imputer_cabinCat.transform(df_test)


## rare label encoding
rare_label = ce.RareLabelCategoricalEncoder(tol=0.01,
                                    n_categories=6,
                                    variables=['cabin_cat'])
rare_label.fit(X_train)
X_train = rare_label.transform(X_train)
X_test = rare_label.transform(X_test)
df_test = rare_label.transform(df_test)


## ordinalencoding
ordinal_enc = OrdinalCategoricalEncoder(
    encoding_method='arbitrary',
    variables=['cabin_cat'])
ordinal_enc.fit(X_train)
X_train = ordinal_enc.transform(X_train)
X_test = ordinal_enc.transform(X_test)
df_test = ordinal_enc.transform(df_test)

X_train = X_train.drop('Cabin', axis=1)
X_test = X_test.drop('Cabin', axis=1)
df_test = df_test.drop('Cabin', axis=1)

X_train['Embarked'].unique()
from feature_engine import missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce

ohe_enc = mdi.CategoricalVariableImputer(variables=['Embarked'])

ohe_enc.fit(X_train)
X_train = ohe_enc.transform(X_train)
X_test = ohe_enc.transform(X_test)

# Embarked for validation set
df_test = ohe_enc.transform(df_test)

ordinal_cat = ce.OrdinalCategoricalEncoder(encoding_method='arbitrary',
                                  variables=['Embarked'])
ordinal_cat.fit(X_train)
X_train = ordinal_cat.transform(X_train)
X_test = ordinal_cat.transform(X_test)
df_test = ordinal_cat.transform(df_test)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
mms = StandardScaler()

X_train[['Age', 'Fare', 'cabin_num']] = mms.fit_transform(X_train[['Age', 'Fare', 'cabin_num']])
X_test[['Age', 'Fare', 'cabin_num']] = mms.fit_transform(X_test[['Age', 'Fare', 'cabin_num']])


# Scaling for validation set
df_test[['Age', 'Fare', 'cabin_num']] = mms.fit_transform(df_test[['Age', 'Fare', 'cabin_num']])
# combine the training and testing set to create model
train_combine = pd.concat([X_train, X_test], ignore_index=True)
test_combine = pd.concat([y_train, y_test], ignore_index=True)
combine_train = train_combine.join(test_combine)
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Machine Learning Algorithm (MLA) Selection and Initialization
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
    XGBClassifier()    
    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%


#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = pd.DataFrame()

#index through MLA and save performance to table
row_index = 0
for alg in MLA:
    
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, train_combine, test_combine, cv  = cv_split, return_train_score = True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    
    #save MLA predictions - see section 6 for usage
    alg.fit(train_combine.reindex(sorted(train_combine.columns), axis=1), test_combine)
    MLA_predict[MLA_name] = alg.predict(df_test.reindex(sorted(df_test.columns), axis=1))
    
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict

MLA_predict
# save the predicted score to csv
ids = test_data['PassengerId'].copy()
new_output = ids.to_frame()
new_output["Survived"]= MLA_predict['GradientBoostingClassifier']
new_output.to_csv("TitanicGBC.csv",index=False)
