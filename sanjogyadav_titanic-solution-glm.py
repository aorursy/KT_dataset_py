# data processing and analysis

import pandas as pd 

print("pandas version: {}". format(pd.__version__))



# scientific computing

import numpy as np 

print("NumPy version: {}". format(np.__version__))



# scientific and publication-ready visualization

import matplotlib 

print("matplotlib version: {}". format(matplotlib.__version__))



# scientific and publication-ready visualization 

import seaborn as sns

print("seaborn version: {}". format(sns.__version__))



# machine learning algorithms

import sklearn 

print("scikit-learn version: {}". format(sklearn.__version__))



# machine learning algorithms

import statsmodels

print("statsmodels version: {}". format(statsmodels.__version__))



# scientific computing and advance mathematics

import scipy as sp 

print("SciPy version: {}". format(sp.__version__))
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


# import train data from file

data = pd.read_csv('../input/titanic/train.csv')



# a dataset should be broken into 3 splits: train, test, and (final) validation

# we will split the train set into train and test data in future sections

data_val  = pd.read_csv('../input/titanic/test.csv')



# to play with our data, create copy

data1 = data.copy(deep = True)



# however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data1, data_val]
data1.head()
data_val.head()
data1.shape
data_val.shape
data1.info()
data1.describe()
print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())
# Data description

data.describe(include = 'all')
for dataset in data_cleaner:    

    # age: median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    # embarked: mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    # fare: median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

    # drop Cabin as it has 687 as null out of 891 (approx 77% of data)

    dataset.drop('Cabin', axis=1, inplace=True)
print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())


# List of variables to map



varlist =  ['Sex']



# Defining the map function

def binary_map(x):

    return x.map({'male': 1, "female": 0})



# Applying the function to the housing list

for dataset in data_cleaner:

    dataset[varlist] = dataset[varlist].apply(binary_map)
data1.head()
data_val.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data1['Embarked'], prefix='Embarked', drop_first=True)

    

# Adding the results to the master dataframe

data1 = pd.concat([data1, dummy1], axis=1)
data1.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data_val['Embarked'], prefix='Embarked', drop_first=True)

    

# Adding the results to the master dataframe

data_val = pd.concat([data_val, dummy1], axis=1)
data_val.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data1['Pclass'], prefix='Pclass', drop_first=True)

    

# Adding the results to the master dataframe

data1 = pd.concat([data1, dummy1], axis=1)
data1.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data_val['Pclass'], prefix='Pclass', drop_first=True)

    

# Adding the results to the master dataframe

data_val = pd.concat([data_val, dummy1], axis=1)
data_val.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data1['Sex'], prefix='Male', drop_first=True)

    

# Adding the results to the master dataframe

data1 = pd.concat([data1, dummy1], axis=1)
data1.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data_val['Sex'], prefix='Male', drop_first=True)

    

# Adding the results to the master dataframe

data_val = pd.concat([data_val, dummy1], axis=1)
data_val.head()
data1['FamilySize'] = data1['SibSp'] + data1['Parch'] + 1

data1.head(2)
data_val['FamilySize'] = data_val['SibSp'] + data_val['Parch'] + 1

data_val.head(2)
# Renaming the column 

data1= data1.rename(columns={ 'Male_1' : 'Male'})

data_val= data_val.rename(columns={ 'Male_1' : 'Male'})
drop_column = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Fare', 'Embarked']

data1.drop(drop_column, axis=1, inplace = True)
data1.head(2)
data_val.head(2)
# Checking for outliers in the continuous variables

cont_col = data1['Age']



# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

cont_col.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Let's see the correlation matrix 

plt.figure(figsize = (10,5))        # Size of the figure

sns.heatmap(data1.corr(),annot = True)

plt.show()
data1.head(2)
# Plot the scatter plot of the data



sns.pairplot(data1, x_vars=['Age'], y_vars='Survived',size=4, aspect=1, kind='scatter')

plt.show()
#graph distribution of quantitative data

plt.figure(figsize=[4,5])



#plt.subplot(231)

plt.boxplot(x=data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age')
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Age']==0]['Age']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age ')

plt.ylabel('# of Passengers')

plt.legend()
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['SibSp'], data1[data1['Survived']==0]['SibSp']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('SibSp Histogram by Survival')

plt.xlabel('SibSp')

plt.ylabel('# of Passengers')

plt.legend()
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['Parch'], data1[data1['Survived']==0]['Parch']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Parch Histogram by Survival')

plt.xlabel('Parch')

plt.ylabel('# of Passengers')

plt.legend()
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['Male'], data1[data1['Survived']==0]['Male']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Sex Histogram by Survival')

plt.xlabel('Sex')

plt.ylabel('# of Passengers')

plt.legend()
# Plot the heatmap of the data to show the correlation



sns.heatmap(data1.corr(), cmap="YlGnBu", annot = True)

plt.show()
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = data1.drop(['Survived'], axis=1)



X.head()
# Putting response variable to y

y = data1['Survived']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
### Checking the Survival Rate

survived = (sum(data1['Survived'])/len(data1['Survived'].index))*100

survived
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train[col])

logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm1.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})

y_train_pred_final['PassengerId'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop(['Parch','SibSp','FamilySize', 'Embarked_S', 'Embarked_Q'], 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})

y_train_pred_final['PassengerId'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survived_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.56 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )

confusion2
X_test = X_test[col]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)



# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)



# Putting CustID to index

y_test_df['PassengerId'] = y_test_df.index



# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)



# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Survived_Prob'})



# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.47 else 0)

y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)
data_val_Id = data_val['PassengerId']

data_val = data_val[col]

data_val.head()
data_val_sm = sm.add_constant(data_val)



y_val_pred = res.predict(data_val_sm)

y_val_pred[:10]
# Converting y_val to a dataframe which is an array

y_val_1 = pd.DataFrame(y_val_pred)



# Let's see the head

y_val_1.head()
# Renaming the column 

y_val_1= y_val_1.rename(columns={ 0 : 'Survived_Prob'})



# Putting CustID to index

y_val_1['PassengerId'] = data_val_Id



y_val_1['final_predicted'] = y_val_1.Survived_Prob.map(lambda x: 1 if x > 0.57 else 0)

y_val_1.head()
output = pd.DataFrame({'PassengerId': y_val_1.PassengerId, 'Survived': y_val_1.final_predicted})

output.to_csv('my_submission_GLM.csv', index=False)

print("Your submission was successfully saved!")