import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import os
os.getcwd()
test=pd.read_csv('../input/titanic/test.csv')

train=pd.read_csv('../input/titanic/train.csv')

# train=pd.read_csv('train.csv')

# train=pd.read_csv('test.csv')
train.head()
test.head()
train.info()
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train[train['Sex']=='female']

men = train[train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

sns.boxplot(x = 'Age', data = train, orient = 'v', ax = ax1)

ax1.set_xlabel('People Age', fontsize=15)

ax1.set_ylabel('Age', fontsize=15)

ax1.set_title('Age Distribution', fontsize=15)

ax1.tick_params(labelsize=15)



# sns.distplot(train['Age'], ax = ax2)

# sns.despine(ax = ax2)

# ax2.set_xlabel('Age', fontsize=15)

# ax2.set_ylabel('Occurence', fontsize=15)

# ax2.set_title('Age x Ocucurence', fontsize=15)

# ax2.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout()
FacetGrid = sns.FacetGrid(train, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
data = [train, test]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

train['not_alone'].value_counts()
axes = sns.factorplot('relatives','Survived', 

                      data=train, aspect = 2.5, )

train=train.drop(['PassengerId'],axis=1)

test=test.drop(['PassengerId'],axis=1)
train.head()
test.head()
def Check_duplicate(df):

    duplicate_row=df.duplicated().sum()

    null_values=df.isnull().sum()

    Total_null_values=sum(null_values)

    if(duplicate_row>0):

        print("Please remove duplicates Row=",duplicate_row)

    elif(Total_null_values>0):

            print("Please deal with Missing Values",null_values)

    else:

        print(duplicate_row,"duplicated and null/Missing vlues",Total_null_values," in this dataFrame")
Check_duplicate(train)
Check_duplicate(test)
# Quartiles

def deal_with_outlier(column,df_name,col_name):

    Q1=column.quantile(q = 0.25)

    Q2=column.quantile(q = 0.50)

    Q3=column.quantile(q = 0.75)

    Q4=column.quantile(q = 1.00)

    print('1º Quartile: ', Q1)

    print('2º Quartile: ', Q2)

    print('3º Quartile: ', Q3)

    print('4º Quartile: ', Q4)

    #Calculate the outliers:

    IQR = Q3 - Q1  # Interquartile range, 

    Lower=Q1 - 1.5 * IQR

    Upper=Q3 + 1.5 * IQR

    print("Lower bound",Lower)

    print("Upper bound",Upper)  

    out=column.quantile(q = 0.75) + 1.5*(column.quantile(q = 0.75) - column.quantile(q = 0.25))

    print(' above: ',out , 'are outliers')

   

    

#      show the percentage of outlier for upper

    print('Number of outliers in upper: ', df_name[column > Upper][col_name].count())

    print('Number of clients: ', len(df_name))

#Outliers in %

    print('Outliers are:', round(df_name[column > Upper][col_name].count()*100/len(df_name),2), '%')

    

    #     show the percentage of outlier for lower

    print('Number of outliers in Lower: ', df_name[column > Lower][col_name].count())

    print('Number of clients: ', len(df_name))

#Outliers in %

    print('Outliers are:', round(df_name[column > Lower][col_name].count()*100/len(df_name),2), '%')

    

#     Deal with outlier



    ## Flooring

    df_name.loc[column < (Q1 - 1.5 * IQR),col_name] = column.quantile(0.05)

    ## Capping 

    df_name.loc[column > (Q3 + 1.5 * IQR),col_name] = column.quantile(0.95)

    

    Boxplot=df_name.boxplot(column=[col_name])

    

#     After deal with outlier

    

#     show the percentage of outlier for upper 

    print('Number of outliers in upper Afer deal: ', df_name[column > Upper][col_name].count())

    print('Number of clients: ', len(df_name))

#Outliers in %

    print('Outliers are Afer deal:', round(df_name[column > Upper][col_name].count()*100/len(df_name),2), '%')

    

    #     show the percentage of outlier for lower

    print('Number of outliers in Lower Afer deal: ', df_name[column > Lower][col_name].count())

    print('Number of clients: ', len(df_name))

#Outliers in %

    print('Outliers are Afer deal:', round(df_name[column > Lower][col_name].count()*100/len(df_name),2), '%')

    return Boxplot
train.boxplot(column=['Age'])

plt.title('we have see outlier in AGE Variable')
deal_with_outlier(train['Age'],train,'Age')



# train.boxplot(column=['Age'])

# plt.title('we have see outlier in AGE Variable')
# Calculating some values to evaluete this independent variable

print('MEAN:', round(train['Age'].mean(), 1))

# A low standard deviation indicates that the data points tend to be close to the mean or expected value

# A high standard deviation indicates that the data points are scattered

print('STD :', round(train['Age'].std(), 1))



print('Median',round(train['Age'].median(),1))

# I thing the best way to give a precisly insight abou dispersion is using the CV (coefficient variation) (STD/MEAN)*100

#    cv < 15%, low dispersion

#    cv > 30%, high dispersion

print('CV  :',round(train['Age'].std()*100/train['Age'].mean(), 1), ', High middle dispersion')
test.boxplot(column=['Age'])

plt.title('We have see outlier in AGE Variable(test dataset)')
deal_with_outlier(test['Age'],test,'Age')



# train.boxplot(column=['Age'])

# plt.title('we have see outlier in AGE Variable')
train['Age']=train['Age'].fillna(train['Age'].mean())

test['Age']=test['Age'].fillna(test['Age'].mean())
Check_duplicate(train)
Check_duplicate(test)
train['Cabin'].mode()
# import re

# deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

# data = [train, test]



# for dataset in data:

#     dataset['Cabin'] = train['Cabin'].fillna("U0")

#     dataset['Deck'] = train['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

#     dataset['Deck'] = train['Deck'].map(deck)

#     dataset['Deck'] = train['Deck'].fillna(0)

#     dataset['Deck'] = train['Deck'].astype(int)

# we can now drop the cabin feature

train = train.drop(['Cabin'], axis=1)

test= test.drop(['Cabin'], axis=1)
Check_duplicate(train)
train = train.dropna()
train=train.reset_index(drop=True)
Check_duplicate(train)
Check_duplicate(test)
train.boxplot(column=['Fare'])

plt.title('we have see outlier in Fare Variable')
deal_with_outlier(train['Fare'],train,'Fare')



# train.boxplot(column=['Fare'])

# plt.title('we have see outlier in Fare Variable')
test.boxplot(column=['Fare'])

plt.title('we have see outlier in Fare Variable(test dataset)')
deal_with_outlier(test['Fare'],test,'Fare')



# train.boxplot(column=['Fare'])

# plt.title('we have see outlier in Fare Variable')
train['Fare']=train['Fare'].fillna(train['Fare'].mean())

test['Fare']=test['Fare'].fillna(test['Fare'].mean())
Check_duplicate(train)
Check_duplicate(test)
train.info()
train.head()
Check_duplicate(test)  # now age have no duplicte and no outlier
data = [train, test]



for dataset in data:

#     dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
train.info()
test.info()
data = [train, test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train= train.drop(['Name'], axis=1)

test= test.drop(['Name'], axis=1)
genders = {"male": 0, "female": 1}

data = [train, test]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)

train['Ticket'].describe()
train= train.drop(['Ticket'], axis=1)

test= test.drop(['Ticket'], axis=1)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train, test]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



# let's see how it's distributed train_df['Age'].value_counts()
train.head()
test.head()
y=train.Survived
y.head()
X=train.drop(['Survived'], axis=1)
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Train Dataset',X_train.shape,y_train.shape)

print('Test Dataset',X_test.shape,y_test.shape)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix

import itertools

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train_f = sc_X.fit_transform(X_train)

X_test_f = sc_X.transform(X_test)

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
logmodel = LogisticRegression() 

logmodel.fit(X_train,y_train)

logpred = logmodel.predict(X_test)



# predict our train dataset



cnf_matrix=confusion_matrix(y_test, logpred)

print("Confusion Matrix on Train Dataset:")

print(confusion_matrix(y_test, logpred))

print(round(accuracy_score(y_test, logpred),2)*100)



# Stratified cross validation

LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

print('strtified cross validation accuracy',LOGCV)
from sklearn.metrics import classification_report, confusion_matrix

import itertools

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

print(confusion_matrix(y_test, logpred))

print(round(accuracy_score(y_test, logpred),2)*100)
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['YES=1','NO=0'],normalize= False,  title='Confusion matrix')
logpred = logmodel.predict(test)



# confusion matrix

# cnf_matrix=confusion_matrix(y_train, logpred)

# print(confusion_matrix(y_train, logpred))

# print(round(accuracy_score(y_train, logpred),2)*100)

!pip install pydotplus
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pydotplus

from sklearn import tree, metrics, model_selection, preprocessing

from IPython.display import Image, display

from sklearn.tree import export_graphviz
# train the decision tree

dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

dtree.fit(X_train,y_train)
# use the model to make predictions with the test data

y_pred_test = dtree.predict(X_test)

y_pred_test
# how did our model perform?

# count_misclassified =(y_test!= y_pred).sum()

# print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred_test)

print('Accuracy: {:.2f}'.format(accuracy))
# use the model to make predictions with the test data

y_pred = dtree.predict(test)

y_pred
# from sklearn.cross_validation import KFold



# cv = KFold(n=len(bank_final),  # Number of elements

#            n_folds=10,            # Desired number of cv folds

#            random_state=12) 

cv = KFold(n_splits=12, shuffle=True, random_state=0)
fold_accuracy = []



# titanic_train["Sex"] = encoded_sex



for train_fold, valid_fold in cv.split(X):

    train = X.loc[train_fold] # Extract train data with cv indices

    valid = X.loc[valid_fold] # Extract valid data with cv indices

    

    train_y = y.loc[train_fold]

    valid_y = y.loc[valid_fold]

    

    model = dtree.fit(X = train, 

                           y = train_y)

    valid_acc = model.score(X = valid, 

                            y = valid_y)

    fold_accuracy.append(valid_acc)    



print("Accuracy per fold: ", fold_accuracy, "\n")

print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))
import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

# from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor



# we can achieve the above two tasks using the following codes

# Bagging: using all features

rfc1 = RandomForestClassifier(max_features=8, random_state=1)

rfc1.fit(X_train, y_train)

pred1 = rfc1.predict(X_test)

print(roc_auc_score(y_test, pred1))



# play around with the setting for max_features

rfc2 = RandomForestClassifier(max_features=6, random_state=1)

rfc2.fit(X_train, y_train)

pred2 = rfc2.predict(X_test)

print(roc_auc_score(y_test, pred2))









# applyin on test dataset------------------------------------------



# we can achieve the above two tasks using the following codes

# Bagging: using all features







# rfc1 = RandomForestClassifier(max_features=8, random_state=1)

# rfc1.fit(X_train, y_train)

pred_x = rfc1.predict(X_test)

print(roc_auc_score(y_test, pred_x))



# # play around with the setting for max_features

# rfc2 = RandomForestClassifier(max_features=6, random_state=1)

# rfc2.fit(X_train, y_train)

# pred2 = rfc2.predict(X_train)

# print(roc_auc_score(y_train, pred2))

y_pred_final = dtree.predict(test)
submission=pd.read_csv('../input/titanic/test.csv')
submission['Survived']=y_pred_final
submission= submission.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], axis=1)
submission.head()
submission.to_csv('submission',index=False)