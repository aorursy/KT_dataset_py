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
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.columns
df_train.info()
# let's split are variables between categorical and numerical ones, let's not include
# variables we are not considering using ('PassengerId', 'Cabin')
# two variables need to be reprocessed Name & Ticket if we want to use them
var_cat = [df_train['Survived'], df_train['Pclass'], df_train['Sex'], df_train['Embarked']]
var_num_d = [df_train['SibSp'], df_train['Parch']]
var_num_c = [df_train['Age'], df_train['Fare']]
# let's explore graphically the varables
# categorical variables
for var in var_cat:
    sns.barplot(x=var.value_counts().index, y=var.value_counts().values)
    plt.title(var.name)
    plt.show()
# We see that a majority of passengers were 3rd class (lowest class) and male
# at first sight Pclass, Sex seems to have played a natural role on the surviving odds. for Embarked
# it is not clear at all
# continous numerical variables
for var in var_num_c:
    sns.distplot(var)
    plt.title(var.name)
    plt.show()
    
# Age is pretty normally distributed
# Fare is skewed we may want to normalize it 
# discrete numerical variables
for var in var_num_d:
    sns.barplot(x=var.value_counts().index, y=var.value_counts().values, palette='BuPu_r')
    plt.title(var.name)
    plt.show()
# skewed 
# also the range is very different from the continous variables we may want to do some feature scalling
# on the numerical variables 
# let's engeener 'name' & ticket variables so we can maybe extract some value from those features
# Name:
# By exploring the data, we can see different titles in the names Mr. , Mrs., Miss., Master., Rev.
# let's extract those title and create a new column 'title'
title_ls = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Rev.', 'Dr.', ' ']

def get_title(name):
    for t in title_ls:
        if t in name:
            return t
        else:
            continue

df_train['title'] = df_train['Name'].apply(lambda x: get_title(x))
# Now let's explore the Names to which we assigned a ' ' value to see if we didn't missed any
# valuable title
df_train.loc[df_train['title']== ' ', 'Name']
# We did miss Dr. (frequent enough), the other titles are note frequent enough to be valuable 
# lets modify back are function above to add Dr

# let's assign 'Other' to the last rows where we assigned a ' ' title 
df_train.loc[df_train['title']== ' ', 'title'] = 'Other'
# Tickets
df_train['Ticket']
# No interesting pattern seems to come up, let's just don't use ticket variable in our model for now
# let's deal with missing values:
# Age : a reasonable amount of nan : let's find a way to replace them
# Cabin : too much missing values : let's for the moment not bother using this variable
# Embarked : Only two missing values let's just drop the rows containing those missing values
# Nan Embarked
# drop the rows of the df_train which has a Nan in Embarked
df_train = df_train[~df_train['Embarked'].isnull()]
# Nan Age
# let's replace the missing Age of the passengers, by the average age of the passengers sharing the 
# same title -> it should provide a sufficiently robust estimation

# we need to be able to get the average age of each title classes

title_ls  = df_train.groupby('title').mean()['Age'].index
age_ls  = df_train.groupby('title').mean()['Age'].values

def get_age(title):
    for t, a in zip(title_ls, age_ls):
        if title == t:
            return a 

# select from df_train the rows where age is missing & replace their value using get_age function
df_train.loc[df_train['Age'].isnull(), 'Age'] = df_train.loc[df_train['Age'].isnull(), 'title'].apply(lambda x: get_age(x))

# A baseline of features to consider seems to have now emerged :
# Pclass, title, Sex** (maybe title, provides a good replacement or Sex, since titles are exclusive
# to often one sex), SibSp, Parch, Fare.
# Some interesting features might be later explored : Embarked, Cabin, Ticket 

# Since I am considering building a baseline model using a simple logisitc regression , I don't see
# the use for now to normalize my numerical features, as well as to scale them.

# Lets Build the baseline algo using all the features considered so we later:
# -> understand better feature importance & do some feature selection
# -> see if including some higher polynomials can help
# -> see if we are slightly overfitting / underfitting
# -> see if a non linear model is required 
# first separate the predictive variable ('Survival')(y_train) from the predictors (X_train)

# prepare train variables 
y_train = df_train['Survived']
X_train = df_train.loc[:,['Pclass', 'title', 'Sex', 'SibSp', 'Parch', 'Fare']]
X_train

# prepare test variables

# get titles for test set
df_test['title'] = df_test['Name'].apply(lambda x: get_title(x))
# let's assign 'Other' to the last rows where we assigned a ' ' title 
df_test.loc[df_test['title']== ' ', 'title'] = 'Other'

X_test = df_test.loc[:,['Pclass', 'title', 'Sex', 'SibSp', 'Parch', 'Fare']]

# change dtype of categorical variables which are ints as objects so pd.get_dummies work
X_test['Pclass'] = X_test['Pclass'].astype(object)
X_train['Pclass'] = X_train['Pclass'].astype(object)
# get dummies
X_test = pd.get_dummies(X_test)
X_train = pd.get_dummies(X_train)
# replace the unique nan value in X_test for "Fare" by the average Fare of the train set
X_test.loc[X_test['Fare'].isnull(), 'Fare'] = X_train['Fare'].mean()
# Note that with sklearn Logistic regression is implemented with regularization by default
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_train)

train_accuracy = clf.score(X_train, y_train)
train_accuracy
from sklearn.metrics import plot_confusion_matrix, classification_report

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, X_train, y_train,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
# classification report 
report = classification_report(y_train, y_pred)
print(report)

print(X_train.columns)
#X_test.insert(2, "Age", [21, 23, 24, 21], True)
X_test.insert(11,'title_Other', 0, True) 
print(X_test.columns)
# let's analyse the accuracy score on the test set
predictions = clf.predict(X_test)
# let's submit this baseline result so we can improve on it 
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('baseline_submission.csv', index=False)
print("Your submission was successfully saved!")
# Conclusion of Baseline : 

# training_accuracy = 0.83
# test_accuracy = 0.77 
# the gap between train & test don't seem to large, however the training accuracy is still pretty low
# so i am guessing that our algorithm suffers a bit from underfitting.

# Now the baseline submission has been made, let's try to figure it out where
# the algo struggled to classify correctly the data.

# here are the 152 rows where the classifier got things wrong
df_base_w = df_train.loc[y_train != y_pred]
# Based on the confusion matrix above the baseline classifier fails the most predicting that a person 
# died while it survived -> we should focus on this

# it seems that the alogrithm don't class correctly a few female passengers that died
# maybe if the algorithm gives more weight to the person's class could he get more things right
# Analysis of the false negatives :
# on all those rows the algorithm predicted 0, and was wrong
df_false_neg = df_base_w.loc[df_base_w['Survived']==1]
# The majority of the false negatives is about "male" passengers ! 
df_false_neg['Sex'].value_counts()
df_false_neg[df_false_neg['Sex']== 'male']['Pclass'].value_counts() # Pclass isn't of any help explaining
# the survival of those man 
avg_f_male_fn = df_false_neg[df_false_neg['Sex']== 'male']['Fare'].mean()
avg_f_male = df_train[df_train['Sex'] == 'male']['Fare'].mean()
print('Comparisson of avg fare: ', 'avg fare male fn: ' + str(avg_f_male_fn),
      'avg fare male: ' + str(avg_f_male), sep='\n')
# average fare was higher for man who where missclassified as non surviving # Fare might need some more weight
print('avg age man fn:',df_false_neg[df_false_neg['Sex']== 'male']['Age'].mean())
print('avg age man:', df_train[df_train['Sex'] == 'male']['Age'].mean()) # just a bit older but not so
# significant

# for false negatives paying attention to 'Fare' seems like a good solution
# Analysis of false positives
df_fp = df_base_w.loc[df_base_w['Survived']==0]
# the majority of fp concerns female passengers : 
df_fp['Sex'].value_counts()

# let's take a closer look ...
# it seems that misses were twicely more classfied as survying while they didn't. 
df_fp[df_fp['Sex'] == 'female']['title'].value_counts() # maybe title here should be took more in account

df_fp[df_fp['Sex'] =='female']['Pclass'].value_counts() # OKay ! Pclass plays a huge role here, 
# the xtreme majority of female that were classified as surviving but actually died were in 3rd class.
# we should pay more attention to 'Pclass'

# maybe they payed a less greater fare, such as man who survied seem to have payed a higher fare
# less check
avgf_femfp = df_fp[df_fp['Sex'] =='female']['Fare'].mean()
avgf_fem = df_train[df_train['Sex'] == 'female']['Fare'].mean()
print('avg fare of female passengers that were fp:', avgf_femfp)
print('avg fare of female passengers', avgf_fem)
# WOOOOW again huge difference !!!!! female that unfortunely died and were not classified correctly as 
# so did pay far less fare than global female passengers !!! 
# We definitely need to pay attention to 'Fare' & Pclass in our next model ! 
# The goal is to build a new model which : 
# tries to underfit less the data (maybe more features)
# so we might wanna try a polynomial logistic regression

# We also wan't to help our algorithm 

# but first let's take a look to outliers & see if removing them can significantly improve our model

# Fare
sns.lmplot(x='PassengerId', y='Fare',data=df_train, fit_reg=False, hue='Survived', legend=True)
plt.show()
# We have seen that Fare plays a great role in survival so we might don't wanna delete those three outliers,
# hmm.
# Age
sns.lmplot(x='PassengerId', y='Age',data=df_train, fit_reg=False, hue='Survived', legend=True)
plt.show()
sns.boxplot(x='Age',y='Sex', data=df_train) # outliers don't seem to worrysome
# According to our false negatives / positives analysis. The algorithm classified badly as survying women
# that paid a low fare. We might wanna check outliers for multi-variable 'Sex'='Female' & Fare. we might 
# want to remove women surviving while paying a very low fare.
df_train_female = df_train[df_train['Sex'] == 'female']

sns.set(style="whitegrid")
g = sns.lmplot(x='PassengerId', y='Fare', hue='Survived', data=df_train_female, fit_reg=False)
g.set(ylim=(0, 10))
plt.plot()
# don't seem much outlier to clean here hmm...
# try to find another type of outlier women based on Pclass
g = sns.lmplot(x='Fare', y='Pclass', data=df_train_female, hue='Survived', fit_reg=False)
g.set(xlim=(0, 10))
g.set(ylim=(2.9, 3.1))
plt.plot()
# here we see that a few women survied while being in 3rd class and paying a low Fare -> I might want to
# consider them as outlier but I am not sure for now. 
# we were caring about the the fp (women dying but not classfied as so), let's focus on the fn (men 
# classfied as dying but who actually survied) in fact our algorithm did way more fn.
# we found at that does man missclasifed as dying tend to have paid higher fares on average.
# to help our model we might want to get rid of outliers where man paying very high fares ended dying. 
# let's see if we can find any?
df_train_men = df_train[df_train['Sex'] == 'male']

sns.set(style="whitegrid")
g = sns.lmplot(x='PassengerId', y='Fare', hue='Survived', data=df_train_men, fit_reg=False)
g.set(ylim=(0, None))
plt.plot()
# hmm interesting we see a bunch of 6 outliers where men payed a fare between 200 & 300 but ended up dying 
# we might wanna remove these ! On the other hand we don't want to remove for the moment the top two orange
# outlier since we believe that paying higher fare tend to increase a lot men's survival chances
# After analysis I didn't found very clear outliers for female passengers.
# However for male passengers, it seems to be 6 clear outliers of male paying a havy fare but still dying.
# let's try removing them and see if it improves our model

# these male passenger outlier have payed a fare between 200 & 300 while dying
df_train2 = df_train.copy()
male_outliers = df_train2[(df_train['Sex'] == 'male') &  (df_train2['Fare'] > 200) & (df_train2['Fare'] < 300) ]
male_outliers
# filter the df so I remove the six outliers
df_train2 = df_train2[~df_train2.index.isin(male_outliers.index)]
df_train2
# extract X_train


features = ['Pclass', 'title', 'Sex', 'SibSp', 'Parch', 'Fare']
X_train2 = df_train2[features]
X_train2['Pclass'] = X_train2['Pclass'].astype(object)
X_train2 = pd.get_dummies(X_train2)

# get y_test
y_train2 = df_train2['Survived']
# we still use the same X_test

clf2 = LogisticRegression(solver='liblinear', random_state=0).fit(X_train2, y_train2)

y_pred = clf.predict(X_train2)

train_accuracy = clf.score(X_train2, y_train2)
train_accuracy
# the training accuracy has improved of 0.002 (0.2%) 
# let's submit this model to see if there's significative improvement on the test set 
# let's analyse the accuracy score on the test set
predictions = clf2.predict(X_test)
# let's submit this baseline result so we can improve on it 
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('logreg_outlier1_sub.csv', index=False)
print("Your submission was successfully saved!")
# the performance on the test set is actually WORSE ;( ! 
# let's analyse if the type of error made on the training set did improved
from sklearn.metrics import plot_confusion_matrix, classification_report

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf2, X_train2, y_train2,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
# ok seems that the change was so small it barely affected the algorithm ...
# let's try to hunt more outliers and transform more the data set ! -> we are also going to revert that change since it pretty much didn't improve our model.
features = ['Pclass', 'title', 'SibSp', 'Parch', 'Fare']
X_train3 = df_train[features]
X_train3['Pclass'] = X_train3['Pclass'].astype(object)
X_train3 = pd.get_dummies(X_train3)

# get y_test
y_train3 = df_train['Survived']
#X_train3.columns

X_test3 = X_test.drop(columns=['Sex_female', 'Sex_male'], axis=1)
# let's build a model without the sex variable since 'title' provides the same kind of information more precisely
clf3 = LogisticRegression(solver='liblinear', random_state=0).fit(X_train3, y_train3)

y_pred = clf3.predict(X_train3)

train_accuracy = clf3.score(X_train3, y_train3)
train_accuracy
# no sign of improvement but let's try & submit it to see on the test set
# let's submit this model to see if there's significative improvement on the test set 
# let's analyse the accuracy score on the test set
predictions = clf3.predict(X_test3)
# let's submit this baseline result so we can improve on it 
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('logreg_no_sex_sub.csv', index=False)
print("Your submission was successfully saved!")
# the performance on the test set is actually WORSE ;( ! 
# There's a slight improvement on the test set !  let's keep this change.