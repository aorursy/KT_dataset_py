import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

from sklearn.preprocessing import normalize

from sklearn.pipeline import Pipeline

#ignore warnings

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(6)
train.info()
for n in train.columns:

    print('The number of different values in ', n, 'are:', len(train[n].unique()))

print('-------------------------------------')

#I will search numbers of Nan values

for n in train.columns:

    if train[n].isnull().values.any() == True:

        print('There is' , train[n].isna().sum(), 'null values in', n, 'column')
for n in test.columns:

    print('The number of different values in ', n, 'are:', len(test[n].unique()))

print('-------------------------------------')

#I will search numbers of Nan values

for n in test.columns:

    if test[n].isnull().values.any() == True:

        print('There is' , test[n].isna().sum(), 'null values in', n, 'column')
train = train.drop(['Name','Ticket','PassengerId'], axis = 1)

test = test.drop(['Name','Ticket','PassengerId'], axis = 1)
#First, we will create a copy:

train_bf = train.copy()

test_bf = test.copy()
print("The mean Age value for the train DataFrame is:",train['Age'].mean(),"\nThe Standard Deviation for the Age is:",train['Age'].std())

print("The mean Age value for the test DataFrame is:",test['Age'].mean(),"\nThe Standard Deviation for the Age is:",test['Age'].std())
#Creating a function that fill NaN values of train and test

def fill_nan_w_mean_std(df,col='Age'):

    nan = df[df[col].isna()]

    min = df[col].mean() - df[col].std()

    max = df[col].mean() + df[col].std()

    for i in nan.index:

        random_num = random.uniform(min,max)

        df[col].loc[i] = random_num
fill_nan_w_mean_std(train)

fill_nan_w_mean_std(test)
#We will transform the values to integers

train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)
plt.style.use('ggplot')

fig = plt.figure(figsize = (25,10))

fig.subplots_adjust(hspace=0.6, wspace=0.15)

ax = fig.add_subplot(1,2,1)

ax.set_ylim([0,0.04])

ax.set_title('Before filling NaN values')

sns.distplot(train_bf['Age'].dropna(), bins=20)

ax_2 = fig.add_subplot(1,2,2)

ax_2.set_ylim([0,0.04])

ax_2.set_title('Data with filled values')

sns.distplot(train['Age'],bins=20)
#First, we have to visualize the Embarked column in train data: 

sns.countplot(x='Embarked',data=train)
#Now we see that most of the data embarked from S port we can fill with this class

nan_emb = train[train['Embarked'].isna()]

for i in nan_emb.index:

    train['Embarked'].loc[i] = 'S'
#We deal with that one nan value in fare column of test data

nan_fare = test[test['Fare'].isna()]

for i in nan_fare.index:

    test['Fare'].loc[i] = test['Fare'].mean()
train['Family_members'] = train['SibSp'] + train['Parch'] + 1 #him/her

test['Family_members'] = test['SibSp'] + test['Parch'] + 1



#Let's create the Alone feature

train['not_Alone'] = 0 #0 if he/she is alone

train['not_Alone'].loc[train['Family_members'] > 1] = 1



test['not_Alone'] = 0 #1 if he/she is alone

test['not_Alone'].loc[train['Family_members'] > 1] = 1
train.head()
#We will create a third string in gender: child. Because in a catastrophe like this children are first.

def agg_child(df,col='Age'):

    mask = (df[col] <= 15)

    df.loc[mask,'Sex'] = 'child'



#We apply the function

agg_child(train)

agg_child(test)
def agg_cabin_bin(df,col='Cabin'):

    df['with_Cabin'] = 0

    mask = (df[col].isna() == False)

    df.loc[mask,'with_Cabin'] = 1

#0 if the passenger had not a Cabin, 1 if he has a cabin
agg_cabin_bin(train)

agg_cabin_bin(test)



del train['Cabin']

del test['Cabin']
train.head()
sns.barplot(x='Sex',y='Survived',data=train, order=['female','child','male'])

plt.title('Sex vs Survived')
gs = plt.GridSpec(2,3,wspace=0.45, hspace=0.8)

plt.figure(figsize=(12,10))

ax1 = plt.subplot2grid((3,3),(0,0),rowspan=2,colspan=2)

plt.title('Age vs Survived vs Sex')

sns.swarmplot(x = 'Survived',y='Age', 

              data=train, linewidth=1,hue='Sex', palette = 'muted')

ax2 = plt.subplot2grid((3,3),(0,2))

plt.title('Embarked vs Survived')

sns.barplot(x='Embarked',y='Survived',

            data=train,order=['C','Q','S'])

ax3 = plt.subplot2grid((3,3),(1,2))

plt.title('Pclass vs Survived')

sns.barplot(x='Pclass',y='Survived',

            data=train, palette = 'muted')
#Now we can say that Embarked class/Pclass/sex have priorities for survive. So we can transform the categorical data to numbers

cat_to_nums = {"Embarked":  {"S": 0, "Q": 1, "C":2},

               "Sex": {"male":0,"child":1,"female":2}}

#We will use replace to convert the values

train.replace(cat_to_nums, inplace = True)
test.replace(cat_to_nums, inplace = True)
train.head()
train.corr()
def correlation_heatmap(df): #from "A Data Science Framework: To Achieve 99% Accuracy" kernel by LD Freedman

    s , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    g = sns.heatmap(df.corr(), cmap = colormap,square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax, annot=True,linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 })

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
correlation_heatmap(train)
# Normalize 'Fare' & 'Age' values for test and train dataset

X = [train['Fare'],

     train['Age']]

X_normalize = normalize(X)



X_2 = [test['Fare'],

      test['Age']]

X_2_normalize = normalize(X_2)



train = train.assign(Fare = X_normalize[0])



train = train.assign(Age = X_normalize[1])



test = test.assign(Fare = X_2_normalize[0])



test = test.assign(Age = X_2_normalize[1])
#Now we have all numerical values!

train.head()
X_train = train.loc[:,'Pclass':]

y_train = train.loc[:,'Survived']

X_test_final = test
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV #We will use gridsearchCV



#Dividing the data before tuning the model

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size = 0.3, 

                                                    random_state=21)
#Logistic Regression:



logreg = LogisticRegression()



param_grid = {'C' : [x for x in range(1,5000,5)]  }

#finding the best parameter:

searcher = GridSearchCV(logreg, param_grid)



searcher.fit(X_train,y_train)
# Report the best parameters and the corresponding score

print("Best CV params", searcher.best_params_)

print("Best CV accuracy", searcher.best_score_)



# Report the test accuracy using these best parameters

print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
pred = searcher.predict(X_test_final)
from sklearn.svm import SVC

svc = SVC()

parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1,1,10,100],

              'C':[x for x in np.linspace(0.1,10,100)]}

ssp = GridSearchCV(svc,parameters)



ssp.fit(X_train,y_train)
# Report the best parameters and the corresponding score

print("Best CV params", ssp.best_params_)

print("Best CV accuracy", ssp.best_score_)



# Report the test accuracy using these best parameters

print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
pred23 = ssp.predict(X_test)

len(y_test)
y_test
from sklearn.linear_model import SGDClassifier
# We set random_state=0 for reproducibility 

linear_classifier = SGDClassifier(random_state=0)



# Instantiate the GridSearchCV object and run the search

parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 

             'loss':['hinge', 'log'], 'penalty':['l1','l2']}

searcher_sgd = GridSearchCV(linear_classifier, parameters, cv=10)

searcher_sgd.fit(X_train, y_train)



# Report the best parameters and the corresponding score

print("Best CV params", searcher.best_params_)

print("Best CV accuracy", searcher.best_score_)

print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier, export_graphviz #importing the module

from sklearn.metrics import accuracy_score
#Finding our optimize max_depth

for i in range(1,18):

    tree_clf = DecisionTreeClassifier(max_depth=i)

    tree_clf.fit(X_train,y_train)

    y_pred = tree_clf.predict(X_test)

    print("Test accuracy of the Decision Trees model:", accuracy_score(y_pred,y_test), "for max_depth: ", i)
#Basic Decision Tree model

tree_clf = DecisionTreeClassifier(max_depth = 2)

tree_clf.fit(X_train,y_train) #training the model



#Predicting with the model

y_pred = tree_clf.predict(X_test)



#We will try to evaluate the accuracy score

print("Test accuracy of the Decision Trees model:", accuracy_score(y_pred,y_test))
pred = tree_clf.predict(X_test_final)
from sklearn.ensemble import RandomForestClassifier
#Random Forest Classifier model

rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1)

rnd_clf.fit(X_train,y_train)



#Predicting

y_pred = rnd_clf.predict(X_test)



#Evaluate the accuracy score

print("Test accuracy of the Decision Trees model:", accuracy_score(y_pred,y_test))
pred = rnd_clf.predict(X_test_final)
from sklearn.ensemble import VotingClassifier
#We will now try an ensemble method

voting_clf = VotingClassifier(estimators = [('lr',searcher),('rf',rnd_clf),('svc',ssp),

                                            ('sgd',searcher_sgd),('dt',tree_clf)],

                             voting = 'hard')

voting_clf.fit(X_train,y_train)
y_pred = voting_clf.predict(X_test)
#Evaluate the accuracy score

print("Test accuracy of the Decision Trees model:", accuracy_score(y_pred,y_test))

pred = voting_clf.predict(X_test_final)
from sklearn.ensemble import AdaBoostClassifier
adb_clf = AdaBoostClassifier(base_estimator = tree_clf,

                             n_estimators = 100)

adb_clf.fit(X_train,y_train)

y_pred = adb_clf.predict(X_test_final)

y_pred
from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical
y_train2 = to_categorical(y_train)
n_cols = X_train.shape[1]

model = Sequential()

model.add(Dense(100,activation='relu',input_shape = (n_cols,)))

model.add(Dense(100,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(optimizer='sgd', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
model.fit(X_train, y_train2, epochs = 100)
preds = model.predict(X_test)