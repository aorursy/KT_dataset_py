import numpy as np 

import os

import pandas as pd    

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

from numpy import loadtxt

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
%matplotlib inline

%config InlineBackend.figure_format = 'svg'
sns.set_style("whitegrid") 
train= pd.read_csv('../input/titanic/train.csv',encoding = 'utf-8')

test= pd.read_csv('../input/titanic/test.csv',encoding = 'utf-8')
(train.isnull().sum() / len(train)) * 100
train
train['Age'].describe()
train['Age'].mean()
test['Age'].mean()
train['Age'].groupby(train['Pclass']).mean()

#train['Age'].groupby(train['Pclass']).mean()
fill_mean = lambda g: g.fillna(g.mean())

train['Age']=train['Age'].groupby(train['Pclass']).apply(fill_mean)
train['Fare'].groupby(train['Pclass']).mean()
train['Fare']=train['Fare'].groupby(train['Pclass']).apply(fill_mean)
train['Deck']=train['Cabin'].str[0]

test['Deck']=test['Cabin'].str[0]

train['Deck'].fillna('M', inplace=True)

test['Deck'].fillna('M', inplace=True)

train['Deck'].replace(['G','T'],'M', inplace=True)

test['Deck'].replace(['G','T'],'M', inplace=True)

train['Deck'].unique()

test['Deck'].unique()

#train['Deck'].value_counts()

sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=train[train['Deck'].notnull()],

                    kind="count", size=2.5, aspect=.8)
sns.boxplot(x="Deck", y="Fare", data=train).set(

    xlabel='Position', 

    ylabel='Fare')
def convert_dummy(df, feature,rank=0):

    pos = pd.get_dummies(df[feature], prefix=feature)

    mode = df[feature].value_counts().index[rank]

    biggest = feature + '_' + str(mode)

    pos.drop([biggest],axis=1,inplace=True)

    df.drop([feature],axis=1,inplace=True)

    df=df.join(pos)

    return df
train = convert_dummy(train,'Deck')

test = convert_dummy(test,'Deck')
train['Age'].groupby(train['Embarked']).mean()
test['Age'].groupby(test['Pclass']).mean()
test['Age']=test['Age'].groupby(test['Pclass']).apply(fill_mean)

test['Fare']=test['Fare'].groupby(train['Pclass']).apply(fill_mean)
(test.isnull().sum() / len(test)) * 100
len(train)
train.columns
test['Fare'] = np.log1p(test['Fare'])

Y = train['Survived']

X = train.drop(['Survived','Name','PassengerId','Cabin','Ticket'],axis=1)

x_test = test.drop(['Name','PassengerId','Cabin','Ticket'],axis=1)
X.columns
X['Sex']=X['Sex'].replace(['female','male'],[0,1])

x_test['Sex']=x_test['Sex'].replace(['female','male'],[0,1])
X['Pclass'].value_counts() #Passenger's class (1st, 2nd, or 3rd)
X = convert_dummy(X,'Pclass')

x_test = convert_dummy(x_test,'Pclass')
X['Age'].value_counts(bins=6,normalize=True,sort=False) #cut into bins
X['Age'].plot(kind='hist',bins=6,density=True)
X['SibSp'].value_counts() #Number of siblings/spouses aboard the Titanic
X.loc[X['SibSp'] > 0, 'SibSp'] = 1

x_test.loc[x_test['SibSp'] > 0, 'SibSp'] = 1
X['Parch'].value_counts() #Number of parents/children aboard the Titanic

#X['Parch']=X['Parch'].astype(int)
X.loc[X['Parch'] > 0, 'Parch'] = 1

x_test.loc[x_test['Parch'] > 0, 'Parch'] = 1
X['Fare'].describe() #Fare paid for ticket
train['Fare'].plot(kind='hist',bins=30,density=True) 
X['Fare'] = np.log1p(X['Fare'])
X['Fare'].plot(kind='hist',bins=30,density=True) 
X['Fare']=pd.cut(X['Fare'],4,labels=['1','2','3','4'])

x_test['Fare']=pd.cut(x_test['Fare'],4,labels=['1','2','3','4'])

X = convert_dummy(X,'Fare')

x_test = convert_dummy(x_test,'Fare')
X['Embarked'].value_counts() #Where the passenger got on the ship 
X = convert_dummy(X,'Embarked')

x_test = convert_dummy(x_test,'Embarked')
#x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify=Y, test_size=0.3, random_state=0)
param_set = {

 'max_depth':range(10,14),

 'min_child_weight':range(1,3),

 'subsample':[i/10.0 for i in range(5,8)],

 'colsample_bytree':[i/10.0 for i in range(5,8)],

 'reg_alpha':[0.001],

 'gamma':[i/10.0 for i in range(6,7)]

}
model = xgb.XGBClassifier(n_estimators=50)

#gridcv = GridSearchCV(model,param_grid = param_set,cv=10)

#gridcv.fit(X, Y)
#gridcv.best_params_, gridcv.best_score_ 
#bestcv = gridcv.best_estimator_

bestfit = model.fit(X, Y)

test['Survived'] = bestfit.predict(x_test)

predY = bestfit.predict(X)
test
submission = test[['PassengerId','Survived']]

submission
#submission.to_csv('C:/Users/xsong/Desktop/table/submission.csv',index = False)
from xgboost import plot_importance

fig,ax = plt.subplots(figsize=(8,7))

plot_importance(bestfit,

                ax=ax,

                height=0.5,

                max_num_features=64).set(xlabel='feature importance',title='',

ylabel='feature')
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

    plt.xticks(tick_marks, classes)

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
from sklearn.metrics import confusion_matrix



sns.set_style("white") 

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

class_names = ['0','1']

plot_confusion_matrix(confusion_matrix(Y,predY),classes=class_names, normalize=True, 

                      title='Normalized Confusion Matrix: RandomForestClassifier')