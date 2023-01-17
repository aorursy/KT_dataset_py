%matplotlib inline

%config InlineBackend.figure_format = 'svg'



import warnings

warnings.filterwarnings('ignore')

import os

import numpy as np

import pandas as pd    

import matplotlib.pyplot as plt

from numpy import loadtxt

import xgboost as xgb

from lightgbm import plot_metric

    

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

from sklearn import metrics

import itertools



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



fill_mean = lambda x: x.fillna(x.mean())



def convert_dummy(df, feature, rank=0):

    '''Xgboost doesn't have method to process categorical features

    we should use dummy-encode'''

    pos = pd.get_dummies(df[feature], prefix=feature)

    mode = df[feature].value_counts().index[rank]

    biggest = feature + '_' + str(mode)

    pos.drop([biggest],axis=1,inplace=True)

    df.drop([feature],axis=1,inplace=True)

    df = df.join(pos)

    return df
train['Age'] = train['Age'].groupby(train['Pclass']).apply(fill_mean)

test['Age'] = test['Age'].groupby(test['Pclass']).apply(fill_mean)



train['Fare'] = train['Fare'].groupby(train['Pclass']).apply(fill_mean)

test['Fare'] = test['Fare'].groupby(test['Pclass']).apply(fill_mean)



train['Fare'] = np.log1p(train['Fare'])

test['Fare'] = np.log1p(test['Fare'])



train['Deck']=train['Cabin'].str[0]

test['Deck']=test['Cabin'].str[0]

train['Deck'].fillna('M', inplace=True)

test['Deck'].fillna('M', inplace=True)

train['Deck'].replace(['G','T'],'M', inplace=True)

test['Deck'].replace(['G','T'],'M', inplace=True)

train['Deck'].unique()

test['Deck'].unique()



train = convert_dummy(train,'Deck')

test = convert_dummy(test,'Deck')



train['Sex'] = train['Sex'].replace(['female','male'],[0,1])

test['Sex'] = test['Sex'].replace(['female','male'],[0,1])



train = convert_dummy(train,'Pclass')

test = convert_dummy(test,'Pclass')



train.loc[train['SibSp'] > 0, 'SibSp'] = 1

test.loc[test['SibSp'] > 0, 'SibSp'] = 1



train.loc[train['Parch'] > 0, 'Parch'] = 1

test.loc[test['Parch'] > 0, 'Parch'] = 1



train = convert_dummy(train,'Embarked')

test = convert_dummy(test,'Embarked')



y = train['Survived']

features = ['Pclass_1', 'Pclass_2','Sex', 'Age', 'SibSp','Parch','Fare', 'Embarked_C', 'Embarked_Q', 'Deck_A', 'Deck_B',

 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F']
from sklearn.preprocessing import RobustScaler



def robust_transfer(df):

    Scaler = RobustScaler().fit(df) 

    newdf = Scaler.transform(df)

    df0 = pd.DataFrame(newdf,columns = df.columns)

    return df0



train0 = robust_transfer(train[features])

test0 = robust_transfer(test[features])
x_train, x_val, y_train, y_val = train_test_split(train0, y, 

                                                  stratify=y, 

                                                  test_size=0.3,

                                                  random_state=2020)



model = xgb.XGBClassifier(n_estimators = 500,

                          learning_rate = 0.05,   

                          objective = 'binary:logistic',

                          max_depth=8,

                          min_child_weight=1, # 叶子上的最小样本数

                          colsample_bytree=0.8, 

                          subsample=0.8,

                          seed=64)



model.fit(x_train, y_train,

          verbose=True,

          eval_set=[(x_train, y_train), (x_val, y_val)],

          eval_metric='error',

          early_stopping_rounds = 20)



evals_result = model.evals_result()

ax = plot_metric(evals_result, metric = 'error')

plt.title('Xgboost Learning Curve')

plt.show()



y_val_pred = model.predict(x_val)

y_test_pred = model.predict(test0)



val_acc = metrics.accuracy_score(y_val, y_val_pred)

print('Out of folds accuracy_score is {:.4f}'.format(val_acc))
train.to_csv('train1.csv',index = False)

test.to_csv('test1.csv',index = False)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_test_pred})

submission.to_csv('submission.csv',index = False)

submission.head(10)
fig,ax = plt.subplots(figsize=(8,7))

xgb.plot_importance(model,

                ax=ax,

                height=0.5).set(xlabel='feature importance',

                                         title='',

                                         ylabel='feature')
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    '''    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    '''

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
plt.style.use('seaborn-white')

class_names = ['0','1']

plot_confusion_matrix(confusion_matrix(y_val, y_val_pred),classes=class_names, normalize=True, 

                      title='Normalized Confusion Matrix: Xgboost')