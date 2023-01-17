import pandas as pd

import numpy as np

import os

import seaborn as sns

import xgboost as xgb

import matplotlib.pyplot as plt



from matplotlib import style

from sklearn import metrics

from sklearn.metrics import accuracy_score
path = '../input/titanic'

train = pd.read_csv(os.path.join(path,'train.csv'))

test = pd.read_csv(os.path.join(path,'test.csv'))

gender = pd.read_csv(os.path.join(path,'gender_submission.csv'))
train.head(10)
train.info()
f, ax = plt.subplots(1, 2, figsize=(18,8))  # 1 x 2 subplots



survived = train['Survived'].value_counts()

survived.plot.pie(explode=[0, 0.1], 

                  autopct='%1.1f%%', 

                  ax=ax[0], 

                  shadow=True)



ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])

ax[1].set_title('Survived')



plt.show()
f, ax = plt.subplots(figsize=(18,8))



sns.violinplot("Pclass", "Age", 

               hue="Survived", 

               data=train, 

               split=True, 

               ax=ax)



ax.set_title('Pclass and Age vs Survived')

ax.set_yticks(range(0, 110, 10))



plt.show()
f, ax = plt.subplots(figsize=(18,8))



sns.violinplot("Sex", "Age", 

               hue="Survived", 

               data=train, 

               split=True, 

               ax=ax)



ax.set_title('Sex and Age vs Survived')

ax.set_yticks(range(0, 110, 10))



plt.show()
one_hot = pd.get_dummies(train[['Sex','Embarked']])

one_hot_test = pd.get_dummies(test[['Sex','Embarked']])
train = pd.concat([train,one_hot],1)

test = pd.concat([test,one_hot_test],1)
train = train.drop(['Name','Ticket','Cabin','Sex','Embarked'],1)

test = test.drop(['Name','Ticket','Cabin','Sex','Embarked'],1)
train_num = int(train.shape[0]*0.8)
val = train[train_num:]

train_parts = train[:train_num]

val_y = val.pop('Survived')

train_y = train_parts.pop('Survived')
dtrain = xgb.DMatrix(train_parts.values, label=train_y.values,feature_names=train_parts.columns.values.tolist())

dval = xgb.DMatrix(val.values,label=val_y.values,feature_names=val.columns.values.tolist())

dtest = xgb.DMatrix(test.values,feature_names=test.columns.values.tolist())
list_num = list()

list_accuracy = list()



xgb_params = {

    # 二値分類問題

    'objective': 'binary:logistic',

    # 評価指標

    'eval_metric': 'auc'

}

for i in range(10,300,10):

    bst = xgb.train(xgb_params,

                    dtrain,

                    num_boost_round=i,  

                    )



    predict_y = bst.predict(dval)



    #fpr, tpr, thresholds = metrics.roc_curve(val_y, predict_y)

    #auc = metrics.auc(fpr,tpr)

    predict_y = np.where(predict_y > 0.5, 1, 0)

    accuracy = accuracy_score(val_y,predict_y)



    list_num.append(i)

    list_accuracy.append(accuracy)



    print('\nnum_boost_round:{0}'.format(i))

    #print('auc:{0}'.format(auc))

    print('accuracy:{0}'.format(accuracy))



#max_index = np.argmax(np.array(list_auc))

max_index = np.argmax(np.array(list_accuracy))



bst = xgb.train(xgb_params,

                dtrain,

                num_boost_round=max_index,

                )



print('best accuracy:{0}'.format(list_accuracy[max_index]))

bst.predict(dtrain)
predict_y = bst.predict(dtest)
test_num = test['PassengerId']
predict_y = np.where(predict_y > 0.5, 1, 0)
submission  = pd.DataFrame({'PassengerId': test['PassengerId'], 

                            'Survived': predict_y.astype(int)})



submission.to_csv('my_submission.csv', index=False)