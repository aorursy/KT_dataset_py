#import 

import numpy as np

import pandas as pd

import hyperopt

from catboost import Pool, CatBoostClassifier, cv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#get the train and test data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
#show the train data

train_df.info()
#show how many the null value for each column

train_df.isnull().sum()
#for the train data ,the age ,fare and embarked has null value,so just make it -999 for it

#and the catboost will distinguish it

train_df.fillna(-999,inplace=True)

test_df.fillna(-999,inplace=True)
#now we will get the train data and label

x = train_df.drop('Survived',axis=1)

y = train_df.Survived

#show what the dtype of x, note that the catboost will just make the string object to categorical 

#object inside

x.dtypes
#choose the features we want to train, just forget the float data

cate_features_index = np.where(x.dtypes != float)[0]
#make the x for train and test (also called validation data) 

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=.85,random_state=1234)
#let us make the catboost model, use_best_model params will make the model prevent overfitting

model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
#now just to make the model to fit the data

model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
#for the data is not so big, we need use the cross-validation(cv) for the model, to find how

#good the model is ,I just use the 10-fold cv

cv_data = cv(model.get_params(),Pool(x,y,cat_features=cate_features_index),fold_count=10)
#show the acc for the model

print('the best cv accuracy is :{}'.format(np.max(cv_data["b'Accuracy'_test_avg"])))
#show the model test acc, but you have to note that the acc is not the cv acc,

#so recommend to use the cv acc to evaluate your model!

print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))
#last let us make the submission,note that you have to make the pred to be int!

pred = model.predict(test_df)

pred = pred.astype(np.int)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':pred})
#make the file to yourself's directory

submission.to_csv('catboost.csv',index=False)