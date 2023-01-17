import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.model_selection import train_test_split, ParameterGrid

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler



import warnings

warnings.filterwarnings('ignore')
X = pd.read_csv("../input/train.csv")

print('\n shape of raw training:', X.shape)



known_ids = X['ID_code']

y = X['target']

X = X.drop(['target', 'ID_code'], axis=1).values

print('\n shape of ids:', known_ids.shape)

print('\n shape of labels:', y.shape)

print('\n shape of training data:', X.shape)

print("\n train data loaded!")
def my_cv(X, y, model, folds=5, rand_st=1):

    scores = []

    for i in range(1,folds+1):

        print('\n fold: ', i)

        X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                            test_size=1/folds, 

                                                            random_state=i+rand_st)

        # scaler = MinMaxScaler(feature_range=(-1,1))

        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)

        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)

        model_pred = model.predict(X_test)

        # model_pred_proba =model.predict_proba(X_test)

        scores.append(roc_auc_score(model_pred,y_test))

        if i == 1:

            conf_mat = confusion_matrix(y_test, model_pred)

        else:

            conf_mat += confusion_matrix(y_test, model_pred)

    print('\n',scores,'\n',conf_mat)

    return scores, conf_mat  
model = SGDClassifier(loss='hinge',

                      class_weight='balanced', 

                      penalty = 'l1',

                      l1_ratio = 0.7,

                      alpha = 5e-4, 

                      max_iter = 1000,

                      early_stopping=True,

                      tol = 1e-3,

                      n_iter_no_change = 10,

                      verbose=1)

scores, conf_mat = my_cv(X, y, model, folds=5)

print('class 0 accuracy: ', conf_mat[0,0]/sum(conf_mat[0,]))

print('class 1 accuracy: ', conf_mat[1,1]/sum(conf_mat[1,]))
print(scores)

print('\n', sum(scores)/len(scores))

print('\n', conf_mat)
test = pd.read_csv('../input/test.csv')

submit = test[['ID_code']]

X_test = test.drop(columns=['ID_code'])



scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)



model.fit(X,y)

preds = model.predict(X_test)



submit['target'] = preds

print(submit.head)
submit.to_csv('sgdclassifier.csv', index=False)