import numpy as np 

import pandas as pd 

import seaborn as sb

import xgboost as xgb





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#explore the data



train = pd.read_csv("../input/train.csv")

print(train.head())



test = pd.read_csv("../input/test.csv")
#sum features

print (len(train))

featuresSum = train.sum()

#print(featuresSum)

#print(type(train))

#plot the sum of features in desceding order

featuresSum.drop(['target', 'id']).order().plot(kind='barh', figsize=(8,16))
#Clases are String like "Class_1" => we map them as numbers from 0-8 (9 classes)



class_range = range(1, 10)

class_dict = {}



for n in class_range:

    class_dict['Class_{}'.format(n)] = n-1



#print(class_dict)

#print(train.head())

train['target'] = train['target'].map(class_dict)

#print(train['target'].head())



#next we plot the count for each class unsing seaborn libary

sb.countplot(x='target', data= train)
#next we create training and testing sets

X_train = train.drop(["id", "target"], axis=1)

Y_train = train["target"].copy()

X_test = test.drop("id", axis = 1).copy()
from sklearn.model_selection import StratifiedKFold



params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}



#perform K-fold validation with k = 10



iter = [110, 120, 130, 140, 150, 160, 170]



for i in iter:

    skf = StratifiedKFold(n_splits=10)

    score = 0

    for train, test in skf.split(X_train, Y_train):

        X_train_k = X_train.iloc[train]

        Y_train_k = Y_train.iloc[train]

        X_test_k = X_train.iloc[test]

        Y_test_k = Y_train.iloc[test] 

        T_train_xgb = xgb.DMatrix(X_train_k, Y_train_k)

        X_test_xgb  = xgb.DMatrix(X_test_k)

        #number of iterations - initial 20, 

        gbm = xgb.train(params, T_train_xgb, i)

        Y_pred = gbm.predict(X_test_xgb)

        val_score =0

        Y_k = Y_test_k.as_matrix()

        for i in range(len(Y_pred)):

            if( Y_k[i] == Y_pred[i].argmax()):

                val_score +=1

        val_score = val_score/len(Y_pred)

        score += val_score

    score = score/10

    print(i)

    print (score)

    



score = score/10

print (score)

#Make a submission based on model "sample_submission.csv"



submission = pd.DataFrame({ "id": test["id"]})



i = 0



for num in class_range:

    col_name = str("Class_{}".format(num))

    submission[col_name] = Y_pred[:,i]

    i = i + 1

    

submission.to_csv('otto1.csv', index=False)