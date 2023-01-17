import numpy as np

import pandas as pd 

import sklearn.linear_model as linear_model

from sklearn.preprocessing import LabelEncoder



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



y_train = train['SalePrice']

train = pd.concat((train,test)).reset_index(drop=True)

train.drop(['Id','SalePrice'], axis = 1, inplace = True)



qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

train[qualitative] = train[qualitative].fillna('Missing')

for c in qualitative:  

    le = LabelEncoder().fit(list(train[c].values)) 

    train[c] = le.transform(list(train[c].values))

    

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

for item in quantitative:

    train[item] = np.log1p(train[item].values)



X_train = train[:len(y_train)].fillna(0)

X_test = train[len(y_train):].fillna(0)

                        

model = linear_model.LassoLarsCV()

model.fit(X_train, np.log(y_train))



prediction = pd.DataFrame({"Id": test["Id"], "SalePrice": np.exp(model.predict(X_test))})

prediction.to_csv('house_submission1.csv', index=False)   



print(np.sqrt(np.sum(np.square(np.log(y_train)-model.predict(X_train)))/len(y_train)))
