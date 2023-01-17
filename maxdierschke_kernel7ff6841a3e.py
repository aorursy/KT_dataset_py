import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input")

file_read ="../input/train.csv"

test_read="../input/test.csv"

df=pd.read_csv(file_read)

df=df.fillna(df.mean())

df=df.drop(columns=['Name','PassengerId','Ticket','Cabin'])

dummies=df.select_dtypes(include=['object']).columns.values

df=pd.get_dummies(df,columns=dummies,drop_first=True)

columns=df.drop(columns="Survived").columns

print(columns)

df.head()
corr=df.corr()

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize=(16,12))

sns.heatmap(corr, annot = True)
X=df.drop(columns="Survived").values

y=df["Survived"]
import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import mean_absolute_error as mse

splits=5

repeats=10

model = XGBClassifier()



kf=RepeatedKFold(n_splits=splits,n_repeats=repeats)

error=0

score=0

for train_index, test_index in kf.split(X):

    X_test=X[test_index]

    X_train=X[train_index]

    y_test=y[test_index]

    y_train=y[train_index]

    model.fit(X_train,y_train,verbose=False)

    print( model.score(X_test,y_test))

    print(mse(y_test,model.predict(X_test)))

    print("--------------------")

    error=error+mse(y_test,model.predict(X_test))

    score=score+model.score(X_test,y_test)

print("--------------------")

print("Error="+str(error/(splits*repeats)))

print("ModelAccuracy="+str(score/(splits*repeats)))

model.fit(X,y)




# read test data file using pandas



test_data=pd.read_csv(test_read)

df_test=pd.read_csv(test_read)

print(df_test.columns)

df_test=df_test.fillna(df.mean())

df_test=df_test.drop(columns=['Name','PassengerId','Ticket','Cabin'])

dummies=df_test.select_dtypes(include=['object']).columns.values

print(dummies)

df_test=pd.get_dummies(df_test, columns=dummies,drop_first=False)



df_test=df_test[columns].values

test_preds =model.predict(df_test)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)