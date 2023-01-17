# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import model_selection, preprocessing

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')
#Conversion using label encoder

clean_df = train_set.copy()



for f in clean_df.columns:

    if clean_df[f].dtype == 'object':

        label = preprocessing.LabelEncoder()

        label.fit(list(clean_df[f].values))

        clean_df[f] = label.transform(list(clean_df[f].values))

clean_df.head()
train_y = clean_df.y.values

train_x = clean_df.drop(["y"],axis=1)

train_x = train_x.drop(["ID"],axis=1)

train_x = train_x.values
#Fitting XGB regressor 

model = xgb.XGBRegressor()

model.fit(train_x,train_y)

print (model)



#Transforming the testset

id_vals = test_set.ID.values



clean_test = test_set.copy()

for f in clean_test.columns:

    if clean_test[f].dtype == 'object':

        label = preprocessing.LabelEncoder()

        label.fit(list(clean_test[f].values))

        clean_test[f] = label.transform(list(clean_test[f].values))

clean_test.fillna((-999), inplace=True)

test = clean_test.drop(['ID'],axis=1)

x_test = test.values



#testset is ready
#Predict 

output = model.predict(data=x_test)

final_df = pd.DataFrame()

final_df["ID"] = id_vals

final_df["Prediction"] = output

#final_df.to_csv("Output_1.csv")

final_df.head()