# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



os.makedirs("../output", exist_ok=True)



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("train.shape:", train.shape)

print("test.shape:", test.shape)
train.info()
X, y = train.drop(columns="Chance of Admit"), train["Chance of Admit"]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf.fit(X, y)
pred = rf.predict(test)
def submit_predictions(pred):

    pred = pd.Series(pred)

    serialNumber = test["Serial No."]

    submission = pd.concat([serialNumber, pred], axis=1, ignore_index=True)

    submission.rename(columns={0:"Serial No.", 1:"Chance of Admit"}, inplace=True)

    submission.to_csv("submission.csv", index=False)

    
submit_predictions(pred)