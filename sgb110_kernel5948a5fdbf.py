# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

data = pd.read_csv("../input/Toddler Autism dataset July 2018.csv",header=0,usecols=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',

       'Age_Mons', 'Qchat-10-Score', 'Sex', 'Jaundice',

       'Family_mem_with_ASD',  'Class/ASD Traits '])
data["Class"] = data["Class/ASD Traits "].map({"No":0,"Yes":1})

data["Sex_bool"] = data["Sex"].map({"f":0,"m":1})

data["Family_mem_with_ASD_bool"] = data["Family_mem_with_ASD"].map({"no":0,"yes":1})

data["Jaundice_bool"] = data["Jaundice"].map({"no":0,"yes":1})

data["Qchat-10-Score"] = data["Qchat-10-Score"]/np.sum(data["Qchat-10-Score"])

data=data.drop("Sex", axis=1)

data=data.drop("Family_mem_with_ASD", axis=1)

data=data.drop("Jaundice", axis=1)

data=data.drop("Class/ASD Traits ", axis=1)
X = data.loc[:,data.columns !="Class" ]

Y = data.loc[:,data.columns =="Class"]
from sklearn.model_selection import train_test_split

x_train , x_test, y_train , y_test = train_test_split(X,Y ,test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l1', solver='saga',C=.2,

                                      tol=1e-6, max_iter=int(1e6),

                                      warm_start=True).fit(x_train, y_train)
p=model.predict(x_test)

from sklearn.metrics import f1_score

f1_score(y_test,p)