# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('../input/hirirng-decision/Datapoint.csv')
df.head()
df.info()
y = df['Decision'] # Dependent Variable

X = df.drop('Decision',axis=1) # Independent Variables
y.value_counts()
from imblearn.combine import SMOTETomek

oversampling = SMOTETomek()
X_res,y_res = oversampling.fit_sample(X,y)
y_res.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=40)

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train,y_train)



y_predict= model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predict))

print(classification_report(y_test,y_predict))
a = model.predict([[8,6]])



if a ==1 : 

    print("candidate is eligible for interview")

    

else:

    print("candidate is not eligible for interview")

    
b = model.predict([[4,1]])





if b == 1 : 

    print("candidate is eligible for interview")

    

else:

    print("candidate is not eligible for interview")

import joblib
joblib.dump(model,'Candidate_Selection')
selection_model=joblib.load('Candidate_Selection')
c= selection_model.predict([[7,5]])

if c == 1 : 

    print("candidate is eligible for interview")

    

else:

    print("candidate is not eligible for interview")
