import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score

np.random.seed(0)





warnings.filterwarnings('ignore')

sns.set(style="white")

%matplotlib inline 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df.head()
df.info()
sns.countplot(x="Churn", data=df)
from sklearn.model_selection import train_test_split



# Feature + Target

y = df.Churn

X = df.drop(['Churn'], axis=1)



# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.dummy import DummyClassifier



# Dummy Stratege

model = DummyClassifier()



# Train

model.fit(X_train, y_train)
#sklearn.metrics.f1_score

model.score(X_test, y_test)
df_valid = pd.read_csv('../input/test.csv')

df_valid.head()
## Predição do modelo

pred = model.predict(df_valid) # predição aleatório para teste



def save_submission(pred):

    with open('submission_pred.csv', 'w+') as f:

        f.write("UserID,Churn\n")    

        i = 0

        for p in pred:

            f.write("{},{}\n".format(i, p))  

            i = i+1
save_submission(pred)