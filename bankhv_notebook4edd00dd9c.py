from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/creditcard.csv")
df.head(5)
features = df.columns[0:-1]

df[features].corrwith(df[df.columns[-1]])
print(df.mean())
from sklearn.preprocessing import scale

from sklearn.utils import shuffle



df= pd.read_csv("../input/creditcard.csv")

#df = df.drop('Time', 1)

df = df.drop('Amount', 1)

#df = df.drop('V2', 1)

#df = df.drop('V4', 1)

#df = df.drop('V20', 1)

#all_data = all_data.drop('V13', 1)

#all_data = all_data.drop('V15', 1)

#all_data = all_data.drop('V19', 1)

#all_data = all_data.drop('V20', 1)

#all_data = all_data.drop('V21', 1)

#all_data = all_data.drop('V22', 1)

#all_data = all_data.drop('V23', 1)

#all_data = all_data.drop('V24', 1)

#all_data = all_data.drop('V25', 1)

#all_data = all_data.drop('V26', 1)

#all_data = all_data.drop('V27', 1)

#all_data = all_data.drop('V28', 1)

df['feature1'] = df['V1'] * df['V3']



df_shuffled = shuffle(df, random_state = 123)

X = scale(df_shuffled[df_shuffled.columns[:-1]])

y = df_shuffled["Class"]



lr = LogisticRegression()

scores = cross_val_score(lr, X, y, cv = 5)

print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))