import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

# from xgboost import XGBClassifier

# import xgboost

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/mushrooms.csv')

df.shape # (rows=8124, columns=23)

df.describe()
# convert to classes to numbers

label_encoder = preprocessing.LabelEncoder()

for col in df.columns:

    df[col] = label_encoder.fit_transform(df[col])
# compute correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=5, yticklabels=5,

            linewidths=.25, cbar_kws={"shrink": .5}, ax=ax)
y = df['class'].values

df = df.drop('class', 1)

X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1337) # use default .25 test size
models = [SVC(kernel='rbf', random_state=0), SVC(kernel='linear', random_state=0), LogisticRegression()]

model_names = ['SVC - ', 'SVC_linear', 'Logistic Regression']

for i, model in enumerate(models):

    model.fit(X_train, y_train)

    print ('The accurancy of ' + model_names[i] + ' is ' + str(accuracy_score(y_test, model.predict(X_test))) )