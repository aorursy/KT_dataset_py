%matplotlib inline

from IPython.display import display

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)



from sklearn import linear_model
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

usecols = []

for c in df_train.columns:

    if 'cont' in c:

        usecols.append(c)



x_train = df_train[usecols]

x_test = df_test[usecols]

y_train = df_train['loss']



id_test = df_test['id']
for c in df_train.columns:

    if 'cat' in c:

        df_train[c] = df_train[c].astype('category')

        df_test[c] = df_test[c].astype('category')

        x_train[c + '_numeric'] = df_train[c].cat.codes

        x_test[c + '_numeric'] =  df_test[c].cat.codes
x_test.head()
# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(x_train, y_train)





# Predict using the trained model 

y_pred = regr.predict(x_test)
sub = pd.DataFrame()

sub['id'] = id_test

sub['loss'] = y_pred

sub.to_csv('lin_regression.csv', index=False)
# from sklearn.ensemble import RandomForestClassifier



# # Random Forest 

# rf = RandomForestClassifier(n_estimators=30)

# rf.fit(x_train, np.asarray(y_train, dtype="|S6"))

# y_pred = rf.predict(x_test)



# sub = pd.DataFrame()

# sub['id'] = id_test

# sub['loss'] = y_pred

# sub.to_csv('random_forest.csv', index=False)
np.asarray(y_train, dtype="|S6")
# from sklearn.tree import DecisionTreeClassifier, export_graphviz



# dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)

# dt.fit(x_train, np.asarray(y_train, dtype="|S6"))
