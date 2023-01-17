import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import os

print(os.listdir("../input"))
df = pd.read_csv("../input/birds-bones-and-living-habits/bird.csv")
df.head()
def initial_observation(df):

    if isinstance(df, pd.DataFrame):

        total_na = df.isna().sum().sum()

        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))

        print("Total NA Values : %d " % (total_na))

        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))

        col_name = df.columns

        dtyp = df.dtypes

        uniq = df.nunique()

        na_val = df.isna().sum()

        for i in range(len(df.columns)):

            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))

        

    else:

        print("Expect a DataFrame but got a %15s" % (type(df)))
initial_observation(df)
df["type"].value_counts()
sns.catplot(x="type", y="huml", data= df);
sns.catplot(x="type", y="humw", data= df);
sns.catplot(x="type", y="ulnal", data= df);
sns.catplot(x="type", y="ulnaw", data= df);
sns.catplot(x="type", y="feml", data= df);
sns.catplot(x="type", y="femw", data= df);
sns.catplot(x="type", y="tibl", data= df);
sns.catplot(x="type", y="tibw", data= df);
sns.catplot(x="type", y="tarl", data= df);
sns.catplot(x="type", y="tarw", data= df);
df1 = df.copy()
df1 = df1.dropna(axis=0, subset=['huml', "humw", "ulnal", "ulnaw", "feml", "femw", "tibl", "tibw", "tarl", "tarw"])
initial_observation(df1)
df["type"].value_counts()
df1['type'].value_counts().plot(kind='bar');
df_oversample = df1.copy()

df_undersample = df1.copy()

df_smote = df1.copy()
df_oversample['type'].value_counts().plot(kind='bar');
from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler
df2 = df.copy()
x = df1.drop(["type"], axis = 1)

y = df1["type"]
x.head()
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x,y)
X_train_scaled = X_train.copy()

X_val_scaled = X_val.copy()



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train_scaled)

x_train_scaled_1 = scaler.transform(X_train_scaled)

x_val_scaled_1 = scaler.transform(X_val_scaled)
print("X Train shape:" , X_train.shape)

print("X Validation shape:" ,   X_val.shape)

print("Y Train shape:",     Y_train.shape)

print( "Y Validation Shape:",   Y_val.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
rf_parm = dict(n_estimators = [20, 30, 50, 70, 100, 150], max_features = [0.1, 0.2, 0.6, 0.9], max_depth = [10,20,30],min_samples_leaf=[1,10,100, 400, 500, 600],random_state=[0])
rc = RandomForestClassifier()

rf_grid = GridSearchCV(estimator = rc, param_grid = rf_parm)
rf_grid.fit(X_train,Y_train)
print("RF Best Score:", rf_grid.best_score_)

print("RF Best Parameters:", rf_grid.best_params_)
rc_best = RandomForestClassifier(n_estimators = 20,  max_features = 0.9)
rc_best.fit(X_train, Y_train)

rc_tr_pred = rc_best.predict(X_train)

rc_val_pred = rc_best.predict(X_val)
print(rc_val_pred)
from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
print("Precision Score : ",precision_score(Y_val, rc_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))

print("Recall Score : ",recall_score(Y_val, rc_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))

print("F1 Score:",  f1_score(Y_val, rc_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))
from sklearn.metrics import classification_report



print(classification_report(Y_val, rc_val_pred))