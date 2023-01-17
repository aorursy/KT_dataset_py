import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns',None)
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
# Correlation matrix - linear relation among independent attributes and with the Target attribute



sns.set(style="white")



# Compute the correlation matrix

correln = df.corr()



# Generate a mask for the upper triangle

#mask = np.zeros_like(correln, dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(correln,  cmap=cmap, vmax=.3, #mask=mask,

            linewidths=.8, cbar_kws={"shrink": .9})
sns.set()

sns.pairplot(df, size = 2.5)

plt.show()
df["alcohol"].describe()
plt.boxplot(df["alcohol"])

plt.show()
sns.barplot(df["quality"], df["alcohol"])
sns.barplot(df["quality"], df["sulphates"])
x = df.drop(["quality"], axis = 1)

y = df["quality"]
x.head()
y.head()
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

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
rf_parm = dict(n_estimators = [20, 30, 50, 70, 100, 150], max_features = [0.1, 0.2, 0.6, 0.9], max_depth = [10,20,30],min_samples_leaf=[1,10,100, 400, 500, 600],random_state=[0])
rc = RandomForestRegressor()

rf_grid = GridSearchCV(estimator = rc, param_grid = rf_parm)
rf_grid.fit(x_train_scaled_1,Y_train)
rf_grid.best_score_
rf_grid.best_params_
rc_best = RandomForestRegressor(n_estimators = 30,  max_features = 0.6)
rc_best.fit(x_train_scaled_1, Y_train)

rc_tr_pred = rc_best.predict(x_train_scaled_1)

rc_val_pred = rc_best.predict(x_val_scaled_1)
from sklearn.metrics import r2_score



print(r2_score(Y_train, rc_tr_pred))

print(r2_score(Y_val, rc_val_pred))
from sklearn.ensemble import GradientBoostingRegressor



gbrt = GradientBoostingRegressor(random_state = 42)

gbrt_grid = GridSearchCV(estimator = gbrt, param_grid = dict(n_estimators = [2, 3, 5, 7, 10, 15, 20, 25], max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], learning_rate = [0.001, 0.0001, 0.01, 0.1, 1, 100]))
gbrt_grid.fit(x_train_scaled_1, Y_train)
print("Best Boosting Parameters:", gbrt_grid.best_params_)

print("Best Boosting Score:", gbrt_grid.best_score_)
gbrt_best = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 25, random_state = 42)

gbrt_best.fit(x_train_scaled_1, Y_train)
gbrt_best_tr_pred = gbrt_best.predict(x_train_scaled_1)

gbrt_best_val_pred = gbrt_best.predict(x_val_scaled_1)
from sklearn.metrics import r2_score



print(r2_score(Y_train, gbrt_best_tr_pred))

print(r2_score(Y_val, gbrt_best_val_pred))