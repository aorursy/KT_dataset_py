%%time

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
auto = pd.read_csv("/kaggle/input/vehiceldata/AutoData.csv")

print("Dataset with rows {} and columns {}".format(auto.shape[0],auto.shape[1]))

auto.head()
auto.info() # so in this datasets we have 8 float data types, 7 interger data types and 10 object type data.

# to check if there is any null values
auto.isnull().sum(axis=0) # so it's comparebly very clean data with no null values. 

# Now we will move to EDA part of the datasets.
auto.head() # so considering all the features we have to predict the price.
%%time

# here we are seperating object and numerical data types 

obj_col = []

num_col = []

for col in auto.columns:

    if auto[col].dtype=='O':

        obj_col.append(col)

    else:

        num_col.append(col)
print("Object data type features ",obj_col)

print("Numerical data type features ",num_col)

from numpy import median

for col in obj_col[1:]:

    plt.figure(figsize=(10,8))

    sns.violinplot(auto[col],auto["price"])

    plt.title("Price vs "+col,fontsize=20)

    plt.xlabel(col,fontsize=12)

    plt.ylabel("Price",fontsize=12)

    plt.show()

#sns.despine()

# violin plots give best of both worlds 

# it gives boxplot and distribution of data like whether the data is skewed or not.

# if normally distributed then it's the best you can get.

# you can also use barplots in this case.
plt.figure(figsize=(15,12))

sns.heatmap(auto.corr(),annot=True,cmap='RdBu_r')

plt.title("Correlation Of Each Numerical Features")

plt.show()
for col in num_col[:-1]:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = auto[col],y = auto["price"],kind='reg')

    plt.xlabel(col,fontsize = 15)

    plt.ylabel("Prïce",fontsize = 15)

    plt.grid()

    plt.show()
from sklearn.model_selection import train_test_split

X_tr,X_ts,y_tr,y_ts = train_test_split(auto.drop(["price"],axis=1),auto["price"],test_size = 0.2,random_state=42)

print("Train Data shape ",X_tr.shape)

#X_tr.head()

print("Test Data shape ",X_ts.shape)

#X_ts.head()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse = False,handle_unknown="ignore")

X_tr_obj = ohe.fit_transform(X_tr[obj_col])

X_ts_obj = ohe.transform(X_ts[obj_col])

print(X_tr_obj.shape)

print(X_ts_obj.shape)
features = ohe.get_feature_names().tolist()
X_tr_obj = pd.DataFrame(X_tr_obj,columns= features)

X_ts_obj = pd.DataFrame(X_ts_obj,columns= features)
auto["make"].value_counts()
X_tr_obj["x0_Nissan versa"]
from sklearn.preprocessing import MinMaxScaler

min_max = MinMaxScaler()

X_tr = min_max.fit_transform(X_tr[num_col[:-1]])

X_ts = min_max.transform(X_ts[num_col[:-1]])

print(X_tr.shape)

print(X_ts.shape)
X_tr = pd.DataFrame(X_tr,columns=num_col[:-1])

X_ts = pd.DataFrame(X_ts,columns=num_col[:-1])
X_tr = pd.concat([X_tr_obj,X_tr[num_col[:-1]]],axis=1)

X_ts = pd.concat([X_ts_obj,X_ts[num_col[:-1]]],axis=1)

print(X_tr.shape)

print(X_ts.shape)
X_tr.head()
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score



# calculation part

model = LinearRegression()

model.fit(np.array(X_tr["enginesize"]).reshape(-1,1),np.array(y_tr).reshape(-1,1))

y_pred = model.predict(np.array(X_ts["enginesize"]).reshape(-1,1))





# plotting part

plt.figure(figsize=(10,6))

sns.scatterplot(x = X_ts["enginesize"],y = y_ts,label = "Actual Points",palette="set1")

plt.plot(X_ts["enginesize"],y_pred,label = "Estimated Line")

plt.title("Price Vs Engine Size",fontsize=20)

plt.xlabel("Engine Size",fontsize = 15)

plt.ylabel("Price",fontsize = 15)

plt.legend()

plt.grid()

plt.show()

print("R2 Score using engine size features is -->",r2_score(y_ts,y_pred))
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm

from sklearn.feature_selection import RFE
#selecting top 10 features 

lr = LinearRegression(n_jobs=-1)

rfe = RFE(estimator=lr,n_features_to_select=10)

rfe.fit(X_tr,y_tr)
selected_feat = X_tr.columns[rfe.ranking_==1]
selected_feat
X_tr_cat = sm.add_constant(X_tr[selected_feat]) # adding constant 
pd.Series([variance_inflation_factor(X_tr_cat.values, i) 

               for i in range(X_tr_cat.shape[1])], 

              index=X_tr_cat.columns)
model = sm.OLS(np.array(y_tr),X_tr_cat).fit()

model.summary()
X_tr_obj = X_tr_obj[["x6_rear","x7_dohc","x7_l","x7_ohc","x8_eight","x8_twelve","x8_four"]]
lr = LinearRegression(n_jobs=-1)

rfe = RFE(estimator=lr,n_features_to_select=5)

rfe.fit(X_tr[num_col[:-1]],y_tr)
selected_feat = X_tr[num_col[:-1]].columns[rfe.ranking_==1]
selected_feat
X_tr_num = sm.add_constant(X_tr[selected_feat])
pd.Series([variance_inflation_factor(X_tr_num.values, i) 

               for i in range(X_tr_num.shape[1])], 

              index=X_tr_num.columns)
model = sm.OLS(np.array(y_tr),X_tr_num).fit()

model.summary()
X_tr_main = pd.concat([X_tr_obj,X_tr_num],axis=1)

X_tr_main.head()
model = sm.OLS(np.array(y_tr),X_tr_main).fit()

model.summary()
top_features = X_tr_main[["x6_rear","x7_ohc","x8_four","enginesize","stroke"]]
lr = LinearRegression(n_jobs=-1)

lr.fit(top_features,y_tr)

X_ts_main = X_ts[["x6_rear","x7_ohc","x8_four","enginesize","stroke"]]

y_pred = lr.predict(X_ts_main)

y_pred_tr = lr.predict(top_features)

print("r2 score on test data is --> ",r2_score(y_ts,y_pred))

print("r2 score on train data is --> ",r2_score(y_tr,y_pred_tr))