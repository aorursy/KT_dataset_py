# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestRegressor

import statsmodels.formula.api as smf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/housing.csv")

data.head()
data.shape
data.info()
data.describe()
data['total_bedrooms'].isnull().sum()
data['total_bedrooms'].fillna(data['total_bedrooms'].median(),inplace=True)

data['total_bedrooms'].isnull().sum()
fig = plt.subplots(figsize = (8,6))

sns.heatmap(data.corr(),annot=True)
sns.set(style='ticks')

sns.pairplot(data)

#pd.plotting.scatter_plot(data,figsize=(12, 14))

plt.show()
newData = data.join(pd.get_dummies(data[['ocean_proximity']],prefix='Proximity').iloc[:,3:]).drop(data[['ocean_proximity']],axis=1)

housing_features=newData.drop(newData[['median_house_value']],axis=1)

cols = np.full((housing_features.corr().shape[0],), True, dtype=bool)

#print(cols)

for i in range(housing_features.corr().shape[0]):

    for j in range(i+1,housing_features.corr().shape[0]):

        if housing_features.corr().iloc[i,j] >=0.6:

            if cols[j]:

                cols[j]= False

#selected_cols = data.select_dtypes(exclude=[np.object]).columns[cols]

selected_cols = housing_features.columns[cols]

selected_cols
scaler = MinMaxScaler()

colNames = newData.columns.values

scaled_data = pd.DataFrame(scaler.fit_transform(newData,colNames))

scaled_data.columns= colNames

scaled_data = scaled_data.rename(columns={"Proximity_NEAR BAY":"proxim_nearBay","Proximity_NEAR OCEAN":"proxim_nearOcean"})

scaled_data.head()

#features = ['total_rooms','housing_median_age','median_income','Proximity_NEAR BAY','Proximity_NEAR OCEAN']

X = scaled_data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'median_income', 'proxim_nearBay', 'proxim_nearOcean']]

y= scaled_data.median_house_value

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.20)

model = LinearRegression()

model.fit(X_train,y_train)

print(model.intercept_)

print(model.coef_)
score = model.score(X_test,y_test)

y_predicted = model.predict(X_test)

err = mse(y_test,y_predicted)

print(score)

lr_rmse = np.sqrt(err)

lr_r_square = r2_score(y_test,y_predicted)

print("Root Mean Squared Error for Simple Linear Regression is %s",lr_rmse)

print("R square for Simple Linear Regression is %s",lr_r_square)
smf_model = smf.ols(formula='median_house_value ~ longitude+latitude+housing_median_age+total_rooms+median_income+proxim_nearBay+proxim_nearOcean', data=scaled_data)

fitted_model = smf_model.fit()

fitted_model.summary()
def root_mean_squared_error(y_test,y_pred):

    return np.sqrt(mse(y_test,y_pred))



rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)



dt= DecisionTreeRegressor(random_state=2)

dt_params = [{"max_depth":[4,8,12,16,20],"max_leaf_nodes":range(2,20)}]

dt_gs = GridSearchCV(estimator=dt,param_grid=dt_params,scoring=rmse_scorer,cv=10)

dt_gs = dt_gs.fit(X_train,y_train)

print(dt_gs.best_params_)

print(dt_gs.best_estimator_)
dt_model = dt_gs.best_estimator_

dt_model.fit(X_train,y_train)

y_df_pred = dt_model.predict(X_test)

df_rmse = np.sqrt(mse(y_test,y_df_pred))

r_df_sq = r2_score(y_test,y_df_pred)

print("Root Mean Squared Error for DecisionTreeRegressor is %s",df_rmse)

print("R square for DecisionTreeRegressor is %s",r_df_sq)
rf = RandomForestRegressor(random_state=2)

rf_params = [{"n_estimators":range(1,25)}]

rf_gs = GridSearchCV(estimator=rf,param_grid=rf_params,scoring=rmse_scorer,cv=10)

rf_gs = rf_gs.fit(X_train,y_train)

print(rf_gs.best_params_)

print(rf_gs.best_estimator_)
rf_model=rf_gs.best_estimator_

rf_model.fit(X_train,y_train)

y_rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mse(y_test,y_rf_pred))

r_rf_sq = r2_score(y_test,y_rf_pred)

print("Root Mean Squared Error for RandomForestRegressor is %s",rf_rmse)

print("R square for RandomForestRegressor is %s",r_rf_sq)
X_new = scaled_data[['median_income']]

y_new= scaled_data.median_house_value

X_new_train,X_new_test,y_new_train,y_new_test = train_test_split(X_new,y_new,random_state=1,test_size=0.20)

bonus_model = LinearRegression()

bonus_model.fit(X_new_train,y_new_train)

print(bonus_model.intercept_)

print(bonus_model.coef_)

y_bonus_pred = bonus_model.predict(X_new_test)
plt.scatter(X_new_train, y_new_train, color = 'cyan')

plt.plot(X_new_train, bonus_model.predict(X_new_train), color = 'black')

plt.title('Median House Value vs Median Income(Train)')

plt.xlabel('Median Income')

plt.ylabel('Median House Value')

plt.show()
plt.scatter(X_new_test, y_new_test, color = 'yellow')

plt.plot(X_new_test, y_bonus_pred, color = 'blue')

plt.title('Median House Value vs Median Income(Test)')

plt.xlabel('Median Income')

plt.ylabel('Median House Value')

plt.show()
plt.scatter(y_new_test,y_bonus_pred)

plt.title('Median House Value- Actual Vs Predicted')

plt.xlabel('Actual Median House Value')

plt.ylabel('Predicted Median House Value')

plt.show()