import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
data = pd.read_csv('../input/who-suicide-statistics/who_suicide_statistics.csv')
data.head()
# Checking for the total data points

data.shape
data.describe()
# Checking for any null values 

data.isnull().sum()
data = data.sort_values(['year'],ascending=True)

data.head()
print("Number of unique countries :" , data['country'].nunique())
# Checking for the total data points

# after dropping the null rows 

data.dropna(inplace = True)

data.isnull().sum()
data.head()
data[['country','suicides_no']].groupby(['country']).agg('sum').sort_values(by='suicides_no',ascending=False).head(10)
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

plt.rcParams['figure.figsize'] = (15, 10)



xval = pd.DataFrame(data.groupby(['country'])['suicides_no'].sum().reset_index())

xval.sort_values(by = ['suicides_no'], ascending = False, inplace = True)



sns.barplot(xval['country'].head(10),y = xval['suicides_no'].head(10), data= xval, palette = 'winter')

plt.title('Top 10 Countries with maximum number of suicides ', fontsize = 19)

plt.xlabel('Name of the Country')

plt.xticks(rotation = 90)

plt.ylabel('Count ( unit = 1e6 )')

plt.rcParams['font.size'] = 19

plt.show()
data[['country','suicides_no']].groupby(['country']).agg('sum').sort_values(by='suicides_no',ascending=True).head(10)
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

plt.rcParams['figure.figsize'] = (20, 10)



xval = pd.DataFrame(data.groupby(['country'])['suicides_no'].sum().reset_index())

xval.sort_values(by = ['suicides_no'], ascending = True, inplace = True)



sns.barplot(xval['country'].head(10),y = xval['suicides_no'].head(10), data= xval, palette = 'winter')

plt.title('Top 10 Countries with maximum number of suicides ', fontsize = 19)

plt.xlabel('Name of the Country')

plt.xticks(rotation = 90)

plt.ylabel('Count')

plt.rcParams['font.size'] = 19

plt.show()
xvals = data[['year','suicides_no']].groupby('year').agg('sum').sort_values(by='suicides_no',ascending=False)

xvals.head(10)
data[['population','suicides_no','year']].groupby('year').agg('sum').sort_values(by='suicides_no',ascending=False).head(10)
corr= data['suicides_no'].corr(data['population'])

print("Correlation between the number of suicides and population is :",corr)
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

plt.rcParams['figure.figsize'] = (15, 9)



xvals = pd.DataFrame(data.groupby(['year'])['suicides_no'].sum().reset_index())

xvals.sort_values(by = ['suicides_no'], ascending = False , inplace = True)



sns.barplot(x = "year", y = "suicides_no", data= xvals.head(10), palette = 'winter')

plt.title('Top 10 years having the most suicides', fontsize = 20)

plt.xlabel('Year')

plt.rcParams['font.size'] = 19

plt.xticks(rotation = 45)

plt.ylabel('Count')

plt.show()
data[['age','suicides_no']].groupby('age').agg('sum').sort_values(by='suicides_no',ascending=False)
plt.style.use('seaborn-dark')

plt.style.use('dark_background')



x = pd.DataFrame(data.groupby(['age'])['suicides_no'].sum().reset_index())

x.sort_values(by = ['suicides_no'], ascending = False , inplace = True)



sns.barplot(x['age'].head(10), y = x['suicides_no'].head(10), data= x, palette = 'winter')

plt.title('Top age groups with highest number Suicides', fontsize = 20)

plt.xlabel('age group')

plt.xticks(rotation = 45)

plt.ylabel('Count')

plt.show()
data[['sex','suicides_no']].groupby('sex').agg('sum').sort_values(by='suicides_no')
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

plt.rcParams['figure.figsize'] = (9,5)



color = plt.cm.winter(np.linspace(0, 10, 100))

x = pd.DataFrame(data.groupby(['sex'])['suicides_no'].sum().reset_index())

x.sort_values(by = ['suicides_no'], ascending = False , inplace = True)



# sns.pieplot(x['sex'], y = x['suicides_no'], data= x, palette = 'winter')

x.plot.pie(y ='suicides_no')

plt.title('Sex vise division', fontsize = 20)

plt.xlabel('Sex')

plt.ylabel('Count (unit = 1e6)')

plt.show()
bm_df = data.copy()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

bm_df['sex'] = LE.fit_transform(bm_df['sex'])

bm_df['age'] = LE.fit_transform(bm_df['age'])

bm_df['country'] = LE.fit_transform(bm_df['country'])

data['sex'] = LE.fit_transform(data['sex'])
bm_df.head()
bm_df.shape
# correlation betwwen the features

plt.style.use('seaborn-dark')

import matplotlib.pyplot as plt

%matplotlib inline



corrmat = bm_df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

sns.heatmap(bm_df[top_corr_features].corr() ,annot=True,cmap="RdYlGn")

plt.show()
data = pd.get_dummies(data,drop_first = True)
data.head()
data.shape
X= data.drop(['suicides_no'],axis=1)

y= data['suicides_no']



bm_X= bm_df.drop(['suicides_no','country'],axis=1)

bm_y= bm_df['suicides_no']
nplace = True# splitting into train_test_split



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

bm_x_train, bm_x_test, bm_y_train, bm_y_test = train_test_split(bm_X, bm_y, test_size = 0.2, random_state = 45)



print("Shapes of train data :")

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)



print("Shapes of benchmark data :")

print(bm_x_train.shape)

print(bm_x_test.shape)

print(bm_y_train.shape)

print(bm_y_test.shape)
from sklearn.preprocessing import MinMaxScaler



# creating a scaler

mm = MinMaxScaler()



# scaling the independent variables

x_train = mm.fit_transform(x_train)

x_test = mm.transform(x_test)



bm_x_train = mm.fit_transform(bm_x_train)

bm_x_test = mm.transform(bm_x_test)



X = mm.fit_transform(X)
results = pd.DataFrame(columns = ["model_name","MSE","RMSE","r2_score"])



bm_results = pd.DataFrame(columns = ["model_name","MSE","RMSE","r2_score"])
from sklearn.experimental import enable_hist_gradient_boosting



from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, SGDRegressor, Ridge

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor,BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



from sklearn.metrics import r2_score
# creating the model

model = DecisionTreeRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(bm_x_train, bm_y_train)



# predicting the test set results

y_pred = model.predict(bm_x_test)



# calculating the mean squared error

mse = np.mean((bm_y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(bm_y_test, y_pred)

print("r2_score :", r2)



bm_results = bm_results.append({"model_name" : "Desicion Tree Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = LinearRegression()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Linear Regression","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = Lasso()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Losso","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = ElasticNet()

print("Model :",model)

# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Elastic Net","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = SGDRegressor() 

print("Model :",model)

# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "SGD Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = Ridge()

print("Model :",model)

# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Ridge","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = RandomForestRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Random Forest Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = AdaBoostRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Ada Boost Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = GradientBoostingRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Gradient Boosting Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = ExtraTreesRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)





y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Extra Trees Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = HistGradientBoostingRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Histogram Gradient Boosting Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = BaggingRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Bagging Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = DecisionTreeRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "Desicion Tree Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
# creating the model

model = XGBRegressor()

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)



results = results.append({"model_name" : "XGBoost Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
results.head(20)
results.sort_values(by='r2_score',ascending=False)
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

sns.barplot(x = "model_name", y = "MSE" , data = results.sort_values(by = "MSE", ascending = True), palette = 'winter')

plt.title('Comparison of MSE values of different models(Lower the better)', fontsize = 20)

plt.xlabel('Name of Models')

plt.xticks(rotation = 90)

plt.ylabel('Count')

plt.rcParams['figure.figsize'] = (15, 9)

plt.rcParams['font.size'] = 19

plt.show()
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

sns.barplot(x = "model_name", y = "RMSE" , data = results.sort_values(by = "RMSE", ascending = True), palette = 'winter')

plt.title('Comparison of RMSE values of different models.(Lower the better)', fontsize = 20)

plt.xlabel('Name of Models')

plt.xticks(rotation = 90)

plt.ylabel('Count')

plt.rcParams['figure.figsize'] = (15, 9)

plt.show()
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

sns.barplot(x = "model_name", y = "r2_score" , data = results.sort_values(by = "r2_score", ascending = False), palette = 'winter')

plt.title('Comparison of R2 score of models.(Higher the better)', fontsize = 20)

plt.xlabel('Name of Models')

plt.xticks(rotation = 90)

plt.ylabel('Count')

plt.rcParams['figure.figsize'] = (15, 9)

plt.show()
# creating the model

model = ExtraTreesRegressor(n_estimators = 115, n_jobs = 10)

print("Model :",model)



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)
bm_results = bm_results.append({"model_name" : "Tuned Extra Trees Regressor","MSE":mse,"RMSE":rmse,"r2_score":r2},ignore_index=True)
bm_results.head()
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

sns.barplot(x = "model_name", y = "MSE" , data = bm_results.sort_values(by = "MSE", ascending = True), palette = 'winter')

plt.title('Comparison of MSE values of the 2 models (Lower the better)', fontsize = 20)

plt.xlabel('Name of Models')

plt.ylabel('Count')

plt.rcParams['figure.figsize'] = (15, 5)

plt.rcParams['font.size'] = 19

plt.show()
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

sns.barplot(x = "model_name", y = "RMSE" , data = bm_results.sort_values(by = "RMSE", ascending = True), palette = 'winter')

plt.title('Comparison of RMSE values of the 2 models (Lower the better)', fontsize = 20)

plt.xlabel('Name of Models')

plt.ylabel('Count')

plt.rcParams['figure.figsize'] = (15, 5)

plt.rcParams['font.size'] = 19

plt.show()
plt.style.use('seaborn-dark')

plt.style.use('dark_background')

sns.barplot(x = "model_name", y = "r2_score" , data = bm_results.sort_values(by = "r2_score", ascending = True), palette = 'winter')

plt.title('Comparison of R2 score values of the 2 models (Higher the better)', fontsize = 20)

plt.xlabel('Name of Models')

plt.ylabel('Count')

plt.rcParams['figure.figsize'] = (15, 5)

plt.rcParams['font.size'] = 19

plt.show()