import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
audi = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')
bmw = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv')
ford = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv')
hyundai = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv')
mercedes = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/merc.csv')
skoda = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/skoda.csv')
toyota = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/toyota.csv')
vauxhall = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/vauxhall.csv')
vw = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv')
print("Columns in the Audi dataframe:") 
print(list(audi.columns))
print("-" * 50)
print("Columns in the BMW dataframe:")
print(list(bmw.columns))
print("-" * 50)
print("Columns in the Ford dataframe:")
print(list(ford.columns))
print("-" * 50)
print("Columns in the Hyundai dataframe:")
print(list(hyundai.columns))
print("-" * 50)
print("Columns in the Mercedes dataframe:")
print(list(mercedes.columns))
print("-" * 50)
print("Columns in the Skoda dataframe:")
print(list(skoda.columns))
print("-" * 50)
print("Columns in the Toyota dataframe:")
print(list(toyota.columns))
print("-" * 50)
print("Columns in the Vauxhall dataframe:")
print(list(vauxhall.columns))
print("-" * 50)
print("Columns in the VW dataframe:")
print(list(vw.columns))
hyundai.rename({'tax(£)': 'tax'},axis=1,inplace=True)
print(list(hyundai.columns))
audi['make'] = 'Audi'
bmw['make'] = 'BMW'
ford['make'] = 'Ford'
hyundai['make'] = 'Hyundai'
mercedes['make'] = 'Mercedes'
skoda['make'] = 'Skoda'
toyota['make'] = 'Toyota'
vauxhall['make'] = 'Vauxhall'
vw['make'] = 'Volkswagen'
df = pd.concat([audi, bmw, ford, hyundai, mercedes, skoda, toyota, vauxhall, vw], axis=0, ignore_index=True)
df.info()
df = df[['make','model','year','fuelType','mileage','engineSize','transmission','mpg','tax','price']]
df.head()
df.nunique(axis=0)
print('Unique values for "fuelType" column:', sorted(list(df['fuelType'].unique())))
print('Unique values for "transmission" column:', sorted(list(df['transmission'].unique())))

df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
df[df['year'] == 2060]
df = df.drop(df.index[39175])
df[df['year'] == 1970]
indexNames = df[df['year'] == 1970].index
df = df.drop(indexNames)
df[df['engineSize'] == 0]
len(df[df['engineSize'] == 0]) * 100 / len(df)
engineIndex = df[df['engineSize']==0].index
df = df.drop(engineIndex)
df[df['mileage']==1]
len(df[(df['mileage']==1) & (df['year']<= 2019)]) * 100 / len(df)
mileageIndex = df[(df['mileage']==1) & (df['year']<= 2019)].index
df = df.drop(mileageIndex)
df[df['tax'] == 0]
len(df[df['tax'] == 0]) * 100 / len(df)
taxindex = df[df['tax']==0].index
df = df.drop(taxindex)
df[df['mpg'] < 5]
df = df.drop(df[df['mpg']==0.3].index)
df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
df.isnull().sum()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
sns.scatterplot(x='mileage',y='price',data=df)
plt.title('Scatter plot of Mileage against Price')
sns.scatterplot(x='engineSize',y='price',data=df)
plt.title('Scatter Plot of Engine Size against Price')
sns.pairplot(df)
plt.figure(figsize=(10,8))
sns.boxplot(x='make',y='price',data=df)
df[(df['make'] == 'Hyundai') & (df['price'] > 80000)]
hyundai_error = df[(df['make'] == 'Hyundai') & (df['price'] > 80000)].index
df = df.drop(hyundai_error)
df[(df['make'] == 'Skoda') & (df['price']> 80000)]
skoda_error = df[(df['make'] == 'Skoda') & (df['price']> 80000)].index
df = df.drop(skoda_error)
sns.boxplot(x='fuelType',y='price',data=df)
sns.boxplot(x='transmission',y='price',data=df)
sns.distplot(df['price'],bins=50)
sns.distplot(df['year'],bins=50)
sns.distplot(df['mileage'],bins=50)
sns.distplot(df['tax'],bins=50)
sns.distplot(df['mpg'],bins=50)
sns.distplot(df['engineSize'],bins=50)
df['price'] = np.log(df['price'])
df['year'] = np.log(df['year'])
df['mileage'] = np.log(df['mileage'])
df['tax'] = np.log(df['tax'])
df['mpg'] = np.log(df['mpg'])
df['engineSize'] = np.log(df['engineSize'])
df = df.drop('make',axis=1)
transmission = pd.get_dummies(df['transmission'],drop_first=True)
model = pd.get_dummies(df['model'],drop_first=True)
fueltype = pd.get_dummies(df['fuelType'],drop_first=True)
df = pd.concat([df,transmission,model,fueltype],axis=1)
df = df.drop(['transmission','model','fuelType'],axis=1)
df.head()
X = df.drop('price',axis=1)
y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=101)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_preds = linreg.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
linreg_r2 = r2_score(np.exp(y_test),np.exp(linreg_preds))
linreg_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(linreg_preds)))
print("Linear Regression R2 Score: {}".format(linreg_r2))
print("Linear Regression RMSE: {}".format(linreg_RMSE))
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_preds = dtr.predict(X_test)
dtr_r2 = r2_score(np.exp(y_test),np.exp(dtr_preds))
dtr_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(dtr_preds)))
print("Decision Tree R2 Score: {}".format(dtr_r2))
print("Decision Tree RMSE: {}".format(dtr_RMSE))
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_preds = rfr.predict(X_test)
rfr_r2 = r2_score(np.exp(y_test),np.exp(rfr_preds))
rfr_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(rfr_preds)))
print("Random Forest R2 Score: {}".format(rfr_r2))
print("Random Forest RMSE: {}".format(rfr_RMSE))
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
svr_preds = svr.predict(X_test)
svr_r2 = r2_score(np.exp(y_test),np.exp(svr_preds))
svr_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(svr_preds)))
print("Support Vector Regression R2 Score: {}".format(svr_r2))
print("Support Vector Regression RMSE: {}".format(svr_RMSE))
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
mlp_r2 = r2_score(np.exp(y_test),np.exp(mlp_preds))
mlp_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(mlp_preds)))
print("MLP Regressor R2 Score: {}".format(mlp_r2))
print("MLP Regressor RMSE: {}".format(mlp_RMSE))
d = {'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Support Vector Regressor', 'MLP Regressor'],
    'R2 Score': [linreg_r2, dtr_r2, rfr_r2, svr_r2, mlp_r2],
    'RMSE': [linreg_RMSE, dtr_RMSE, rfr_RMSE, svr_RMSE, mlp_RMSE]}
results = pd.DataFrame(data=d)
results
plt.figure(figsize=(10,6))
sns.barplot(x='Model',y='R2 Score',data=results,order=['Support Vector Regressor', 'Linear Regression', 'MLP Regressor','Decision Tree','Random Forest'])
plt.title('R2 Score for Each Model')
sns.scatterplot(x='R2 Score',y='RMSE',data=results,hue='Model')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
s_dtr = DecisionTreeRegressor()
s_dtr.fit(X_train,y_train)
s_dtr_preds = s_dtr.predict(X_test)
s_dtr_r2 = r2_score(np.exp(y_test),np.exp(s_dtr_preds))
s_dtr_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(s_dtr_preds)))
print("Scaled Decision Tree R2 Score: {}".format(s_dtr_r2))
print("Scaled Decision Tree RMSE: {}".format(s_dtr_RMSE))
s_rfr = RandomForestRegressor()
s_rfr.fit(X_train, y_train)
s_rfr_preds = s_rfr.predict(X_test)
s_rfr_r2 = r2_score(np.exp(y_test),np.exp(s_rfr_preds))
s_rfr_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(s_rfr_preds)))
print("Scaled Random Forest R2 Score: {}".format(s_rfr_r2))
print("Scaled Random Forest RMSE: {}".format(s_rfr_RMSE))
s_svr = SVR()
s_svr.fit(X_train, y_train)
s_svr_preds = s_svr.predict(X_test)
s_svr_r2 = r2_score(np.exp(y_test),np.exp(s_svr_preds))
s_svr_RMSE = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(s_svr_preds)))
print("Scaled Support Vector Regression R2 Score: {}".format(s_svr_r2))
print("Scaled Support Vector Regression RMSE: {}".format(s_svr_RMSE))
d = {'Model': ['Scaled Decision Tree', 'Scaled Random Forest', 'Scaled Support Vector Regressor'],
    'R2 Score': [s_dtr_r2,s_rfr_r2,s_svr_r2],
    'RMSE': [s_dtr_RMSE,s_rfr_RMSE,s_svr_RMSE]}
scaled_results = pd.DataFrame(d)
scaled_results
full_results = pd.concat([results,scaled_results],ignore_index=True)
full_results
