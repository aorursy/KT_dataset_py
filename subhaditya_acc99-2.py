import pandas as pd

from sklearn.model_selection import cross_val_score
data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
data.head()
data.describe()
data['x']==0
data = data[(data[['x','y','z']] != 0).all(axis=1)]
for i in data.columns:

    print(i,sum(data[i].isna()))
# z / mean(x, y)

data['volume'] = data['x']*data['y']*data['z']

data['area'] = data['x']*data['y']

data['priceunvol'] = data['price']/data['volume']
data = data.drop(['Unnamed: 0','x','y','z'],axis = 1)

# data = data.drop(['Unnamed: 0'],axis = 1)
data.head()
data['table'].unique()


np.where(data['volume'].values )

data =  pd.get_dummies(data)

data.head()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



# data_fitter =  pd.DataFrame(sc.fit_transform(data[['carat','depth','volume','table']]),columns=['carat','depth','volume','table'],index=data.index)



# data_fitter =  pd.DataFrame(sc.fit_transform(data[['carat','depth','volume','table','area']]),columns=['carat','depth','volume','table','area'],index=data.index)



data_fitter =  pd.DataFrame(sc.fit_transform(data[['carat','depth','volume','table','area','priceunvol']]),columns=['carat','depth','volume','table','area','priceunvol'],index=data.index)
data2 = data.copy(deep=True)

# data2[['carat','depth','volume','table']] = data_fitter[['carat','depth','volume','table']]

# data2[['carat','depth','volume','table','area']] = data_fitter[['carat','depth','volume','table','area']]

data2[['carat','depth','volume','table','area','priceunvol']] = data_fitter[['carat','depth','volume','table','area','priceunvol']]
data2.head()


from sklearn.model_selection import train_test_split

x = data2.drop(["price"],axis=1)

y = data2.price

train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 2,test_size=0.3)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn import linear_model



regr = linear_model.LinearRegression()

regr.fit(train_x,train_y)

y_pred = regr.predict(test_x)

print("accuracy: "+ str(regr.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1

adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
from sklearn.ensemble import AdaBoostRegressor 



clf_rf = AdaBoostRegressor()

clf_rf.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_rf, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred2 = clf_rf.predict(test_x)

print('Score : %.4f' % clf_rf.score(test_x, test_y))

mse = mean_squared_error(test_y, y_pred2)

mae = mean_absolute_error(test_y, y_pred2)

rmse = mean_squared_error(test_y, y_pred2)**0.5

r2 = r2_score(test_y, y_pred2)

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)
from sklearn.ensemble import RandomForestRegressor

clf_rf = RandomForestRegressor()

clf_rf.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_rf, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred2 = clf_rf.predict(test_x)

print('Score : %.4f' % clf_rf.score(test_x, test_y))

mse = mean_squared_error(test_y, y_pred2)

mae = mean_absolute_error(test_y, y_pred2)

rmse = mean_squared_error(test_y, y_pred2)**0.5

r2 = r2_score(test_y, y_pred2)

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)
from tabulate import tabulate

headers = ["name", "score"]

values = sorted(zip(train_x.columns, clf_rf.feature_importances_), key=lambda x: x[1] * -1)

print(tabulate(values, headers, tablefmt="plain"))
from sklearn.ensemble import GradientBoostingRegressor



clf_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls',verbose = 1)

clf_gbr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_gbr, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred2 = clf_gbr.predict(test_x)

print('Score : %.4f' % clf_gbr.score(test_x, test_y))

mse = mean_squared_error(test_y, y_pred2)

mae = mean_absolute_error(test_y, y_pred2)

rmse = mean_squared_error(test_y, y_pred2)**0.5

r2 = r2_score(test_y, y_pred2)

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)