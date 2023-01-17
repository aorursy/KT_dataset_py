import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/Melbourne_housing_FULL.csv')
data.head().columns
data.head(2)
data.describe().T
data.shape
data.isnull().sum()
data.Price.isnull().sum()
7610/34857*100
data.Price.describe().T
data.Price.median()
data.loc[data.Price.isnull(),'Price'] = 1050173
data.drop(data[data.Postcode.isnull()].index, inplace=True)
data.drop(data[data.CouncilArea.isnull()].index, inplace=True)
data.drop(data[data.Regionname.isnull()].index, inplace=True)
data.drop(data[data.Propertycount.isnull()].index, inplace=True)
data.mode().T
data.Bedroom2.fillna(data.Bedroom2.mode()[0], inplace=True)
data.Bathroom.fillna(data.Bathroom.mode()[0], inplace=True)
data.Car.fillna(data.Car.mode()[0], inplace=True)
data.YearBuilt.fillna(data.YearBuilt.mode()[0], inplace=True)
data.Bedroom2.mode()[0]
data.Lattitude.fillna(data.Lattitude.mean(), inplace=True)
data.Longtitude.fillna(data.Longtitude.mean(), inplace=True)
data.Landsize.fillna(data.Landsize.mean(), inplace=True)
data.BuildingArea.fillna(data.BuildingArea.mean(), inplace=True)

plt.figure(figsize=(10,9))
corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0,  annot= True, linewidths=0.1 )
data[['Rooms','Bedroom2']].corr()
data.plot(y='Rooms',x='Price',kind='scatter',figsize=(6,5))
data.plot(y='Distance',x='Price',kind='scatter',figsize=(6,5))
data.plot(y='Car',x='Price',kind='scatter',figsize=(6,5))
data.Suburb.value_counts().head(10).unique
def formatting_columns(x):
    x[0] = '{0:.2f}'.format(x[0])
    x[1] = '{0:.0f}'.format(x[1])
    x[2] = '{0:.0f}'.format(x[2])
    x[3] = '{0:.0f}'.format(x[3])
    return x
data['Price'].groupby(data.Suburb).agg(['mean','count','max','min']).sort_values(by='mean',ascending=False).apply(formatting_columns,axis=1).head(20)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
LR = LinearRegression() 
data.describe(include=['O']).T #To select all numeric types use numpy numpy.number. To select categorical objects use type object
data["Date"] = pd.to_datetime(data["Date"],dayfirst=True)
from datetime import date
dataframe_dr = data.dropna().sort_values("Date")
all_Data = []
##Find out days since start
days_since_start = [(x - dataframe_dr["Date"].min()).days for x in dataframe_dr["Date"]]
dataframe_dr["Days"] = days_since_start
suburb_dummies = pd.get_dummies(dataframe_dr[["Type", "Method"]])
all_Data = dataframe_dr.drop(["Address","Price","Date", "SellerG","Suburb","Type","Method","CouncilArea","Regionname"],axis=1).join(suburb_dummies)
X = all_Data
y = dataframe_dr["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
LR.fit(X_train,y_train)
print(LR.intercept_)
predictions = LR.predict(X_test)
plt.scatter(y_test, predictions)

y = data.Price
X = data[['Rooms','Distance','Bedroom2','Car','Bathroom','Lattitude','Longtitude','Postcode','Landsize','BuildingArea']]
predicted = cross_val_predict(LR, X.values, y.values, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
# prepare the train and test data.
Col_Rooms_Bedrooms2 = data.loc[:,('Rooms','Bedroom2')]
train, test = train_test_split(Col_Rooms_Bedrooms2, test_size=0.2, random_state=42) #test_size = 20% in data
x_train = pd.DataFrame(train.Rooms)
y_train = train.Bedroom2
x_test = pd.DataFrame(test.Rooms)
y_test = test.Bedroom2
# Train the model
LR.fit(x_train, y_train)
# print result
c = LR.coef_[0]
i = LR.intercept_
print("For x is {} and y is {}".format('Rooms', 'Bedroom2'))
print('=> Formula:','y = {}x + {}'.format(c, i) if i>=0 else 'y={}x - {}'.formar(c,i))
from sklearn.metrics import mean_squared_error
# predict the Bedrooms from Rooms in test sample and get Mean Squared Error.
y_predict = LR.predict(x_test)
print("Mean squared error is: %.4f"%mean_squared_error(y_test, y_predict))
y_Datapredict = LR.predict(data.Bedroom2.values.reshape(data.shape[0],1))
x_Datapredict = LR.predict(data.Rooms.values.reshape(data.shape[0],1))
print("Mean squared error is: %.4f"%mean_squared_error(x_Datapredict, y_Datapredict))
Col_Rooms_Bedrooms2.Rooms= Col_Rooms_Bedrooms2.Rooms.astype(dtype='int64')
Col_Rooms_Bedrooms2.Bedroom2= Col_Rooms_Bedrooms2.Bedroom2.astype(dtype='int64')
x, y = pd.Series(Col_Rooms_Bedrooms2.Rooms, name="x_var"), pd.Series(Col_Rooms_Bedrooms2.Bedroom2, name="y_var")
ax = sns.regplot(x=x, y=y, marker="*")
from sklearn.model_selection import cross_val_predict
y = Col_Rooms_Bedrooms2.Rooms.values.reshape(data.shape[0],1)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(LR, Col_Rooms_Bedrooms2.Bedroom2.values.reshape(data.shape[0],1), y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
from sklearn.decomposition import PCA
numeric_data=data.loc[:,('Rooms','Bedroom2')]
pca = PCA(n_components=2,svd_solver='full') 
pca.fit(numeric_data)
print (pca.explained_variance_)  #parameter returns a vector of the variance explained by each dimension
print (pca.explained_variance_ratio_) #gives the variance explained solely by the i+1st dimension.
print (pca.explained_variance_ratio_.cumsum()) # return a vector x such that x[i] returns the cumulative variance explained by the first i+1 dimensions
pca_arpack = PCA(n_components=1,svd_solver='arpack') 
pca_arpack.fit(numeric_data)
print (pca_arpack.explained_variance_)  #parameter returns a vector of the variance explained by each dimension
print (pca_arpack.explained_variance_ratio_) #gives the variance explained solely by the i+1st dimension.
print (pca_arpack.explained_variance_ratio_.cumsum()) # return a vector x such that x[i] returns the cumulative variance explained by the first i+1 dimensions
train, test = train_test_split(numeric_data, test_size=0.2, random_state=42) #test_size = 20% in data
x_train = pd.DataFrame(train.Rooms)
y_train = train.Bedroom2
x_test = pd.DataFrame(test.Rooms)
y_test = test.Bedroom2
# Train the model
LR.fit(x_train, y_train)
# print result
c = LR.coef_[0]
i = LR.intercept_
print("For x is {} and y is {}".format('Rooms', 'Bedroom2'))
print('=> Formula:','y = {}x + {}'.format(c, i) if i>=0 else 'y={}x - {}'.formar(c,i))
# predict the Bedrooms from Rooms in test sample and get Mean Squared Error.
y_predict = LR.predict(x_test)
print("Mean squared error is: %.4f"%mean_squared_error(y_test, y_predict))
y_Datapredict = LR.predict(numeric_data.Bedroom2.values.reshape(data.shape[0],1))
x_Datapredict = LR.predict(numeric_data.Rooms.values.reshape(data.shape[0],1))
print("Mean squared error is: %.4f"%mean_squared_error(x_Datapredict, y_Datapredict))
numeric_data.Rooms= numeric_data.Rooms.astype(dtype='int64')
numeric_data.Bedroom2= numeric_data.Bedroom2.astype(dtype='int64')
x, y = pd.Series(numeric_data.Rooms, name="x_var"), pd.Series(numeric_data.Bedroom2, name="y_var")
ax = sns.regplot(x=x, y=y, marker="*")
y = numeric_data.Rooms.values.reshape(data.shape[0],1)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(LR, numeric_data.Bedroom2.values.reshape(data.shape[0],1), y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
sns.lmplot(x="Rooms", y="Bedroom2",data=Col_Rooms_Bedrooms2,x_estimator=np.mean)
sns.lmplot(x="Rooms", y="Bedroom2",data=Col_Rooms_Bedrooms2,x_estimator=np.mean,hue="Rooms")
X = Col_Rooms_Bedrooms2.Rooms.values.reshape(Col_Rooms_Bedrooms2.shape[0],1)
y = Col_Rooms_Bedrooms2.Bedroom2
predict = LR.predict(X)
a = (y.values - predict)
fig,ax = plt.subplots()
ax.scatter(Col_Rooms_Bedrooms2.Rooms.values, a)
ax.set_xlabel('Rooms')
ax.set_ylabel('Residual')
plt.show()
sns.residplot(x=X,y=y,data=Col_Rooms_Bedrooms2)









