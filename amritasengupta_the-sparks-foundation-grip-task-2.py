import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
df= pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df.head()
df.tail()
df.isnull().sum()
df.duplicated().sum()
df.info()
df.hist('Scores')
df.describe()
plt.scatter(x= df['Hours'],y=df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
df[(df['Hours']>8) & (df['Scores']<80)]
df[(df['Hours']>8) & (df['Scores']<82)]
df['Scores'][(df['Hours']>8) & (df['Scores']<80)]=81
plt.scatter(x= df['Hours'],y=df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
Predictor=['Hours']
TargetVariable=['Scores']
x = df[Predictor].values
y = df[TargetVariable].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=10)
lr.fit(x_train, y_train)
train_prediction= lr.predict(x_train)

print('The value of slope is: ',lr.coef_)
print('The value of Intercept is: ',lr.intercept_)

print('r-squared for training data:',r2_score(train_prediction , y_train))
test_prediction=lr.predict(x_test)
print('r-squared for testing data:',r2_score(test_prediction , y_test))

print('Accuracy (MAPE): ',100- (np.mean(np.abs((y_test - test_prediction) / y_test)) * 100))
import copy
df1=copy.deepcopy(df)
predicted_score=lr.predict(x)
df1['Predicted_Score']=predicted_score
plt.scatter(x=df1['Hours'] , y=df1['Scores'])
plt.plot(df1['Hours'] , df1['Predicted_Score'],color='red')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('BEST FIT LINE')
knr.fit( x_train, y_train )
train_prediction = knr.predict(x_train)

print('r-squared for training data:',r2_score(train_prediction , y_train))
test_prediction=knr.predict(x_test)
print('r-squared for testing data:',r2_score(test_prediction , y_test))

print('Accuracy (MAPE): ',100- (np.mean(np.abs((y_test - test_prediction) / y_test)) * 100))
from sklearn.preprocessing import MinMaxScaler
PredictorScaler=MinMaxScaler()
TargetVarScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(x)
TargetVarScalerFit=TargetVarScaler.fit(y)

# Generating the normalized values of X and y
x_normal=PredictorScalerFit.transform(x)
y_normal=TargetVarScalerFit.transform(y)


# Split the data into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_normal, y_normal, test_size=0.3, random_state=42)
model= lr.fit(x_train, y_train)
train_prediction= model.predict(x_train)

print('The value of slope is: ',model.coef_)
print('The value of Intercept is: ',model.intercept_)

print('r-squared for training data:',r2_score(train_prediction , y_train))
test_prediction=model.predict(x_test)
print('r-squared for testing data:',r2_score(test_prediction , y_test))

print('Accuracy (MAPE): ',100- (np.mean(np.abs((y_test - test_prediction) / y_test)) * 100))
knr.fit( x_train, y_train )
train_prediction = knr.predict(x_train)

print('r-squared for training data:',r2_score(train_prediction , y_train))
test_prediction=knr.predict(x_test)
print('r-squared for testing data:',r2_score(test_prediction , y_test))

print('Accuracy (MAPE): ',100- (np.mean(np.abs((y_test - test_prediction) / y_test)) * 100))
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(x)
TargetVarScalerFit=TargetVarScaler.fit(y)

# Generating the standardized values of X and y
x_standard=PredictorScalerFit.transform(x)
y_standard=TargetVarScalerFit.transform(y)


# Split the data into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_standard, y_standard, test_size=0.3, random_state=42)

model= lr.fit(x_train, y_train)
train_prediction= model.predict(x_train)

print('The value of slope is: ',model.coef_)
print('The value of Intercept is: ',model.intercept_)

print('r-squared for training data:',r2_score(train_prediction , y_train))
test_prediction=model.predict(x_test)
print('r-squared for testing data:',r2_score(test_prediction , y_test))

print('Accuracy (MAPE): ',100- (np.mean(np.abs((y_test - test_prediction) / y_test)) * 100))
knr.fit( x_train, y_train )
train_prediction = knr.predict(x_train)

print('r-squared for training data:',r2_score(train_prediction , y_train))
test_prediction=knr.predict(x_test)
print('r-squared for testing data:',r2_score(test_prediction , y_test))

print('Accuracy (MAPE): ',100- (np.mean(np.abs((y_test - test_prediction) / y_test)) * 100))
