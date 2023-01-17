# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Reading the csv file:
train=pd.read_csv("../input/training-set/Train_A102.csv")
test=pd.read_csv("../input/test-a102csv/Test_A102 (1).csv")
print(train.head())
print(test.head())


#Data stucture of the dataframe:
train.info()

#Converting dtypes from object to float:
#train[0:]=train[0:].astype(float)
#train[11]=train[11].astype(float)
print(train.dtypes)


#Summary of the dataset:
train.describe(include='all')
#print(train.describe())
#Dimensions of the dataset:
print(train.shape)
#Data stucture of the dataframe:
test.info()

#Converting dtypes from object to float:
#train[0:]=train[0:].astype(float)
#train[11]=train[11].astype(float)
print(test.dtypes)

#Summary of the dataset:
test.describe(include='all')
#print(train.describe())


#Dimensions of the dataset:
print(test.shape)

#Adding the missing last column in the test dataset:
test['Item_Outlet_Sales']= np.nan

print(test.head())

#Appending both the datasets:
combined_set=train.append(test)
print(combined_set.head())

#Summary of the combined datsets:
combined_set.describe(include='all')

print(combined_set.shape)

#Maximum  values:
combined_set.max()

#Minimum values of combined set:
print(combined_set.min())
#Replacing the missing values with the categorical variable:
combined_set.replace('',np.NaN)
#print(train.head())

#creating a copy of the combined set:
combined_meanset=combined_set.copy(deep=True)


#Replacing the categorical variable by the mean value:
combined_meanset['Item_Weight'].fillna(value=12.792854 , inplace=True)
print(combined_meanset.head(10))

#creating a copy of the combined set:
combined_medianset=combined_set.copy(deep=True)

#Replacing the categorical variable by the median value:
combined_medianset['Item_Weight'].fillna(value=12.6 , inplace=True)
#combined_medianset=['Outlet_Size'].fillna(value='Medium', inplace=True)
print(combined_medianset.head(10))
print(combined_set.head(10))

#plotting histograms:
combined_medianset['Item_Weight'].hist(color='white', edgecolor='black')
plt.title("Median Histogram")
plt.xlabel("X-axis")
plt.ylabel("Item_Weight")
plt.show()

#Plottinh histogram:
combined_meanset['Item_Weight'].hist(color='white', edgecolor='black')
plt.title("Mean Histogram")
plt.xlabel("X-axis")
plt.ylabel("Item_Weight")
plt.show()
#Plotting histogram:
combined_set['Item_Weight'].hist(color='white', edgecolor='black')
plt.title("True valued Histogram")
plt.xlabel("X-axis")
plt.ylabel("Item_Weight")
plt.show()
#Fianlly filling the nan values with the median value: 
combined_set['Item_Weight'].fillna(value=12.6 , inplace=True)
combined_set.head()
#Scatter plot:
train.plot.density()
plt.show()

#Plotting the boxplot:
 
combined_set.boxplot()
plt.plot()

combined_set.boxplot("Item_Visibility", figsize=(12,8))
plt.plot()
plt.show()

combined_set.boxplot("Outlet_Establishment_Year" , figsize=(12,8))
plt.plot()


plt.show()

#Filling the misiing values in the categorical column of Outlet_Size:

#Grouping the Outlet_Size with other categorical columns:
print(combined_set.groupby('Item_Fat_Content').Outlet_Size.value_counts(dropna=False))
print(combined_set.groupby('Item_Type').Outlet_Size.value_counts(dropna=False))
print(combined_set.groupby('Outlet_Identifier').Outlet_Size.value_counts(dropna=False))
print(combined_set.groupby('Outlet_Type').Outlet_Size.value_counts(dropna=False))
print(combined_set.groupby('Outlet_Location_Type').Outlet_Size.value_counts(dropna=False))

#We observe that the nan values when different categorical columns are grouped with the Outlet_Size column,
#shows a lot of variance, hence we subsitute the Nan values with the mode value:

combined_set['Outlet_Size'].fillna(value='Medium' , inplace=True)
print(combined_set.head())

#Plotting histograms:
combined_set['Outlet_Size'].hist(color='white' , edgecolor='blue')
plt.title("Histogram of Outlet_Size")
plt.xlabel("X-axis")
plt.ylabel("Outlet_Size")
#Replacing the nan values in the last column with any random value:
combined_set['Item_Outlet_Sales'].fillna(value=-999, inplace=True)
print(combined_set.tail())


#New Features:
print(combined_set.shape[0])
abcd=[100000000]
i=0
for i in range(combined_set.shape[0]):
    if (combined_set.iloc[i, 5] <70).any():
        abcd.append('low')
        #print(combined_set.head())
        
        
    elif (combined_set.iloc[i,5]<130).any():
         abcd.append('Medium') 
    
    else:
        abcd.append('high')
        
    i=i+1

array=np.array(abcd[1:])
combined_set['Item_MRP']=array
print(combined_set.head())

#combined_set.drop('Item_MRP', axis=1, inplace=True)
print(combined_set.shape)


#Outliers identification:Feature having outlier is Item_Output_Sales
X=train.iloc[:,0:].values
item_outlet_sales=X[:,11]
Outliers=(item_outlet_sales>6501)
print(train[Outliers])

#Data cleaning in the column Item_Fat_Content:
print(combined_set['Item_Fat_Content'].value_counts())
combined_set['Item_Fat_Content']=combined_set['Item_Fat_Content'].replace({'low fat' : 'LF', 'Low Fat' : 'LF', 'Regular' : 'reg'})
print(combined_set.head())


#Beginning with basic model: Replacing the missing values or random values in the Outlet_Sales by mean of the column:

#creating a copy of the combined set:
combined_mean_modelset=combined_set.copy(deep=True)


#Replacing by mean value:
combined_mean_modelset['Item_Outlet_Sales']=combined_mean_modelset['Item_Outlet_Sales'].replace(to_replace=-999, value=2181.3)
print(combined_mean_modelset.tail())

combined_mean_modelset.tail()
#creating a copy:
combined_median_modelset=combined_set.copy(deep=True)

#Replacing by median value:
combined_median_modelset['Item_Outlet_Sales']=combined_median_modelset['Item_Outlet_Sales'].replace(to_replace=-999, value=1794.3)
print(combined_median_modelset.tail())
combined_median_modelset.tail()
#Extracting the train and test data sets:
train_new=combined_mean_modelset[0:8523]
print(train_new.tail())

test_new=combined_mean_modelset[8523:]
print(test_new.tail())

train_new.shape
test_new.shape
combined_mean_modelset.tail()
#Importing the datasets:
X=combined_mean_modelset.iloc[:, 1:11].values
print(X)
Y=combined_mean_modelset.iloc[:,11].values
print(Y)
#X[10]=X[10].astype('object')

X=pd.DataFrame(X)
X.head()
#X.dtypes
X.dtypes

X.iloc[:,1]=X.iloc[:,1].astype('object')
X.iloc[:,0]=X.iloc[:,0].astype(float)
X.iloc[:,2]=X.iloc[:,2].astype(float)
#X.iloc[:,4]=X.iloc[:,4].astype(float)
X.iloc[:,6]=X.iloc[:,6].astype(float)
X.dtypes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

labelencoder_X=LabelEncoder()
X.iloc[:,1]=labelencoder_X.fit_transform(X.iloc[:,1])
#onehotencoder=OneHotEncoder(categorical_features=[0])
X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:,1])
X_extra.head()
#Adding them to X:
frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
X.head()
X.drop(1)
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

labelencoder_X=LabelEncoder()
X.iloc[:,3]=labelencoder_X.fit_transform(X.iloc[:,3])
#onehotencoder=OneHotEncoder(categorical_features=[0])
X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:,3])
X_extra.head()
frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

#labelencoder_X=LabelEncoder()
X.iloc[:,4]=labelencoder_X.fit_transform(X.iloc[:,4])
#onehotencoder=OneHotEncoder(categorical_features=[0])
#X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:, 4])
X_extra.head()
frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

#labelencoder_X=LabelEncoder()
X.iloc[:,5]=labelencoder_X.fit_transform(X.iloc[:,5])
#onehotencoder=OneHotEncoder(categorical_features=[0])
#X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:, 5])
X_extra.head()
frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

#labelencoder_X=LabelEncoder()
X.iloc[:,7]=labelencoder_X.fit_transform(X.iloc[:,7])
#onehotencoder=OneHotEncoder(categorical_features=[0])
#X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:, 7])
X_extra.head()

frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

#labelencoder_X=LabelEncoder()
X.iloc[:,8]=labelencoder_X.fit_transform(X.iloc[:,8])
#onehotencoder=OneHotEncoder(categorical_features=[0])
#X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:, 8])
X_extra.head()
frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding categorical variable

#labelencoder_X=LabelEncoder()
X.iloc[:,9]=labelencoder_X.fit_transform(X.iloc[:,9])
#onehotencoder=OneHotEncoder(categorical_features=[0])
#X=pd.DataFrame(X)
X_extra=pd.get_dummies(X.iloc[:, 9])
X_extra.head()
frames=[X,X_extra]
X=pd.concat(frames,axis=1)
X.head()
#Splitting into test and training datasets:
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=.399957758, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(X_train.head())
Y_train
#Fitting the linear Regression Model:
from sklearn.linear_model import LinearRegression
linreg=LinearRegression(normalize=True)
linreg.fit(X_train,Y_train)

#CHECHING THE VALIDITY OF ASSUMPTIONS:
#Scatter plot matrix:Linear or any other polynomial relationship existence
sns.pairplot(combined_mean_modelset)
sns.despine()
#All the data is continuous numeric and not categorical.
#Correlations between the different columns of the dataframe:
print(X.corr())
#Hence there is no such correaltion in between the predictor variables
#Checking for missing values in the dataset:
X_train.iloc[:,0:].values
#Hence there doesn't exists any missing values as it returns nothing.
#Checking for Outliers:
train_new.boxplot()
plt.plot()
#Since we can see the column Item_visibility has outliers all lying to the right(above), removing them will remove the majority of the data hence ignored.
#Multiple R square Value:
from sklearn.linear_model import LinearRegression
linreg=LinearRegression(normalize=True)
model=linreg.fit(X_train,Y_train)
print(linreg.score(X_train,Y_train))

X_train.shape

Y_train.shape

Y=pd.DataFrame(Y)
Y.head()
#Predicting the values of y from the model:
Y_pred=linreg.predict(X_train)
Y.shape
res=Y_pred - Y_train
res
#Residual vs fitted plot:To check whether linear model or not; check for heteroska
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
sns.residplot(Y_pred,res, lowess=True, color="g")
sns.despine()

## The plot of linear model
plt.scatter(Y_train, Y_pred)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("Linear Model")
combined_mean_modelset.dtypes

train_new.dtypes
combined_mean_modelset.head()
#Normal Q-Q plot:
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data=combined_mean_modelset.iloc[:,1].values.flatten()
data.sort()
norm=random.normal(0,2,len(data))
norm.sort()
plt.figure(figsize=(12,8),facecolor='1.0') 

plt.plot(norm,data,"o")

#generate a trend line as in http://widu.tumblr.com/post/43624347354/matplotlib-trendline
z = np.polyfit(norm,data, 1)
p = np.poly1d(z)
plt.plot(norm,p(norm),"k--", linewidth=2)
plt.title("Normal Q-Q plot", size=28)
plt.xlabel("Theoretical quantiles", size=24)
plt.ylabel("Expreimental quantiles", size=24)
plt.tick_params(labelsize=16)
plt.show()

#Normal Q-Q plot:
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data=combined_mean_modelset.iloc[:,3].values.flatten()
data.sort()
norm=random.normal(0,2,len(data))
norm.sort()
plt.figure(figsize=(12,8),facecolor='1.0') 

plt.plot(norm,data,"o")

#generate a trend line as in http://widu.tumblr.com/post/43624347354/matplotlib-trendline
z = np.polyfit(norm,data, 1)
p = np.poly1d(z)
plt.plot(norm,p(norm),"k--", linewidth=2)
plt.title("Normal Q-Q plot", size=28)
plt.xlabel("Theoretical quantiles", size=24)
plt.ylabel("Expreimental quantiles", size=24)
plt.tick_params(labelsize=16)
plt.show()
#Normal Q-Q plot:
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data=combined_mean_modelset.iloc[:,7].values.flatten()
data.sort()
norm=random.normal(0,2,len(data))
norm.sort()
plt.figure(figsize=(12,8),facecolor='1.0') 

plt.plot(norm,data,"o")

#generate a trend line as in http://widu.tumblr.com/post/43624347354/matplotlib-trendline
z = np.polyfit(norm,data, 1)
p = np.poly1d(z)
plt.plot(norm,p(norm),"k--", linewidth=2)
plt.title("Normal Q-Q plot", size=28)
plt.xlabel("Theoretical quantiles", size=24)
plt.ylabel("Expreimental quantiles", size=24)
plt.tick_params(labelsize=16)
plt.show()
#Error function:
Error_function=Y_pred - Y_train
Error_function
#Cross validation using K-folds validation with k+10 folds
from sklearn.model_selection import cross_val_predict, cross_val_score
ypred=cross_val_predict(linreg, X_train, Y_train, cv=10)
print(ypred)
#The plot of cross validation
fig, ax = plt.subplots()
ax.scatter(Y_train, ypred, edgecolors=(0, 0, 0))
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('Cross validated plot')
plt.show()
#Evaluating the multiple R scored of the cross validated model:
from sklearn import metrics
accuracy = metrics.r2_score(Y_train, ypred)
accuracy
#Evaluating the scoes of the corss validated model:
scores=cross_val_score(model, X, Y,cv=10)
scores
#Application of Ridge regression:
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=.05,normalize=True)
ridgereg.fit(X_train, Y_train)
   
#Predicted values of Y in the ridge model:
y_pred = ridgereg.predict(X_train)
y_pred
#Fitting the training set in the ridge model
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=.05,normalize=True)
ridgereg.fit(X_train, Y_train)
print(ridgereg.score(X_train, Y_train))
#evaluating the RMSE value of ridge model:
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_train, y_pred))
Y_train.shape
y_pred.shape
## The plot of ridge model
plt.scatter(Y_train, y_pred)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("Ridge Model")
#Applying the Lasso Rigression:
from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=.05,normalize=True, max_iter=1e5)
lassoreg.fit(X_train, Y_train)
#predicted values on the dependent variable Y in Lasso model:
y_predict = lassoreg.predict(X_train)
y_predict

#Fitting the training set in the lasso model:
from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=.05,normalize=True)
lassoreg.fit(X_train, Y_train)
print(lassoreg.score(X_train, Y_train))
#Calculating the RMSE value of the lasso model:
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_train, y_predict))
#Plot of lasso model:
plt.scatter(Y_train, y_predict)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("lasso Model")
#We observed that RMSE value is lower for ridge model: Hence we apply the lasso model on the test set to predict the values
from sklearn.linear_model import Ridge
ridgereg = Lasso(alpha=.05,normalize=True, max_iter=1e5)
ridgereg.fit(X_test, Y_test)
#Preducted values of the dependent variable Y in the test dataset
Ypred = ridgereg.predict(X_test)
Ypred
#Evaluating the score of the test dataset in the lasso model
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=.05,normalize=True)
ridgereg.fit(X_test, Y_test)
print(ridgereg.score(X_test, Y_test))
#The RMSE value of the test dataset:
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, Ypred))
#Residual values of the test dataset:
residual=Ypred - Y_test
residual
Ypred.shape

#Plot of the ridge on test model:
plt.scatter(Y_test, Ypred)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("Ridge Model")
test_new.head()
test_new['Item_Outlet_Sales']=Ypred
test_new.head()
pd.DataFrame(test_new, columns=['Item_Identifier' ,'Outlet_Identifier', 'Item_Outlet_Sales']).to_csv('prediction.csv')