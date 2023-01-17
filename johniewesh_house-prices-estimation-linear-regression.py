import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import pandas as pd

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test.head()
train.head()
#drop all the unstructured data
test1=test.drop(columns=[f for f in test.columns if test[f].dtype =='object'])
#check for missing values
missing=test1.isnull().sum()
missing
#drop collumns with too many missing values
#drop collumns with too many missing values
x_test1=test1.drop(['LotFrontage','MasVnrArea','GarageYrBlt','Id'], axis=1)
#for columns with a few missing values replace with mean
x_test=x_test1.fillna(x_test1.mean())
x_test.head()
#drop all the text variables data
train1=train.drop(columns=[f for f in train.columns if train[f].dtype =='object'])
missing1=train1.isnull().sum()
missing1
y=train1.drop(['LotFrontage','MasVnrArea','GarageYrBlt','Id'], axis=1)
y.head()
## drop the indepedent variable in the set of predictors
x_train=y.drop(['SalePrice'], axis=1)
x_train.head()
##set the depedent variable as the droped
y_train= y.SalePrice
##define the model
model=LinearRegression()
model.fit(x_train, y_train)

predictions=model.predict(x_test)
print(predictions)
##convert to predictiosn to pandas dataframe
df = pd.DataFrame(predictions)
 ##save the predictions in a csv
df.to_csv('C:\\Users\\Wesh\\Desktop\\SalePrice.csv',sep=',')
