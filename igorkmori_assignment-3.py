import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
test =pd.read_csv('../input/test.csv')
train =pd.read_csv('../input/train.csv')
test.shape
train.shape
train.SalePrice.describe()
target = train.SalePrice
numeric_features = train.select_dtypes(include=[np.number])
corr= numeric_features.corr()
corr['SalePrice'].head()
plt.scatter(x=train['GrLivArea'],y=target)
# handle null values and no float values 
nulls =pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns =['Nulls']
nulls.index.name ='Feature'
data =train.select_dtypes(include=[np.number]).interpolate().dropna()
#Define X = data and y = target

y= np.log(train.SalePrice)
X=data.drop(['SalePrice','Id'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=33)
# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
linreg.score(X_test,y_test)
submission = pd.DataFrame()
submission['Id'] = test.Id
f =test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()
predictions = linreg.predict(f)
final_pred =np.exp(predictions) #back to real prices
submission['SalePrice'] =final_pred
submission.head()
submission.to_csv('submission.csv',index=False)
