import  numpy as np

import pandas as pd

test=pd.read_csv('../input/test.csv')

test
ss=pd.read_csv('../input/sample_submission.csv')

ss
Merge=pd.merge(test,ss,on="Id" , how="inner")

Merge
train=pd.read_csv('../input/train.csv')

train
final=pd.concat([train,Merge])

final=final.reset_index()

final
final.info()

final.isnull().sum()

final=final.drop(['LotFrontage','Alley','KitchenQual','Exterior1st','GarageCars','GarageArea','Functional','SaleType','Exterior2nd','MasVnrType','MasVnrArea','FireplaceQu','GarageType','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'],axis=1)

final.head(1)
final.columns
X=final[['LotArea','Street','Neighborhood','HouseStyle','OverallCond','RoofStyle','Electrical','SaleCondition']]

Y=final['SalePrice']

X.info()

X['Electrical']=X['Electrical'].fillna('SBrkr')
X.head(1)
from sklearn.preprocessing import LabelEncoder

X_labelencoder=LabelEncoder()

X.iloc[:,1]=X_labelencoder.fit_transform(X.iloc[:,1])

X.iloc[:,2]=X_labelencoder.fit_transform(X.iloc[:,2])

X.iloc[:,3]=X_labelencoder.fit_transform(X.iloc[:,3])

X.iloc[:,5]=X_labelencoder.fit_transform(X.iloc[:,5])

X.iloc[:,6]=X_labelencoder.fit_transform(X.iloc[:,6])

X.iloc[:,7]=X_labelencoder.fit_transform(X.iloc[:,7])

print(X)

print(Y)
X.isnull().sum()
X_train=X.iloc[:1460,:]

y_train=Y.iloc[:1460]

X_test=X.loc[1460:,:]

y_test=Y.iloc[1460:]

X_train



#1 linear regression

from sklearn.linear_model import LinearRegression

lnr = LinearRegression()

lnr.fit (X_train, y_train )



predicted_y = lnr.predict(X_test)

print("predicted y records:")

print(predicted_y)

#ss

#ss=ss.drop('SalePrice')

#ss['SalePrice']=predicted_y

#ss. to_csv('../input/linearprediction.csv')

#ss
from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, predicted_y))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predicted_y))

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predicted_y)))
import matplotlib.pyplot as plt

plt.scatter(X_test['SaleCondition'], y_test, color = 'red')

plt.scatter(X_test['SaleCondition'],predicted_y, color = 'green')

plt.title ('Results ')

plt.xlabel('SaleCondition')

plt.ylabel('SalePrice')

plt.show()



import seaborn as sns

sns.countplot(X['SaleCondition'])



plt.show()

X.head()

X['Street'].value_counts()
#2 polynomial regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=3)

polyX_train = poly_reg.fit_transform(X_train)

#print(polyX_train)



polyX_test=poly_reg.fit_transform(X_test)

polyX_test
from sklearn.linear_model import LinearRegression



lin_reg_2 = LinearRegression()

lin_reg_2.fit(polyX_train, y_train)



predictValues = lin_reg_2.predict(polyX_test)

print(predictValues)

#ss

#ss=ss.drop('SalePrice')

#ss['SalePrice']=predictValues

#ss. to_csv('../input/polynomialprediction.csv')

#ss
from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, predictValues))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predictValues))

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictValues)))
#bar graph

a=X['OverallCond']

b=Y

plt.bar(a,b)

#plt.axis([0,11,0,800000])

plt.xlim([1,10])



plt.show()
#scatter plot

import matplotlib.pyplot as plt

# Visualising the Polynomial Regression results

plt.scatter(X_test['HouseStyle'], y_test, color = 'red')

plt.scatter(X_test['HouseStyle'],predicted_y, color = 'green')

plt.title('Polynomial Regression result on X and Y')

plt.xlabel('HouseStyle')

plt.ylabel('SalePrice')

plt.show()



#countplot

import seaborn as sns

sns.countplot(X['HouseStyle'])



plt.show()

#3 decision tree

#some columns selected for decision tree algorithm because mean square error value is very large

x=X[['HouseStyle','OverallCond']]



xtrain=x.iloc[:1460,:]

xtest=x.iloc[1460:,:]

from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(random_state = 100,max_depth=2)

decision_tree.fit(xtrain, y_train)

predictValues =decision_tree.predict(xtest)

predictValues
#ss

#ss=ss.drop('SalePrice')

#ss['SalePrice']=predictValues

#ss. to_csv('../input/decisiontreeprediction.csv')

#ss
from sklearn import metrics

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")

print(np.sqrt(metrics.mean_squared_error(y_test, predictValues)))
data_feature_names =['HouseStyle','OverallCond']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=data_feature_names, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))
#4random forest

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_estimators=5, random_state=50, max_depth=2)



model.fit(xtrain,y_train)

pred=model.predict(xtest)

pred

#ss

#ss=ss.drop('SalePrice')

#ss['SalePrice']=pred

#ss. to_csv('../input/randomforestprediction.csv')

#ss
from sklearn import metrics

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")

print(np.sqrt(metrics.mean_squared_error(y_test, pred)))

estimators=model.estimators_[3]

labels=['HouseStyle','OverallCond']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=labels

   , filled = True))

display(SVG(graph.pipe(format='svg')))