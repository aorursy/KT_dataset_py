import numpy as np
import pandas as pd
import sklearn as sk
df=pd.read_csv("../input/kc_house_data.csv",index_col='id')
df.head()
print(df.dtypes)
print(df.shape)
df=df.drop(['date','zipcode'],axis=1)#drop two features
df['basement_present'] = df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not
df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated
df['age']=df['yr_built'].apply(lambda x: 2014-x if x > 0 else 0)#Indicate the age of the building
df.head()
print(df.shape)
# Stotastical properties
df.describe(include="all")
# Check missing dat
print(df.isnull().any())
# seperate price column as target column
y=df.price # y as target
x=df.drop('price',axis=1)
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm
from scipy import stats
#histogram and normal probability plot
sns.distplot(y, fit=norm, bins=10, kde=False)
plt.show()
#applying log transformation
y = np.log(y)
#transformed histogram and normal probability plot
sns.distplot(y, fit=norm,  bins=10, kde=False);
plt.show()
x['sqft_living'] = np.log(x['sqft_living'])
from sklearn.model_selection import train_test_split
#Split train test data, 70% and 30%
x_train, x_test, y_train,y_test=train_test_split(x,y ,test_size=0.3,random_state = 0)
from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(x_train)
scalery= preprocessing.StandardScaler().fit(y_train.values.reshape(-1,1))
x_train=pd.DataFrame(scaler.transform(x_train),columns=list(x_train.columns.values))
y_train=pd.DataFrame(scalery.transform(y_train.values.reshape(-1,1)),columns=["price"])
x_test=pd.DataFrame(scaler.transform(x_test),columns=list(x_test.columns.values))
y_test=pd.DataFrame(scalery.transform(y_test.values.reshape(-1,1)),columns=["price"])
c=x_train.shape[1]
print(c)
plt.figure(0)

for i in range(5):
    for j in range(4):
        ax=plt.subplot2grid((5,4),(i,j))
        ax.scatter(x_train.iloc[:,j+i*4], y_train)
        ax.title(list(x_train.columns.values)[j+i*4])
plt.show()
absCor=abs(x_train.corrwith(y_train['price']))
absCor.sort_values(ascending=False)
t=14
x_train.drop(absCor.index.values[t:],axis=1,inplace=True)
x_test.drop(absCor.index.values[t:],axis=1,inplace=True)
featuresCor=x_train.corr()
f, ax = plt.subplots(figsize=(10, 8))
corr = x_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot=True)
plt.show()
highCor=abs(corr.unstack()).sort_values(ascending=False)[t::2]
highCor.head()
highCorrList=list(set(list(highCor[highCor>0.75].unstack().index)+list(highCor[highCor>0.75].unstack().columns.values)))

print(highCorrList)
DropList=absCor[highCorrList][absCor[highCorrList]<0.65]
DropList=list(DropList.index)
print(DropList)
x_train.drop(DropList,axis=1,inplace=True)
x_test.drop(DropList,axis=1,inplace=True)
print(x_train.dtypes)
print(x_train.shape)
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
# Create linear regression object
regr = LinearRegression()
# Train the model with training sets
regr.fit(x_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The intercept and coefficients, mean-squared error, R-squared score
print(' Intercept & Coefficients: \n', regr.intercept_, regr.coef_)
print("Mean squared error (training): %.2f" % mean_squared_error(y_train, regr.predict(x_train)))
print("Mean squared error (testing): %.2f" % mean_squared_error(y_test, y_pred))
print('R-square score (training): %.2f' % regr.score(x_train, y_train))
print('R-square score (testing): %.2f' % r2_score(y_test, y_pred))
plt.scatter(y_test,y_pred,color='g')
plt.xlabel('true price')
plt.ylabel('predicted price')
plt.show()
import statsmodels.api as sm

x=sm.add_constant(x_train)
model = sm.OLS(y_train,x)
results = model.fit()
# Statsmodels gives R-like statistical output
print(results.summary())
DropList=list(results.pvalues[1:][results.pvalues>0.05].index)
print(DropList)
x_train.drop(DropList,axis=1,inplace=True)
x_test.drop(DropList,axis=1,inplace=True)
x_train.columns

X=x_train
y=y_train
y = np.ravel(y)
features=X.columns.values
results=[]

lr=LinearRegression()
for i in range(1,9):
    selector=RFE(lr,n_features_to_select=i, step=1)
    selector.fit(X,y)
    r2=selector.score(X, y)
    selected_features=features[selector.support_]
    msr=mean_squared_error(y, selector.predict(X))
    results.append([i,r2,msr,",".join(selected_features)])
    
results=pd.DataFrame(results,columns=['no_features','r2','mean square error','selected_features']) 
results
plt.scatter(y, selector.predict(X)-y,  color='blue')
plt.plot([y.min(),y.max()],[0,0],color='black')
plt.title("Residuals' plot:")
plt.xlabel('fitted value (y)')
plt.ylabel('Residual')
plt.show()
FinalFeatures=results.selected_features[2].split(',')
print(FinalFeatures)
x_train2=x_train[FinalFeatures]
x_test2=x_test[FinalFeatures]
# Create linear regression object
regr = LinearRegression()
# Train the model with training sets
regr.fit(x_train2, y_train)
# Make predictions using the testing set
y_pred = regr.predict(x_test2)

# The intercept and coefficients, mean-squared error, R-squared score
print(' Intercept & Coefficients: \n', regr.intercept_, regr.coef_)
print("Mean squared error (training): %.2f" % mean_squared_error(y_train, regr.predict(x_train2)))
print("Mean squared error (testing): %.2f" % mean_squared_error(y_test, y_pred))
print('R-square score (training): %.2f' % regr.score(x_train2, y_train))
print('R-square score (testing): %.2f' % r2_score(y_test, y_pred))
