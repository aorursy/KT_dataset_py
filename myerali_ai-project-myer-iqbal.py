import pandas as pd
import seaborn as sns
import numpy as np
df = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
df.columns
df.head()
df.drop(['Date'],axis=1,inplace=True)
Holiday_1=pd.get_dummies(df['StateHoliday'])
Holiday_1.drop(['a','b','c'],axis=1,inplace=True)
Holiday_2=pd.get_dummies(df['SchoolHoliday'],drop_first=True)
Holiday_2.tail()
Holiday=Holiday_1.join(Holiday_2['1'])
Holiday.head()
Holiday.tail()

df.drop(['StateHoliday','SchoolHoliday'],axis=1,inplace=True)
#df=pd.concat([Holiday],axis=1)
df.head()
#df=pd.concat([Holiday],axis=1)

df=df.join(Holiday['0'])
df.head()
df=df.join(Holiday['1'])
df.head()
df.rename(columns={'0':'isStateHoliday', '1':'isSchoolHoliday'},inplace=True)
df.head()
df_1=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
df_1.head()
df.tail()
df = pd.merge(df,df_1,on='Store')
df.head()
df.drop(['StoreType','Assortment'],axis=1,inplace=True)
df.head()
df.tail()
df.info()
df.info()
df.head()
df.tail()
df.dropna(inplace=True)
df.head()
df.columns
X=df[['Store','Customers' ,'Open', 'Promo','isStateHoliday','isSchoolHoliday', 'CompetitionDistance','CompetitionOpenSinceMonth','Promo2SinceWeek']]
y=df['Sales']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
lm.coef_
X_train.columns
cdf=pd.DataFrame(lm.coef_,X.columns)
cdf
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
cdf
predictions=lm.predict(X_test)
predictions
y_test

predictions=lm.predict(X_test)
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
mean_sqr=metrics.mean_squared_error(y_test,predictions)
print('Mean Squared Error Through Regression os '+str(mean_sqr)+' good fitted')
np.sqrt(metrics.mean_squared_error(y_test,predictions))
Coeff_of_r2=metrics.r2_score(y_test,predictions)*100
print('Our model is '+str(Coeff_of_r2)+' good fitted')
#In general, the higher the R-squared, the better the model fits your data.



