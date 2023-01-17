import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)
df.head()
# Arrange by rating (high to low):



df.sort_values(by='average_rating',ascending=False)
# Count null values (if any):



df.isnull().sum()
plt.figure(figsize=(15,10))

sns.barplot(data=df,x='language_code',y='average_rating')

plt.xlabel('Languages')

plt.ylabel('Average Rating')

plt.show()
plt.figure(figsize=(15,10))

sns.scatterplot(data=df,x='# num_pages',y='average_rating')

plt.xlabel('No. of Pages')

plt.ylabel('Average Rating')

plt.show()
plt.figure(figsize=(15,6))

sns.distplot(df['average_rating'],bins=20, color='red')

plt.title('Distribution of Average Ratings')

plt.xlabel('Average Rating')

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(data=df,y='# num_pages',x='language_code')

plt.xlabel('Language')

plt.ylabel('No. of Pages')

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),square=True,vmax=0.1,annot=True)

plt.xlabel('Language')

plt.ylabel('No. of Pages')

plt.show()
plt.figure(figsize=(15,7))

sns.distplot(df['# num_pages'],bins=20,color='gold')

plt.title('Distribution of Number of Pages')

plt.xlabel('No. of Pages')

plt.show()
df.describe()
cols = list(df.select_dtypes(exclude=['object']).columns)

df1 = df.loc[:,cols]
print("The Range of the dataset is : \n\n",df1.max()-df1.min())
IQR = df1.quantile(0.75)-df1.quantile(0.25)

IQR
ct= pd.crosstab(df['average_rating'],df['language_code'],normalize=False)

ct.head()
df[df['average_rating']==0]
df[df['average_rating']==0].count().loc['bookID']
df.groupby('language_code')['average_rating'].agg(['mean'])
df.groupby('language_code')['bookID'].agg(['count'])
df['Ranks']=df['average_rating'].rank(ascending=0,method='dense')
df
df.sort_values(by='average_rating',ascending=False)
df['Rank By Language']=df.groupby('language_code')['average_rating'].rank(ascending=0,method='dense')

df.sort_values(by='average_rating',ascending=False)
plt.figure(figsize=(10,10))

sns.regplot(data=df,y="average_rating",x="# num_pages",marker='*',color='k')

plt.xlabel('No. of Pages')

plt.ylabel('Average Rating')

plt.show()
df.head()
df1 = df.drop(['bookID','title','authors','isbn','isbn13','Ranks','Rank By Language'],axis=1)

df1.head()
lang = list(df1['language_code'].value_counts().head(6).index)
df1['language_code']=np.where(df1['language_code'].isin(lang),df1['language_code'],'others')
df1['language_code'].value_counts()
df1 = pd.get_dummies(df1, columns=['language_code'],drop_first=True)
df1.head()
import statsmodels.api as sm

X = df1.drop('average_rating',axis=1)

y =df1['average_rating']

xc = sm.add_constant(X)

lr = sm.OLS(y,xc).fit()

lr.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values,i) for i in range (X.shape[1])]

pd.DataFrame({'vif':vif}, index = X.columns)
X.head()
X = X.drop('text_reviews_count',axis=1)

X.head()
xc = sm.add_constant(X)

lr = sm.OLS(y,xc).fit()

lr.summary()
X = X.drop('language_code_en-US',axis=1)
xc = sm.add_constant(X)

lr = sm.OLS(y,X).fit()

lr.summary()
X = X.drop('ratings_count',axis=1)
xc = sm.add_constant(X)

lr = sm.OLS(y,X).fit()

lr.summary()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
xc = sm.add_constant(X_train)

lr = sm.OLS(y_train,X_train).fit()

lr.summary()
y_pred = lr.predict(X_test)
xc = sm.add_constant(X_test)

lr = sm.OLS(y_test,X_test).fit()

lr.summary()
plt.figure(figsize=(7,5))

plt.scatter(y_test,y_pred, color='y')

plt.plot(y,y,color='b')

plt.show()
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV



rf = RandomForestRegressor()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))

print('R-squared', r2_score(y_test,y_pred))
n_estimators = [int(x) for x in np.linspace(start = 10, stop=200, num=10)]

max_depth = [int(x) for x in np.linspace(10,100,num=10)]

min_samples_split = [2, 3, 4, 5, 10]

min_samples_leaf = [1, 2, 4, 10]



random_grid = {'n_estimators': n_estimators,

              'max_depth': max_depth,

              'min_samples_leaf': min_samples_leaf,

              'min_samples_split': min_samples_split}



print(random_grid)
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid, cv=3)



rf_random.fit(X_train, y_train)
rf_random.best_params_
rf = RandomForestRegressor(**rf_random.best_params_)

rf
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test) 



# Predict values of y by applying the Random Forest model generated through train data to test data.
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))

print('R-squared:',r2_score(y_test,y_pred))
plt.figure(figsize=(7,5))

plt.scatter(y_test,y_pred, color='r')

plt.plot(y,y,color='b')

plt.show()