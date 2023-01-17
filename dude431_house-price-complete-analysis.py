# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

data = pd.read_csv("../input/kc_house_data.csv")
data_copy = data.copy()
data.sample(5)
#Let's get some basic info of data types
data.info()
# It seems there are no Null Values.
# Let's Confirm

print(data.isnull().any().sum(),'/',len(data.columns))
print(data.isnull().any(axis=1).sum(), '/', len(data))
data.describe()
sns.catplot(data=data , kind='box' , height=7, aspect=2.5)
plt.show()
corr = data.corr()
plt.figure(figsize=(20,16))
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
plt.show()
from scipy.stats import pearsonr
#It helps to measures the linear relationship between two datasets

features = data.iloc[:,3:].columns.tolist()
target = data.iloc[:,2].name
correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head()
columns = data[['sqft_living','grade','sqft_above','sqft_living15','bathrooms','price']]

sns.pairplot(columns, kind="scatter", palette="Set1")
plt.show()
plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.title('Living Area Distribution')
sns.distplot(data['sqft_living'])
          
plt.subplot(122)
plt.title('Living Area in 2k15 Distribution')
sns.distplot(data['sqft_living15'])

plt.show()

plt.figure(figsize = (12, 6))

plt.title('Upper Area Distribution')
sns.distplot(data['sqft_above'])
plt.show()
sns.catplot(x='grade', data=data , kind='count',aspect=2.5 )
plt.show()


sns.catplot(x='grade', y='price', data=data, kind='violin' ,aspect=2.5 )
plt.show()
data["bathrooms"] = data['bathrooms'].round(0).astype(int)

sns.catplot(x='bathrooms', data=data , kind='count',aspect=2.5 )
plt.show()

sns.catplot(x='bathrooms', y='price', data=data, kind='box' ,aspect=2.5 )
plt.show()
data["bathrooms"] = data['bathrooms'].round(0).astype(int)

labels = data.bathrooms.unique().tolist()
sizes = data.bathrooms.value_counts().tolist()

print(labels)


explode = (0.1,0.0,0.1,0.1,0.0,0.2,0.4,0.6,0.8)
plt.figure(figsize=(20,20))
plt.pie(sizes, explode=explode, labels=labels,autopct='%2.2f%%', startangle=0)
plt.axis('equal')
plt.title("Percentage of Clarity Categories")

plt.legend(labels,
          title="No. of bathrooms",
          loc="center left")
          #bbox_to_anchor=(1, 0, 0.5, 1))
plt.plot()
plt.show()
data2 = data.drop(['date','id'],axis=1)
#I dont think that date and id help us in predicting prices
#Divide the Dataset into Train and Test, So that we can fit the Train for Modelling Algos and Predict on Test.
from sklearn.model_selection import train_test_split
x = data2.drop(['price'], axis=1)
y = data2['price']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
from sklearn.model_selection import cross_val_score

lr=LinearRegression()
lr.fit(x_train,y_train)
score1= lr.score(x_test,y_test)
accu1 = cross_val_score(lr,x_train,y_train,cv=5)
print("___Linear Regresion____\n")
print(score1)
print(accu1)
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,BaggingRegressor
rf = RandomForestRegressor()
rf.fit(x_train,y_train)

score2 = rf.score(x_test,y_test)
accu2 = cross_val_score(rf,x_train,y_train,cv=5)
print("____ Random Forest Regressor____\n")
print(score2)
print(accu2)
br = BaggingRegressor()
br.fit(x_train,y_train)

score3 = br.score(x_test,y_test)
accu3= cross_val_score(br,x_train,y_train,cv=5)
print("____Bagging Regressor____\n")
print(score3)
print(accu3)
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
gb = GradientBoostingRegressor(n_estimators=1000)
gb.fit(x_train,y_train)

score4 = gb.score(x_test,y_test)
accu4 = cross_val_score(gb,x_train,y_train,cv=5)
print("____ Gradient Boosting Regressor____\n")
print(score4)
print(accu4)
et = ExtraTreesRegressor(n_estimators=1000)
et.fit(x_train,y_train)

score5 = et.score(x_test,y_test)
accu5 = cross_val_score(et,x_train,y_train,cv=5)
print("____ Extra Tree Regressor____\n")
print(score5)
print(accu5)
Models = ['Linear Regression','RandomForest Regression','Bagging Regressor','Gradient Boosting Regression','ExtraTree Regression']
Scores = [score1,score2,score3,score4,score5]
compare = pd.DataFrame({'Algorithms' : Models , 'Scores' : Scores})
compare.sort_values(by='Scores' ,ascending=False)
sns.factorplot(x='Algorithms', y='Scores' , data=compare, size=6 , aspect=2)
plt.show()
