import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler
data_dir = '/kaggle/input/eeedatathoncompetition/'
diamonds = pd.read_csv(data_dir+"train.csv",index_col = False)

diamonds
diamonds.info()
diamonds["price"] = diamonds["price"].astype(float)

diamonds
diamonds.hist(bins = 50, figsize = (20, 15))

plt.show()
diamonds.corr()
corr_matrix = diamonds.corr()

plt.subplots(figsize = (10, 8))

sns.heatmap(corr_matrix, annot = True)

plt.show()
sns.pairplot(data = diamonds)
sns.catplot(x='cut', data=diamonds , kind='count',aspect = 2)
sns.catplot(x='cut', y='price', data=diamonds, kind='box' ,aspect = 2)
sns.catplot(x='cut', y='price', data=diamonds, kind='violin' ,aspect = 2)
sns.catplot(x='color', data=diamonds , kind='count',aspect = 2)
sns.catplot(x='color', y='price', data=diamonds, kind='box' ,aspect = 2)
sns.catplot(x='color', y='price', data=diamonds, kind='violin' ,aspect = 2)
labels = diamonds.clarity.unique().tolist()

sizes = diamonds.clarity.value_counts().tolist()

explode = (0.1, 0.0, 0.1, 0, 0.1, 0, 0.1,0)

plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True)

plt.title("Percentage of Clarity Categories")

plt.plot()

plt.show()
sns.catplot(x='clarity', y='price', data=diamonds, kind='box' ,aspect = 2)
sns.catplot(x='clarity', y='price', data=diamonds, kind='violin' ,aspect = 2)
scaler = StandardScaler()

print("Cut: ",set(diamonds["cut"]))

print("Color: ",set(diamonds["color"]))

print("Clarity: ",set(diamonds["clarity"]))
diamonds['price/wt']=diamonds['price']/diamonds['carat']

print(diamonds.groupby('cut')['price/wt'].mean().sort_values())

print(diamonds.groupby('color')['price/wt'].mean().sort_values())

print(diamonds.groupby('clarity')['price/wt'].mean().sort_values())
diamonds['cut']=diamonds['cut'].map({'Ideal':1,'Good':2,'Very Good':3,'Fair':4,'Premium':5})

diamonds['color']=diamonds['color'].map({'E':1,'D':2,'F':3,'G':4,'H':5,'I':6,'J':7})

diamonds['clarity']=diamonds['clarity'].map({'VVS1':1,'IF':2,'VVS2':3,'VS1':4,'I1':5,'VS2':6,'SI1':7,'SI2':8})
diamonds['cut/wt']=diamonds['cut']/diamonds['carat']

diamonds['color/wt']=diamonds['color']/diamonds['carat']

diamonds['clarity/wt']=diamonds['clarity']/diamonds['carat']

diamonds['dimension']=diamonds['x']*diamonds['y']*diamonds['z']

diamonds['width_top']=diamonds['table']*diamonds['y']

diamonds = diamonds.drop(['cut','color','clarity','price/wt','table','x','y','z'], axis=1)
diamonds
diamonds.corr()
X=diamonds.drop(['price'],axis=1)

Y=diamonds['price']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

from sklearn.ensemble import RandomForestRegressor

clf_rf = RandomForestRegressor()

clf_rf.fit(X_train , Y_train)

Y_pred = clf_rf.predict(X_test)



print('')

print('###### Random Forest ######')

print('Score : %.6f' % clf_rf.score(X_test, Y_test))



mse = mean_squared_error(Y_test, Y_pred)

mae = mean_absolute_error(Y_test, Y_pred)

rmse = mean_squared_error(Y_test, Y_pred)**0.5

r2 = r2_score(Y_test, Y_pred)



print('')

print('MSE    : %0.6f ' % mse)

print('MAE    : %0.6f ' % mae)

print('RMSE   : %0.6f ' % rmse)

print('R2     : %0.6f ' % r2)
diamonds_submission = pd.read_csv(data_dir+"test.csv",index_col = False)
diamonds_submission['cut']=diamonds_submission['cut'].map({'Ideal':1,'Good':2,'Very Good':3,'Fair':4,'Premium':5})

diamonds_submission['color']=diamonds_submission['color'].map({'E':1,'D':2,'F':3,'G':4,'H':5,'I':6,'J':7})

diamonds_submission['clarity']=diamonds_submission['clarity'].map({'VVS1':1,'IF':2,'VVS2':3,'VS1':4,'I1':5,'VS2':6,'SI1':7,'SI2':8})
diamonds_submission['cut/wt']=diamonds_submission['cut']/diamonds_submission['carat']

diamonds_submission['color/wt']=diamonds_submission['color']/diamonds_submission['carat']

diamonds_submission['clarity/wt']=diamonds_submission['clarity']/diamonds_submission['carat']

diamonds_submission['dimension']=diamonds_submission['x']*diamonds_submission['y']*diamonds_submission['z']

diamonds_submission['width_top']=diamonds_submission['table']*diamonds_submission['y']

diamonds_submission = diamonds_submission.drop(['cut','color','clarity','table','x','y','z'], axis=1)

diamonds_submission
diamonds_submission_pred = clf_rf.predict(diamonds_submission)

diamonds_submission_pred
price_submission = pd.read_csv(data_dir+"submission_sample.csv",index_col = False)
price_submission['price'] = diamonds_submission_pred

price_submission
price_submission.to_csv("DiamondPricePrediction_byTweight(FINAL).csv", index=False)