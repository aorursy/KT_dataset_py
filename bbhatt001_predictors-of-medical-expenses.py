%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 500)
data=pd.read_csv('../input/insurance.csv')
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
print(data.head())
print(data.info())
print(data.describe())
sns.relplot(x='age', y='charges', hue= 'sex', data=data, palette='husl')
plt.title('Effect of Age on Charges')
sns.relplot(x='age', y='charges', col='sex',data=data, palette='husl')
sns.relplot(x='age', y='charges', hue='smoker', style= 'sex', data=data, palette='husl')
plt.title('Combined effect of Age and Smoking on Charges')
sns.lmplot(x='age', y='charges', hue='smoker', col='sex',data=data, palette='husl')
sns.violinplot(x="sex", y='charges', hue="smoker", data=data, palette='Dark2')
plt.title('Effect of Smoking on Charges of males and females')
data_grouped=data.groupby(['smoker', 'sex']).agg({'charges':'sum','sex':'count'})
data_grouped['mean_charges']= data_grouped['charges']/data_grouped['sex']
data_grouped=data_grouped.rename(columns={'sex':'number_in_gender'})
data_grouped.index=[0,1,2,3]
data_grouped['smoker']=['no','no','yes','yes']
data_grouped['sex']=['female','male','female','male']
data_grouped=data_grouped[['smoker', 'sex','number_in_gender','charges','mean_charges']]
data_grouped
sns.catplot(x='sex',y='mean_charges',hue='smoker',kind='bar',data=data_grouped, palette='Dark2')
sns.catplot(x='sex',y='number_in_gender',hue='smoker',kind='bar',data=data_grouped, palette='Dark2')
sns.relplot(x='bmi',y='charges',style='sex',data=data)
plt.title('Effect of BMI on Charges')
sns.relplot(x='bmi',y='charges',hue='smoker',style='sex',data=data)
sns.lmplot(x='bmi',y='charges',hue='smoker', col='sex',data=data)
sns.pairplot(data, vars= ['age','bmi','children','charges'], hue='sex')
sns.catplot(x="children", y='charges', hue='sex', kind='box',data=data, palette= 'Accent')
plt.title('Charges vs number of children')
sns.catplot(x="children", y='charges', hue='smoker', kind='box',data=data , palette= 'Paired')
data_grouped2=data.groupby('children').agg({'charges':'sum','sex':'count'})
#data_grouped['mean_charges']= data_grouped['charges']/data_grouped['sex']
data_grouped2['mean_charges2']=data_grouped2['charges']/data_grouped2['sex']
data_grouped2['median_charges']=data.groupby('children')['charges'].median()
data_grouped2


data_grouped3=data.groupby(['children','sex','smoker']).agg({'sex':'count', 'charges':'sum'})
data_grouped3['mean_charges2']=data_grouped3['charges']/data_grouped3['sex']
data_grouped3
sns.violinplot(x="region", y='charges', data=data)
sns.violinplot(x="region", y='charges', hue="smoker", data=data)
data_grouped4=data.groupby('region').agg({'charges':'sum','sex':'count'})
data_grouped4['mean_charges3']=data_grouped4['charges']/data_grouped4['sex']
data_grouped4
data_grouped5=data.groupby(['region','smoker']).agg({'sex':'count', 'charges':'sum'})
data_grouped5['mean_charges']=data_grouped5['charges']/data_grouped5['sex']
data_grouped5
print(data.dtypes)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(data['sex'].drop_duplicates())
data['sex']=encoder.transform(data['sex'])
encoder.fit(data['smoker'].drop_duplicates())
data['smoker']=encoder.transform(data['smoker'])
data1=pd.get_dummies(data['region'], prefix='region')
data= pd.concat([data,data1], axis=1).drop(['region'],axis=1)
print(data.head(2))
print(data.dtypes)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
y= data['charges']
X = data.drop(['charges'], axis=1)
lin_reg=LinearRegression()
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
lin_reg.fit(train_X,train_y)
pred_y=lin_reg.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print("RMSE: %f" % (rmse))
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import numpy as np

y= data['charges']
X = data.drop(['charges'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
train_X = pd.DataFrame(data=train_X, columns=X.columns)
test_X = pd.DataFrame(data=test_X, columns=X.columns)

model_x = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model_x.fit(train_X, train_y, early_stopping_rounds=5,eval_set=[(test_X, test_y)], verbose=False)
predictions = model_x.predict(test_X)

rmse = np.sqrt(mean_squared_error(test_y, predictions))
print("RMSE: %f" % (rmse))
plot_importance(model_x)
y= data['charges']
X = data.drop(['charges','sex','region_northeast','region_northwest','region_southeast','region_southwest'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
train_X = pd.DataFrame(data=train_X, columns=X.columns)
test_X = pd.DataFrame(data=test_X, columns=X.columns)

model_x = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model_x.fit(train_X, train_y, early_stopping_rounds=5,eval_set=[(test_X, test_y)], verbose=False)
predictions = model_x.predict(test_X)

rmse = np.sqrt(mean_squared_error(test_y, predictions))
print("RMSE: %f" % (rmse))

y= data['charges']
X = data.drop(['charges','sex','region_northeast','region_northwest','region_southeast','region_southwest','children'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
model_x = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model_x.fit(train_X, train_y, early_stopping_rounds=5,eval_set=[(test_X, test_y)], verbose=False)
predictions = model_x.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_y, predictions))
print("RMSE: %f" % (rmse))