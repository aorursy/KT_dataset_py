# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, t, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
df.info()
df.describe()
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)
df['roomsPerHousehold'] = df["total_rooms"] / df["households"]
df['populationPerHousehold'] = df["population"] / df["households"]
df['beedroomsPerRoom'] = df["total_bedrooms"] / df["total_rooms"]
df.describe()
df.corr()
_ = df['population'].loc[df['population'] > 5000].hist(bins='auto')
#df['population'].loc[df['population'] > 10000].count()
df['population'].quantile(0.95)
df['ocean_proximity'].value_counts()
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

california_img = mpimg.imread('../input/california-housing-feature-engineering/california.png')
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=df['population']/100, 
        label='population', figsize=(22,14), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.show()
_ = df.hist(bins='auto', figsize=(25,18))
dflog = np.log(df[[cl for cl in df.columns if df[cl].dtype !='object']])
_ = dflog.hist(bins='auto', figsize=(25,18))
#columns = ['households', 'median_income', 'total_bedrooms', 'total_rooms', 'population']
#print(df[columns].describe())
#df[columns] = np.log(df[columns])
#df[columns].describe()
features = [cl for cl in df.columns if df[cl].dtype !='object']

for f in features:
    if df[f].skew() > 1:
        # df[f] = np.where(df[f] < df[f].quantile(0.05), df[f].quantile(0.00), df[f])
        df[f] = np.where(df[f] > df[f].quantile(0.98), df[f].quantile(0.98), df[f])
    print(f"For feature {f} the skew = {df[f].skew()}")

#_ = df.hist(bins='auto', figsize=(25,18))
#df.describe()
for f in list(set(features) - set(['median_house_value'])):
    df[f + '_2'] = df[f]**2
    #df[f + '_3'] = df[f]**3
    _
                         
features = [cl for cl in df.columns if df[cl].dtype !='object']
import matplotlib.pyplot as plt
import seaborn as sns
    
def plotGraph(pdf, pscaled_df):
    fig, (a, b) = plt.subplots(ncols=2, figsize=(16, 5))
    a.set_title("Before the scaler")
    for col in pdf.columns:
        sns.kdeplot(pdf[col], ax=a)
    b.set_title("after the scaler")
    for col in pdf.columns:
        sns.kdeplot(pscaled_df[col], ax=b)
    plt.show()
target = ['median_house_value']
predict = list(set(features) - set(target))
predict = list(set(predict) - set([cl for cl in predict if 'itude' in cl]))
predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[predict])
scaled_df = pd.DataFrame(scaled_df, columns=predict)
plotGraph(df[predict], scaled_df)
_ = scaled_df.hist(bins='auto', figsize=(25,18))
scaled_df.describe()
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True, prefix='OP', dtype='float64')
for cl in df.columns: 
    if "OP_" in cl:
        scaled_df[cl] = df[cl]
    
scaled_df.info()
scaled_df[target] = df[target]
scaled_df.corr()
scaled_df.drop(['median_house_value'], axis=1, inplace=True)
scaled_df.columns
from sklearn.model_selection import train_test_split

#selected_feautures = ['median_income', 'median_income_2', 'median_income_3', 'OP_INLAND', 'OP_ISLAND',
#       'OP_NEAR BAY', 'OP_NEAR OCEAN', 'roomsPerHousehold', 'populationPerHousehold', 'beedroomsPerRoom']
x_train, x_test, y_train, y_test = train_test_split(scaled_df, 
                                                    df[target], test_size=0.33, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lr = LinearRegression()
lr.fit(x_train, y_train)
print(f"Coefficients of the régression \nIntercept                = {lr.intercept_.item():12.2f}")
for cl, cf in zip(np.array(scaled_df.columns).reshape(-1,1),lr.coef_.reshape(-1,1)):
      print(f"{cl.item():<24} = {cf.item():12.2f}")
rr = Ridge(alpha=0.01)
rr.fit(x_train, y_train) 
print(f"Coefficients of the régression \nIntercept                = {rr.intercept_.item():12.2f}")
for cl, cf in zip(np.array(scaled_df.columns).reshape(-1,1),rr.coef_.reshape(-1,1)):
      print(f"{cl.item():<24} = {cf.item():12.2f}")
lm = Lasso(alpha=0.01, max_iter=100000)
lm.fit(x_train, y_train) 
print(f"Coefficients of the régression \nIntercept                = {lm.intercept_.item():12.2f}")
for cl, cf in zip(np.array(scaled_df.columns).reshape(-1,1),lm.coef_.reshape(-1,1)):
      print(f"{cl.item():<24} = {cf.item():12.2f}")
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
print("========  Linear regression  OLS methode   ======")
print("--------          Score training           ------")
pred_train_lr= lr.predict(x_train)
print(f"{np.sqrt(mean_squared_error(y_train,pred_train_lr)):.2f}")
print(f"{r2_score(y_train, pred_train_lr):.2%}")

print("--------          Score testing           ------")
pred_test_lr= lr.predict(x_test)
print(f"{np.sqrt(mean_squared_error(y_test,pred_test_lr)):.2f}") 
print(f"{r2_score(y_test, pred_test_lr):.2%}")
print("========       Ridge Regression Model      ======")
print("--------          Score training           ------")
pred_train_rr= rr.predict(x_train)
print(f"{np.sqrt(mean_squared_error(y_train,pred_train_rr)):.2f}")
print(f"{r2_score(y_train, pred_train_rr):.2%}")

print("--------          Score testing            ------")
pred_test_rr= lr.predict(x_test)
print(f"{np.sqrt(mean_squared_error(y_test,pred_test_rr)):.2f}") 
print(f"{r2_score(y_test, pred_test_rr):.2%}")
print("========            Lasso Model            ======")
print("--------          Score training           ------")
pred_train_lm= lm.predict(x_train)
print(f"{np.sqrt(mean_squared_error(y_train,pred_train_lm)):.2f}")
print(f"{r2_score(y_train, pred_train_lm):.2%}")

print("--------          Score testing            ------")
pred_test_lm= lr.predict(x_test)
print(f"{np.sqrt(mean_squared_error(y_test,pred_test_lm)):.2f}") 
print(f"{r2_score(y_test, pred_test_lm):.2%}")