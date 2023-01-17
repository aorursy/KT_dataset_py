import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/small-dataset-about-used-fiat-500-sold-in-italy/Used_fiat_500_in_Italy_dataset.csv")
df.head()
df.info()
df.describe().T
df.corr()
# sns.pairplot(df,kind='reg');
df.columns
df.drop(['previous_owners', 'lat', 'lon'],axis = 1,inplace = True)
tran = df.transmission.values.reshape(-1,1)
model = df.model.values.reshape(-1,1)
e_pow = df.engine_power.values.reshape(-1,1)
# LabelEncoder sınıfını import ettik.
from sklearn.preprocessing import LabelEncoder
# LabelEncoder sınıfından bir nesne türettik.
lb = LabelEncoder()
# Encode işlemini gerçekleştirdik. Artık model ve transmission kolonları sayısal değerlerde
tran[:,0] = lb.fit_transform(tran[:,0])
model[:,0] = lb.fit_transform(model[:,0])
e_pow[:,0] = lb.fit_transform(e_pow[:,0])
ktran = tran.astype('int64',copy = False)
kmodel = model.astype ('int64',copy = False)
epow = e_pow.astype ('int64',copy = False)
df.drop(['model','transmission','engine_power'],axis = 1,inplace = True)
dftran = pd.DataFrame(data=ktran[:,:1],index=range(len(ktran)),columns=['transmission',])
dfmodel = pd.DataFrame(data=kmodel,index=range(len(kmodel)),columns=['model',])
dfepow = pd.DataFrame(data=epow,index=range(len(epow)),columns=['engine_power',])
# Verileri birleştirdik.
new_df = pd.concat([dfmodel,dfepow,df,dftran],axis=1)
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
x = new_df['age_in_days'].values.reshape(-1,1)
y = new_df['price'].values.reshape(-1,1)
t = new_df.copy()
linear_reg.fit(x,y)

b0 = linear_reg.predict([[5000]])
print("b0:",b0)

b0_ = linear_reg.intercept_
print("b0_:",b0_)

b1 = linear_reg.coef_
print("b1: ", b1)

print('tahmin',linear_reg.predict([[1000]]) )
array = t['age_in_days'].values.reshape(-1,1)

plt.scatter(x,y)
y_head = linear_reg.predict(array)

plt.plot(array,y_head, color="red")

plt.show
import statsmodels.regression.linear_model as sm
lin = sm.OLS(x,y)
model = lin.fit()
model.summary()
print(new_df.engine_power.value_counts())
print(new_df.model.value_counts())
print(new_df.transmission.value_counts())
new_df.corr()
new_df.describe().T
# sns.pairplot(new_df,kind='reg');
plt.figure(figsize=(8,6))
sns.scatterplot(x='age_in_days', y='price', data=new_df, hue='km');
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
X = new_df.drop(['price'],axis = 1)
y =  new_df['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 40)

# lr = LinearRegression()

# lr.fit(X_train,y_train)

# tahmin = lr.predict(X_test)

# plt.scatter(X_train,y_train)
# plt.plot(X_test,tahmin, color='red')
# plt.show()
import statsmodels.regression.linear_model as sm
lm = sm.OLS(y_train,X_train)
model = lm.fit()
model.summary()
# model.mse_model
