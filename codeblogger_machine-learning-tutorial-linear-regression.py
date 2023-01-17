# import the necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



plt.style.use("seaborn-whitegrid")

warnings.filterwarnings("ignore")
# load data

data = "../input/insurance/insurance.csv"

df = pd.read_csv(data)



# show data (6 row)

df.head(6)
df.describe().T
df.info()
df.columns[df.isnull().any()]
df.isnull().sum()
data = df.copy()

data = data.select_dtypes(include=["float64","int64"])

data.head()
column_list = ['age', 'bmi', 'children', 'charges']

for col in column_list:

    sns.boxplot(x = data[col])

    plt.xlabel(col)

    plt.show()
f= plt.figure(figsize=(16,5))



ax=f.add_subplot(121)

sns.distplot(df['charges'],bins=50,color='r',ax=ax)

ax.set_title('Distribution of insurance charges')



ax=f.add_subplot(122)

sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)



plt.show()
f = plt.figure(figsize=(14,6))

ax = f.add_subplot(121)

sns.violinplot(x='sex', y='charges',data=df,palette='Wistia',ax=ax)

ax.set_title('Violin plot of Charges vs sex')



ax = f.add_subplot(122)

sns.violinplot(x='smoker', y='charges',data=df,palette='magma',ax=ax)



plt.show()
sns.jointplot(x="bmi",y="charges",data=df,kind="reg")

plt.show()
sns.jointplot(x="age",y="charges",data=df,kind="reg")

plt.show()
sns.jointplot(x="children",y="charges",data=df,kind="reg")

plt.show()
# import the necessary packages

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from scipy.stats import boxcox

from sklearn import metrics



df_encode = df.copy()
df_encode = pd.get_dummies(data = df_encode, columns = ['sex','smoker','region'])

df_encode.head()
# normalization

y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)

df_encode['charges'] = np.log(df_encode['charges'])



df_encode.head()
X = df_encode.drop('charges',axis=1) 

y = df_encode['charges']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)
X = df_encode['bmi'].values.reshape(-1,1)  # Independet variable

y = df_encode['charges'] # dependent variable



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

predictions = lin_reg.predict(X_test)



print("intercept: ", model.intercept_)

print("coef: ", model.coef_)

print("RScore. ", model.score(X_test,y_test))
plt.figure(figsize=(12,6))

plt.scatter(y_test,predictions)

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.figure(figsize=(12,6))

g = sns.regplot(x=df_encode['bmi'],y=df_encode["charges"],ci=None,scatter_kws = {'color':'r','s':9})

g.set_title("Model Equation")

g.set_ylabel("charges")

g.set_xlabel('bmi')

plt.show()
plt.figure(figsize=(12,6))

g = sns.regplot(x=df_encode['age'],y=df_encode["charges"],ci=None,scatter_kws = {'color':'r','s':9})

g.set_title("Model Equation")

g.set_ylabel("charges")

g.set_xlabel('age')

plt.show()