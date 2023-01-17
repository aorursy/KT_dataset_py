import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import ElasticNet, LassoLars, Ridge, LinearRegression, Lasso

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures

%matplotlib inline

plt.style.use('ggplot')
df = pd.read_csv('../input/diamonds.csv',delimiter=',')

df.head()
df.isnull().sum()
df.info()
df.drop(['Unnamed: 0','x','y','z'],inplace=True, axis=1)

df.head()
f, ax = plt.subplots(3, figsize=(12,18))

sns.countplot('cut', data=df, ax=ax[0])

sns.countplot('color', data=df, ax=ax[1])

sns.countplot('clarity', data=df, ax=ax[2])

ax[0].set_title('Diamond cut')

ax[1].set_title('Colour of the diamond')

ax[2].set_title('Clarity of the diamond')
df.describe()
f, ax = plt.subplots(4, figsize=(12,24))

sns.distplot(df.carat,color='c',ax=ax[0])

sns.distplot(df.depth,color='c',ax=ax[1])

sns.distplot(df.table,color='c',ax=ax[2])

sns.distplot(df.price,color='c',ax=ax[3])

ax[0].set_title('Diamond carat distribution')

ax[1].set_title('Total depth distribution')

ax[2].set_title('Table width distribution')

ax[3].set_title('Price distribution')
g = sns.pairplot(df,vars=['carat', 'depth',

      'table', 'price'])
f, ax = plt.subplots(3,figsize=(12,16))

# sns.violinplot(x='carat',y='price',data=df,ax=ax[3])

sns.violinplot(x='clarity',y='price',data=df,ax=ax[2])

sns.violinplot(x='color',y='price',data=df,ax=ax[1])

sns.violinplot(x='cut',y='price',data=df,ax=ax[0])

ax[0].set_title('Cut vs Price')

ax[1].set_title('Color vs Price')

ax[2].set_title('Clarity vs Price')

# ax[3].set_title('carat vs Price')
sns.jointplot(x='carat',y='price',data=df,color='c')


corrmat = df.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True)
le = LabelEncoder()

df.cut = le.fit_transform(df.cut)

df.color = le.fit_transform(df.color)

df.clarity = le.fit_transform(df.clarity)

df.info()
df
x = df.drop('price',axis=1)

y = df.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=28)
sc = StandardScaler()

sc.fit(x_train)

sc.transform(x_train)

sc.transform(x_test)
clf = LinearRegression()

clf.fit(x_train,y_train)

from sklearn.linear_model import LinearRegression

from yellowbrick.regressor import PredictionError

f, ax = plt.subplots(figsize=(30, 15))

model = LinearRegression() # Instantiate the linear model and visualizer

visualizer = PredictionError(model=clf, identity=False)

visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

visualizer.poof()             # Draw/show/poof the data