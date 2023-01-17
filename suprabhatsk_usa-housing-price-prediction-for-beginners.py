import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df  = pd.read_csv('../input/USA_Housing.csv')
df.info()
# So, we have 6 countinuos variable and one categorical variable. 
# Now, lets look at some of the data.
df.head()
df.describe()
# lets see if there is any null/missing values in the datasets or not. It's important to remove or
# replace all the missing values before moving further. 
df.isna().sum()
# We don't have any missing values. So, we are good to go. 
# Now, let's understand the correlation between variable by plotting correlation plot.
df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True)
# As we can see that price is more correlated to Avg. Income Area, House Age and Area Population than Number of Bedrooms and Rooms. Lets see these metrics in tabular format.
df.corr().Price.sort_values(ascending=False)
sns.pairplot(df)
# As we can see here in the last line of graph that all the features seems to be in a linear relationship with price except Avg. Area Number of Bedroom.
# We can also see this by plotting a separate graph

plt.scatter(df.Price, df[['Avg. Area Income']])
sns.distplot(df.Price)
# We can see the price plot seems like a bell shaped curve and all the price is normally distributed.
df = df.drop(['Address'], axis=1)
df.head()
from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()
feature = df.drop(['Price'], axis = 1)
label = df.Price

# Now, we have feature and label for machine learning algorithms. Now, we can scale the data by using standard scaler.

feature = pre_process.fit_transform(feature)
feature 
#this is how the scaled data looks like.
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, label.values, test_size = 0.2, random_state = 19)
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(feature_train, label_train)
from sklearn.metrics import r2_score, mean_squared_error

score = r2_score(linear_regression.predict(feature_train), label_train)
error = mean_squared_error(linear_regression.predict(feature_train), label_train)
score, error
linear_regression.coef_
linear_regression.intercept_
pd.DataFrame(linear_regression.coef_, index=df.columns[:-1], columns=['Values'])
# Applying this on test data.
score_test = r2_score(linear_regression.predict(feature_test), label_test)
score_test
ransac = linear_model.RANSACRegressor()
ransac.fit(feature_train, label_train)

# Scoring the Ransac model

ransac_r2_score = r2_score(ransac.predict(feature_test), label_test)
ransac_r2_score
ridge_model = linear_model.Ridge()
ridge_model.fit(feature_train, label_train)

# Scoring the Ridge Regression

ridge_r2_score = r2_score(ridge_model.predict(feature_test), label_test)
ridge_r2_score

from sklearn import tree
tree_model = tree.DecisionTreeRegressor()
tree_model.fit(feature_train, label_train)

# Scoring the Ridge Regression

tree_r2_score = r2_score(tree_model.predict(feature_test), label_test)
tree_r2_score
from sklearn.ensemble import RandomForestRegressor
random_model = RandomForestRegressor()
random_model.fit(feature_train, label_train)

# Scoring the Ridge Regression

random_r2_score = r2_score(tree_model.predict(feature_test), label_test)
random_r2_score
data = [score_test, ransac_r2_score, ridge_r2_score, tree_r2_score,random_r2_score]
index = ['Linear Regression', 'Ransac Regression', 'Ridge Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
pd.DataFrame(data, index=index, columns=['Scores']).sort_values(ascending = False, by=['Scores'])
