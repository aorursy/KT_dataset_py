import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Import Data
df = pd.read_csv('../input/real-estate-datacsv/real_estate_data.csv')
df.columns
# Drop Unwanted Columns
df = df.drop(['No', 'X1 transaction date', 'X5 latitude', 'X6 longitude'], axis=1)
df.columns
# Rename Columns
df.columns = ['house_age',
       'distance_to_the_nearest_MRT_station',
       'number_of_convenience_stores', 'house_price_of_unit_area']

df.head()
# Data Ser Info
df.info()
df.describe()
sns.pairplot(df)
sns.distplot(df['house_price_of_unit_area'], hist_kws=dict(edgecolor="black", linewidth=2))
sns.boxplot(x='number_of_convenience_stores', y='house_price_of_unit_area', data=df)
mat = df.corr()
mat
sns.heatmap(mat, annot=True)
# One hot Encoding Dummies
df['number_of_convenience_stores'] = df['number_of_convenience_stores'].astype('category')
## 3. Dummy Coding ##
col_dummies = pd.get_dummies(df['number_of_convenience_stores'])
df = pd.concat([df, col_dummies], axis=1)
del df['number_of_convenience_stores']
df.head()
# Features and target variable
X = df.drop(['house_price_of_unit_area'], axis=1)
y = df['house_price_of_unit_area']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.linear_model import LinearRegression
# Instantiate Model
model = LinearRegression()
print(model)
# Train Model
model.fit(X_train, y_train)
# Intercept
model.intercept_
# Coeffcients
model.coef_
# cdf values
pd.DataFrame(model.coef_, X.columns, columns=['coef'])
predictions = model.predict(X_test)
y_test.isnull().sum()
y_test = y_test.reset_index(drop=True)
pred_df = pd.concat([y_test, pd.Series(predictions)], axis=1, ignore_index=True)
pred_df.columns = ['y_test', 'predicted']
pred_df
plot_df = pred_df.head(50)
plot_df.plot(kind='bar',figsize=(12,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
ax = sns.scatterplot(x='y_test', y='predicted', data=pred_df)
# Distribution of Residuals
sns.distplot(y_test-predictions, hist_kws=dict(edgecolor="black", linewidth=2))
plt.title('Distribution of Residuals')
from sklearn.metrics import mean_squared_error
# Root Mean Squared Error
np.sqrt(mean_squared_error(y_test, predictions))



