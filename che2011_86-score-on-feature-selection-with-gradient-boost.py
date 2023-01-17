import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
# Open file and inspect first five rows.
df = pd.read_csv('../input/Melbourne_housing_FULL.csv')
df.head()
# Check number of rows.
df.shape
# Run descriptive statistics.
df.describe()
# Confirm number of records with nulls for each column.
df.isnull().sum()
# Return column data type.
df.dtypes
# Check correlations.
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot= True, annot_kws={"size": 10})
# Drop unnecessary, redundant and highly correlated variables (i.e., Bedroom2 vs. Rooms).
df = df.drop(['Address', 'SellerG', 'Date', 'Postcode', 'Bedroom2', 'CouncilArea', 'Suburb'], axis=1)
# Create a feature set consisting of the float columns.
flo_feat = df.select_dtypes(exclude=['object'])

# Create the zero-imputed dataframe.
df_fz = flo_feat.fillna(0)

# Split the dataframe into a train and test set.
from sklearn.model_selection import train_test_split
X_train_fz, X_test_fz, y_train_fz, y_test_fz = train_test_split(df_fz.drop(['Price'],axis=1), df_fz['Price'], test_size=0.3, random_state=123)

# Fit and run a linear regression model.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_fz, y_train_fz)
lr_pred_fz = lr.predict(X_test_fz)

# Return the mean absolute error.
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test_fz, lr_pred_fz))
# Create the mean-imputed dataframe.
df_im = flo_feat.fillna(flo_feat.mean())

# Split the dataframe into a train and test set.
X_train_im, X_test_im, y_train_im, y_test_im = train_test_split(df_im.drop(['Price'],axis=1), df_im['Price'], test_size=0.3, random_state=123)

# Fit and run a linear regression model.
lr.fit(X_train_im, y_train_im)
lr_pred_im = lr.predict(X_test_im)

# Return the mean absolute error.
print('MAE: ', metrics.mean_absolute_error(y_test_im, lr_pred_im))
# Create the dataframe where null rows are dropped.
df_d = flo_feat.dropna()

# Split the dataframe into a train and test set.
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(df_d.drop(['Price'],axis=1), df_d['Price'], test_size=0.3, random_state=123)

# Fit and run a linear regression model.
lr.fit(X_train_d, y_train_d)
lr_pred_d = lr.predict(X_test_d)

# Return the mean absolute error.
print('MAE: ', metrics.mean_absolute_error(y_test_d, lr_pred_d))
# Create the final dataframe with the the desired features and null rows dropped.
df_dr = df.dropna()
df_dr.shape
# Create dummies for object features.
df_final = pd.concat([df_dr, pd.get_dummies(df_dr[['Regionname', 'Type', 'Method']])], axis=1)

# Create an age feature based on the year built column.
df_final['age'] = 2018 - df_final['YearBuilt']
df_final['age'] = np.where(df_final['age']>20, 1, 0)

# Drop the redundant features. 
df_final = df_final.drop(['YearBuilt', 'Regionname', 'Type', 'Method'], axis=1)
df_final.shape
# Show distribution of bedrooms and bathrooms.
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,sharey=True, figsize=(20, 8))

df_final['Rooms'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_title('Number of Bedrooms')
ax1.set_ylabel('Count')

df_final['Bathroom'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_title('Number of Bathrooms')
plt.show()
# Visualize the home locations based on latitude and longitude.
plt.figure(figsize=(10,10))
sns.jointplot(x=df_final.Lattitude.values, y=df_final.Longtitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
# Create scatterplots showing numeric feature relationships.
g = sns.PairGrid(df_d)
g = g.map(plt.scatter)
# Split the dataframe into a train and test set.
X = df_final.drop('Price',axis=1)
Y = df_final['Price']
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, Y, test_size=0.3, random_state=123)

# Fit and run a linear regression model.
lr.fit(X_train_final, y_train_final)
lr_pred_final = lr.predict(X_test_final)

# Print score and errors.
print('Score: ', lr.score(X_test_final, y_test_final))
print('MAE: ', metrics.mean_absolute_error(y_test_final, lr_pred_final))
print('MSE: ', metrics.mean_squared_error(y_test_final, lr_pred_final))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test_final, lr_pred_final)))
# Fit and run a gradient boost model.
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='huber')
clf.fit(X_train_final, y_train_final)
clf_pred = clf.predict(X_test_final)

# Print score and errors.
print('Score: ', clf.score(X_test_final, y_test_final))
print('MAE: ', metrics.mean_absolute_error(y_test_final, clf_pred))
print('MSE: ', metrics.mean_squared_error(y_test_final, clf_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test_final, clf_pred)))
# Fit and run a random forest model.
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train_final, y_train_final)
rf_pred = rf.predict(X_test_final)

# Print score and errors.
print('Score: ', rf.score(X_test_final, y_test_final))
print('MAE: ', metrics.mean_absolute_error(y_test_final, rf_pred))
print('MSE: ', metrics.mean_squared_error(y_test_final, rf_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test_final, clf_pred)))
# Convert latitude to absolute to prevent negative value error. 
df_final['Lattitude'] = df_final['Lattitude'].abs()
X = df_final.drop('Price',axis=1)

# Find the optimum number of features (k) for SelectKBest.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
for k in range(1,df_final.shape[1]):
    X_skb = SelectKBest(chi2, k=k).fit_transform(X, Y)
    X_train_skb, X_test_skb, y_train_skb, y_test_skb = train_test_split(X_skb, Y, test_size = .3, random_state=25)
    clf.fit(X_train_skb, y_train_skb)
    clf_pred_skb = clf.predict(X_test_skb)
    print('Score at k =', k, ':', clf.score(X_test_skb, y_test_skb))
