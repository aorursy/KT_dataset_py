import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import os 
import statsmodels.api as sm 
import seaborn as sns
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import warnings
from sklearn import metrics
%matplotlib inline
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
df = pd.read_csv('../input/placementdata/Placement_Data_Full_Class.csv')
df.info()
del df['sl_no']
df.fillna(0,inplace=True)
df.head()
df.describe()
df_placed = df.query('status == "Placed"')
df_placed.shape
sns.boxplot(data=df_placed,x=df['salary']);
z=np.abs(stats.zscore(df_placed.salary))
print(z)
print(np.where(z>3))
df_placed=df_placed[(z< 3)]
df_placed.shape
sns.boxplot(data=df_placed,x=df['hsc_p']);
z=np.abs(stats.zscore(df_placed.hsc_p))
print(z)
print(np.where(z>3))
df_placed=df_placed[(z< 3)]
df_placed.shape
sns.boxplot(data=df_placed,x=df['degree_p']);
z=np.abs(stats.zscore(df_placed.degree_p))
print(z)
print(np.where(z>3))
df_placed=df_placed[(z< 3)]
df_placed.shape
df.status.value_counts()
sns.countplot(x='status', data=df)
plt.show()
%matplotlib inline
pd.crosstab(df.workex,df.status).plot(kind='bar')
plt.title('work ex status for placed and not placed students ')
plt.xlabel('workex')
plt.ylabel('Placement status');
%matplotlib inline
pd.crosstab(df.gender,df.status).plot(kind='bar')
plt.title('gender status for placed and not placed students ')
plt.xlabel('gender')
plt.ylabel('Placement status');
%matplotlib inline
pd.crosstab(df.specialisation,df.status).plot(kind='bar')
plt.title('specialisation status for placed and not placed students ')
plt.xlabel('specialisation')
plt.ylabel('Placement status');
sns.lineplot(x = 'etest_p', y = 'salary', data = df_placed);
sns.lineplot(x = 'degree_p', y = 'salary', data = df_placed);
sns.lineplot(x = 'mba_p', y = 'salary', data = df_placed);
df_temp = df_placed[['hsc_b','hsc_s','salary']]
df_group = df_temp.groupby(['hsc_b','hsc_s'],as_index=False).mean()
df_group
correlation_matrix = df_placed.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
df[['drop','Placed']] = pd.get_dummies(df.status)
df.drop('drop', axis=1, inplace = True)
df.head()
df[['Female','drop']] = pd.get_dummies(df.gender)
df.drop('drop', axis=1, inplace = True)
df.head()
df[['drop','Mkt&HR']] = pd.get_dummies(df.specialisation)
df.drop('drop', axis=1, inplace = True)
df.head()
df[['Central_hsc','drop']] = pd.get_dummies(df.hsc_b)
df.drop('drop', axis=1, inplace = True)
df.head()
df[['Central_ssc','drop']] = pd.get_dummies(df.ssc_b)
df.drop('drop', axis=1, inplace = True)
df.head()
df[['Arts','Commerce','Science']] = pd.get_dummies(df.hsc_s)
df.drop('Arts', axis=1, inplace = True)
df.head()
df[['Comm&Mgmt','drop','Sci&Tech',]] = pd.get_dummies(df.degree_t)
df.drop('drop', axis=1, inplace = True)
df.head()
df[['drop','WorkEx']] = pd.get_dummies(df.workex)
df.drop('drop', axis=1, inplace = True)
df.head()
df.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status'], axis=1, inplace = True)
df.head()
correlated_features = set()
correlation_matrix = df.drop('Placed', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
correlated_features
df.drop(['Sci&Tech','Science'], axis=1, inplace = True)
X = df.drop('Placed', axis=1)
target = df['Placed']

rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)
print('Optimal number of features: {}'.format(rfecv.n_features_))
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()
lgm=sm.Logit(df['Placed'],df[['Female', 'Mkt&HR', 'Central_hsc', 'Central_ssc','Commerce','Comm&Mgmt','WorkEx']])
result=lgm.fit()
result.summary()
X = df_placed['degree_p'].values.reshape(-1,1)
y = df_placed['salary'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df_test = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_test
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
X = df_placed[['ssc_p', 'hsc_p', 'etest_p', 'mba_p', 'degree_p']]
y = df_placed['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df
y_pred = regressor.predict(X_test)
y_pred
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df2
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
