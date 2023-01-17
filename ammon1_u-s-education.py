import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/states_all.csv')
df.isna().sum()*100/df.shape[0]

corr = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
fig, ax = plt.subplots(figsize=(15, 15))
sns.violinplot(x="TOTAL_REVENUE", y="STATE", data=df)
df['average_reveneue']=df['TOTAL_REVENUE']/df['GRADES_ALL_G']
df['average_expenditure']=df['TOTAL_EXPENDITURE']/df['GRADES_ALL_G']
fig, ax = plt.subplots(figsize=(15, 15))
sns.violinplot(x="average_reveneue", y="STATE", data=df[df.STATE!='VIRGINIA'])
fig, ax = plt.subplots(figsize=(15, 15))
sns.violinplot(x="average_expenditure", y="STATE", data=df[df.STATE!='VIRGINIA'])
sns.jointplot("average_expenditure", "AVG_MATH_4_SCORE", data=df, kind="reg")
sns.jointplot("TOTAL_REVENUE", "AVG_MATH_4_SCORE", data=df, kind="reg")
df1=df.drop(['PRIMARY_KEY','AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE','ENROLL'],axis=1)
df2=df1.dropna()
df3 = pd.get_dummies(df2, columns=['STATE'])
df4=(df3-df3.mean())/df3.std()
y=df4.loc[:,'AVG_MATH_4_SCORE'].values
X=df4.drop(['AVG_MATH_4_SCORE'],axis=1).loc[:,:].values
print(X.shape,' ',y.shape)
rf=RandomForestRegressor()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',rf)
print('Score ',rf.score(X, y))
Y_rf=rf.predict(X)
plt.plot(Y_rf, y, 'ro')
plt.show()
feature_importances_rf = pd.DataFrame(rf.feature_importances_,
                                   index = df4.drop(['AVG_MATH_4_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_rf.head()
from sklearn.linear_model import Lasso, ElasticNet
clf = Lasso()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'tol': [0.00001,0.0001,0.001, 0.01]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',clf)
print('Score ',clf.score(X, y))
Y_clf=clf.predict(X)
plt.plot(Y_clf, y, 'ro')
plt.show()
feature_importances_clf = pd.DataFrame(clf.coef_,
                                   index = df4.drop(['AVG_MATH_4_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_clf.head(10)
eln = ElasticNet()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'l1_ratio': [0.0001,0.001,0.01, 0.1],
              'tol': [0.00001,0.0001,0.001, 0.01],
              'max_iter': [1000,2000,5000, 10000],
             }

# Run the grid search
grid_obj = GridSearchCV(eln, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
eln = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',eln)
print('Score ',eln.score(X, y))
Y_eln=eln.predict(X)
plt.plot(Y_eln, y, 'ro')
plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
plt.plot(Y_eln, y, 'ro',label='Elastic Net')#red Elastic Net
plt.plot(Y_clf, y, 'bs',label='Lasso')#blue Lasso
plt.plot(Y_rf, y, 'g^',label='Random Forest')#green Random Forest
plt.plot([-5,2],[-5,2])#PREDICTION LIKE IT SHOUD BE
plt.legend()
plt.show()
feature_importances_eln = pd.DataFrame(eln.coef_,
                                   index = df4.drop(['AVG_MATH_4_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_eln.head(10)
fig, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(x="YEAR", y="AVG_MATH_4_SCORE", data=df)
df1=df.drop(['PRIMARY_KEY','AVG_MATH_4_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE','ENROLL'],axis=1)
df2=df1.dropna()
df3 = pd.get_dummies(df2, columns=['STATE'])
df4=(df3-df3.mean())/df3.std()
y=df4.loc[:,'AVG_MATH_8_SCORE'].values
X=df4.drop(['AVG_MATH_8_SCORE'],axis=1).loc[:,:].values
rf=RandomForestRegressor()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',rf)
print('Score ',rf.score(X, y))
Y_rf=rf.predict(X)
plt.plot(Y_rf, y, 'ro')
plt.show()
feature_importances_rf = pd.DataFrame(rf.feature_importances_,
                                   index = df4.drop(['AVG_MATH_8_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_rf.head()
clf = Lasso()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'tol': [0.00001,0.0001,0.001, 0.01]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',clf)
print('Score ',clf.score(X, y))
Y_clf=clf.predict(X)
plt.plot(Y_clf, y, 'ro')
plt.show()
feature_importances_clf = pd.DataFrame(clf.coef_,
                                   index = df4.drop(['AVG_MATH_8_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_clf.head(10)
eln = ElasticNet()

parameters = {'alpha': [0.00001,0.0001,0.001, 0.01],
              'l1_ratio': [0.0001,0.001,0.01, 0.1],
              'tol': [0.00001,0.0001,0.001, 0.01],
              'max_iter': [1000,2000,5000, 10000],
             }

# Run the grid search
grid_obj = GridSearchCV(eln, parameters, cv=5)
grid_obj = grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
eln = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
print('Params ',eln)
print('Score ',eln.score(X, y))
Y_eln=clf.predict(X)
plt.plot(Y_eln, y, 'ro')
plt.show()
feature_importances_eln = pd.DataFrame(eln.coef_,
                                   index = df4.drop(['AVG_MATH_8_SCORE'],axis=1).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_eln.head(10)
fig, ax = plt.subplots(figsize=(15, 15))
plt.plot(Y_eln, y, 'ro',label='Elastic Net')#red Elastic Net
plt.plot(Y_clf, y, 'bs',label='Lasso')#blue Lasso
plt.plot(Y_rf, y, 'g^',label='Random Forest')#green Random Forest
plt.plot([-5,2],[-5,2])#PREDICTION LIKE IT SHOUD BE
plt.legend()
plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
sns.boxplot(x="AVG_MATH_8_SCORE", y="STATE", data=df)
states=df.groupby(['STATE']).mean()
normalized_df=(states-states.mean())/states.std()
fig, ax = plt.subplots(figsize=(15, 15))
normalized_df['AVG_MATH_8_SCORE'].plot(kind='bar');