import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
df_train.describe()
df_train.shape, df_test.shape
test_index=df_test['Unnamed: 0']
features = []
for i in range (3, 18):
    features.append("F" + str(i))
#checking if any column has null values
df_train.isna().sum()
df_train.info()
categorical_cols = [ftr for ftr in features if len(df_train[ftr].unique()) < 25]
continuous_cols = [ftr for ftr in features if ftr not in categorical_cols]
#count plots for categorical columns
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.countplot(x='F3', palette = 'winter', data = df_train, ax=ax[0][0])
sns.countplot(x='F4', palette = 'winter', data = df_train, ax=ax[0][1])
sns.countplot(x='F5', palette = 'winter', data = df_train, ax=ax[1][0])
sns.countplot(x='F7', palette = 'winter', data = df_train, ax=ax[1][1])
sns.countplot(x='F8', palette = 'winter', data = df_train, ax=ax[2][0])
sns.countplot(x='F9', palette = 'winter', data = df_train, ax=ax[2][1])
sns.countplot(x='F11', palette = 'winter', data = df_train, ax=ax[3][0])
sns.countplot(x='F12', palette = 'winter', data = df_train, ax=ax[3][1])
plt.show()
#KDE plots for continuous features
fig, ax = plt.subplots(4, 2, figsize=(16,16))
sns.kdeplot(data = df_train['F6'], color='salmon', shade=True, ax=ax[0][0])
sns.kdeplot(data = df_train['F10'], color='salmon', shade=True, ax=ax[0][1])
sns.kdeplot(data = df_train['F13'], color='salmon', shade=True, ax=ax[1][0])
sns.kdeplot(data = df_train['F14'], color='salmon', shade=True, ax=ax[1][1])
sns.kdeplot(data = df_train['F15'], color='salmon', shade=True, ax=ax[2][0])
sns.kdeplot(data = df_train['F16'], color='salmon', shade=True, ax=ax[2][1])
sns.kdeplot(data = df_train['F17'], color='salmon', shade=True, ax=ax[3][0])
fig.delaxes(ax[3,1])
#scatterplots for continuous variables
fig, ax = plt.subplots(4,2, figsize=(16,16))
k = 0
for i in range(4):
    for j in range(2):
        if i != 3 or j != 1:
            sns.scatterplot(x=continuous_cols[k], y='O/P', data=df_train, palette='coolwarm', ax=ax[i][j])
            k+=1
fig.delaxes(ax[3,1])
plt.show()
#correlation between different columns
corrDf = df_train.corr()
matrixMask = np.triu(corrDf, k=1)
plt.figure(figsize=(16,12))
sns.heatmap(corrDf, annot=True, mask=matrixMask, cmap='copper')
plt.title("Correlation Matrix")
plt.show()
#trying standardization and seeing if it makes a difference
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_train[continuous_cols] = scaler.fit_transform(df_train[continuous_cols])
df_train
#scatterplots for continuous variables
fig, ax = plt.subplots(4,2, figsize=(16,16))
k = 0
for i in range(4):
    for j in range(2):
        if i != 3 or j != 1:
            sns.scatterplot(x=continuous_cols[k], y='O/P', data=df_train, palette='coolwarm', ax=ax[i][j])
            k+=1
fig.delaxes(ax[3,1])
plt.show()
#correlation between different columns
corrDf = df_train.corr()
matrixMask = np.triu(corrDf, k=1)
plt.figure(figsize=(16,12))
sns.heatmap(corrDf, annot=True, mask=matrixMask, cmap='copper')
plt.title("Correlation Matrix")
plt.show()
scaler = MinMaxScaler()
df_train[continuous_cols] = scaler.fit_transform(df_train[continuous_cols])
df_train
#correlation between different columns
corrDf = df_train.corr()
matrixMask = np.triu(corrDf, k=1)
plt.figure(figsize=(16,12))
sns.heatmap(corrDf, annot=True, mask=matrixMask, cmap='copper')
plt.title("Correlation Matrix")
plt.show()
df_train.drop(['Unnamed: 0','F1', 'F2', 'F6', 'F10'], axis = 1, inplace = True)
df_test.drop(['F6', 'F10'], axis = 1, inplace = True)  
#removed F1, F2 as they are index and dates respectively, F6 and F10 have very little impact on output
X = df_train[[f for f in features if f not in ['F6', 'F10']]]
Y = df_train['O/P']
X,Y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = .35, random_state= 43)

#from sklearn.model_selection import RandomizedSearchCV as RCV, GridSearchCV as GCV
from sklearn.metrics import mean_squared_error
# creating a random grid for Random Search (doing Random Search first to get an idea)
#n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#max_features = ['auto', 'sqrt']
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
#min_samples_split = [2, 5, 10]
#min_samples_leaf = [1, 2, 4]
#bootstrap = [True, False]
# Create the random grid
#random_grid = {'n_estimators': n_estimators,
 #              'max_features': max_features,
  #             'max_depth': max_depth,
   #            'min_samples_split': min_samples_split,
    #           'min_samples_leaf': min_samples_leaf,
     #          'bootstrap': bootstrap}
#print(random_grid)
#rf = RandomForestRegressor()
#rf_random = RCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 5, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(x_train, y_train)
#rf_random.best_params_
#base_mod = RandomForestRegressor(n_estimators = 50, random_state = 43)
#base_mod.fit(x_train, y_train)
#pred = base_mod.predict(x_test)
#base_mse = mean_squared_error(y_test, pred)
#base_mse
#best_rand = rf_random.best_estimator_
#best_pred = best_rand.predict(x_test)
#best_mse = mean_squared_error(y_test, best_pred)
#best_mse
#Grid Search for better evaluation
#param_grid = {
 #   'bootstrap': [True],
  #  'max_depth': [20, 30],
   # 'max_features': ['auto'],
    #'min_samples_leaf': [2, 3, 4, 5],
    #'min_samples_split': [2, 4, 6],
    #'n_estimators': [200, 300, 400, 500, 600, 700, 800, 1200]
#}
#rf = RandomForestRegressor()
#grid_search = GCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
#grid_search.fit(x_train, y_train)
#grid_search.best_params_
gs = RandomForestRegressor(n_estimators=600, random_state=43, max_depth=20, min_samples_leaf=2)
gs = gs.fit(x_train, y_train)
gs_pred = gs.predict(x_test)
gs_mse = mean_squared_error(y_test, gs_pred)
gs_mse
#best parameters acc. to Grid Search have been used
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.03, 
                         max_depth=8, objective='reg:squarederror', 
                         random_state=43, subsample=0.7)
xgb_model.fit(x_train, y_train)
y_pred_xgb = xgb_model.predict(x_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mse_xgb
#parameters decided by manual tuning 

#mse is way better, so fitting it to the actual training set
xgb_model.fit(X, Y)

df_test = df_test.loc[:, 'F3':'F17']
pred_xgb = xgb_model.predict(df_test)
pred_xgb
rf = RandomForestRegressor(n_estimators=200, random_state=43)
rf.fit(X, Y)

df_test = df_test.loc[:, 'F3':'F17']
pred = rf.predict(df_test)
print(pred)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred_xgb)
result.head()
result.to_csv('output.csv', index=False)
