import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.feature_selection as feat_sel
import sklearn.ensemble as ensemble
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# Nº of rows and colums
print('Train: Rows - '+str(len(df_train)) + ' Columns - ' + str(len(df_train.columns)))
print('Test: Rows - '+str(len(df_test)) + ' Columns - ' + str(len(df_test.columns)))

# Type of columns
train_col_types = df_train.dtypes
test_col_types = df_train.dtypes
print('-'*60)
print('Train: Type of columns')
print('-'*60)
print(train_col_types.groupby(train_col_types).count())
print('-'*60)
print('Test: Type of columns')
print('-'*60)
print(test_col_types.groupby(test_col_types).count())

# Missing values?
print('-'*60)
list = []
counts = []
for i in df_train.columns:
    list.append(i)
    counts.append(sum(df_train[i].isnull()))
print('Train: Nº of columns with missing values')
print('-'*60)
print(sum(counts))
print('-'*60)
list = []
counts = []
for i in df_test.columns:
    list.append(i)
    counts.append(sum(df_test[i].isnull()))
print('Test: Nº of columns with missing values')
print('-'*60)
print(sum(counts))
# Columns with all rows zero
columns_train_sum = pd.DataFrame(df_train.sum(),columns=['Sum of Row'])
print('Train: Nº of columns with all rows zero train: ' + str(columns_train_sum[columns_train_sum==0].count()))
# Is any ID on the test dataset?
for i in df_train.ID.values:
    c = 0
    if i in df_test.ID.values:
        c = c + 1
print('Nº of ID''s on the test dataset: ' + str(c))
# There's any visibla outlier on the target?
#plt.scatter(range(df_train.shape[0]), np.sort(df_train['target'].values))
plt.plot(df_train.ID, np.sort(df_train.target.values))
plt.xlabel('ID')
plt.ylabel('Target')
plt.title('ID vs Target')
plt.show()
# How is the target distribuition
df_train.target.hist()
plt.xlabel('Target')
plt.ylabel('Nº of ID''s')
plt.title('Histogram of target')
plt.show()
# How is the log target distribuition
df_train['target_log'] = np.log(df_train.target)
df_train.target_log.hist()
plt.xlabel('Target')
plt.ylabel('Nº of ID''s')
plt.title('Histogram of log target')
plt.show()
## Drop columns will all zero values
list_columns_train_drop=[]
for i in columns_train_sum[columns_train_sum['Sum of Row']==0].index:
    list_columns_train_drop.append(i)
df_train = df_train.drop(columns=list_columns_train_drop)
len(df_train.columns)
## Verify the correlatin between the target and the variables
corr_train_target_values = []
corr_train_target_column = []
for i in df_train.columns:
    if i in ['ID','target']:
        None
    else:
        corr = df_train[['target',i]].corr(method='spearman')
        corr_train_target_values.append(corr.target[1])
        corr_train_target_column.append(i)

corr_train_target = pd.DataFrame(corr_train_target_values,index=corr_train_target_column)
corr_train_target.describe()
X = df_train.drop(columns=['target','target_log','ID'])
variable_mean = X.mean()
variable_std = X.std()
variable_name = X.columns
high_indices = np.argsort(variable_mean)[::-1][:50]
low_indices = np.argsort(variable_mean)[:50]
plt.bar(range(len(variable_mean[high_indices])),variable_mean[high_indices],yerr=variable_std[high_indices])
#plt.xticks(range(len(variable_mean[high_indices])),variable_name[high_indices],rotation='vertical')
plt.show()
plt.bar(range(len(variable_mean[low_indices])),variable_mean[low_indices],yerr=variable_std[low_indices])
#plt.xticks(range(len(variable_mean[low_indices])),variable_name[low_indices],rotation='vertical')
plt.show()
# Implementing a PCA to reduze the amount of variables and standardize data

from sklearn import preprocessing
from sklearn.decomposition import PCA

X = df_train.drop(columns=['target','ID','target_log'])
X = preprocessing.scale(X)
list_n_comp=[]
list_var_ratio=[]
n_comp = 100
max_list_var_ratio = 0.0
while max_list_var_ratio<0.8: #n_comp <= 1000:
    print(n_comp)
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    list_n_comp.append(n_comp)
    list_var_ratio.append(sum(pca.explained_variance_ratio_))
    max_list_var_ratio = max(list_var_ratio)
    print(max_list_var_ratio)
    n_comp = n_comp + 100
#list_n_comp,list_var_ratio

plt.plot(list_n_comp, list_var_ratio)
plt.xlabel('Number of components')
plt.ylabel('Variance Ratio')
plt.title('PCA')
plt.ylim([0,1])
plt.axhline(0.8,color='r')
plt.show()
# PCA with 80% explained ratio is with 900 components
from sklearn import preprocessing
from sklearn.decomposition import PCA

X = df_train.drop(columns=['target','ID','target_log'])
X = pd.DataFrame(preprocessing.scale(X),columns = X.columns)
pca = PCA(n_components=900)
X_pca = pd.DataFrame(pca.fit_transform(X))

df_train_pca = df_train[['ID','target','target_log']]
df_train_pca[X_pca.columns.values]= X_pca

X = df_test.drop(columns=['ID'])
X = pd.DataFrame(preprocessing.scale(X),columns = X.columns)
pca = PCA(n_components=900)
X_pca = pd.DataFrame(pca.fit_transform(X))

df_test_pca = pd.DataFrame(df_test['ID'],columns=['ID'])
df_test_pca[X_pca.columns.values]= X_pca
# Split train data into test and train
X = df_train_pca.drop(columns=['target','ID','target_log'])
y = df_train_pca.target_log
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Finding the best model out of the box
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

def score(y_true,y_pred):
    score = np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred)))
    return score
my_score = make_scorer(score,greater_is_better=False)

list_model = [Ridge(),SVR(),GradientBoostingRegressor(),RandomForestRegressor(),AdaBoostRegressor(),MLPRegressor()]
for i in list_model:
    print(i)
    print(cross_val_score(i, X_train, y_train, cv=5, scoring = my_score))
    print('-'*60)
# Finding best hyperparameters

parameters = {'n_estimators':[50,100,150],'max_depth':[3,5]}
model = GradientBoostingRegressor()
grid = GridSearchCV(model, parameters,scoring=my_score)
grid_result = grid.fit(X_train,y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Using best hyperparameters and evaluate test set

model = GradientBoostingRegressor(n_estimators=100,max_depth=3)
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
results = {'rmse train': np.sqrt(mean_squared_log_error(np.exp(y_train), np.exp(y_pred_train))),
           'rmse test': np.sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_test))),
           'std y_train': np.exp(y_train).std(),
           'mean y_train': np.exp(y_train).mean(),
           'std y_test': np.exp(y_test).std(),
           'mean y_test': np.exp(y_test).mean(),
           'std y_pred_train': np.exp(y_pred_train).std(),
           'mean y_pred_train': np.exp(y_pred_train).mean(),
           'std y_pred_test': np.exp(y_pred_test).std(),
          }
results
# make prediction with gradient boost
X = df_test_pca.drop(columns=['ID'])
df_test_pca['target'] = np.exp(model.predict(X))
df_test_pca[['ID','target']].to_csv('subsmission_gb.csv',index=False,sep=',')
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

# Split train data into test and train
X = df_train_pca.drop(columns=['target','ID','target_log'])
y = df_train_pca.target_log
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Finding best hyperparameters

from sklearn.metrics import make_scorer
def score(y_true,y_pred):
    score = np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred)))
    return score
my_score = make_scorer(score,greater_is_better=False)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def create_model():
    model = Sequential()
    model.add(Dense(2,input_dim=900,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='relu'))
    model.compile(loss=root_mean_squared_error,optimizer='SGD',metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=10,batch_size=3)

epochs = [5,10,20]
batch_size = [3,5,7]
verbose = [0]
parameters = dict(epochs=epochs,batch_size=batch_size,verbose=verbose)

grid = GridSearchCV(model, parameters,scoring=my_score)
grid_result = grid.fit(X_train,y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Using best hyperparameters and evaluate test set

model = Sequential()
model.add(Dense(2,input_dim=900,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='relu'))
model.compile(loss=root_mean_squared_error,optimizer='SGD')

model.fit(X_train, y_train, epochs=10, batch_size=7,verbose=1)
y_pred_train = model.predict(X_train, batch_size=32)
y_pred_test = model.predict(X_test, batch_size=32)

results = {'rmse train': np.sqrt(mean_squared_log_error(np.exp(y_train), np.exp(y_pred_train))),
           'rmse test': np.sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_test))),
           'std y_train': np.exp(y_train).std(),
           'mean y_train': np.exp(y_train).mean(),
           'std y_test': np.exp(y_test).std(),
           'mean y_test': np.exp(y_test).mean(),
           'std y_pred_train': np.exp(y_pred_train).std(),
           'mean y_pred_train': np.exp(y_pred_train).mean(),
           'std y_pred_test': np.exp(y_pred_test).std(),
          }
results
# make prediction with nueral network
X = df_test_pca.drop(columns=['ID','target'])
df_test_pca['target'] = np.exp(model.predict(X))
df_test_pca[['ID','target']].to_csv('subsmission_nn.csv',index=False,sep=',')
