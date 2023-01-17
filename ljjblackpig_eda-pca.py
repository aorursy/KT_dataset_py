import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Show all the rows & Columns of data

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
#Read in the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(3)
test.head(3)
#Checking shape of train & test data

print(train.shape, test.shape)
#Checking missing values

print('Is the training data contains any missing values? ' + str(train.isnull().any().any()) + '\n'

     + 'Is the testing data contains any missing values? ' + str(test.isnull().any().any()))
#Checking column types

train.info(verbose = 1)
train.describe()
#Visualizing the response variable

sns.countplot(train['target'])
#Take out the response variable and sample 3 dataset

target = train.iloc[:, train.columns == 'target']

sample1 = pd.concat([train.iloc[:, 2:10], target], axis = 1)

sample2 = pd.concat([train.iloc[:, 25:35], target], axis = 1)
sample1.head(3)
cor1 = sample1.corr()

cor2 = sample2.corr()

f, ax = plt.subplots(1, 2, figsize = (12, 8))

sns.heatmap(cor1, vmax = 0.9, annot = True, square = True, fmt = '.2f', ax=ax[0])

sns.heatmap(cor2, vmax = 0.9, annot = True, square = True, fmt = '.2f', ax=ax[1])
#Checking correlation between features

#referening to : https://www.kaggle.com/allunia/santander-customer-transaction-eda

train_correlations = train.drop(["target"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_correlations, color="Red", label="train")

sns.distplot(test_correlations, color="Green", label="test")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
#Take out the data without id & response variable

train_tran = train.iloc[:, 2:]
#Doing PCA for our model

from sklearn.decomposition import PCA

from sklearn import preprocessing



pca = PCA()

pca.fit(train_tran)

pca.data = pca.transform(train_tran)





#Percentage variance of each pca component stands for

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

#Create labels for the scree plot

labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]



#Plot the data

plt.figure(figsize = (12, 10))

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label = labels)

plt.ylabel('percentage of Explained Variance')

plt.xlabel('Principle Component')

plt.title('Scree plot')

plt.show()
print('PC1 + PC150 add up to ' +  str(sum(per_var[:150])) + ' % of the variance')
#Extract the top 150 pc information

pc_columns = []

for i in range(150):

    pc_columns.append('PC' + str(i + 1))



PC_train = pd.DataFrame(data=pca.data[:,:150], columns = pc_columns)
PC_train.head(3)
PC_train.shape
#Setting the seed for calculation

seed = 2019
#train test split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(PC_train, target,

                                                   test_size = 0.25, random_state = seed)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train)  # Don't cheat - fit only on training data

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)  # apply same transformation to test data
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



rf_model = RandomForestClassifier(n_estimators=300, max_depth=10,

                                  oob_score=True,

                              random_state=seed)

rf_model.fit(X_train, y_train.values.ravel())
from sklearn import metrics



rf_pred = rf_model.predict(X_test)

print(metrics.accuracy_score(y_test, rf_pred))
#Get a validation dataset

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,

                                                   test_size = 0.25, random_state = seed)
import lightgbm as lgb



# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
#Tunning in hyperparameters

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'binary_logloss',

    'max_depth': 6,

    'learning_rate': 0.06,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.9,

    'bagging_freq': 5,

    'max_bin': 255,

    'verbose': 1

}
num_round = 2000

lgbm_model = lgb.train(params, lgb_train, num_round, valid_sets = lgb_eval, early_stopping_rounds = 10)
lgbm_pred = lgbm_model.predict(X_test, num_iteration = lgbm_model.best_iteration)

print(metrics.accuracy_score(y_test, lgbm_pred.round()))