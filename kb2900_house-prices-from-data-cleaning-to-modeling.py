import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

sns.set_context('notebook', font_scale=1.5)

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
train.shape
test.head()
test.shape
train_num = train.select_dtypes(exclude=['object']).columns

train_num
train.describe()
from scipy.stats import shapiro

# apply shapiro test

stat, p = shapiro(train['BsmtFinSF1'])

print('Skewness=%.3f' %train['BsmtFinSF1'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



# set alpha to 0.05: when p > 0.05, accept the null hypothesis; when p < 0.05, reject the null

alpha = 0.05

if p > alpha:

    print('Data looks normal (fail to reject H0)')

else:

    print('Data does not look normal (reject H0)')



sns.distplot(train['BsmtFinSF1']);
stat, p = shapiro(train['BsmtFullBath'])

print('Skewness=%.3f' %train['BsmtFullBath'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(train['BsmtFullBath']);
stat, p = shapiro(train['LotArea'])

print('Skewness=%.3f' %train['LotArea'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(train['LotArea']);
stat, p = shapiro(np.log(train['LotArea']))

print('After log transformation...')

print('Skewness=%.3f' %np.log(train['LotArea']).skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(np.log(train['LotArea']));
stat, p = shapiro(train['MasVnrArea'].dropna())

print('Skewness=%.3f' %train['MasVnrArea'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(train['MasVnrArea'].dropna());
masvnrarea_std = (train['MasVnrArea'] - np.mean(train['MasVnrArea'])) / np.std(train['MasVnrArea'])

stat, p = shapiro(masvnrarea_std.dropna())

print('Skewness=%.3f' %masvnrarea_std.skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(masvnrarea_std.dropna());
stat, p = shapiro(train['SalePrice'])

print('Skewness=%.3f' %train['SalePrice'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(train['SalePrice']);
stat, p = shapiro(np.log(train['SalePrice']))

print('After log transformation...')

print('Skewness=%.3f' %np.log(train['SalePrice']).skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(np.log(train['SalePrice']));
train_num_corr = train[train_num].drop(['Id'], axis=1)

corr = pd.DataFrame(train_num_corr.corr(method = 'pearson')['SalePrice'])

corr.sort_values(['SalePrice'], ascending= False)
cmap = sns.cubehelix_palette(light = 0.95, as_cmap = True)

sns.set(font_scale=1.2)

plt.figure(figsize = (9, 9))

sns.heatmap(abs(train_num_corr.corr(method = 'pearson')), vmin = 0, vmax = 1, square = True, cmap = cmap);
train_cat = train.select_dtypes(include=['object']).columns

train_cat
pd.set_option('display.max_rows', 300)

df_output = pd.DataFrame()

# loop through categorical variables, and append calculated stats together

for i in range(len(train_cat)):

    c = train_cat[i]

    df = pd.DataFrame({'Variable':[c]*len(train[c].unique()),

                       'Level':train[c].unique(),

                       'Count':train[c].value_counts(dropna = False)})

    df['Percentage'] = 100 * df['Count']  / df['Count'].sum()

    df_output = df_output.append(df, ignore_index = True)

    

df_output
sns.set(style = 'whitegrid', rc = {'figure.figsize':(10,7), 'axes.labelsize':12})

sns.boxplot(x = 'MSZoning', y = 'SalePrice', palette = 'Set2', data = train, linewidth = 1.5);
col_order = train.groupby(['Neighborhood'])['SalePrice'].aggregate(np.median).reset_index().sort_values('SalePrice')

p = sns.boxplot(x = 'Neighborhood', y = 'SalePrice', palette = 'Set2', data = train, order=col_order['Neighborhood'], linewidth = 1.5)

plt.setp(p.get_xticklabels(), rotation=45);
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', palette = 'Set2', data = train, linewidth = 1.5);
sns.scatterplot(x = 'YearBuilt', y = 'SalePrice', data = train, hue = 'HouseStyle', style = 'HouseStyle', palette = 'colorblind');
print(train.isnull().sum())
# training data

# calculate percentage of missing values

train_missing = pd.DataFrame(train.isnull().sum()/len(train.index) * 100)

train_missing.columns = ['percent']



# flag columns whose missing percentage are larger than 15%

train_missing.loc[train_missing['percent'] > 15, 'column_select'] = True

train_col_select = train_missing.index[train_missing['column_select'] == True].tolist()

train_col_select
# test data

test_missing = pd.DataFrame(test.isnull().sum()/len(test.index) * 100)

test_missing.columns = ['percent']

test_missing.loc[test_missing['percent'] > 15, 'column_select'] = True

test_col_select = test_missing.index[test_missing['column_select'] == True].tolist()

test_col_select
# drop LotFrontage

train_col_select.pop(0)

test_col_select.pop(0)
train.drop(train_col_select, inplace = True, axis = 1, errors = 'ignore')

test.drop(test_col_select, inplace = True, axis = 1, errors = 'ignore')



train.head()
train.shape
test.head()
test.shape
from sklearn.base import TransformerMixin

class MissingDataImputer(TransformerMixin):

    def fit(self, X, y=None):

        """Extract mode for categorical features and median for numeric features"""        

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self



    def transform(self, X, y=None):

        """Replace missingness with the array got from fit"""

        return X.fillna(self.fill)
train_nmissing = MissingDataImputer().fit_transform(train.iloc[:,1:-1])

test_nmissing = MissingDataImputer().fit_transform(test.iloc[:,1:])

train_nmissing.head()
print(train_nmissing.isnull().sum())
train_cat = train_nmissing.select_dtypes(include=['object']).columns

test_cat = test_nmissing.select_dtypes(include=['object']).columns

train_cat.difference(test_cat)
train_w_dummy = pd.get_dummies(train_nmissing, prefix_sep='_', drop_first=True, columns=train_cat)

test_w_dummy = pd.get_dummies(test_nmissing, prefix_sep='_', drop_first=True, columns=test_cat)



# find all dummy variables in the training set

cat_dummies = [col for col in train_w_dummy 

               if '_' in col 

               and col.split('_')[0] in train_cat]
# drop dummy variables in test set but not in training set

for col in test_w_dummy.columns:

    if ("_" in col) and (col.split("_")[0] in train_cat) and col not in cat_dummies:

        test_w_dummy.drop(col, axis=1, inplace=True)



# add dummy variables in training set but not in test set, and assign them 0

for col in cat_dummies:

    if col not in test_w_dummy.columns:

        test_w_dummy[col] = 0        
train_cols = list(train_w_dummy.columns[:])

test_w_dummy = test_w_dummy[train_cols]
train_w_dummy.shape
test_w_dummy.shape
train_num = train_nmissing.select_dtypes(exclude=['object']).columns

test_num = test_nmissing.select_dtypes(exclude=['object']).columns

test_num.difference(train_num)
train_num_std = [col for col in train_num if abs(train_w_dummy[col].skew()) <= 1]

train_num_yjt = [col for col in train_num if abs(train_w_dummy[col].skew()) > 1]
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer



# standardization

scaler = StandardScaler().fit(train_w_dummy[train_num_std].values)

train_w_dummy[train_num_std] = scaler.transform(train_w_dummy[train_num_std].values)

test_w_dummy[train_num_std] = scaler.transform(test_w_dummy[train_num_std].values)



# power transform

pt = PowerTransformer().fit(train_w_dummy[train_num_yjt].values)

train_w_dummy[train_num_yjt] = pt.transform(train_w_dummy[train_num_yjt].values)

test_w_dummy[train_num_yjt] = pt.transform(test_w_dummy[train_num_yjt].values)
test_w_dummy.head()
from sklearn.decomposition import PCA

pca = PCA().fit(train_w_dummy)

plt.figure(figsize = (6, 4))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components', fontsize = 14)

plt.ylabel('cumulative explained variance', fontsize = 14)



plt.grid(True);
pca = PCA(n_components = 100)

x_train = pca.fit_transform(train_w_dummy) 

x_test = pca.transform(test_w_dummy) 
feature_names = []

for i in range(100):

    # get the index of the feature with the largest absolute value

    feature_idx = np.abs(pca.components_[i]).argmax()

    feature_names.append(train_w_dummy.columns[feature_idx])

    

feature_dict = {'PC{}'.format(i+1): feature_names[i] for i in range(100)}

pd.DataFrame(list(feature_dict.items()), columns=['PC', 'Name']).head(25)
# do not forget to log transform our response variable

y_train = train['SalePrice'].values

y_train_log = np.log1p(train['SalePrice']).values



y_test_data = pd.read_csv('../input/sample_submission.csv')

y_test = y_test_data['SalePrice'].values

y_test_log = np.log1p(y_test_data['SalePrice']).values
from sklearn.ensemble import RandomForestRegressor

from pprint import pprint

rf_base = RandomForestRegressor(n_estimators=400)



# look at parameters used by our base forest

pprint(rf_base.get_params())
# base model result

from sklearn import metrics



# model with original y_train

rf_base.fit(x_train, y_train)

y_pred_rf_base = rf_base.predict(x_test)

# create a dictionary to store mse for comparison, since the results are not stable

mse_dict = {'rf_base': metrics.mean_squared_error(y_test, y_pred_rf_base)}

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_base))  

print('Mean Squared Error:', mse_dict['rf_base'])  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_base)))
rf_base.score(x_train, y_train)
# model with log transformed y_train

rf_base.fit(x_train, y_train_log)

# need to take exponential before calculating mse, etc. in order to compare

y_pred_rf_base_log = np.exp(rf_base.predict(x_test))

mse_dict.update({'rf_base_log': metrics.mean_squared_error(y_test, y_pred_rf_base_log)})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_base_log))  

print('Mean Squared Error:', mse_dict['rf_base_log'])  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_base_log)))
rf_base.score(x_train, y_train_log)
y_pred_rf_base_log
from sklearn.model_selection import RandomizedSearchCV



# params that will be sampled from

max_depth = [int(x) for x in np.linspace(40, 80, num = 5)]

max_depth.append(None)

random_params = {'n_estimators': [200, 400, 600, 800, 1000, 1200],

                'max_depth': max_depth,

                'min_samples_split': [2, 5],

                'min_samples_leaf': [1, 2, 4]}



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_params, n_iter = 100, cv = 3, n_jobs = -1, verbose = 2, random_state = 1)

rf_random.fit(x_train, y_train_log)

rf_random.best_params_
rf_random.best_score_
# random search with best performance parameters

rf_random_best = rf_random.best_estimator_

y_pred_rf_random = np.exp(rf_random_best.predict(x_test))

mse_dict.update({'rf_random': metrics.mean_squared_error(y_test, y_pred_rf_random)})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_random))  

print('Mean Squared Error:', mse_dict['rf_random'])  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_random)))
y_pred_rf_random
from sklearn.model_selection import GridSearchCV

grid_params = {'max_depth': [45, 50, 55, None],

               'min_samples_leaf': [1, 2],

               'min_samples_split': [2, 4, 5],

               'n_estimators': [800, 900, 1000]}



rf_grid = GridSearchCV(estimator = rf, param_grid = grid_params, cv = 3, n_jobs = -1, verbose = 2)

rf_grid.fit(x_train, y_train_log)

rf_grid.best_estimator_
rf_grid.best_score_
# grid search with best performance parameters

rf_grid_best = rf_grid.best_estimator_

y_pred_rf_grd = np.exp(rf_grid_best.predict(x_test))

mse_dict.update({'rf_grd': metrics.mean_squared_error(y_test, y_pred_rf_grd)})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf_grd))  

print('Mean Squared Error:', mse_dict['rf_grd'])  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_grd)))
# keep the output with the lowest mse

rf_model = min(mse_dict, key = lambda x: mse_dict.get(x))

prediction = pd.DataFrame(globals()["y_pred_" + rf_model], columns = ['SalePrice'])

result = pd.concat([y_test_data['Id'], prediction], axis = 1)

result.to_csv('./submission.csv', index = False)
from xgboost import XGBRegressor

xgb_base = XGBRegressor()



# current parameters used by XGBoost

pprint(xgb_base.get_params())
# model with original y_train

xgb_base.fit(x_train, y_train)

y_pred_xgb_base = xgb_base.predict(x_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_base))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_base))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_base)))
xgb_base.score(x_train, y_train)
# model with log transformed y_train

xgb_base.fit(x_train, y_train_log)

y_pred_xgb_base_log = np.exp(xgb_base.predict(x_test))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_base_log))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_base_log))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_base_log)))
xgb_base.score(x_train, y_train_log)
random_params = {'learning_rate': [0.01],

                 'n_estimators': [400, 800, 1000, 1200],

                 'max_depth': [3, 5, 8],

                 'min_child_weight': [4, 6, 8],

                 'subsample': [0.8],

                 'colsample_bytree': [0.8],

                 'reg_alpha': [0, 0.005, 0.01],

                 'seed': [12]}



xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = random_params, n_iter = 100, cv = 3, n_jobs = -1, verbose = 2, random_state = 12)

xgb_random.fit(x_train, y_train_log)

xgb_random.best_params_
xgb_random.best_score_
xgb_random_best = xgb_random.best_estimator_

y_pred_xgb_random = np.exp(xgb_random_best.predict(x_test))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_random))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_random))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_random)))
grid_params = {'learning_rate': [0.01],

               'n_estimators': [400, 800, 1000, 1200],

               'max_depth': [3, 5, 8],

               'min_child_weight': [4, 6, 8],

               'subsample': [0.8],

               'colsample_bytree': [0.8],

               'reg_alpha': [0, 0.005, 0.01],

               'seed': [12]}



xgb_grid = GridSearchCV(estimator = xgb_base, param_grid = grid_params, cv = 3, n_jobs = -1, verbose = 2)

xgb_grid.fit(x_train, y_train_log)

xgb_grid.best_estimator_
xgb_grid.best_score_
# grid search with best performance parameters

xgb_grid_best = xgb_grid.best_estimator_

y_pred_xgb_grd = np.exp(xgb_grid_best.predict(x_test))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb_grd))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb_grd))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb_grd)))
import tensorflow as tf



input_dim = x_train.shape[1]

learning_rate = 0.002

n_nodes_l1 = 25

n_nodes_l2 = 25



x = tf.placeholder("float")

y = tf.placeholder("float")



def neural_net_model(data, input_dim):

    # 2 hidden layer feed forward neural net

    layer_1 = {'weights':tf.Variable(tf.random_normal([input_dim, n_nodes_l1])),

               'biases':tf.Variable(tf.random_normal([n_nodes_l1]))}



    layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_l1, n_nodes_l2])),

               'biases':tf.Variable(tf.random_normal([n_nodes_l2]))}

    

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_l2, 1])),

                    'biases':tf.Variable(tf.random_normal([1]))}

    # affine function

    l1 = tf.add(tf.matmul(tf.cast(data, tf.float32), layer_1['weights']), layer_1['biases'])

    # relu activation

    l1 = tf.nn.relu(l1)



    l2 = tf.add(tf.matmul(l1, layer_2['weights']), layer_2['biases'])

    l2 = tf.nn.relu(l2)

    

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])



    return output
# train neural network with y_train

# get predictions, define loss and optimizer

prediction = neural_net_model(x_train, input_dim)

cost = tf.reduce_mean(tf.square(prediction - y_train))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)



batch_size = 100

epochs = 500

display_epoch = 100

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Training cycle

    for epoch in range(epochs):

        avg_cost = 0

        total_batch = int(x_train.shape[0]/batch_size)

        for i in range(total_batch-1):

            batch_x = x_train[i*batch_size:(i+1)*batch_size]

            batch_y = y_train[i*batch_size:(i+1)*batch_size]

            

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})               

            avg_cost += c/total_batch

            

        if epoch % display_epoch == 0:    

            print("Epoch:", (epoch + 1), " mse =", "{:.6f}".format(avg_cost)) 

    

    # running test set

    results = sess.run(prediction, feed_dict={x: x_test})

    test_cost = sess.run(cost, feed_dict={x: x_test, y: y_test})

    print('test cost: {:.6f}'.format(test_cost))

    

    # calculate r^2

    total_error = tf.reduce_sum(tf.square(y_test - tf.reduce_mean(y_test)))

    unexplained_error = tf.reduce_sum(tf.square(y_test - results))

    R_squared = 1.0 - tf.div(total_error, unexplained_error)

    print(R_squared.eval())    
# train neural network with y_train_log

# get predictions, define loss and optimizer

prediction = neural_net_model(x_train, input_dim)

cost = tf.reduce_mean(tf.square(prediction - y_train_log))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)



batch_size = 100

epochs = 500

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Training cycle

    for epoch in range(epochs):

        avg_cost = 0        

        total_batch = int(x_train.shape[0]/batch_size)

        for i in range(total_batch-1):

            batch_x = x_train[i*batch_size:(i+1)*batch_size]

            batch_y = y_train_log[i*batch_size:(i+1)*batch_size]

            

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})               

            avg_cost += c/total_batch



        if epoch % display_epoch == 0:    

            print("Epoch:", (epoch + 1), " mse =", "{:.6f}".format(avg_cost))

    

    # running test set

    results = sess.run(prediction, feed_dict={x: x_test})

    test_cost = tf.reduce_mean(tf.square(tf.math.exp(results) - y_test))

    print(test_cost.eval())

    

    total_error = tf.reduce_sum(tf.square(y_test_log - tf.reduce_mean(y_test_log)))

    unexplained_error = tf.reduce_sum(tf.square(y_test_log - results))

    R_squared = 1.0 - tf.div(total_error, unexplained_error)

    print(R_squared.eval())