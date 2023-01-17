# Importing libraries and packages
import numpy as np ; import pandas as pd 
import matplotlib.pyplot as plt ; import seaborn as sns; import graphviz
from matplotlib.pyplot import figure;
import matplotlib.ticker as ticker
import warnings ;import time ;import sys ;import datetime;import gc; warnings.simplefilter(action='ignore')
import plotly.offline as py;py.init_notebook_mode(connected=True);import plotly.graph_objs as go; import plotly.tools as tls

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, learning_curve, validation_curve
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb 
# Copied function to reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
%%time
new_transactions = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=['purchase_date'])
historical_transactions = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])

org_train = pd.read_csv('../input/train.csv')
org_test = pd.read_csv('../input/test.csv')
# Reduce memory usage
historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)
print('historical_transactions contains ' + str(len(historical_transactions)) + ' transactions ' + 'for ' + str(len(historical_transactions.card_id.unique())) + ' card_ids')
print('new_marchants_transactions contains ' + str(len(new_transactions)) + ' transactions ' + 'for ' + str(len(new_transactions.card_id.unique())) + ' card_ids')
fig, ax = plt.subplots(1, 3, figsize = (14, 4));
org_train['feature_1'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='#099EE8', title='feature_1', fontsize = 5);
org_train['feature_2'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='red', title='feature_2', fontsize = 5);
org_train['feature_3'].value_counts().sort_index().plot(kind='pie', ax=ax[2], colors = ['#099EE8', '#FFE600'], title='feature_3', explode=[0,0.1], autopct='%1.1f%%',shadow=False)
train_corr_matrix = org_train.corr()
# Generate mask for the upper triangle
mask = np.zeros_like(train_corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 5))
cmap = sns.diverging_palette(10, 255)
sns.heatmap(train_corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
d1 = org_train['first_active_month'].value_counts().sort_index()
d2 = org_test['first_active_month'].value_counts().sort_index()
data = [go.Scatter(x=d1.index, y=d1.values, name='train', mode = 'lines+markers', marker={'color': 'red'}), go.Scatter(x=d2.index, y=d2.values, name='test', mode = 'lines+markers', marker={'color': '#099EE8'})]
layout = go.Layout(dict(title = "Counts of first active",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Count'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))
# Descriptive statistics summary
print(org_train['target'].describe())
# Histogram
plt.figure(figsize=(12,8))
sns.distplot(org_train.target.values, bins=50, color="red", vertical = False, kde=True, kde_kws={"color": "#FFE600"})
plt.title("Histogram of Loyalty score", fontsize = 18)
plt.ylabel('Loyalty score', fontsize=12)
plt.show()
# Read files, change date format and calculate 'elapsed_time'
def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df


# Binarize categorical values
def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

org_train = read_data('../input/train.csv')
org_test = read_data('../input/test.csv')

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)
# Dummy encoding for catogories
historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])
# Purchase_month format to month
historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month
#'nunique' -> Returns number of unique elements in the group
def aggregate_data(transactions):
    transactions.loc[:, 'purchase_date'] = pd.DatetimeIndex(transactions['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    # Aggregate columns by:
    aggregations = {
        'category_1': ['sum', 'mean'],'category_2_1.0': ['mean'],'category_2_2.0': ['mean'],'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],'category_2_5.0': ['mean'],'category_3_A': ['mean'],'category_3_B': ['mean'],'category_3_C': ['mean'],
        'merchant_id': ['nunique'],'merchant_category_id': ['nunique'],'state_id': ['nunique'],'city_id': ['nunique'],'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'], 'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'], 'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['min', 'max']
        }
    
    # Group rows by 'card_id'
    agg_transactions = transactions.groupby(['card_id']).agg(aggregations)
    agg_transactions.columns = ['_'.join(col).strip() for col in agg_transactions.columns.values]
    agg_transactions.reset_index(inplace=True)
    
    #
    data = (transactions.groupby('card_id').size().reset_index(name='transactions_count'))
    agg_transactions = pd.merge(data, agg_transactions, on='card_id', how='left')
    return agg_transactions
# Aggregate historical_transactions
hist_agg = aggregate_data(historical_transactions)
hist_agg.columns = ['hist_' + c if c != 'card_id' else c for c in hist_agg.columns]
# Aggregate new_transactions
new_agg = aggregate_data(new_transactions)
new_agg.columns = ['hist_' + c if c != 'card_id' else c for c in new_agg.columns]
# Join with train- and test-dataset
final_train = pd.merge(org_train, hist_agg, on='card_id', how='left')
final_test = pd.merge(org_test, hist_agg, on='card_id', how='left')

final_train = pd.merge(final_train, new_agg, on='card_id', how='left')
final_test = pd.merge(final_test, new_agg, on='card_id', how='left')

print("Train original:", org_train.shape); print("Test original:", org_test.shape)
print("Historical Transactions:", historical_transactions.shape); print("New Transactions:", new_transactions.shape); 
print("Test merged:", final_train.shape); print("Train merged:", final_test.shape)
print("Number of Rows through Aggregation  {:0.2f}%".format(final_train.shape[0]/(historical_transactions.shape[0] + new_transactions.shape[0])))
# Get features
features = [c for c in final_train.columns if c not in ['card_id', 'first_active_month','target']]
categorical_feats = [c for c in features if 'feature_' in c]

# Full load
target= final_train.target
train = final_train[features]
test = final_test[features]

# Data to work with 
work_target=final_train.target
work_data=final_train.sample(n=10000)
work_train=final_train[features]

# Training and test-dataset
X_train,X_test,y_train,y_test=train_test_split(work_train,work_target,test_size=0.33,random_state=42)
print("X_train:",X_train.shape," y_train:",y_train.shape," X_test:",X_test.shape," y_test:",y_test.shape)
# Compute correlation matrix
train_corr = final_train.corr()

# Generate mask for the upper triangle
mask = np.zeros_like(train_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(train_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Get correlation of features to target
corr_matr=final_train.corr().abs().sort_values(by=['target'],ascending=False)['target'];
corr_matr.head(10)
# checkin data set for NaN's
X_train.isna().sum().sum()
# Fill NaN's with median value
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(X_train)

# For training data
X_train_numpy=imputer.transform(X_train)
X_train_filled=pd.DataFrame(X_train_numpy)

# For test data
X_test_numpy=imputer.transform(X_test)
X_test_filled=pd.DataFrame(X_test_numpy)

# Checkin data set for NaN's
X_train_filled.isna().sum().sum()
# Creating scaled data sets
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(); train_scale=X_train; test_scale=X_test
X_train_scaled = scaler.fit_transform(train_scale)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=features)
X_test_scaled = scaler.fit_transform(test_scale)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
# Some models required a multiclass label instead of a continous one
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)
y_test_encoded = lab_enc.fit_transform(y_test)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train_encoded))
org_train = pd.read_csv('../input/train.csv')
org_test = pd.read_csv('../input/test.csv')
# Adding a column to classify outliers
identify_outlier_train = org_train.assign(outliers = org_train.target)
identify_outlier_train.shape

identify_outlier_train.loc[identify_outlier_train.target < -31, 'outliers'] = 1
identify_outlier_train.loc[identify_outlier_train.target > -31, 'outliers'] = 0

# Only outliers
dirty_train = identify_outlier_train[identify_outlier_train.outliers == 1]
# Without outliers
clean_train = identify_outlier_train[identify_outlier_train.outliers == 0]

# Training and test data without outliers
clean_train = clean_train.drop(['first_active_month','card_id'],axis=1)
clean_target = clean_train['target']
X_train_clean,X_test_clean,y_train_clean,y_test_clean=train_test_split(clean_train,clean_target,test_size=0.33,random_state=42)

print('Number of outliers in train data is: ' + str(len(dirty_train)))
print('Number of non-outliers in train data is: ' + str(len(clean_train)))
print('Proportion of outliers in train dataset is {:0.2%}'.format((len(dirty_train)/len(identify_outlier_train))))
# define function to calculate RMSE
def rmse(y_test,prediction):
    rmse=np.sqrt(mean_squared_error(y_test,prediction))
    return rmse;
# create a dummy prediciton with the average, in order to compare the model to the most simplest one
y_dummy=pd.Series(data=np.full((y_test.shape),np.average(target)))
rmse_dummy=rmse(y_test,y_dummy)
print('Dummy-Model RMSE: {:0.2f}'.format(rmse_dummy))
# define function in order to calculate RMSE and Improvement of Model
def evaluation(model,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test):
    """
   This function evaluates the model, calculates the RMSE and measures its imporvement in comparison to the dummy-prediction.
   If no specific Training and Test-set is given, it defaults to X_train,y_train,X_test and y_test.
   """
    model.fit(X_train,y_train)
    model_pred=model.predict(X_test)
    rmse_new=rmse(y_test,model_pred)
    #print("Number of features used:",model.coef_!=0 )
    print("Model:",model.__class__.__name__,";",'RMSE: {:0.2f}'.format(rmse_new),";",'Improvement of: {:0.2f}%'.format(100 * (rmse_dummy- rmse_new) / rmse_new))
# define RMSE for sklearn-models
from sklearn.metrics import make_scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)
# Testing Linear Models on Dataset
# https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet

linReg=LinearRegression()
logReg=LogisticRegression()
lasso=Lasso(alpha=0.9)
ridge=Ridge(alpha=0.9) # higher the alpha value, more restriction on the coefficients; 
                       # low alpha > more generalization, coefficients are barely
elasNet=ElasticNet(0.5)

for reg in (linReg,lasso,ridge,elasNet):
    evaluation(reg,X_train=X_train_filled,X_test=X_test_filled);
# add significane test
import statsmodels.api as sm
import statsmodels.formula.api as smf

est = smf_linReg = smf.ols('target ~ feature_1 + feature_2 + feature_3', final_train).fit()
est.summary().tables[1]
# Training Decision Tree and first Evaluation of Performance
tree1 = DecisionTreeRegressor(max_depth=3)
evaluation(tree1, X_train = X_train_filled, X_test = X_test_filled)

# Visualize simple decision tree
dot_tree = tree.export_graphviz(tree1, out_file=None)
tree1_graph = graphviz.Source(dot_tree); tree1_graph
# Evaluation Training- and Test Error
# Create CV training and test scores for various training set sizes
depth_range = np.linspace(1, 10, 10) # Range of depth for Decison Tree's

train_scores,test_scores = validation_curve(DecisionTreeRegressor(),X_train_filled, y_train,
                                            param_name ="max_depth", param_range = depth_range, cv=3 ,
                                            scoring=rmse_scorer, n_jobs=-1)

# Create means and standard deviations of training set scores
train_scores_mean = np.mean(train_scores, axis=1); test_scores_mean = np.mean(test_scores, axis=1)

# Calculating optimal regularization parameter
i_depth_range_optim = np.argmax(test_scores_mean)
depth_range_optim = depth_range[i_depth_range_optim]
print("Optimal regularization parameter : %s" % depth_range_optim )
tree2=DecisionTreeRegressor(max_depth=depth_range_optim); evaluation(tree2,X_train=X_train_filled,X_test=X_test_filled)

# Plotting the Validation Curve
figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.title("Validation Curve with DecisionTreeRegressor - Complexity");plt.xlabel("Complexity/max_depth");plt.ylabel("RMSE")
plt.plot(depth_range,train_scores_mean,  label="Training score",linewidth=4,color='blue') ;
plt.plot(depth_range,test_scores_mean, label="Cross-validation score",linewidth=4,color='red')
plt.vlines(depth_range_optim, plt.ylim()[0], np.max(test_scores_mean), color='k',linewidth=2,linestyles='--',label='Optimum on test')
plt.legend(loc="best"); plt.show()
# Evaluation Training Size
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(DecisionTreeRegressor(max_depth=3), 
                                                        X_train_filled, y_train,cv=3,scoring=rmse_scorer, n_jobs=-1, 
                                                        # 20 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 20))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(train_sizes, train_mean, '--', color='blue',  label="Training score")
plt.plot(train_sizes, test_mean, color= 'red', label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
# Create plot -> change y-label
plt.title("Learning Curve with DecisionTreeRegressor - Training Size")
#y_labels = ax.get_yticks()
#ax.yaxis.set_major_formatter(ticker.)
plt.xlabel("Training Set Size"), plt.ylabel("RMSE"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
# Create first random forrest 
from sklearn.ensemble import RandomForestRegressor

ranFor1 = RandomForestRegressor(max_depth=3, random_state=0,
                            n_estimators=20)
evaluation(ranFor1,X_train=X_train_filled,X_test=X_test_filled)
# Evaluation of n_estimators = n_trees
# Create CV training and test scores for various training set sizes
estimator_range = np.arange(2, 40, 2)
train_scores,test_scores = validation_curve(RandomForestRegressor(max_depth=3),X_train_filled, y_train,
                                                          param_name="n_estimators", param_range=estimator_range,
                                                          cv=3,scoring=rmse_scorer, n_jobs=-1)

# Create means and standard deviations of training set scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

i_estimator_range_optim = np.argmax(test_scores_mean)
estimator_range_optim = estimator_range[i_estimator_range_optim]
print("Optimal regularization parameter :",estimator_range_optim )
ranFor2=RandomForestRegressor(n_estimators=estimator_range_optim)
evaluation(ranFor2,X_train=X_train_filled,X_test=X_test_filled)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Validation Curve with RandomForestRegressor");plt.xlabel("Number of Trees");plt.ylabel("RMSE")
plt.plot(estimator_range,train_scores_mean,  label="Training score",linewidth=4,color='blue') ;
plt.plot(estimator_range,test_scores_mean, label="Cross-validation score",linewidth=4,color='red')
plt.vlines(depth_range_optim, plt.ylim()[0], np.max(test_scores_mean), color='k',linewidth=2,linestyles='--',label='Optimum on test')
plt.legend(loc="best"); plt.show()
from sklearn.model_selection import GridSearchCV
ranFor3=RandomForestRegressor()

# Create the parameter grid based on the results of random search 
param_grid_ranFor = {
    'bootstrap': [True], 'max_depth': [2,8], 'max_features': [2,10],
    'min_samples_leaf': [2,20], 'min_samples_split': [4,20], 'n_estimators': [10,20]
}
# Instantiate the grid search model
grid_search_ranFor = GridSearchCV(estimator = ranFor3, param_grid = param_grid_ranFor, 
                                  cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

# Fit the grid search to the data
grid_search_ranFor.fit(X_train_filled,y_train);

# Best Parameters of GridSearch
best_param_ranFor=grid_search_ranFor.best_params_ ; print(best_param_ranFor)

# Evaluation of RandomForrest with GirdSearch Parameters
ranFor3 = RandomForestRegressor(**best_param_ranFor) ; evaluation(ranFor3,X_train=X_train_filled,X_test=X_test_filled)
# Default parameters for LightGBM
param1 = {'application': "regression", "boosting": "gbdt", "metric": 'rmse', 'max_depth': 3,  
          'learning_rate': 0.1, 'num_leaves':31, 'min_data_in_leaf': 20, "random_state": 2019,
          'min_gain_to_split':0.5,      # default =0
          'feature_fraction': 0.5,      # default =1 
          'bagging_fraction': 0.5,      # default =1 
         }
# define function in order to calculate RMSE and Improvement of Model
def lgb_evaluation(model,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test):
    pred_lgb_model=model.predict(X_test)
    rmse_new=rmse(y_test,pred_lgb_model)
    print("Model: LGB;",'RMSE: {:0.2f}'.format(rmse_new),";",'Improvement of: {:0.2f}%'.format(100 * (rmse_dummy- rmse_new) / rmse_new))
#Setting up the first LGBM Model
rounds = 10
training_data = lgb.Dataset(data = X_train, label = y_train, params = param1, 
                          categorical_feature = categorical_feats, free_raw_data = False)
#Training the Model
%time lgb1 = lgb.train(train_set = training_data, params = param1, num_boost_round = rounds)

#Making Predictions
pred_lgb1=lgb1.predict(X_test)

#Evaluation of first LGB
lgb_evaluation(model=lgb1)
from bayes_opt import BayesianOptimization

# Define Bayes Optimization function for LGBM
def bayes_parameter_opt_lgb(training, testing, init_round=5, opt_round=10, n_folds=4, random_seed=2019, 
                            n_estimators=100, learning_rate=0.05, output_process=False):
    
    # Training data
    training_data = lgb.Dataset(data= train, label= target, categorical_feature = categorical_feats, free_raw_data=False)
    
    # Parameters to optimize
    def lgb_evaluation(learning_rate, num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1,min_data_in_leaf,min_split_gain):
        
        params = {'application':'regression','num_iterations': n_estimators,  
                  'early_stopping_round':100, 'metric':'rmse'}
        # Bayes opt's outputs are always float
        params['learning_rate'] = max(min(feature_fraction, 1), 0)
        params["num_leaves"] = int(round(num_leaves)) # Rounding of parameters did not work
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['min_split_gain'] = min_split_gain
        cv_result = lgb.cv(params, training_data, nfold=n_folds, seed=random_seed, stratified=False, 
                           verbose_eval =200, metrics=['rmse'])
        return max(cv_result['rmse-mean'])
   
    # Parameter Range
    lgbBO = BayesianOptimization(lgb_evaluation, {'learning_rate': (0.0001, 0.1),
                                            'num_leaves': (150, 300),
                                            'feature_fraction': (0.01, 0.2),
                                            'bagging_fraction': (0.2, 1),
                                            'max_depth': (1, 3),
                                            'lambda_l1': (0, 2),
                                            'min_data_in_leaf': (200,400),
                                            'min_split_gain': (0.001, 0.2)})
    # Optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # Output optimization process to CSV
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result1.csv")
    
    # return best parameters
    return lgbBO.max
# Using optimization function on training data
%time 
opt_params_lgb = bayes_parameter_opt_lgb(train, target)
param1=opt_params_lgb['params'];param1
# Results of different rounds of Bayesian Optimization
param1 = {"boosting": "gbdt", 'objective':'regression',"metric": 'rmse', "random_state": 2019, "verbosity": -1,
          'bagging_fractions':0.7768, 'feature_fraction':0.3249, 'lambda_l1':0.0460, 'learning_rate':0.067,
          'max_depth':3, 'min_data_in_leaf':176, 'min_split_gain':0.0297, 'num_leaves':77}
param2 = {'num_leaves': 111, 'min_data_in_leaf': 149,'objective':'regression','max_depth': 9,
         'learning_rate': 0.005, "boosting": "gbdt", "feature_fraction": 0.7522, "bagging_freq": 1,
         "bagging_fraction": 0.7083 , "bagging_seed": 11, "metric": 'rmse', "lambda_l1": 0.2634,
         "random_state": 133, "verbosity": -1}
param3 = {"boosting": "gbdt", 'objective':'regression',"metric": 'rmse', "random_state": 2019, "verbosity": -1,
          'bagging_fractions':1, 'feature_fraction':0.1, 'lambda_l1':1.0, 'learning_rate':0.001,
          'max_depth':2, 'min_data_in_leaf':250, 'min_split_gain':0.001, 'num_leaves':160}
param4 = {"boosting": "gbdt", 'objective':'regression',"metric": 'rmse', "random_state": 2019, "verbosity": -1,
          'bagging_fractions':0.7, 'feature_fraction':0.02, 'lambda_l1':1.75, 'learning_rate':0.05,
          'max_depth':3, 'min_data_in_leaf':305, 'min_split_gain':0.01, 'num_leaves':300}
#train/test indices to split data in train/test sets -> each fold is then used once as a validation while the k - 1 remaining folds form the training set
folds             = KFold(n_splits=4,shuffle=True,random_state=15)
X                 = train
y                 = target
param             = param3
num_boost_rounds  = 10000
early_stopping    = 100
out_of_folds      = np.zeros(len(X))
predictions       = np.zeros(len(y))
feature_importance= pd.DataFrame()
evals_result      = {}

# for-loop to split data in n training and validation sets and perfom LGBM on every fold
for fold, (train_ind,val_ind) in enumerate(folds.split(X,y)):
    print("fold n°{}".format(fold))
    
    #training&validation sets
    training=lgb.Dataset(X.iloc[train_ind],label=y.iloc[train_ind])
    validation=lgb.Dataset(X.iloc[val_ind],label=y.iloc[val_ind])
    
    #training for each fold:
    lgb2=lgb.train(param1,training,valid_sets = [training,validation], verbose_eval=10,num_boost_round=num_boost_rounds,
                   early_stopping_rounds = early_stopping  , evals_result = evals_result)
    
    #predictions on validation fold -> OOF:"Out-of-fold" -> using k-fold validation in which the predictions from each set of folds are grouped together into one group of 1000 predictions
    out_of_folds[val_ind] = lgb2.predict(X.iloc[val_ind], num_iteration=lgb2.best_iteration)
    
    #storing in feature importance
    fold_importance= pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = lgb2.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
lgb_evaluation(lgb2)
lgb2.best_score
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.title("Validation Curve with RandomForestRegressor");plt.xlabel("Number of Trees");plt.ylabel("RMSE")
#plt.legend(loc="best"); plt.show()
ax = lgb.plot_metric(evals_result, metric='rmse')
plt.show()
feature_importance.sort_values(by=['importance'],ascending=False).head(10)
# Group by feature
agg_features = (feature_importance[["feature","importance"]].groupby("feature").mean().sort_values(by="importance",ascending=False)[:1000].index)
best_features = feature_importance.loc[feature_importance.feature.isin(agg_features)]

# Plot feature importance
plt.figure(figsize=(12,20))
sns.barplot(x="importance",y="feature",data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (avg over folds)');plt.tight_layout()
#plt.savefig('lgbm_importances.png')
predictions = lgb2.predict(test)
elo_sub_XX = pd.DataFrame({"card_id":org_test["card_id"].values})
elo_sub_XX["target"] = predictions
elo_sub_XX.to_csv("elo_subX.csv", index=False)
#train/test indices to split data in train/test sets -> each fold is then used once as a validation while the k - 1 remaining folds form the training set
folds             = KFold(n_splits=5,shuffle=True,random_state=15)
X                 = clean_train
y                 = clean_target
param             = param3
num_boost_rounds  = 100
early_stopping    = 10
out_of_folds      = np.zeros(len(X))
predictions       = np.zeros(len(y))
feature_importance= pd.DataFrame()
evals_result      = {}

# for-loop to split data in n training and validation sets and perfom LGBM on every fold
for fold, (train_ind,val_ind) in enumerate(folds.split(X,y)):
    print("fold n°{}".format(fold))
    
    #training&validation sets
    training=lgb.Dataset(X.iloc[train_ind],label=y.iloc[train_ind])
    validation=lgb.Dataset(X.iloc[val_ind],label=y.iloc[val_ind])
    
    #training for each fold:
    lgb_clean=lgb.train(param1,training,valid_sets = [training,validation], verbose_eval=10,num_boost_round=num_boost_rounds,
                   early_stopping_rounds = early_stopping  , evals_result = evals_result)
    
    #predictions on validation fold -> OOF:"Out-of-fold" -> using k-fold validation in which the predictions from each set of folds are grouped together into one group of 1000 predictions
    out_of_folds[val_ind] = lgb_clean.predict(X.iloc[val_ind], num_iteration=lgb_clean.best_iteration)

    
lgb_evaluation(lgb_clean ,X_train = X_train_clean, y_train = y_train_clean,
           X_test = X_test_clean, y_test = y_test_clean );
lgb_clean.best_score
linReg=LinearRegression()
logReg=LogisticRegression()
lasso=Lasso(alpha=0.9)
ridge=Ridge(alpha=0.9) # higher the alpha value, more restriction on the coefficients; 
                       # low alpha > more generalization, coefficients are barely
elasNet=ElasticNet(0.5)

for reg in (linReg,lasso,ridge,elasNet):
    evaluation(reg ,X_train = X_train_clean, y_train = y_train_clean,
               X_test = X_test_clean, y_test = y_test_clean );
#predictions = lgb_clean.predict(test)
#elo_sub_clean = pd.DataFrame({"card_id":org_test["card_id"].values})
#elo_sub_clean["target"] = predictions
#elo_sub_clean.to_csv("elo_sub_clean.csv", index=False)