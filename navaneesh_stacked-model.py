# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
# Input data files are available in the "../input/" directory.
#Input of Data 
train = pd.read_csv('../input/ProcessedDataTrain.csv',header=0,engine='python')
test = pd.read_csv('../input/ProcessedDataTest.csv',header=0,engine='python' )
#test = pd.read_csv('../input/test.csv', header=0)

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['id']
test_ID = test['id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("id", axis = 1, inplace = True)
test.drop("id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
##display the first five rows of the train dataset.
train.head(5)
##display the first five rows of the test dataset.
test.head(5)
fig, ax = plt.subplots()
ax.scatter(x = train['bedrooms'], y = train['log_price'])
plt.ylabel('log_price', fontsize=13)
plt.xlabel('bedrooms', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = train['beds'], y = train['log_price'])
plt.ylabel('log_price', fontsize=13)
plt.xlabel('beds', fontsize=13)
plt.show()
#Outlier removal 
train = train[train['log_price'] !=0]

#train = train[(!train['log_price'] < 4) & (!train['bedrooms']) > 7 ]
train = train[train['beds'] < 17]
sns.distplot(train['log_price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['log_price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['log_price'], plot=plt)
plt.show()
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.log_price.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['log_price'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_data["NewScore"] = all_data['number_of_reviews']*0.75 + all_data['review_scores_rating']*0.25
all_data.dtypes
%matplotlib inline
import matplotlib.pyplot as plt
all_data.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()
corr_matrix = train.corr()
corr_matrix["log_price"].sort_values(ascending=False)
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
all_data = all_data.select_dtypes(include=numerics).fillna(0)

train = all_data[:ntrain]
test = all_data[ntrain:]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
#Modelling

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_rf = RandomForestRegressor(random_state = 3, n_estimators = 500, criterion='mse',n_jobs=-1 )
#score = rmsle_cv(GBoost)
#print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#score = rmsle_cv(model_xgb)
#print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#score = rmsle_cv(model_rf)
#print("RF score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (GBoost, model_xgb, model_rf))

averaged_models.fit(train.values, y_train)
#averaged_train_pred = averaged_models.predict(train.values)
averaged_pred = averaged_models.predict(test.values)

submissionAVG = pd.DataFrame(np.column_stack([test_ID, averaged_pred]), columns = ['id','log_price'])
submissionAVG.to_csv("submissionAVG.csv", index = False)

model_xgb.fit(train, y_train)
#xgb_train_pred = model_xgb.predict(train)
xgb_pred = model_xgb.predict(test)
#print(rmsle(y_train, xgb_train_pred))
submissionXGB = pd.DataFrame(np.column_stack([test_ID, xgb_pred]), columns = ['id','log_price'])
submissionXGB.to_csv("submissionXGB.csv", index = False)

model_rf.fit(train, y_train)
#rf_train_pred = model_rf.predict(train)
rf_pred = model_rf.predict(test.values)

submissionRF = pd.DataFrame(np.column_stack([test_ID, rf_pred]), columns = ['id','log_price'])
submissionRF.to_csv("submissionRF.csv", index = False)
#print(rmsle(y_train, lgb_train_pred))
GBoost.fit(train, y_train)
#GBoost_train_pred = GBoost.predict(train)
GBoost_pred = GBoost.predict(test.values)

submissionGBoost = pd.DataFrame(np.column_stack([test_ID, GBoost_pred]), columns = ['id','log_price'])
submissionGBoost.to_csv("submissionGBoost.csv", index = False)
'''RMSE on the entire Train data when averaging'''

#print('RMSLE score on train data:')
#print(rmsle(y_train,stacked_train_pred*0.70 +
 #              xgb_train_pred*0.15 + rf_train_pred*0.15 ))
ensemble = averaged_pred*0.70 + xgb_pred*0.15 + rf_pred*0.15
sub = pd.DataFrame()
sub['id'] = test_ID
sub['log_price'] = ensemble
sub.to_csv('submissionEnsemble.csv',index=False)