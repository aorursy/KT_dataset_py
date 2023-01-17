%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import os

import datetime as dt

import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import KFold, train_test_split

from scipy.signal import savgol_filter

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sys import stdout

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.cross_decomposition import PLSRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_predict

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from sklearn.preprocessing import LabelEncoder, StandardScaler

from tqdm import tqdm
train_users = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')

test_users = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')

print("There were", train_users.shape[0], "observations in the training set and", test_users.shape[0], "in the test set.")

print("In total there were", train_users.shape[0] + test_users.shape[0], "observations.")
plt.figure(figsize=(12,10))

cor = train_users.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_users.head(10)
test_users.head(10)
train_users.isnull().sum()
train_users['City'].unique(), train_users['Hour'].unique()
intersaction_city=train_users[['City','IntersectionId']].drop_duplicates()



intersaction_city.groupby(['City'])['IntersectionId'].aggregate('count').reset_index().sort_values('IntersectionId', ascending=False)
def unique_counts(train_users):

   for i in train_users.columns:

       count = train_users[i].nunique()

       print(i, ": ", count)

unique_counts(train_users)
plt.figure(figsize=(12,6))

sns.distplot(train_users.Hour.dropna(), rug=True)

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='Month', data=train_users)

plt.xlabel('Month')

plt.ylabel('Number of observations')

sns.despine()
plt.figure(figsize=(12,6))

sns.countplot(x='Hour', data=train_users)

plt.xlabel('Hour')

plt.ylabel('Number of observations')

sns.despine()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="Hour", data=train_users, hue='City', dodge=True)

g.set_title("Hour Count Distribution by City", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Hours of Day", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)



g.set_ylim(0, max(sizes) * 1.15)

plt.show()
plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="Month", data=train_users, hue='City', dodge=True)

g.set_title("Month Count Distribution by City", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Month", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)



g.set_ylim(0, max(sizes) * 1.15)

plt.show()
train_users.info()
y1 = train_users['TotalTimeStopped_p20']

y2 = train_users['TotalTimeStopped_p50']

y3 = train_users['TotalTimeStopped_p80']

y4 = train_users['DistanceToFirstStop_p20']

y5 = train_users['DistanceToFirstStop_p50']   

y6 = train_users['DistanceToFirstStop_p80']   
Entry = pd.get_dummies(train_users["EntryHeading"],prefix = 'EN')

Exit = pd.get_dummies(train_users["ExitHeading"],prefix = 'EX')

C = pd.get_dummies(train_users["City"])
train_users = train_users.merge(

    train_users.groupby('City')[['Latitude', 'Longitude']].mean(),

    left_on='City', right_index=True, suffixes=['', 'Dist']

)

train_users.LatitudeDist = (np.abs(train_users.Latitude - train_users.LatitudeDist)).round(2)

train_users.LongitudeDist = (np.abs(train_users.Longitude - train_users.LongitudeDist)).round(2)

train_users['CenterDistL1'] = (train_users.LatitudeDist + train_users.LongitudeDist).round(2)

train_users['CenterDistL2'] = (np.sqrt((train_users.LatitudeDist ** 2 + train_users.LongitudeDist ** 2))).round(2)
train_users = pd.concat([train_users,Entry],axis=1)

train_users = pd.concat([train_users,Exit],axis=1)

train_users = pd.concat([train_users,C],axis=1)
X=train_users[['Hour','Weekend','Month','EN_S','EN_SW','EN_SE','EN_N','EN_NW','EN_NE','EN_W','EN_E','EX_S','EX_SW','EX_SE','EX_N','EX_NW','EX_NE','EX_W','EX_E','Atlanta','Boston','Chicago','Philadelphia']]
def optimise_pls_cv(X, y, n_comp, plot_components=True):

    '''Run PLS including a variable number of components, up to n_comp,

       and calculate MSE '''

    mse = []

    component = np.arange(1, n_comp)

    for i in component:

        pls = PLSRegression(n_components=i)

        # Cross-validation

        y_cv = cross_val_predict(pls, X, y, cv=10)

        mse.append(mean_squared_error(y, y_cv))

        comp = 100*(i+1)/40

        # Trick to update status on the same line

        stdout.write("\r%d%% completed" % comp)

        stdout.flush()

    stdout.write("\n")

    # Calculate and print the position of minimum in MSE

    msemin = np.argmin(mse)

    print("Suggested number of components: ", msemin+1)

    stdout.write("\n")

    if plot_components is True:

        with plt.style.context(('ggplot')):

            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')

            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')

            plt.xlabel('Number of PLS components')

            plt.ylabel('MSE')

            plt.title('PLS')

            plt.xlim(left=-1)

        plt.show()

    # Define PLS object with optimal number of components

    pls_opt = PLSRegression(n_components=msemin+1)

    # Fir to the entire dataset

    pls_opt.fit(X, y)

    y_c = pls_opt.predict(X)

    # Cross-validation

    y_cv = cross_val_predict(pls_opt, X, y, cv=10)

    # Calculate scores for calibration and cross-validation

    score_c = r2_score(y, y_c)

    score_cv = r2_score(y, y_cv)

    # Calculate mean squared error for calibration and cross validation

    mse_c = mean_squared_error(y, y_c)

    mse_cv = mean_squared_error(y, y_cv)

    print('R2 calib: %5.3f'  % score_c)

    print('R2 CV: %5.3f'  % score_cv)

    print('MSE calib: %5.3f' % mse_c)

    print('MSE CV: %5.3f' % mse_cv)

    # Plot regression and figures of merit

    rangey = max(y) - min(y)

    rangex = max(y_c) - min(y_c)

    # Fit a line to the CV vs response

    z = np.polyfit(y, y_c, 1)

    with plt.style.context(('ggplot')):

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(y_c, y, c='red', edgecolors='k')

        #Plot the best fit line

        ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)

        #Plot the ideal 1:1 line

        ax.plot(y, y, color='green', linewidth=1)

        plt.title('$R^{2}$ (CV): '+str(score_cv))

        plt.xlabel('Predicted $^{\circ}$Brix')

        plt.ylabel('Measured $^{\circ}$Brix')

        plt.show()

    return
#optimise_pls_cv(X,y1, 20, plot_components=True)

#The lowest RMSE
#optimise_pls_cv(X,y2, 20, plot_components=True)

#R2 calib: 0.014

#R2 CV: 0.010

#MSE calib: 238.604

#MSE CV: 239.453
#optimise_pls_cv(X,y3, 20, plot_components=True)

#R2 calib: 0.020

#R2 CV: 0.013

#MSE calib: 775.134

#MSE CV: 780.114
#optimise_pls_cv(X,y4, 20, plot_components=True)

#R2 calib: 0.003

#R2 CV: 0.001

#MSE calib: 781.981

#MSE CV: 783.262
#optimise_pls_cv(X,y5, 20, plot_components=True)

#R2 calib: 0.006

#R2 CV: 0.003

#MSE calib: 5111.828

#MSE CV: 5130.587
#optimise_pls_cv(X,y6, 20, plot_components=True)

#R2 calib: 0.006

#R2 CV: -0.000

#MSE calib: 23172.832

#MSE CV: 23313.593
results=sm.OLS(y1,X).fit()

results.summary()
test_users.columns # This will show all the column names

test_users.head(10) # Show first 10 records of dataframe

test_users.describe() #You can look at summary of numerical fields by using describe() function
Entry2 = pd.get_dummies(test_users["EntryHeading"],prefix = 'EN')

Exit2 = pd.get_dummies(test_users["ExitHeading"],prefix = 'EX')

C2 = pd.get_dummies(test_users["City"])
test_users = test_users.merge(

    test_users.groupby('City')[['Latitude', 'Longitude']].mean(),

    left_on='City', right_index=True, suffixes=['', 'Dist']

)

test_users.LatitudeDist = (np.abs(test_users.Latitude - test_users.LatitudeDist)).round(2)

test_users.LongitudeDist = (np.abs(test_users.Longitude - test_users.LongitudeDist)).round(2)

test_users['CenterDistL1'] = (test_users.LatitudeDist + test_users.LongitudeDist).round(2)

test_users['CenterDistL2'] = (np.sqrt((test_users.LatitudeDist ** 2 + test_users.LongitudeDist ** 2))).round(2)
test_users = pd.concat([test_users,Entry2],axis=1)

test_users = pd.concat([test_users,Exit2],axis=1)

test_users = pd.concat([test_users,C2],axis=1)
x=test_users[['Hour','Weekend','Month','EN_S','EN_SW','EN_SE','EN_N','EN_NW','EN_NE','EN_W','EN_E','EX_S','EX_SW','EX_SE','EX_N','EX_NW','EX_NE','EX_W','EX_E','Atlanta','Boston','Chicago','Philadelphia']]
from catboost import CatBoostRegressor

cb_model= CatBoostRegressor(iterations=700,

                             learning_rate=0.02,

                             depth=12,

                             eval_metric='RMSE',

                             random_seed = 23,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 75,

                             od_wait=100)
#cb_model.fit(X, y1)

#pred_CB1=cb_model.predict(x)

#cb_model.fit(X, y2)

#pred_CB2=cb_model.predict(x)

#cb_model.fit(X, y3)

#pred_CB3=cb_model.predict(x)

#cb_model.fit(X, y4)

#pred_CB4=cb_model.predict(x)

#cb_model.fit(X, y5)

#pred_CB5=cb_model.predict(x)

#cb_model.fit(X, y6)

#pred_CB6=cb_model.predict(x)
#prediction_CB = []

#for i in range(len(pred_CB1)):

    #for j in [pred_CB1,pred_CB2,pred_CB3,pred_CB4,pred_CB5,pred_CB6]:

        #prediction_CB.append(j[i])

        

#submission_CB = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

#submission_CB["Target"] = prediction_CB

#submission_CB.to_csv("Submission_CB.csv",index = False)

#RMSE=79
train_users['IsTrain'] = 1

test_users['IsTrain'] = 0
whole = pd.concat([train_users, test_users], sort=True)
whole['random'] = np.random.rand(len(whole))
column_stats = pd.concat([

    pd.DataFrame(whole.count()).rename(columns={0: 'cnt'}),

    pd.DataFrame(whole.nunique()).rename(columns={0: 'unique'}),

], sort=True, axis=1)

column_stats.sort_values(by='unique')
train_columns = list(column_stats[column_stats.cnt < 10 ** 6].index)
target_columns = [

    'TotalTimeStopped_p20',

    'TotalTimeStopped_p50',

    'TotalTimeStopped_p80',

    'DistanceToFirstStop_p20',

    'DistanceToFirstStop_p50',

    'DistanceToFirstStop_p80',

]
do_not_use = train_columns + ['IsTrain', 'Path', 'RowId', 'IntersectionId',

                              'random','City','EntryHeading','ExitHeading','EntryStreetName','ExitStreetName']
feature_columns = [c for c in whole.columns if c not in do_not_use]
fix = {

    'lambda': 1., 'nthread': 4, 'booster': 'gbtree',

    'silent': 1, 'eval_metric': 'rmse',

    'objective': 'reg:squarederror'}

config = dict(min_child_weight=20,

              eta=0.05, colsample_bytree=0.6,

              max_depth=20, subsample=0.8)

config.update(fix)

nround = 200
TRAIN_SAMPLE_SIZE = 0.7
whole = whole.dropna()
total_mse = 0.0

submission_parts = []

for i, target in enumerate(target_columns):

    train_idx = whole.random < TRAIN_SAMPLE_SIZE

    valid_idx = whole.random >= TRAIN_SAMPLE_SIZE

    Xtr = whole[train_idx][feature_columns]

    Xv = whole[valid_idx][feature_columns]

    ytr = whole[train_idx][target].values

    yv = whole[valid_idx][target].values

    print(Xtr.shape, ytr.shape, Xv.shape, yv.shape)

    dtrain = xgb.DMatrix(Xtr, label=ytr)

    dvalid = xgb.DMatrix(Xv, label=yv)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(config, dtrain, nround, evals=watchlist,

                      verbose_eval=50, early_stopping_rounds=50)

    pv = model.predict(dvalid)

    mse = np.mean((yv - pv) ** 2)

    total_mse += mse / 6

    print(target, 'rmse', np.sqrt(mse))

    df = pd.DataFrame({

        'Target': model.predict(xgb.DMatrix(test_users[feature_columns])),

        'TargetId': test_users.RowId.astype(str) + '_' + str(i)})

    submission_parts.append(df)
rmse = np.sqrt(total_mse)

print('Total rmse', rmse)

submission = pd.concat(submission_parts, sort=True)

submission.to_csv('submission.csv', index=False)

#RMSE 41
#def run_lgb_f(train_users, test_users):

    #all_preds = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : []}

    #all_target = [y1, y2, y3, y4, y5, y6]

    #nfold = 5

    #kf = KFold(n_splits=nfold, random_state=228, shuffle=True)

    #for i in range(len(all_preds)):

        #print('Training and predicting for target {}'.format(i+1))

        #oof = np.zeros(len(train_users))

        #all_preds[i] = np.zeros(len(test_users))

        #n = 1

        #for train_index, valid_index in kf.split(all_target[i]):

            #print("fold {}".format(n))

            #xg_train = lgb.Dataset(train_users.iloc[train_index],

                                   #label=all_target[i][train_index]

                                   #)

            #xg_valid = lgb.Dataset(train_users.iloc[valid_index],

                                   #label=all_target[i][valid_index]

                                   #)

            #clf = lgb.train(param, xg_train, 100000, valid_sets=[xg_train, xg_valid], 

                            #verbose_eval=500, early_stopping_rounds=100)

            #oof[valid_index] = clf.predict(train_users.iloc[valid_index], num_iteration=clf.best_iteration) 



            #all_preds[i] += clf.predict(test_users, num_iteration=clf.best_iteration) / nfold

            #n = n + 1

            #print("\n\nCV RMSE: {:<0.4f}".format(np.sqrt(mean_squared_error(all_target[i], oof))))

    #return all_preds
#param = {'application': 'regression', 

         #'learning_rate': 0.05, 

         #'metric': 'rmse', 

         #'seed': 42, 

         #'bagging_fraction': 0.7, 

         #'feature_fraction': 0.9, 

         #'lambda_l1': 0.0, 

         #'lambda_l2': 5.0, 

         #'max_depth': 30, 

         #'min_child_weight': 50.0, 

         #'min_split_gain': 0.1, 

         #'num_leaves': 230}
#all_preds = run_lgb_f(train_users[feature_columns], test_users[feature_columns])
#submission = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/sample_submission.csv')

#data = pd.DataFrame(all_preds).stack()

#data = pd.DataFrame(data)

#submission['Target'] = data[0].values

#submission.to_csv('submission.csv', index=False)

#RMSE 80-84