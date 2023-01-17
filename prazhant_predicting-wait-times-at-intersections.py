import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")

sample_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/sample_submission.csv")
train_df.head()
test_df.head()
train_df.info()
#total number of intersections in our dataset

print("The total number of unique intersections in our dataset are : {}".format(train_df['IntersectionId'].nunique()))

print("The total number of Cities in our dataset are : {}".format(train_df['City'].nunique()))
#Number of intersections in each city

train_df.groupby('City')['IntersectionId'].nunique().plot(kind='barh')
print("The maximum number of entry streets for intersections in our dataset is : {}".format(train_df.groupby('IntersectionId')['EntryStreetName'].nunique().max()))

print("The average number of entry streets for intersections in our dataset is : {}".format(train_df.groupby('IntersectionId')['EntryStreetName'].nunique().mean()))
print("The maximum number of exit streets for intersections in our dataset is : {}".format(train_df.groupby('IntersectionId')['ExitStreetName'].nunique().max()))

print("The average number of exit streets for intersections in our dataset is : {}".format(train_df.groupby('IntersectionId')['ExitStreetName'].nunique().mean()))
#Transforming CAtegorical features
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

corr = train_df.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)

# Drop rowID

train_df.drop('RowId', axis=1, inplace=True)

test_df.drop('RowId', axis=1, inplace=True)
#Check cardinality of all the categorical variables

for y in train_df.columns:

    if(train_df[y].dtype == object):

          print("The caridinality for {} is : {}".format(y, train_df[y].nunique()))
X = train_df[['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName', 

              'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'Month', 'Path','City']]

y1 = train_df["TotalTimeStopped_p20"]

y2 = train_df["TotalTimeStopped_p50"]

y3 = train_df["TotalTimeStopped_p80"]

y4 = train_df["DistanceToFirstStop_p20"]

y5 = train_df["DistanceToFirstStop_p50"]

y6 = train_df["DistanceToFirstStop_p80"]
#Creating Dummies for train Data

dfen = pd.get_dummies(X["EntryHeading"],prefix = 'entry')

dfex = pd.get_dummies(X["ExitHeading"],prefix = 'exit')

city = pd.get_dummies(X["City"],prefix = 'city')



X = pd.concat([X,dfen],axis=1)

X = pd.concat([X,dfex],axis=1)

X = pd.concat([X,city],axis=1)



X.drop("EntryHeading", axis=1, inplace=True)

X.drop("ExitHeading", axis=1, inplace=True)

X.drop("City", axis=1, inplace=True)



#Creating Dummies for test Data

dfent = pd.get_dummies(test_df["EntryHeading"],prefix = 'entry')

dfext = pd.get_dummies(test_df["ExitHeading"],prefix = 'exit')

city_test = pd.get_dummies(test_df["City"],prefix = 'city')



test_df = pd.concat([test_df,dfent],axis=1)

test_df = pd.concat([test_df,dfext],axis=1)

test_df = pd.concat([test_df,city_test],axis=1)



test_df.drop("EntryHeading", axis=1, inplace=True)

test_df.drop("ExitHeading", axis=1, inplace=True)

test_df.drop("City", axis=1, inplace=True)
X.head()
#Visualizing rows having NaN values for EntryStreetName and ExitStreetName

#Path being concatenation of EntryStreetName_EntryHeading_ExitStreetName_ExitHeading

train_df[train_df.isnull().any(axis=1)].head()
#filling rows with NaN's

X.fillna("nan", inplace=True)

test_df.fillna("nan", inplace=True)
X.drop('Path', axis=1, inplace=True)

test_df.drop('Path', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['EntryStreetName'] = le.fit_transform(X['EntryStreetName'])

test_df['EntryStreetName'] = le.fit_transform(test_df['EntryStreetName'])



X['ExitStreetName'] = le.fit_transform(X['ExitStreetName'])

test_df['ExitStreetName'] = le.fit_transform(test_df['ExitStreetName'])



# X['Path'] = le.fit_transform(X['Path'])

# test_df['Path'] = le.fit_transform(test_df['Path'])
X.shape
y1.shape
# import lightgbm as lgb

# from sklearn.model_selection import KFold, StratifiedKFold

# def kfold_lightgbm(target, num_folds= 10):

#     print("Starting LightGBM. Train shape: {}, test shape: {}".format(X.shape, test_df.shape))

#     folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

#     # Create arrays and dataframes to store results

#     oof_preds = np.zeros(X.shape[0])

#     sub_preds = np.zeros(test_df.shape[0])



#     for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X,target)):

#         train_x, train_y = X.iloc[train_idx], target.iloc[train_idx]

#         valid_x, valid_y = X.iloc[valid_idx], target.iloc[valid_idx]



#         # LightGBM parameters found by Bayesian optimization

#         clf = lgb.LGBMRegressor(

#             nthread=4,

#             n_estimators=10000,

#             learning_rate=0.001,

#             num_leaves=34,

#             colsample_bytree=0.9497036,

#             subsample=0.8715623,

#             max_depth=8,

#             reg_alpha=0.041545473,

#             reg_lambda=0.0735294,

#             min_split_gain=0.0222415,

#             min_child_weight=39.3259775,

#             silent=-1,

#             verbose=-1)



#         clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

#             eval_metric= 'rmse', verbose= 500, early_stopping_rounds= 200)



#         oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)

#         sub_preds += clf.predict(test_df, num_iteration=clf.best_iteration_) / folds.n_splits

#         return sub_preds
# %%time

# pred1 = kfold_lightgbm(y1)

# pred2 = kfold_lightgbm(y2)

# pred3 = kfold_lightgbm(y3)

# pred4 = kfold_lightgbm(y4)

# pred5 = kfold_lightgbm(y5)

# pred6 = kfold_lightgbm(y6)
# # Appending all predictions

# prediction = []

# for i in range(len(pred1)):

#     for j in [pred1,pred2,pred3,pred4,pred5,pred6]:

#         prediction.append(j[i])

        

# sample_df["Target"] = prediction

# sample_df.to_csv("Submission_CB.csv",index = False)
import h2o

from h2o.automl import H2OAutoML

h2o.init()
df = h2o.import_file("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

h2o_test = h2o.import_file("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
y1 = "TotalTimeStopped_p20"

y2 = "TotalTimeStopped_p50"

y3 = "TotalTimeStopped_p80"

y4 = "DistanceToFirstStop_p20"

y5 = "DistanceToFirstStop_p50"

y6 = "DistanceToFirstStop_p80"
splits = df.split_frame(ratios = [0.8], seed = 1)

train = splits[0]

test = splits[1]
aml_1 = H2OAutoML(max_runtime_secs = 300, seed = 1)

aml_1.train(y = y1, training_frame = df)

print(aml_1.leader.model_performance(test))

aml_1.leaderboard.head()
aml_2 = H2OAutoML(max_runtime_secs = 300, seed = 1)

aml_2.train(y = y2, training_frame = df)

print(aml_2.leader.model_performance(test))

aml_2.leaderboard.head()
aml_3 = H2OAutoML(max_runtime_secs = 300, seed = 1)

aml_3.train(y = y3, training_frame = df)

print(aml_3.leader.model_performance(test))

aml_3.leaderboard.head()
aml_4 = H2OAutoML(max_runtime_secs = 300, seed = 1)

aml_4.train(y = y4, training_frame = df)

print(aml_4.leader.model_performance(test))

aml_4.leaderboard.head()
aml_5 = H2OAutoML(max_runtime_secs = 300, seed = 1)

aml_5.train(y = y5, training_frame = df)

print(aml_5.leader.model_performance(test))

aml_5.leaderboard.head()
aml_6 = H2OAutoML(max_runtime_secs = 300, seed = 1)

aml_6.train(y = y6, training_frame = df)

print(aml_6.leader.model_performance(test))

aml_6.leaderboard.head()
y1 = train_df["TotalTimeStopped_p20"]

y2 = train_df["TotalTimeStopped_p50"]

y3 = train_df["TotalTimeStopped_p80"]

y4 = train_df["DistanceToFirstStop_p20"]

y5 = train_df["DistanceToFirstStop_p50"]

y6 = train_df["DistanceToFirstStop_p80"]
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 6.0)

y6_val = pd.DataFrame({"y6":train_df["DistanceToFirstStop_p80"], "log(y6 + 1)":np.log1p(train_df["DistanceToFirstStop_p80"])})

y6_val.hist()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model,y_value):

    rmse= np.sqrt(-cross_val_score(model, X, y_value, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.01, 0.05, 0.1, 0.3]

cv_ridge = [rmse_cv(Ridge(alpha = alpha),y1).mean() 

            for alpha in alphas]



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge_2 = [rmse_cv(Ridge(alpha = alpha),y2).mean() 

            for alpha in alphas]



cv_ridge_2 = pd.Series(cv_ridge_2, index = alphas)

cv_ridge_2.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge_3 = [rmse_cv(Ridge(alpha = alpha),y3).mean() 

            for alpha in alphas]



cv_ridge_3 = pd.Series(cv_ridge_3, index = alphas)

cv_ridge_3.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge_4 = [rmse_cv(Ridge(alpha = alpha),y4).mean() 

            for alpha in alphas]



cv_ridge_4 = pd.Series(cv_ridge_4, index = alphas)

cv_ridge_4.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge_5 = [rmse_cv(Ridge(alpha = alpha),y5).mean() 

            for alpha in alphas]



cv_ridge_5 = pd.Series(cv_ridge_5, index = alphas)

cv_ridge_5.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
y6 = np.log1p(train_df["DistanceToFirstStop_p80"])

cv_ridge_6 = [rmse_cv(Ridge(alpha = alpha),y6).mean() 

            for alpha in alphas]



cv_ridge_6 = pd.Series(cv_ridge_6, index = alphas)

cv_ridge_6.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")