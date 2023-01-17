# Make                Car Make

# Model               Car Model

# Year                Car Year (Marketing)

# Engine Fuel Type    Engine Fuel Type

# Engine HP           Engine Horse Power (HP)

# Engine Cylinders    Engine Cylinders

# Transmission Type   Transmission Type

# Driven_Wheels       Driven Wheels

# Number of Doors     Number of Doors

# Market Category     Market Category

# Vehicle Size        Size of Vehicle

# Vehicle Style       Type of Vehicle

# highway MPG         Highway MPG

# city mpg            City MPG

# Popularity          Popularity (Twitter)

# MSRP                Manufacturer Suggested Retail Price
import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from collections import defaultdict



import xgboost as xgb

import lightgbm as lgb

from sklearn import model_selection, ensemble

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import Imputer, LabelEncoder

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



import time

import shap
df_cleaned = pd.read_csv("../input/cars_cleaned.csv").drop('Unnamed: 0', axis=1)

df_cleaned.head(5)
# Check missing values

len(df_cleaned) - df_cleaned.count()
d = defaultdict(LabelEncoder)

columns_to_encode = [

    "Make",

    "Model",

    "Engine Fuel Type",

    "Transmission Type",

    "Driven_Wheels",

    "Market Category",

    "Transmission Type",

    "Vehicle Size",

    "Vehicle Style"]



df_cleaned.loc[:,columns_to_encode] = df_cleaned.loc[:,columns_to_encode].apply(lambda x: d[x.name].fit_transform(x.fillna('0')))

df_cleaned.info()
# GET DUMMIES, FEATURES SELECTION, PROPERLY CLEANED AND INPUTED DATAS



df_cleaned_dummies = df_cleaned.copy()

df_cleaned_dummies = pd.get_dummies(df_cleaned_dummies)



X = df_cleaned_dummies.drop(['MSRP'], axis=1)

y = df_cleaned_dummies['MSRP']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)



# Need index of car prices to predict, whatever the strategy is

indexes_cars_predicted = X_test.index
def rmsle(predicted, real):

    sum=0.0

    for x in range(len(predicted)):

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5



NUM_OF_FEATURES = X.shape[1]



train_X_train, train_X_test, train_y_train, train_y_test = train_test_split(X_train, y_train, test_size=0.2)



model = ensemble.RandomForestRegressor(n_jobs=-1, n_estimators = 100, random_state=42)

model.fit(train_X_train, train_y_train)



# Graphs section

fig = plt.figure(figsize=(15,5))

ax1 = plt.subplot(111)

plt.plot(np.cumsum(model.feature_importances_))

plt.axhline(0.85,color= 'r')



NUM_OF_FEATURES = 200



col = pd.DataFrame({'importance': model.feature_importances_, 'feature': X_train.columns}).sort_values(

    by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values



X_train = X_train[col]

X_test = X_test[col]
# Define evaluation method for a given model. we use k-fold cross validation on the training set. 

# The loss function is root mean square logarithm error between target and prediction

# Note: train and y_train are feeded as global variables



NUM_FOLDS = 5

def rmsle_cv(model,strategy):

        kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(X_train.values)

        rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))

        return(rmse)



# Ensemble method: model averaging

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # We define clones of the original models to fit the data in

    # the reason of clone is avoiding affect the original base models

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]  

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)

        return self

    

    # Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([ model.predict(X) for model in self.models_ ])

        return np.mean(predictions, axis=1)



model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, 

                             gamma=1.5, 

                             learning_rate=0.02, 

                             max_depth=12, 

                             n_estimators=1000,

                             subsample=0.7, 

                             objective='reg:linear',

                             booster='gbtree',

                             reg_alpha=0.0, 

                             eval_metric = 'rmse', 

                             silent=1, 

                             random_state =7,

                            )



model_lgb = lgb.LGBMRegressor(objective='regression',

                              num_leaves=144,

                              learning_rate=0.005, 

                              n_estimators=1000, 

                              max_depth=12,

                              metric='rmse',

                              is_training_metric=True,

                              max_bin = 55, 

                              bagging_fraction = 0.8,

                              verbose=-1,

                              bagging_freq = 5, 

                              feature_fraction = 0.9)



l_m = []

l_std = []

time_d = {}



print("GET DUMMIES, FEATURES SELECTION, KEEP MISSING VALUES\n")



start = time.time()

score_1 = rmsle_cv(model_xgb,1)

end = time.time()



l_m.append(score_1.mean())

l_std.append(score_1.std())

time_d['XGB']= end - start

print("    Xgboost score : {:.4f} ({:.4f})".format(score_1.mean(), score_1.std()))



start = time.time()

score_1 = rmsle_cv(model_lgb,1)

end = time.time()



l_m.append(score_1.mean())

l_std.append(score_1.std())

time_d['LGB']= end - start

print("    LGBM score    : {:.4f} ({:.4f})" .format(score_1.mean(), score_1.std()))



averaged_models_1 = AveragingModels(models = (model_xgb, model_lgb))



score_1 = rmsle_cv(averaged_models_1,1)



l_m.append(score_1.mean())

l_std.append(score_1.std())

print("    Averaged score: {:.4f} ({:.4f})\n" .format(score_1.mean(), score_1.std()))



print('\nLGB faster than XGB ?',time_d['LGB']>time_d['XGB'])
model_lgb.fit(X_train.values, y_train)

pred = model_lgb.predict(X_test.values)
# load JS visualization code to notebook

shap.initjs()
# Using a random sample of the dataframe for better time computation

X_sampled = X_train.sample(100, random_state=10)
# explain the model's predictions using SHAP values

# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

explainer = shap.TreeExplainer(model_lgb)

shap_values = explainer.shap_values(X_sampled)
# visualize the first prediction's explanation

shap.force_plot(explainer.expected_value, shap_values[0,:], X_sampled.iloc[0,:])
# visualize the training set predictions

shap.force_plot(explainer.expected_value, shap_values, X_train)
# summarize the effects of all the features

shap.summary_plot(shap_values, X_sampled)
shap.summary_plot(shap_values, X_sampled, plot_type="bar")
pred = np.array(pred)

original = np.array(df_cleaned_dummies.loc[indexes_cars_predicted,'MSRP'])



df_cleaned_dummies.loc[indexes_cars_predicted,'MSRP']



def average_gap(l1,l2):

    resu=0

    for i in range(len(l1)):

        resu += np.abs(l1[i]-l2[i])

    resu = resu/len(l1)

    return(resu)



print("Over",len(pred),"cars, the average gap between the predicted price and the real price is",round(average_gap(pred,original),0),"$")



plt.figure(figsize=(15,7))

sns.distplot(pred, color="blue", label="Distrib Predictions", hist = False)

sns.distplot(original, color="red", label="Distrib Original", hist = False)

plt.title("Distribution of pred and original MSRP")

plt.legend()