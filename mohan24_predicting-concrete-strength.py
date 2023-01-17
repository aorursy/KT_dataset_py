import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

from scipy.stats import iqr 

%matplotlib inline
# Loading the Dataset

data = pd.read_csv('/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')

# lets take a look on dataset

data.head()
#Renaming feature names

col_map = {'Cement (component 1)(kg in a m^3 mixture)': 'cement',

 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'slag',

 'Fly Ash (component 3)(kg in a m^3 mixture)': 'ash',

 'Water  (component 4)(kg in a m^3 mixture)': 'water',

 'Superplasticizer (component 5)(kg in a m^3 mixture)': 'superplastic',

 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarseagg',

 'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fineagg',

 'Age (day)': 'age',

 'Concrete compressive strength(MPa, megapascals) ': 'strength'}
data.rename(columns=col_map,inplace=True)

data.head()
# Checking for null values in dataset

data.info()
# Five Point Statistics Summary

#data.describe().T

summary = data.describe().T

summary['Diff'] = summary['mean'] - summary['50%']

summary
sns.pairplot(data)
## Plotting BoxPlot to perform univariate analysis

data.plot(kind='box',figsize=(15,8))
# Function to find the Upper Cut-off & Median Value for the Given Variable

def outlier_cap(df):

    IQR = iqr(df)

    Q3 = np.percentile(df,75)

    ucap = IQR*1.5 + Q3

    median = df[df<ucap].median()

    return ucap,median
# Treating fineagg outlier

f,((ax_box,ax_box_post),(ax_hist,ax_hist_post)) = plt.subplots(2,2,gridspec_kw={'height_ratios':(0.15,0.85)},figsize=(10,7))

sns.boxplot(data['fineagg'],ax=ax_box).set_title("fineagg_Pre")

sns.distplot(data['fineagg'],ax=ax_hist)

ucap_fineagg,median_fineagg = outlier_cap(data['fineagg'])

data.loc[data['fineagg']>ucap_fineagg,'fineagg'] = median_fineagg

sns.boxplot(data['fineagg'],ax=ax_box_post).set_title("fineagg_Post")

sns.distplot(data['fineagg'],ax=ax_hist_post)
# Treating age outlier

f,((ax_box,ax_box_post),(ax_hist,ax_hist_post)) = plt.subplots(2,2,gridspec_kw={'height_ratios':(0.15,0.85)},figsize=(10,7))

sns.boxplot(data['age'],ax=ax_box).set_title("age_Pre")

sns.distplot(data['age'],ax=ax_hist)

ucap_age,median_age = outlier_cap(data['age'])

data.loc[data['age']>ucap_age,'age'] = median_age

sns.boxplot(data['age'],ax=ax_box_post).set_title("age_Post")

sns.distplot(data['age'],ax=ax_hist_post)
#%matplotlib notebook

plt.figure(figsize=(9,5))

corr = data.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

#ax = sns.heatmap(corr,annot=True,linewidth=0.5)

with sns.axes_style("white"):

    ax = sns.heatmap(corr,annot=True,linewidth=2,

                mask = mask,cmap="magma")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

# Feature Engineering

data['water_cement'] = data['cement']/data['water']

data.corr()
#Splitting the dataset in X & y

X = data.drop(['strength','cement','water'],axis=1)

y = data['strength']
# Importing preprocessing & sklearn libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# Splitting the dataset into training & testing dataset

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=400)



#Scaling the dataset

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Importing linear , Tree & Ensemble model libraries

from sklearn.linear_model import Lasso,Ridge

from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor
#Lets create a Lasso object & try to fit our model

reg_lasso = Lasso(alpha=0.1,max_iter=10e5)

reg_lasso.fit(X_train_scaled,y_train)

#Checking the train & test score

print("Train Score {} & Test Score {}".format(reg_lasso.score(X_train_scaled,y_train),reg_lasso.score(X_test_scaled,y_test)))
# Lets check the Feature Coefficent of Lasso Model

values = np.argsort(reg_lasso.coef_)

val = sorted(reg_lasso.coef_)

columns = X.columns.tolist()

col = [columns[i] for i in values]

plt.bar(col,val)

plt.title("Lasso Co-efficient Plot")

plt.xticks(rotation=45)
#Lets create a Ridge object & try to fit our model

reg_ridge = Ridge(alpha=0.01,max_iter=10e5)

reg_ridge.fit(X_train_scaled,y_train)

print("Train Score {} & Test Score {}".format(reg_ridge.score(X_train_scaled,y_train),reg_ridge.score(X_test_scaled,y_test)))
# Lets check the Feature Coefficent of Ridge Model

values = np.argsort(reg_ridge.coef_)

val = sorted(reg_ridge.coef_)

columns = X.columns.tolist()

col = [columns[i] for i in values]

plt.title("Ridge Co-efficient Plot")

plt.bar(col,val)

plt.xticks(rotation=45)
#Modelling with DecisionTreeRegressor

reg_dt = DecisionTreeRegressor()

reg_dt.fit(X_train,y_train)

print("Train Score {:.2f} & Test Score {:.2f}".format(reg_dt.score(X_train,y_train),reg_dt.score(X_test,y_test)))
#Tuning Hyperparameter max_depth & min_sam_split of DecisionTreeRegressor

max_d = list(range(1,10))

min_sam_split = list(range(10,50,15))

from sklearn.model_selection import GridSearchCV

gridcv = GridSearchCV(reg_dt,param_grid={'max_depth':max_d,'min_samples_split':min_sam_split},n_jobs=-1)

gridcv.fit(X_train,y_train)
print("Parameters :",gridcv.best_params_)

print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv.score(X_train,y_train),gridcv.score(X_test,y_test)))
# Lets find out the feature importance based on DecisionTree Model

importances = reg_dt.feature_importances_

col = X.columns.tolist()

indices = importances.argsort()[::-1]

names = [col[i] for i in indices]



# Plotting Feature importance Chart

plt.title("Decision Tree: Feature Importance")

plt.bar(range(X.shape[1]),importances[indices])

plt.xticks(range(X.shape[1]),names,rotation=45);
#Modelling with RandomForestRegressor

reg_rfe = RandomForestRegressor()

reg_rfe.fit(X_train,y_train)

print("Train Score {:.2f} & Test Score {:.2f}".format(reg_rfe.score(X_train,y_train),reg_rfe.score(X_test,y_test)))
# Tuning Hyperparameter of DecisionTreeRegressor

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Maximum number of levels in tree

max_depth = list(range(10,110,10))

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

from sklearn.model_selection import GridSearchCV

gridcv_rf = GridSearchCV(reg_rfe,param_grid={'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf},n_jobs=-1)

gridcv_rf.fit(X_train,y_train)
print("Best Parameters:",gridcv_rf.best_params_)

print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv_rf.score(X_train,y_train),gridcv_rf.score(X_test,y_test)))
import xgboost

reg_xgb = xgboost.XGBRegressor()

# Fitting model 

reg_xgb.fit(X_train,y_train)

print("Train Score {:.2f} & Test Score {:.2f}".format(reg_xgb.score(X_train,y_train),reg_xgb.score(X_test,y_test)))
import warnings

warnings.filterwarnings("ignore")



gridcv_xgb = GridSearchCV(reg_xgb,param_grid={'n_estimators':list(range(100,1000,100)),'max_depth':list(range(1,10,1)),

                                     'learning_rate':[0.0001,0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3]})

gridcv_xgb.fit(X_train,y_train);
print("Best Parameter:",gridcv_xgb.best_params_)

print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv_xgb.score(X_train,y_train),gridcv_xgb.score(X_test,y_test)))