import warnings

warnings.simplefilter("ignore")
import os

import pandas as pa

from datetime import datetime as dt

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
PLOTS_PATH = "../input/bike-sharing-demand/plots/"

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(PLOTS_PATH, fig_id + "." + fig_extension)

    print("Saving plot", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)

    print("plot saved", fig_id)
DATA_DIR_PATH = "../input/bike-sharing-demand/"

TRAIN_FILE = "train.csv"

TEST_FILE = "test.csv"



def load_csv_data(file_path, file_name):

    return pa.read_csv(os.path.join( file_path, file_name) )



train_data = load_csv_data(DATA_DIR_PATH,TRAIN_FILE)
# Copy dataset for analysis and manupulations.

analysis_data_set = train_data.copy()
# get the metadata of the data.

analysis_data_set.info()
analysis_data_set.head()
# "weekday","month","hour","year" extraction

def transform(X):

    c_datetime = X["datetime"].unique()

    formatted_dt = [dt.strptime(x, "%Y-%m-%d %H:%M:%S") for x in c_datetime]

    X["weekday"] = extract_part_from_date(formatted_dt,"%w")

    X["month"] = extract_part_from_date(formatted_dt,"%m")

    X["hour"] = extract_part_from_date(formatted_dt,"%H")

    X["year"] = extract_part_from_date(formatted_dt,"%Y")



    int_features = ["weekday","month","hour","year"]

    for feature in int_features:

        X[feature] = X[feature].astype(np.int64)



    return X



def extract_part_from_date(dates, code_str):

    return [date.strftime(code_str) for date in dates]



analysis_data_set = transform(analysis_data_set)    

analysis_data_set = analysis_data_set.drop(["datetime"], axis=1)



# print first 6 instances of the dataset

analysis_data_set.head(6)
import missingno as msno

msno.matrix(analysis_data_set,figsize=(12,5))
corr_matrix=analysis_data_set.corr()

corr_matrix["count"].sort_values(ascending=False)
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.heatmap(corr_matrix, cmap=cmap)
features = ["temp","season","windspeed","humidity","weather","count"]

g = sns.pairplot(analysis_data_set,

                 vars= features,

                 palette = 'husl',

                 height=1.4,

                 aspect = 1.5,

                 kind='reg',

                 diag_kind = 'kde',

                 diag_kws=dict(shade=True),

                 plot_kws=dict(scatter_kws={'s':.2}),

                 

            )

g.set(xticklabels=[])
# "temp","season","windspeed","humidity","weather","count"

def bxplot(data):

    fig, axes =  plt.subplots(nrows=3,ncols=2)

    fig.set_size_inches(12,10)

    sns.boxplot(y=data['count'],ax=axes[0][0])

    sns.boxplot(x=data['temp'],y=data['count'],ax=axes[0][1])

    sns.boxplot(x=data['season'],y=data['count'],ax=axes[1][0])

    sns.boxplot(x=data['windspeed'],y=data['count'],ax=axes[1][1])

    sns.boxplot(x=data['humidity'],y=data['count'],ax=axes[2][0])

    sns.boxplot(x=data['weather'],y=data['count'],ax=axes[2][1])

    
bxplot(analysis_data_set)
from scipy.stats import zscore

analysis_data_set = analysis_data_set[(np.abs(zscore(analysis_data_set)) < 3).all(axis=1)]
bxplot(analysis_data_set)
# hour 

sns.catplot(x='hour',y='count', hue="season", data=analysis_data_set, kind="point", aspect=2)
# day

sns.catplot(x='hour',y='count',hue="weekday",data=analysis_data_set, kind="point", aspect=2)
# month wise rent

sns.barplot(x="month", y="count", data=analysis_data_set, capsize=0.2)
# train and test split 

from sklearn.model_selection import StratifiedShuffleSplit



stratified = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_i, validation_i in stratified.split(train_data, train_data['season']):

    strat_train_set = train_data.loc[train_i]

    strat_validation_set = train_data.loc[validation_i]
from sklearn.model_selection import train_test_split

train, validation = train_test_split(train_data, test_size=0.2, random_state=42)
train_X = strat_train_set.drop(["count"],axis=1).copy()

train_Y = strat_train_set["count"].copy()
train_X.info()
from sklearn.base import BaseEstimator, TransformerMixin



class DatePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        return None

    

    def fit(self,X,y=None):

        return self

    

    def transform(self, X, y=None):

        c_datetime = X["datetime"].unique()

        formatted_dt = [dt.strptime(x, "%Y-%m-%d %H:%M:%S") for x in c_datetime]

        X["weekday"] = self.extract_part_from_date(formatted_dt,"%w")

        X["month"] = self.extract_part_from_date(formatted_dt,"%m")

        X["hour"] = self.extract_part_from_date(formatted_dt,"%H")

        X["year"] = self.extract_part_from_date(formatted_dt,"%Y")

        

        int_features = ["weekday","month","hour","year"]

        for feature in int_features:

            X[feature] = X[feature].astype(np.int64)

        return X

    

    def extract_part_from_date(self, dates, code_str):

        return [date.strftime(code_str) for date in dates]
class DropColumns(BaseEstimator, TransformerMixin):

    

    def __init__(self, columns):

        self.columns = columns

    

    def fit(self,X):

        return self

    

    def transform(self, X):

        X = X.drop(self.columns, axis=1)

        return X 
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

drop_cols = ['datetime','registered','casual','atemp']

num_pipline = Pipeline([

        ("date_processor", DatePreprocessor()),

        ("drop_cols", DropColumns(columns = drop_cols) )

])
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

num_attribs = list(train_X)



cat_attribs = ["season","weather"]



cat_pipeline = ColumnTransformer([

    ("num_pipline", num_pipline, num_attribs), # Handling Text and Categorical Attributes

    ("cat_pipline", OneHotEncoder(sparse=False, categories='auto'), cat_attribs) 

])



train_prepared = cat_pipeline.fit_transform(train_X)
# train_prepared = train_prepared[(np.abs(zscore(train_prepared)) < 3).all(axis=1)]
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(train_prepared,train_Y)
validation_x = validation.drop(['count'],axis=1)

validation_y = validation['count'].copy()



num_attribs = list(validation_x)

cat_attribs = ["season","weather"]

drop_cols = ['datetime','registered','casual','atemp','holiday']

validation_x_prepared = cat_pipeline.transform(validation_x)

rfr_prediction = rfr.predict(validation_x_prepared)
rfr_prediction[:2], validation_y[:2]
from sklearn.metrics import mean_squared_error



mse = mean_squared_error(validation_y,rfr_prediction)

lin_rmse = np.sqrt(mse)

lin_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rfr, validation_x_prepared, validation_y, cv=10, scoring="neg_mean_squared_error")

rmse_scores = np.sqrt(-scores)
scores
scores.mean()
scores.std()
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]

model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
validation_x = strat_validation_set.drop(['count'],axis=1)

validation_y = strat_validation_set['count'].copy()

num_attribs = list(validation_x)

cat_attribs = ['season','weather']

drop_cols = ['datetime','registered','casual','atemp','holiday']



rmses=[]

stds = []

d={}

trained_models = []

for model in models:

    model.fit(train_prepared,train_Y)

    trained_models.append(model)

    validation_x_prepared = cat_pipeline.transform(validation_x)

    prediction = model.predict(validation_x_prepared)

    # validation

    scores = cross_val_score(model, validation_x_prepared, validation_y, cv=10, scoring="neg_mean_squared_error")

    rmse_scores = np.sqrt(-scores)

    rmses.append(rmse_scores.mean())

    stds.append(rmse_scores.std())

    

d = {"Models": model_names, "mean rsme score": rmses, "Standard deviation": stds}

df = pa.DataFrame(d)

df
df.loc[df['mean rsme score'].idxmin()]
model = trained_models[0]

model
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3,10,20,30]}

]



rfr = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(rfr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_prepared,train_Y)
grid_search.best_params_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

    'n_estimators': randint(1,200)

}



rfr = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(rfr, param_distributions=param_distribs,cv=10, scoring='neg_mean_squared_error')

random_search.fit(train_prepared,train_Y)
random_search.best_estimator_
random_search.best_params_
cvres = random_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):

    print(np.sqrt(-mean_score), params)
test_data = load_csv_data(DATA_DIR_PATH,TEST_FILE)

test_data.columns

test_data.head()
validation_x.head()
final_model = random_search.best_estimator_



X_test = test_data.copy()

X_test['casual'] = 0

X_test['registered'] = 0



num_attribs = list(X_test)



X_prepared = cat_pipeline.transform(X_test)

prediction = final_model.predict(X_prepared)
prediction
len(prediction),len(X_test)
list(prediction[:10])
out_dict = {"date":test_data.datetime, "count" : prediction }

out_dict_df = pa.DataFrame(out_dict)

out_dict_df.head()
# out_dict_df.to_csv('../input/bike-sharing-demand/bike-sharing-demand-estimation.csv',index=False)