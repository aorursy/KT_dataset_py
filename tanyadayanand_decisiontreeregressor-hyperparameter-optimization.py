# data manuipulation

import numpy as np

import pandas as pd



# modeling utilities

from sklearn import metrics

from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split





# plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Plotting parameters tuning

sns.set_style('whitegrid')

sns.set_context('talk')

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (30, 10),

          'axes.labelsize': 'x-large',

          'axes.titlesize':'x-large',

          'xtick.labelsize':'x-large',

          'ytick.labelsize':'x-large'}



plt.rcParams.update(params)
hour_df = pd.read_csv("../input/bike-sharing-dataset/hour.csv")

hour_df.info()
# Renaming columns names to more readable names

hour_df.rename(columns={'instant':'rec_id',

                        'dteday':'datetime',

                        'holiday':'is_holiday',

                        'workingday':'is_workingday',

                        'weathersit':'weather_condition',

                        'hum':'humidity',

                        'mnth':'month',

                        'cnt':'total_count',

                        'hr':'hour',

                        'yr':'year'},inplace=True)



###########################

# Setting proper data types

###########################

# date time conversion

hour_df['datetime'] = pd.to_datetime(hour_df.datetime)



# categorical variables

hour_df['season'] = hour_df.season.astype('category')

hour_df['is_holiday'] = hour_df.is_holiday.astype('category')

hour_df['weekday'] = hour_df.weekday.astype('category')

hour_df['weather_condition'] = hour_df.weather_condition.astype('category')

hour_df['is_workingday'] = hour_df.is_workingday.astype('category')

hour_df['month'] = hour_df.month.astype('category')

hour_df['year'] = hour_df.year.astype('category')

hour_df['hour'] = hour_df.hour.astype('category')
# Defining categorical variables encoder method

def fit_transform_ohe(df,col_name):



    # label encode the column

    le = preprocessing.LabelEncoder()

    le_labels = le.fit_transform(df[col_name])

    df[col_name+'_label'] = le_labels

    # one hot encoding

    ohe = preprocessing.OneHotEncoder()

    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()

    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]

    features_df = pd.DataFrame(feature_arr, columns=feature_labels)

    return le,ohe,features_df



# given label encoder and one hot encoder objects, 

# encode attribute to ohe

def transform_ohe(df,le,ohe,col_name):

   

    # label encode

    col_labels = le.transform(df[col_name])

    df[col_name+'_label'] = col_labels

    

    # ohe 

    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()

    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]

    features_df = pd.DataFrame(feature_arr, columns=feature_labels)

    

    return features_df
# Divide the dataset into training and testing sets

X, X_test, y, y_test = train_test_split(hour_df.iloc[:,0:-3],

                                        hour_df.iloc[:,-1],

                                        test_size=0.33,

                                        random_state=42)

X.reset_index(inplace=True)

y = y.reset_index()



X_test.reset_index(inplace=True)

y_test = y_test.reset_index()

X
hour_df.shape
# Encoding all the categorical features

cat_attr_list = ['season','is_holiday',

                 'weather_condition','is_workingday',

                 'hour','weekday','month','year']

# though we have transformed all categoricals into their one-hot encodings, note that ordinal

# attributes such as hour, weekday, and so on do not require such encoding.

numeric_feature_cols = ['temp','humidity','windspeed',

                        'hour','weekday','month','year']

subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']



###############

# Train dataset

###############

encoded_attr_list = []

for col in cat_attr_list:

    return_obj = fit_transform_ohe(X,col)

    encoded_attr_list.append({'label_enc':return_obj[0],

                              'ohe_enc':return_obj[1],

                              'feature_df':return_obj[2],

                              'col_name':col})





feature_df_list  = [X[numeric_feature_cols]]

feature_df_list.extend([enc['feature_df'] \

                        for enc in encoded_attr_list \

                        if enc['col_name'] in subset_cat_features])



train_df_new = pd.concat(feature_df_list, axis=1)

print("Train dataset shape::{}".format(train_df_new.shape))

print(train_df_new.head())



##############

# Test dataset

##############

test_encoded_attr_list = []

for enc in encoded_attr_list:

    col_name = enc['col_name']

    le = enc['label_enc']

    ohe = enc['ohe_enc']

    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,

                                                              le,ohe,

                                                              col_name),

                                   'col_name':col_name})

    

    

test_feature_df_list = [X_test[numeric_feature_cols]]

test_feature_df_list.extend([enc['feature_df'] \

                             for enc in test_encoded_attr_list \

                             if enc['col_name'] in subset_cat_features])



test_df_new = pd.concat(test_feature_df_list, axis=1) 

print("Test dataset shape::{}".format(test_df_new.shape))

print(test_df_new.head())
# Constructing train dataset

X = train_df_new

y= y.total_count.values.reshape(-1,1)



# Constructing test dataset

X_test = test_df_new

y_test = y_test.total_count.values.reshape(-1,1)

print(X.shape,y.shape)
dtm = DecisionTreeRegressor(max_depth=4,

                           #min_samples_split=5,

                           #max_leaf_nodes=10

                           )



dtm.fit(X,y)

print("R-Squared on train dataset={}".format(dtm.score(X_test,y_test)))



dtm.fit(X_test,y_test)   

print("R-Squared on test dataset={}".format(dtm.score(X_test,y_test)))
!pip install --upgrade scikit-learn==0.21.3

!pip install pydotplus

#!pip install scikit-learn joblib

#!pip3 install --user sklearn
# Importing required packages for visualization

from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydotplus, graphviz
# Putting features

features = list(train_df_new.columns[:])

features

#hour_df
# plotting the tree

dot_data = StringIO()  

export_graphviz(dtm, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 40)}



# instantiate the model

dtree = DecisionTreeRegressor(criterion = "mse", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds,

                    return_train_score=True,

                   scoring="r2")

tree.fit(X, y)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting r2 with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training r2")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test r2")

plt.xlabel("max_depth")

plt.ylabel("r2")

plt.legend()

plt.show()

# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeRegressor(criterion = "mse", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                    return_train_score=True,

                   scoring="r2")

tree.fit(X, y)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting r2 with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training r2")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test r2")

plt.xlabel("min_samples_leaf")

plt.ylabel("r2")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeRegressor(criterion = "mse", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   return_train_score=True,

                   scoring="r2")

tree.fit(X, y)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_score"], 

         label="training r2")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test r2")

plt.xlabel("min_samples_split")

plt.ylabel("r2")

plt.legend()

plt.show()

param_grid = {

    'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["mse", "mae"]

}





grid_cv_dtm = GridSearchCV(dtm, param_grid, cv=5)



grid_cv_dtm.fit(X,y)





print("R-Squared::{}".format(grid_cv_dtm.best_score_))

print("Best Hyperparameters::\n{}".format(grid_cv_dtm.best_params_))
df = pd.DataFrame(data=grid_cv_dtm.cv_results_)

df.head()
# model with optimal hyperparameters

clf = DecisionTreeRegressor(criterion = "mse", 

                                  random_state = 100,

                                  max_depth=10, 

                                  min_samples_leaf=50,

                                  min_samples_split=50)



clf.fit(X, y)

#clf_gini.fit(X_train, y_train){'criterion': 'mse', 'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 50}
# plotting tree with max_depth=10

dot_data = StringIO()  

export_graphviz(clf, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# model with optimal hyperparameters

clf = DecisionTreeRegressor(criterion = "mse", 

                                  random_state = 100,

                                  max_depth=3, 

                                  min_samples_leaf=50,

                                  min_samples_split=50)



clf.fit(X, y)
# plotting tree with max_depth=3 or 4

dot_data = StringIO()  

export_graphviz(clf, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# Checking the training model scores

r2_scores = cross_val_score(grid_cv_dtm.best_estimator_, X, y, cv=10)

mse_scores = cross_val_score(grid_cv_dtm.best_estimator_, X, y, cv=10,scoring='neg_mean_squared_error')



print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))

print("MSE::{:.3f}".format(np.mean(mse_scores)))
best_dtm_model = grid_cv_dtm.best_estimator_



y_pred = best_dtm_model.predict(X_test)

residuals = y_test.flatten() - y_pred





r2_score = best_dtm_model.score(X_test,y_test)

print("R-squared:{:.3f}".format(r2_score))

print("MSE: %.2f" % metrics.mean_squared_error(y_test, y_pred))