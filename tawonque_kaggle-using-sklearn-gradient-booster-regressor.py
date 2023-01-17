import os

import numpy as np

import pandas as pd



import sklearn.linear_model as lm

import sklearn.cross_validation as cv

import sklearn.preprocessing as pp

from  sklearn import metrics, tree, grid_search

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor 



import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
os.getcwd()
os.chdir('../input/')

realestate = pd.read_csv('../input/train.csv')

realestate_test = pd.read_csv('../input/test.csv')



#data_description = open('../input/data_description.txt', 'r')

#print (data_description.read())
realestate.info()

realestate_test.info()



realestate = realestate.dropna(how='all')

realestate_test = realestate_test.dropna(how='all')





#columns withe very few values

few_values = ['Alley', 'PoolQC', 'MiscFeature']

realestate = realestate.drop(few_values, axis=1)

realestate_test = realestate_test.drop(few_values, axis=1)
print(realestate.info(), realestate.shape)

print(realestate_test.info(), realestate_test.shape)
#some columns with categorical variables are not 'object'

realestate.MSSubClass.astype('object', inplace=True)

realestate_test.MSSubClass.astype('object', inplace=True)

print('Done')
#some columns could be treated as quantitative variables

def change_scale(legend, scale, column_to_replace):

    j = 0

    for i in legend:

        command = column_to_replace + '.replace(to_replace="' + i + '"' + ', value=' + scale[j].astype('str') + ', inplace=True)'

        #print(command)        

        exec(command)

        j += 1

        if j == len(scale):

            break

    return 



legend = ['Ex', 'Gd', 'TA', 'Fa', 'Po']

scale = np.arange(5,0,-1)

column_to_replace = 'realestate.HeatingQC'

change_scale(legend, scale, column_to_replace)

change_scale(legend, scale, 'realestate_test.HeatingQC')



legend = ['Ex', 'Gd', 'TA', 'Fa', 'Po']

scale = np.arange(5,0,-1)

column_to_replace = 'realestate.KitchenQual'

change_scale(legend, scale, column_to_replace)

change_scale(legend, scale, 'realestate_test.KitchenQual')
#Let's separate categorical from quantitative variables and make two dataframes

#Thanks to the kernel by BreadenFitz-Gerald for the idea



df = realestate

categorical = []

for col in df.columns.values:

    if df[col].dtype == 'object':

        categorical.append(col)



df_category = df[categorical]

df_quant = df.drop(categorical, axis=1)



df_category_test = realestate_test[categorical]

df_quant_test = realestate_test.drop(categorical, axis=1)
#We can calculate the correlation coefficients among variables and flag those with extremely high values

corr = df_quant[df_quant.columns[1:39]].corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



fig = plt.figure(figsize=(8,8))

plt.subplot2grid((1,1), (0,0))

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=.9, square=True, annot=False)
#We can also calculate the skewness and notice that there are many variables

skew = df_quant[df_quant.columns[1:40]].skew()

print(skew)
df_quant.info()
#Function to eliminate columns with more than N null values and substitute null values in the remaining with median:

def null_value_treatment(dataframe, thresh_null):

    for col in dataframe.columns.values:

        if np.sum(dataframe[col].isnull()) > thresh_null:

            dataframe.drop(col, axis=1, inplace=True)

            print(col)

        elif np.sum(dataframe[col].isnull()) > 0:

            median = dataframe[col].median()

            idx = np.where(dataframe[col].isnull())[0]

            dataframe[col].iloc[idx] = median

    return



#We could do the same operation for the test dataset, but in reality, we want to keep the same number of predictors
null_value_treatment(df_quant, 150)

null_value_treatment(df_quant_test, 150)
def transform_skew(dataframe, skew_thresh):

    for col in dataframe.columns.values: 

        if (dataframe[col].skew()) > skew_thresh:

            dataframe[col] = np.log(dataframe[col])

            dataframe[col] = dataframe[col].apply(lambda x: 0 if x == (-1*np.inf) else x)

#           df_quant[col] = Normalizer().fit_transform(df_quant[col].reshape(1,-1))[0]



transform_skew(df_quant, 1.0)

transform_skew(df_quant_test, 1.0)
def null_value_treatment_categorical(dataframe, thresh_null):

    for col in dataframe.columns.values:

        if np.sum(dataframe[col].isnull()) > thresh_null:

            dataframe.drop(col, axis=1, inplace=True)

            print(col)

        elif np.sum(dataframe[col].isnull()) > 0:

            dataframe[col] = dataframe[col].fillna('MIA', inplace=True)

    return



null_value_treatment_categorical(df_category, 150)

print('----------------')

null_value_treatment_categorical(df_category_test, 150)
cat_variables = df_category.columns.values

cat_variables_test = df_category_test.columns.values



df_dummies = pd.get_dummies(df_category, columns=cat_variables)

df_dummies_test = pd.get_dummies(df_category_test, columns=cat_variables)
#Dummies are different size because of missing dimensions/values within some of the predictors

print(df_category.shape)

print('------------------')

print(df_category_test.shape)

print('------------------')

print(df_dummies.shape)

print('------------------')

print(df_dummies_test.shape)

print('------------------')

print(df_quant.shape) #one predictor more than test because it still contains the 'predicted' feature.

print('------------------')

print(df_quant_test.shape)
#Let's check if we have the same columns

print(df_category.columns)

df_category_test.columns
#Here we have to merge categorical datasets, then run the dummies, then separate.

df_category_joint = pd.concat([df_category, df_category_test])

df_category_joint.shape
df_dummies_joint = pd.get_dummies(df_category_joint, columns=cat_variables, drop_first=True)

df_dummies_joint.shape



df_dummies = df_dummies_joint[0:1460]

df_dummies_test = df_dummies_joint[1460:2919]
#Let's verify...

print(df_dummies_joint[0:1460].shape)

print(df_dummies_joint[1460:2919].shape)
y_train = df_quant['SalePrice']



X_train = df_dummies.join(df_quant)

X_train = X_train.drop(['SalePrice', 'Id'], axis=1)



X_test = df_dummies_test.join(df_quant_test)

X_test = X_test.drop(['Id'], axis=1)
X_train.head()
X_test.head()
y_train.head()

y_train.shape

#---xxx---
# Train a random forest with XXX decision trees

model_rf1 = RandomForestRegressor(n_estimators=100, max_depth=15)
#Fit the training data

model_rf1.fit(X_train, y_train)
# Define folds for cross-validation

kf = cv.KFold(1460, n_folds=10, shuffle=True)

#kf = 5

scores = cv.cross_val_score(model_rf1, X_train, y_train, cv=kf)

print(scores, ' and the mean score is = ', scores.mean())
# Investigate importances of predictors

###model_rf1.feature_importances_

feature_importance = model_rf1.feature_importances_
# make importances relative to max importance

feature_importance = 100 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

featimp = feature_importance[sorted_idx]

feat = X_train.columns[sorted_idx]

pos = np.arange(sorted_idx.shape[0]) + .5



a = 0 

b = 50 #To limit the number of features

c = b - a



featimp= featimp[::-1][a:b]

feat = feat[::-1][a:b]

pos = pos[::-1][a:b]



fig = plt.figure(figsize=(7,7))

plt.subplot2grid((1,1), (0,0))

with sns.axes_style("white"):

    ax = sns.barplot(y=feat, x=featimp)

plt.xlabel('Relative Importance')

plt.title('Variable Importance (first {0} variables)'.format(c))

plt.show()
#Let's check for a second the names of the columns to be sure what we have...

X_train.columns.values
#redundant_variables = ['CentralAir_N'] #undecided if eliminate 'GarageArea' as it is highly correlated with 'GarageCars'

#However, in the context of an American house, perhaps the area of the garage and the number of cars fitting it are considered separately by the consumer?

#A 1-car garage but huge might not be the same than a 1-car gerage barely fittign the car. 

#With some research we could elucidate this and take an informed decision. 



#X_train = X_train.drop(redundant_variables, axis=1)

#X_test = X_test.drop(redundant_variables, axis=1)
print(X_test.shape, X_train.shape)
def check_classifiers(X, y):

    """

    Returns a sorted list of accuracy scores from fitting and scoring passed data

    against several algorithms.

    """

    params = 100

    _cv = kf

    classifier_score = {}

    

    scores = cv.cross_val_score(RandomForestRegressor(n_estimators=params), X, y, cv=_cv)

    classifier_score['Random Forest Regressor'] = scores.mean()

    

    scores = cv.cross_val_score(BaggingRegressor(n_estimators=params), X, y, cv=_cv)

    classifier_score['Bagging Regressor'] = scores.mean()

    

    scores = cv.cross_val_score(ExtraTreesRegressor(n_estimators=params), X, y, cv=_cv)

    classifier_score['ExtraTrees Regressor'] = scores.mean()

    

    scores = cv.cross_val_score(AdaBoostRegressor(n_estimators=params), X, y, cv=_cv)

    classifier_score['AdaBoost Regressor'] = scores.mean()

    

    scores = cv.cross_val_score(GradientBoostingRegressor(n_estimators=params), X, y, cv=_cv)

    classifier_score['Gradient Boost Regressor'] = scores.mean()



    #return sorted(classifier_score.items(), key=operator.itemgetter(1), reverse=True)

    return sorted(classifier_score.items(), reverse=True)



check_classifiers(X_train, y_train)
model_rf3 = GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='ls',

             max_depth=3, max_features=None, max_leaf_nodes=None,

             min_samples_leaf=1, min_samples_split=2,

             min_weight_fraction_leaf=0.0, n_estimators=500,

             presort='auto', random_state=None, subsample=1.0, verbose=0,

             warm_start=False)

model_rf3.fit(X_train, y_train)
scores = cv.cross_val_score(model_rf3, X_train, y_train, cv=kf)

scores.mean()
y_test = model_rf3.predict(X_test)
y = np.exp(y_test)

y
#Let's group some data...

train_diag = X_train.join(np.exp(y_train))

train_diag['dataset'] = 'train'

train_diag['Id'] = realestate['Id'

                             ]

test_diag = X_test

test_diag['SalePrice'] = y

test_diag['dataset'] = 'test'

test_diag['Id'] = realestate_test['Id']



total_diag = pd.concat([train_diag, test_diag])



print(train_diag.shape, test_diag.shape, total_diag.shape)

test_diag.head()

#total_diag.columns.values
fig = plt.figure(figsize=(6,15))



plt.subplot2grid((2,1), (0,0))

with sns.axes_style("white"):

    sns.violinplot(x='OverallQual', y='SalePrice', hue='dataset', data=total_diag, split=True,

               inner='quart', palette={'train': 'r', 'test': 'y'})

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Overall quality')

plt.ylabel('Sale price')

plt.title('Sale price and overall quality')



'''

plt.subplot2grid((2,1), (0,1))

with sns.axes_style("white"):

    ax = sns.jointplot(X_test['OverallQual'], y, kind="hex", stat_func=kendalltau, color="#4CB391")

plt.xlabel('Relative Importance')

#plt.title('Variable Importance (first {0} variables)'.format(c))'''



plt.show()
submission = test_diag[['Id', 'SalePrice']]

print('Done! Your comments are welcome')

submission.to_csv("submission_realestate_tawonque.csv", index=False)