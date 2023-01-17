# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



print(train.head(), "\n")

print(train.info())
# we dont want to include our target SalePrice in our data, 

# hence we pick all columns except SalePrice



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))   





# now we plot the original and the log1p-transformed target SalePrice to see how it looks 



prices = pd.DataFrame({"price":train["SalePrice"], "log(1+price)":np.log1p(train["SalePrice"])})  

prices.hist()
#log1p transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



# exclude all columns that have 'object' as dtype to get numeric columns 

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index





# compute skewness of numeric features, but

# drop NaNs since they cause errors in the skewness computation



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) 





# Now we want to plot the skewness of all numeric features

# you can also use sns.barplot(), but i find this version prettier than the sns.barplot



skewed_feats.head(30).plot(kind='barh', figsize=(10,6))    





# As you maybe already know, positive skewness can be transformed nicely with a log plot

# to make the feature 'more' normal distributed.

# Here we choose a threshold of 0.75 for the skewness, threshold must be exceeded,

# such that this skewed feature gets log1p transformed.





skewed_feats = skewed_feats[skewed_feats > 0.75]



all_data[skewed_feats.index] = np.log1p(all_data[skewed_feats.index])  



#print(all_data)
#  dummy encoding of our entire dataframe, it's necessary for the fitting of models.

all_data = pd.get_dummies(all_data)





#  filling NA's with the mean of the column, removing missing values/NaNs is always recommended:

all_data = all_data.fillna(all_data.mean())



print("all_data.shape: ", all_data.shape, "\n")

print("successfully dummy encoded!")
#  Now we split up our entire dataset into train data, test data and our target y



X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice



del all_data



print(X_train.shape)

print(X_test.shape)
print(numeric_feats, "\n")

print("number of numeric features: ", len(numeric_feats), "\n")



numeric_data = X_train[numeric_feats]
corrmat = numeric_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1.0, square=True);

plt.show()
#####################################################################################

#     3.1.)  Ridge model with L2 regularization

#####################################################################################



from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Ridge





selector = SelectFromModel(estimator = Ridge()).fit(X_train, y)



coefs = selector.estimator_.coef_



coefs = pd.Series(coefs, index = X_train.columns)



important_coefs = pd.concat([coefs.sort_values().head(15),

                     coefs.sort_values().tail(15)])



matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)





important_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")
# the top 10 features of the Ridge model are:



# GrLivArea

# RoofMatl_WdShngl

# Neighborhood_StoneBr

# GarageQual_Ex

# PoolQC_Ex

# Condition2_PosA

# Functional_Typ

# Neighborhood_Crawfor

# MSZoning_FV

# Condition2_Feedr
#####################################################################################

#     3.2.)  Lasso model with L1 regularization

#####################################################################################





from sklearn.linear_model import Lasso



selector = SelectFromModel(estimator = Lasso()).fit(X_train, y)



coefs = selector.estimator_.coef_



coefs = pd.Series(coefs, index = X_train.columns)



important_coefs = pd.concat([coefs.sort_values().head(15),

                     coefs.sort_values().tail(15)])



matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)





important_coefs.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
selector = SelectFromModel(estimator = Lasso(alpha = 0.001)).fit(X_train, y)



coefs = selector.estimator_.coef_



coefs = pd.Series(coefs, index = X_train.columns)



important_coefs = pd.concat([coefs.sort_values().head(15),

                     coefs.sort_values().tail(15)])



matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)





important_coefs.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
################################################################################################

#     4.1.)  feature_selection.SelectKBest    with f_regression

################################################################################################



#  SelectKBest works with univariate statistical tests, 

#  that calculate how the features relate with the target

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression      



feature_cols = X_train.columns





#  f_regression stands for the ANOVA F-value test and is used for regression tasks

selector = SelectKBest(score_func = f_regression, k=10)    

X_new = selector.fit_transform(X_train[feature_cols], y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                 index = X_train.index,

                                 columns=feature_cols)



selected_columns = selected_features.columns[selected_features.var() != 0]  

print("selected_columns: ", selected_columns, "\n")





scores = pd.Series(selector.scores_)



sorted_scores = sorted(scores)[::-1][0:10]



print("sorted_scores: ", sorted_scores, "\n")



#print(selected_columns)

sns.barplot(sorted_scores, selected_columns, orient = "h")
# top10 of SelectKBest, f_regression:          



#'OverallQual', 

#'YearBuilt', 

#'YearRemodAdd', 

#'1stFlrSF', 

#'GrLivArea',

#'FullBath', 

#'GarageCars', 

#'GarageArea', 

#'ExterQual_TA',

#'KitchenQual_TA'
################################################################################################

#     4.1.)  feature_selection.SelectKBest  with mutual_info_regression

################################################################################################





#  mutual_info_regression 

selector = SelectKBest(score_func = mutual_info_regression, k=10)    



print("started fitting.....\n")

X_new = selector.fit_transform(X_train[feature_cols], y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                 index = X_train.index,

                                 columns=feature_cols)



selected_columns = selected_features.columns[selected_features.var() != 0]  

print("selected_columns: ", selected_columns, "\n")





scores = pd.Series(selector.scores_)



sorted_scores = sorted(scores)[::-1][0:10]



print("sorted_scores: ", sorted_scores, "\n")



#print(selected_columns)

sns.barplot(sorted_scores, selected_columns, orient = "h")
# top10 of SelectKBest, mutual_info_regression:     # top10 of f_regression:      



#'MSSubClass'                                        #'OverallQual', 

#'OverallQual',                                      #'YearBuilt', 

#'YearBuilt',                                        #'YearRemodAdd', 

#'TotalBsmtSF'                                       #'1stFlrSF', 

#'1stFlrSF',                                         #'GrLivArea',

#'GrLivArea',                                        #'FullBath', 

#'GarageYrBlt',                                      #'GarageCars', 

#'GarageCars',                                       #'GarageArea', 

#'GarageArea',                                       #'ExterQual_TA',

#'ExterQual_TA',                                     #'KitchenQual_TA'
################################################################################################

#   4.2.)  Permutation importance

################################################################################################





from IPython.display import display

import eli5

from eli5.sklearn import PermutationImportance
# important features so far:



# top 10 of Lasso model:         top 10 of selectKBest f_regression:  



# GrLivArea                               #'OverallQual', 

# Neighborhood_StoneBr                    #'YearBuilt', 

# Neighborhood_Crawfor                    #'YearRemodAdd', 

# Neighborhood_NoRidge                    #'1stFlrSF', 

# Functional_Typ                          #'GrLivArea',

# LotArea                                 #'FullBath',  

# Neighborhood_NridgHt                    #'GarageCars',   

# Exterior1st_BrkFace                     #'GarageArea', 

# KitchenQual_Ex                          #'ExterQual_TA',

# OverallQual                             #'KitchenQual_TA'
################################################################################################

#   4.2.1)  Permutation importance with Ridge

################################################################################################



#  which alpha to choose





X_train_short = X_train.loc[:, ['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea',

                                'FullBath', 'GarageCars', 'GarageArea', 'ExterQual_TA', 'KitchenQual_TA', 

                                'Neighborhood_StoneBr', 'Functional_Typ', 'LotArea', 'Exterior1st_BrkFace', 

                                'KitchenQual_Ex']]     



print("fitting model...")

my_model = Ridge().fit(X_train_short, y)



#  for performance reasons i will only use n_estimators = 100 in this tutorial,

#  i have tested it with 500 as well, the results didnt really change, 

#  but everything took much much longer to compute.





print("calculating permutation importance...(takes about 1-2 minutes)")

perm = PermutationImportance(my_model, random_state=1).fit(X_train_short, y)



display(eli5.show_weights(perm, feature_names = X_train_short.columns.tolist()))
# top10 of Ridge coefficients:          top10 of permutation importance:



# GrLivArea                                 # GrLivArea 

# RoofMatl_WdShngl                          # OverallQual

# Neighborhood_StoneBr                      # YearBuilt

# GarageQual_Ex                             # LotArea

# PoolQC_Ex                                 # YearRemodAdd

# Condition2_PosA                           # 1stFlrSF

# Functional_Typ                            # GarageCars

# Neighborhood_Crawfor                      # Functional_Typ

# MSZoning_FV                               # KitchenQual_Ex

# Condition2_Feedr                          # Exterior1st_BrkFace
################################################################################################

#   4.2.2)  Permutation importance with Lasso 

################################################################################################





X_train_short = X_train.loc[:, ['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea',

                                'FullBath', 'GarageCars', 'GarageArea', 'ExterQual_TA', 'KitchenQual_TA', 

                                'Neighborhood_StoneBr', 'Functional_Typ', 'LotArea', 'Exterior1st_BrkFace', 

                                'KitchenQual_Ex']]     



print("fitting model...")

my_model = Lasso().fit(X_train_short, y)



#  for performance reasons i will only use n_estimators = 100 in this tutorial,

#  i have tested it with 500 as well, the results didnt really change, 

#  but everything took much much longer to compute.





print("calculating permutation importance...(takes about 1-2 minutes)")

perm = PermutationImportance(my_model, random_state=1).fit(X_train_short, y)



display(eli5.show_weights(perm, feature_names = X_train_short.columns.tolist()))
print("fitting model...")

my_model = Lasso(alpha = 0.001).fit(X_train_short, y)



#  for performance reasons i will only use n_estimators = 100 in this tutorial,

#  i have tested it with 500 as well, the results didnt really change, 

#  but everything took much much longer to compute.





print("calculating permutation importance...(takes about 1-2 minutes)")

perm = PermutationImportance(my_model, random_state=1).fit(X_train_short, y)



display(eli5.show_weights(perm, feature_names = X_train_short.columns.tolist()))
################################################################################################

#   4.2.3)  Permutation importance with RandomForestRegressor

################################################################################################



from sklearn.ensemble import RandomForestRegressor



X_train_short = X_train.loc[:, ['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea',

                                'FullBath', 'GarageCars', 'GarageArea', 'ExterQual_TA', 'KitchenQual_TA', 

                                'Neighborhood_StoneBr', 'Functional_Typ', 'LotArea', 'Exterior1st_BrkFace', 

                                'KitchenQual_Ex']]     



print("fitting model...")

my_model = RandomForestRegressor(n_estimators=100, random_state=1).fit(X_train_short, y)



#  for performance reasons i will only use n_estimators = 100 in this tutorial,

#  i have tested it with 500 as well, the results didnt really change, 

#  but everything took much much longer to compute.





print("calculating permutation importance...(takes about 1-2 minutes)")

perm = PermutationImportance(my_model, random_state=1).fit(X_train_short, y)



display(eli5.show_weights(perm, feature_names = X_train_short.columns.tolist()))
################################################################################################

#  4.3.)  SHAP Values

################################################################################################





import shap # package used to calculate Shap values

#from sklearn                        import metrics, svm

from sklearn import preprocessing

from sklearn import utils



#  This cell takes about 2 minutes to run

print("This cell takes about 2 minutes to run. \n")





#  fit model

print("fitting model...")

my_model = RandomForestRegressor(n_estimators = 100, random_state=1).fit(X_train, y) 





# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# Calculate Shap values

shap_values = explainer.shap_values(X_train)



shap.initjs()



display(shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0,:]))
display(shap.summary_plot(shap_values, X_train, plot_type="bar"))
display(shap.summary_plot(shap_values, X_train))
#################################################################

#  4.4  Recursive  Feature Elimination   (RFE)

#################################################################



'''

from sklearn.feature_selection import RFE



print("doing  recursive feature elimination...")

rfe = RFE(my_model, n_features_to_select=1)





print("fitting...")

rfe.fit(X_train, y)



from operator import itemgetter



# you have to pass the list 'features' with all the features which you trained the model with

for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):   

    print(x, y)     # this line will print the rank x and the name of the feature y

'''



###########################################################################################################################



#  top 15  of the one run on my PC



#1 OverallQual

#2 GrLivArea

#3 TotalBsmtSF

#4 YearBuilt

#5 GarageCars

#6 1stFlrSF

#7 BsmtFinSF1

#8 GarageArea

#9 LotArea

#10 OverallCond

#11 YearRemodAdd

#12 CentralAir_Y

#13 LotFrontage

#14 BsmtUnfSF

#15 2ndFlrSF