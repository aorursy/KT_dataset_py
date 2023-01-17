import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
df = pd.read_csv("train.csv")
df.head()
df1 = pd.read_csv("test.csv")
df1.head()
# Missing values treatment #

df['condition'] = df['condition'].fillna(3)
df.isnull().sum()
df["listing_date"] = (df["listing_date"].str.split(" ", n = 1, expand = True))[0] 
df["listing_date"].head()
df["issue_date"] = (df["issue_date"].str.split(" ", n = 1, expand = True))[0] 
df["issue_date"].head()
df['listing_date'] =pd.to_datetime(df['listing_date'], format='%Y-%m-%d')
df['listing_date'].head()
df['issue_date'] =pd.to_datetime(df['issue_date'], format='%Y-%m-%d')
df['issue_date'].head()
df['No_days'] = (df['listing_date']-df['issue_date']).dt.days  # This will convert no. of days into integer format $#
df['No_days'].head()
# there is no requirement of issues date and listing date, as well the pet_id so will drop this columns #

df = df.drop(['pet_id','issue_date','listing_date'],axis=1)
df.head()
df.head()

# Dropping the length and height as they do not have a significanr relationship with the yarget variables

df = df.drop(['height(cm)','length(m)'],axis=1)
df.head()

df.info()

#Handling the categorical variable colour #
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x_cat = df['color_type']
x_cat.head()
X_cat = pd.DataFrame(le.fit_transform(x_cat))
X_cat.head()
# We need ro concart this new colour series to our original data set #

X = pd.concat([df,X_cat],axis=1)
X.head()
X.drop(['color_type'],axis=1,inplace=True)
X.head()
# Also we needd to drop the two target variables pet_catergory and breed_category #

X.rename(columns = {0:'color'}, inplace = True) 

X.head()
# We will drop both the target variables from this # Independent cariables #

x = X.drop(['breed_category','pet_category'],axis=1)
x.head()
# Identifying yhe categort variables, as they need to be mentioned separately for applying the catboosting algorithm #

cat_features_index = [0,4]   # Condition & Color are categorical variables 
x['condition'] = x['condition'].astype('int64')
x.info()
x.head()
# Target variables : Breed_categpru & Pet categpory 
# Differentating yhe independnetn & dependent variables #
y = X['breed_category']  # 1. target variable : Breed_category
y.head()
y1 = X['pet_category']  # 2. target variable : Pet_category
y1.head()
# Applying the same above prepreccing steps for the test data set # # test data set - df1

# Removing the time portion for issue data and listing dates 

df1["listing_date"] = (df1["listing_date"].str.split(" ", n = 1, expand = True))[0] 
df1["listing_date"].head()
df1["issue_date"] = (df1["issue_date"].str.split(" ", n = 1, expand = True))[0] 
df1["issue_date"].head()
df1['listing_date'] =pd.to_datetime(df1['listing_date'], format='%d-%m-%Y')
df1['listing_date'].head()
df1['issue_date'] =pd.to_datetime(df1['issue_date'], format='%d-%m-%Y')
df1['issue_date'].head()
# Identifuy the no. of days between listuing date and issue date #

df1['No_days'] = (df1['listing_date']-df1['issue_date']).dt.days
df1['No_days'].head()
df1.head()
# Replacing the null values in the condition with a value of 3 

df1['condition'] = df1['condition'].fillna(3)
df1.isnull().sum()
df2 = df1.drop(['pet_id','issue_date','listing_date','length(m)','height(cm)'],axis=1)

df2.head()
x_cat1 = df2['color_type']
x_cat1.head()
# Converting the color_type into labels with help of label encoder #


cat = pd.Series(le.transform(x_cat1))
X1 = pd.concat([df2,cat],axis=1)
X1.head()
X1 = X1.drop(['color_type'],axis=1)
X1.head()

X1.rename(columns = {0:'color'}, inplace = True) 

X1.head()
X1['condition'] = X1['condition'].astype('int64')
X1.info()
df1.head()
X1.info()
# Feature scaling : converting into standard scaler form #

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Lets build the logistic regression model #

lm = LogisticRegression()
mo = lm.fit(x,y)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# classification report and confusion matrix #
y_pred = mo.predict(x)

con = confusion_matrix(y_pred,y)
print(con)


acc = accuracy_score(y_pred,y)
print(acc)
# Random Search Algorithm #
params_gr = {'n_estimators':[103,104,105,106],'min_samples_leaf':[18,19,20],'max_depth':[9,10,11],'criterion':['entropy'],'max_leaf_nodes':[16,17,18,19],'max_features':[4]}
from sklearn.ensemble import RandomForestClassifier
class_weight = dict({0:1,1:2,1:100})
r_wcl= RandomForestClassifier(criterion='entropy', max_depth=9, max_features=4, max_leaf_nodes=19, min_samples_leaf=18)
score_wo = cross_val_score(r_wcl,x,y,cv=3)
score_wo.mean()
from sklearn.model_selection import cross_val_score
r_cl= RandomForestClassifier(class_weight=class_weight,n_estimators=103,criterion='entropy', max_depth=9, max_features=4, max_leaf_nodes=19, min_samples_leaf=18)
score_wi = cross_val_score(r_cl,x,y,cv=3)
score_wi.mean()
# Adding the weights, as the problem seems to little unbalanced #
# Classes Present for Breed category : (0 : 9000 , 1: 8357 , 2: 1477), giving importance to class 2

r= RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

r= RandomForestClassifier()
grid = GridSearchCV(r,params_gr,cv=7)
mo_rf_gr = grid.fit(x,y)
print(mo_rf_gr.best_estimator_)

print(mo_rf_gr.best_params_)

print(mo_rf_gr.best_score_)
#{'criterion': 'entropy', 'max_depth': 9, 'max_features': 4, 'max_leaf_nodes': 19, 'min_samples_leaf': 18, 'n_estimators': 105}
# Model for breed category (name :breed_rf1.pkl ) , Score : 0.9071356242844848
import pickle
# Saving this random forest model for breed category #

pickle.dump(mo_rf_gr,open('breed_rf1.pkl','wb'))
p = pickle.load(open('breed_rf.pkl','rb'))

# Identifying he values of y_test for model#

y_test_br = p.predict(X1)

y_test_br = pd.Series(y_test_br,name='breed_category')
y_test_br.head()
# applying the XG Boosting for the Pet Category #
! pip install xgboost
import xgboost
from xgboost import XGBClassifier
classifier = XGBClassifier()

params_xg={
 "learning_rate"    : [0.04,0.05, 0.10, 0.13,0.15, 0.17,0.20] ,
 "max_depth"        : [ 5,6,7,8,10],
 "min_child_weight" : [0.28,0.3,0.32,0.35,0.4,0.45],
 "gamma"            : [ 0.3,0.4,0.45,0.5,0.55],
 "colsample_bytree" : [ 0.5,0.6,0.65,0.7,0.75,0.8 ]
    
}

# learning rate : 0.05, colsample_bytree:0.7, min_child_weight:0.5
random_search=RandomizedSearchCV(classifier,param_distributions=params_xg,n_iter=100,n_jobs=-1,cv=7,verbose=3)
model1 = random_search.fit(x,y1)
print(model1.best_score_)
print(model1.best_params_)
print(model1.best_estimator_)
# Model : pet_xgboost : 90.54, pet_xgboost_n : 90.60, pet_xgboost_n1 : 90.61
# Model : pet_xgboost_n1 : 90.61, cv=7
# Parameters : {'min_child_weight': 0.4, 'max_depth': 8, 'learning_rate': 0.15, 'gamma': 0.45, 'colsample_bytree': 0.65}

# Saving the model of Xgboosting # - Pet Categpry #

pickle.dump(model1,open('pet_xgboost_n1.pkl','wb'))
# loading the model #

q = pickle.load(open('pet_xgboost_n1.pkl','rb'))
y_pet_xg = q.predict(X1)

y_pet_xg= pd.Series(y_pet_xg,name='pet_category')
y_pet_xg.head()
df_rf_xg= pd.concat([df1['pet_id'],y_test_br,y_pet_xg],axis=1) 
df_rf_xg.head()
# Saving the output in the CSV file # 
# Breed - breed_rf (90.75)
#Pet - pet_xgboost_n1 (90.61)
# Score :90.59
df_rf_xg.to_csv('submission_37.csv', index=False) 
# Saving the output in the CSV file # 
# Breed - breed_rf1 (90.71)
#Pet - pet_xgboost_n1 (90.61)
# Score :90.63
df_rf_xg.to_csv('submission_36.csv', index=False) 
# Saving the output in the CSV file #

df_rf_xg.to_csv('submission_35.csv', index=False) 
df_rf_xg.to_csv('submission_31.csv', index=False)
# Saving the output in the CSV file #

# Best Model with an overall score : 90.82

df_rf_xg.to_csv('submission_27.csv', index=False)   
! pip install catboost
# Starting for Cat Boosting #
from catboost import Pool, CatBoostClassifier 
model = CatBoostClassifier(iterations=10,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass')
train_dataset = Pool(data=x,
                     label=y,
                     cat_features=cat_features_index)
model.fit(train_dataset)
# Identifying he values of y_test for breed_category#

y_cat_br = model.predict(X1)
y_cat_br
y_cat_br= pd.DataFrame(y_cat_br)
y_cat_br.head()

y_cat_br.rename(columns = {0:'breed_category'}, inplace = True) 

y_cat_br.head()
train_dataset1 = Pool(data=x,
                     label=y1,
                     cat_features=cat_features_index)
model1 = CatBoostClassifier(iterations=10,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass')
model1.fit(train_dataset1)
# Identifying he values of y_test for pet_category#

y_cat_pet = model1.predict(X1)
y_cat_pet= pd.DataFrame(y_cat_pet)
y_cat_pet.head()
y_cat_pet.rename(columns = {0:'pet_category'}, inplace = True) 

y_cat_pet.head()
# Concatenating the output for breed and pet category #

df_cat= pd.concat([df1['pet_id'],y_cat_br['breed_category'],y_cat_pet['pet_category']],axis=1) 
df_cat.head()
df_cat.to_csv('submission_28.csv', index=False)
# Applying the Hyper parameter tuning in catboost Algorithm #
from sklearn.model_selection import GridSearchCV
dataset = Pool(data=x,
                     label=y,
                     cat_features=cat_features_index)
cb1 = CatBoostClassifier(loss_function='MultiClass')
params = {'depth': [7, 10],
          'learning_rate' : [0.03, 0.1],
         'l2_leaf_reg': [1,4]}
cb_model = cb1.grid_search(params,dataset, cv=2, shuffle=True, search_by_train_test_split=True, verbose=2)


y_test = cb1.predict_proba(X1)
y_test
y_test_br = cb1.predict(X1)
y_test_br
y_test_br= pd.DataFrame(y_test_br)
y_test_br.head()
y_test_br.rename(columns = {0:'breed_category'}, inplace = True) 

y_test_br.head()
cb2 = CatBoostClassifier(loss_function='MultiClass')
cb2.best_score_
cb2.classes_
dataset1 = Pool(data=x,
                     label=y1,
                     cat_features=cat_features_index)
cb_model1 = cb2.grid_search(params,dataset1, cv=2, shuffle=True, search_by_train_test_split=True, verbose=2)

y_test_pet = cb2.predict(X1)
y_test_pet
y_test_pet = pd.DataFrame(y_test_pet)
y_test_pet.head()
y_test_pet.rename(columns={0:'pet_category'}, inplace=True)
y_test_pet.head()


# Concatenating the output for breed and pet category #

df_cat_1= pd.concat([df1['pet_id'],y_test_br['breed_category'],y_test_pet['pet_category']],axis=1) 
df_cat_1.head()
df_cat_1.to_csv('submission_29.csv', index=False)
#Applning a new model with randomized search #

#Parameters for tuning
ra_grid = {'learning_rate': [0.03,0.25,0.5,0.1,1],
        'depth': [4,6,8,10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}
randomized_search(grid,
                  X,
                  y=None,
                  cv=3,
                  n_iter=10,
                  partition_random_seed=0,
                  calc_cv_statistics=True, 
                  search_by_train_test_split=True,
                  refit=True, 
                  shuffle=True, 
                  stratified=None, 
                  train_size=0.8, 
                  verbose=True)
c = CatBoostClassifier(loss_function='MultiClass')
#Applyning a new model with randomized search #

randomized_breed = c.randomized_search(grid,
                  dataset,
                  cv=3,
                  n_iter=50,
                  partition_random_seed=0,
                  calc_cv_statistics=True, 
                  search_by_train_test_split=True,
                  refit=True, 
                  shuffle=True, 
                  stratified=None, 
                  verbose=True)
c1 = CatBoostClassifier(loss_function='MultiClass')
randomized_pet = c1.randomized_search(grid,
                  dataset1,
                  cv=3,
                  n_iter=50,
                  partition_random_seed=0,
                  calc_cv_statistics=True, 
                  search_by_train_test_split=True,
                  refit=True, 
                  shuffle=True, 
                  stratified=None, 
                  verbose=True)
y_te_br = c.predict(X1)
y_te_br
y_te_br = pd.DataFrame(y_te_br)
y_te_br.head()
y_te_br.rename(columns={0:'breed_category'},inplace=True)
y_te_br.head()
y_te_pet = c1.predict(X1)
y_te_pet
y_te_pet = pd.DataFrame(y_te_pet)
y_te_pet.head()
y_te_pet.rename(columns={0:'pet_category'},inplace=True)
y_te_pet.head()
df_c = pd.concat([df1['pet_id'],y_te_br['breed_category'],y_te_pet['pet_category']],axis=1)  
df_c.head()
df_c.to_csv('submission_30.csv', index=False)
# sacing yhe model to be used at later stages #

import pickle
!pip install catboost
from catboost import CatBoostClassifier
pickle.dump(c, open('breed_cat.pkl', 'wb'))
pickle.dump(c1, open('pet_cat.pkl', 'wb'))
m = pickle.load(open('pet_cat.pkl','rb'))
n = pickle.load(open('breed_cat.pkl','rb'))
# Breed_category

y_t = n.predict(X1)
y_t
y_t = pd.DataFrame(y_t)
y_t.head()
# Breed Category #

y_t.rename(columns={0:'breed_category'},inplace=True)
y_t.head()
# Pet Categry

y_t1 = m.predict(X1)
y_t1
y_t1
y_t1 = pd.DataFrame(y_t1)
y_t1.head()
# Pet Category #

y_t1.rename(columns={0:'pet_category'},inplace=True)
y_t1.head()
# Concatenating the results of per and breed #

df_m1 = pd.concat([df1['pet_id'],y_t['breed_category'],y_t1['pet_category']],axis=1)  
df_m1.head()
# Submission file #

df_m1.to_csv('submission_32.csv', index=False)  #score : 90.68
# Creating a combined model #
# Breed category - catboost Model (breed_cat)
# Pet Category - Xgboost Model (pet_xgboost)
# Breed Category # Predicted data is y_t
y_t.head()
# Pet Category # Predicted Data is  y_pet_xg
y_pet_xg.head()
# Concatenating this data frames #


df_m2 = pd.concat([df1['pet_id'],y_t,y_pet_xg],axis=1)  
df_m2.head()

df_m2.to_csv('submission_33.csv', index=False) #score : 90.71
# New Model #

# Creating a combined model #
# Breed category - Random Forest Model (breed_rf), y_test_br
# Pet Category - Cat Model (pet_cat), y_t1
# Concatenating this data frames #

df_m3 = pd.concat([df1['pet_id'],y_test_br,y_t1],axis=1)  
df_m3.head()
df_m3.to_csv('submission_34.csv', index=False) #score :90.57
# Best Model which gave an overall accuracy score of 90.82
# Breed Category - (Algo. : Random Forest with hyper parameter tuning) 
# Pet Category - (Algo. : Xgboosting with hyper parameter tuning)