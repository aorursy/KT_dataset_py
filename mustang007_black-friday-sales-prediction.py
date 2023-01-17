# Importing Important Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,  KBinsDiscretizer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
Train = pd.read_csv('/kaggle/input/black-friday-sales/Black_Friday_sale/train.csv')
Test = pd.read_csv('/kaggle/input/black-friday-sales/Black_Friday_sale/test.csv')
Sub = pd.read_csv('/kaggle/input/black-friday-sales/Black_Friday_sale/sample_submission_V9Inaty.csv')
Train.head(10)
Test.head(10)
sns.pairplot(Train)
Train.info()
# Numerical Columns
num = []
a = Train.describe()
for i in a:
    i = i
    num.append(i)
print('List of Numerical Columns \n',num)
# Categorical Columns
cat = []
a = Train.describe(include='O')
for i in a:
    i = i
    cat.append(i)
print('List of Categorical Columns \n',cat)
Train.skew()
for i in Train.columns:
    a = Train[i].isnull().sum()
    if a > 0:
        print('This {} column has '.format(i),a,' null values')
        
# It has lot of Null Values so dropping them is not a Solution , so lets do first basic feature engineering
# Train['Product_Category_2'].dropna()
# Train['Product_Category_3'].dropna()
gender_dict = {'F':0, 'M':1}
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
city_dict = {'A':0, 'B':1, 'C':2}
stay_dict = {'0':0, '1':1, '2':2, '3':3, '4+':4}
 
Train["Gender"] = Train["Gender"].apply(lambda x: gender_dict[x])
Test["Gender"] = Test["Gender"].apply(lambda x: gender_dict[x])
 
Train["Age"] = Train["Age"].apply(lambda x: age_dict[x])
Test["Age"] = Test["Age"].apply(lambda x: age_dict[x])
 
Train["City_Category"] = Train["City_Category"].apply(lambda x: city_dict[x])
Test["City_Category"] = Test["City_Category"].apply(lambda x: city_dict[x])
 
Train["Stay_In_Current_City_Years"] = Train["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])
Test["Stay_In_Current_City_Years"] = Test["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])

columns_list = ["Product_ID"]
for var in columns_list:
    lb = LabelEncoder()
    full_var_data = pd.concat((Train[var],Test[var]),axis=0).astype('str')
    temp = lb.fit_transform(np.array(full_var_data))
    Train[var] = lb.transform(np.array( Train[var] ).astype('str'))
    Test[var] = lb.transform(np.array( Test[var] ).astype('str'))
# Data after basic Feature enigneering
Train.head(10)
X = Train.drop(columns='Purchase')
y = Train['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# evaluate each strategy on the dataset
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
results = list()
strategies = ['mean', 'median', 'most_frequent', 'constant']
for s in strategies:
# create the modelin g pipeline
    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestRegressor())])
    # evaluate the model
    pipeline.fit(X_train, y_train)
    scores = pipeline.score(X_test, y_test)
    # store results
    results.append(scores)
    print('>%s %.3f' % (s, np.mean(scores)))
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
pipeline = Pipeline(steps=[('i', imp), ('m', RandomForestRegressor())])
pipeline.fit(X_train, y_train)
scores = pipeline.score(X_test, y_test)
print('%.3f' % (scores))
a = X_train.fillna(-999)
b = X_test.fillna(-999)
pipeline = Pipeline(steps=[('m', RandomForestRegressor())])
pipeline.fit(a, y_train)
X_test.fillna(-999)
scores = pipeline.score(b, y_test)
print('%.3f' % (scores))
dtr = RandomForestRegressor()
dtr.fit(a,y_train)
y_pred = dtr.predict(b)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
dtr = xgb.XGBRegressor()
dtr.fit(a,y_train)
y_pred = dtr.predict(b)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
feature_important = dtr.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(X_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
Train['User_ID']
Train['Product_Category_2'] = Train['Product_Category_2'].fillna(-999)
Train['Product_Category_3'] = Train['Product_Category_3'].fillna(-999)
Test['Product_Category_2'] = Test['Product_Category_2'].fillna(-999)
Test['Product_Category_3'] = Test['Product_Category_3'].fillna(-999)
Train["User_ID_MeanPrice"]  = Train.groupby(['User_ID'])['Purchase'].transform('mean')
Train
userID_mean_dict = Train.groupby(['User_ID'])['Purchase'].mean().to_dict()
# for i, j in userID_mean_dict.items():
#     Test["User_ID_MeanPrice"] = Test["User_ID_MeanPrice"].replace(i, j) #test['User_ID'].apply(lambda x:userID_mean_dict.get(x,0))

Train["Product_ID_MeanPrice"]  = Train.groupby(['Product_ID'])['Purchase'].transform('mean')
Train
# userID_mean_dict = Train.groupby(['Product_ID'])['Purchase'].mean().to_dict()
# Test['Product_ID_MeanPrice'] = Test['Product_ID']
# for i, j in userID_mean_dict.items():
#     Test["Product_ID_MeanPrice"] = Test["Product_ID_MeanPrice"].replace(i, j) #test['User_ID'].apply(lambda x:userID_mean_dict.get(x,0))

X = Train.drop(columns='Purchase')
y = Train['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr = RandomForestRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Earlier RMSE Was RMSE Error: 2756.2351610303654
#                    R2 Score: 0.6985656174193418
feature_important = dtr.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(X_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances

Train["User_ID_MinPrice"] = Train.groupby(['User_ID'])['Purchase'].transform('min')
Train["User_ID_MaxPrice"] = Train.groupby(['User_ID'])['Purchase'].transform('max')
Train["Product_ID_MinPrice"] = Train.groupby(['Product_ID'])['Purchase'].transform('min')
Train["Product_ID_MaxPrice"] = Train.groupby(['Product_ID'])['Purchase'].transform('max')
X = Train.drop(columns='Purchase')
y = Train['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr = RandomForestRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Earlier RMSE Was RMSE Error: 2535.0195136408065
#                           R2 Score: 0.7450101673389087
feature_important = dtr.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(X_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
Train["Product_Cat1_MaxPrice"] = Train.groupby(['Product_Category_1'])['Purchase'].transform('max')
Train["Product_Cat1_MeanPrice"] = Train.groupby(['Product_Category_1'])['Purchase'].transform('mean')
Train["Age_Count"] = Train.groupby(['Age'])['Age'].transform('count')
Train["Occupation_Count"] = Train.groupby(['Occupation'])['Occupation'].transform('count')
Train["Product_Category_1_Count"] = Train.groupby(['Product_Category_1'])['Product_Category_1'].transform('count')
Train["Product_Category_2_Count"] = Train.groupby(['Product_Category_2'])['Product_Category_2'].transform('count')
Train["Product_Category_3_Count"] = Train.groupby(['Product_Category_3'])['Product_Category_3'].transform('count')
Train["User_ID_Count"] = Train.groupby(['User_ID'])['User_ID'].transform('count')
Train["Product_ID_Count"] = Train.groupby(['Product_ID'])['Product_ID'].transform('count')
Train["Occupation__Mean_Price"]  = Train.groupby(['Occupation'])['Purchase'].transform('mean')
Train
X = Train.drop(columns='Purchase')
y = Train['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr = RandomForestRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Earlier RMSE Was RMSE Error: 2535.0195136408065
#                           R2 Score: 0.7450101673389087

dtr =  xgb.XGBRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)   
feature_important = dtr.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(X_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
Train["Stay_In_Current_City_Years_Mean_price"]  = Train.groupby(['Stay_In_Current_City_Years'])['Purchase'].transform('mean')

# Another Method of Advance Feature engineering
# Bin the ID column and add as feature
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
Train["User_ID_ID_Bin"] = est.fit_transform(np.reshape(Train["User_ID"].values, (-1,1)))
Train["Product_ID_ID_Bin"] = est.fit_transform(np.reshape(Train["Product_ID"].values, (-1,1)))
Train
X = Train.drop(columns='Purchase')
y = Train['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr = RandomForestRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Earlier result
    # RMSE Error: 2480.985464001923
    # R2 Score: 0.7557645559747941

dtr =  xgb.XGBRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)   
Train["User_ID_Product_mean"] = Train.groupby(['User_ID'])['Product_ID_MeanPrice'].transform('mean')
Train["User_ID_Product_max"] = Train.groupby(['User_ID'])['Product_ID_MeanPrice'].transform('max')
feature_important = dtr.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100. / total for value in feature_important]
new = np.round(new,2)
keys = list(X_train.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
a = Train.drop(columns=['User_ID_ID_Bin','Product_ID_ID_Bin','Product_ID_MinPrice','Product_Category_3_Count','Product_Category_2'])
cor = a.corr()
plt.figure(figsize=(20,15))
sns.heatmap(cor<-0.9, annot=True)

X = a.drop(columns='Purchase')
y = a['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr =  xgb.XGBRegressor()
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)  

#RMSE Error: 2409.370507700712
#R2 Score: 0.7696610247065712

# RMSE Error: 2407.6520541081927
# R2 Score: 0.7699894803583995
from xgboost import plot_importance

plot_importance(dtr)
test_model = xgb.XGBRegressor(
            eta = 0.03,
            n_estimators = 1500 
)
#model.fit(X_train, y_train)
test_model.fit(X_train, y_train, eval_metric='rmse', 
          eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=100)
dtr =  xgb.XGBRegressor(
                eta = 0.03,
                n_estimators=400,
                max_depth=8,
                min_child_weight=0.9
)
dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2) 

# RMSE Error: 2407.9961756684083
# R2 Score: 0.7699237256450062
# RMSE Error: 2406.639751382179
# R2 Score: 0.7701828565786376
# 2393
from sklearn.model_selection import GridSearchCV
test_model = xgb.XGBRegressor(
                        eta = 0.03,
#                         max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        seed=27,
                        n_estimators = 600
)

param_grids = {
           'max_depth':[3,5,7,9,10]
#  'min_child_weight':[1,2,3,5]
}

grid = GridSearchCV(estimator=test_model,
            
                    param_grid=param_grids, n_jobs=-1)

grid.fit(X_train, y_train)
grid.best_params_
test_model = xgb.XGBRegressor(
            max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                     eta = 0.03,
            n_estimators = 1000 ,
                  subsample=0.8,
                        colsample_bytree=0.8,
                        seed=27
)
#model.fit(X_train, y_train)
test_model.fit(X_train, y_train, eval_metric='rmse', 
          eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=100)
test_model = xgb.XGBRegressor(
            max_depth=8,
                        min_child_weight=1,
                        gamma=5,
                     eta = 0.3,
            n_estimators = 1000 ,
                  subsample=0.8,
                        colsample_bytree=0.8,
                        seed=27
)
#model.fit(X_train, y_train)
test_model.fit(X_train, y_train, eval_metric='rmse', 
          eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=100)
test_model = xgb.XGBRegressor(
            max_depth=6,
                        min_child_weight=1,
                        gamma=0,
                     eta = 0.3,
            n_estimators = 1000 ,
                  subsample=0.8,
                        colsample_bytree=0.8,
                        seed=27
)
#model.fit(X_train, y_train)
test_model.fit(X_train, y_train, eval_metric='rmse', 
          eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=100)