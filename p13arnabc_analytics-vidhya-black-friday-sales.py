#Necessary imports for viewing data
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
#Loading test and train data set
train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')
gender_dict = {'F':0, 'M':1}
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
city_dict = {'A':0, 'B':1, 'C':2}
stay_dict = {'0':0, '1':1, '2':2, '3':3, '4+':4}
train_df["Gender"] = train_df["Gender"].apply(lambda x: gender_dict[x])
train_df["Age"] = train_df["Age"].apply(lambda x: age_dict[x])
train_df["City_Category"] = train_df["City_Category"].apply(lambda x: city_dict[x])
train_df["Stay_In_Current_City_Years"] = train_df["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])
def getCountofVar(dataset_df, var_name):
    var_name_grouped = dataset_df.groupby(var_name)
    count_dict = {}
    for name, group in var_name_grouped:
        count_dict[name] = group.shape[0]
    count_list = []
    for index, row in dataset_df.iterrows():
        name = row[var_name]
        count_list.append(count_dict.get(name, 0))
    return count_list
train_df["User_ID_Count"] = getCountofVar(train_df,"User_ID")
train_df["Product_ID_Count"] = getCountofVar(train_df,"Product_ID")
train_df["Gender_Count"] = getCountofVar(train_df,"Gender")
train_df["Age_Count"] = getCountofVar(train_df,"Age")
train_df["Occupation_Count"] = getCountofVar(train_df,"Occupation")
train_df["City_Count"] = getCountofVar(train_df,"City_Category")
train_df["Stay_Count"] = getCountofVar(train_df,"Stay_In_Current_City_Years")
train_df["Marital_Status_Count"] = getCountofVar(train_df,"Marital_Status")
train_df["PC1_Count"] = getCountofVar(train_df,"Product_Category_1")
train_df["PC2_Count"] = getCountofVar(train_df,"Product_Category_2")
train_df["PC3_Count"] = getCountofVar(train_df,"Product_Category_3")
train_df.fillna(0, inplace=True)
def getPurchaseStats(target_df,compute_df, feature_name):
    feature_grouped = compute_df.groupby(feature_name)
    min_dict = {}
    max_dict = {}
    mean_dict = {}
    twentyfive_dict = {}
    fifty_dict = {}
    seventyfive_dict = {}
    for name, group in feature_grouped:
        min_dict[name] = min(np.array(group["Purchase"]))
        max_dict[name] = max(np.array(group["Purchase"]))
        mean_dict[name] = np.mean(np.array(group["Purchase"]))
        twentyfive_dict[name] = np.percentile(np.array(group["Purchase"]),25)
        fifty_dict[name] = np.percentile(np.array(group["Purchase"]),50)
        seventyfive_dict[name] = np.percentile(np.array(group["Purchase"]),75)
    min_list = []
    max_list = []
    mean_list = []
    twentyfive_list = []
    fifty_list = []
    seventyfive_list = []
    for index, row in target_df.iterrows():
        name = row[feature_name]
        min_list.append(min_dict.get(name,0))
        max_list.append(max_dict.get(name,0))
        mean_list.append(mean_dict.get(name,0))
        twentyfive_list.append( twentyfive_dict.get(name,0))
        fifty_list.append( fifty_dict.get(name,0))
        seventyfive_list.append( seventyfive_dict.get(name,0))
    return min_list, max_list, mean_list, twentyfive_list, fifty_list, seventyfive_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "User_ID")
train_df["User_ID_Min_Purchase"] = min_price_list
train_df["User_ID_Max_Purchase"] = max_price_list
train_df["User_ID_Mean_Purchase"] = mean_price_list
train_df["User_ID_25Per_Purchase"] = twentyfive_price_list
train_df["User_ID_50Per_Purchase"] = fifty_price_list
train_df["User_ID_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Product_ID")
train_df["Product_ID_Min_Purchase"] = min_price_list
train_df["Product_ID_Max_Purchase"] = max_price_list
train_df["Product_ID_Mean_Purchase"] = mean_price_list
train_df["Product_ID_25Per_Purchase"] = twentyfive_price_list
train_df["Product_ID_50Per_Purchase"] = fifty_price_list
train_df["Product_ID_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df, train_df, "Gender")
train_df["Gender_Min_Purchase"] = min_price_list
train_df["Gender_Max_Purchase"] = max_price_list
train_df["Gender_Mean_Purchase"] = mean_price_list
train_df["Gender_25Per_Purchase"] = twentyfive_price_list
train_df["Gender_50Per_Purchase"] = fifty_price_list
train_df["Gender_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Age")
train_df["Age_Min_Purchase"] = min_price_list
train_df["Age_Max_Purchase"] = max_price_list
train_df["Age_Mean_Purchase"] = mean_price_list
train_df["Age_25Per_Purchase"] = twentyfive_price_list
train_df["Age_50Per_Purchase"] = fifty_price_list
train_df["Age_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Occupation")
train_df["Occupation_Min_Purchase"] = min_price_list
train_df["Occupation_Max_Purchase"] = max_price_list
train_df["Occupation_Mean_Purchase"] = mean_price_list
train_df["Occupation_25Per_Purchase"] = twentyfive_price_list
train_df["Occupation_50Per_Purchase"] = fifty_price_list
train_df["Occupation_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "City_Category")
train_df["City_Min_Purchase"] = min_price_list
train_df["City_Max_Purchase"] = max_price_list
train_df["City_Mean_Purchase"] = mean_price_list
train_df["City_25Per_Purchase"] = twentyfive_price_list
train_df["City_50Per_Purchase"] = fifty_price_list
train_df["City_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Stay_In_Current_City_Years")
train_df["Stay_Min_Purchase"] = min_price_list
train_df["Stay_Max_Purchase"] = max_price_list
train_df["Stay_Mean_Purchase"] = mean_price_list
train_df["Stay_25Per_Purchase"] = twentyfive_price_list
train_df["Stay_50Per_Purchase"] = fifty_price_list
train_df["Stay_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Marital_Status")
train_df["Marital_Min_Purchase"] = min_price_list
train_df["Marital_Max_Purchase"] = max_price_list
train_df["Marital_Mean_Purchase"] = mean_price_list
train_df["Marital_25Per_Purchase"] = twentyfive_price_list
train_df["Marital_50Per_Purchase"] = fifty_price_list
train_df["Marital_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Product_Category_1")
train_df["PC1_Min_Purchase"] = min_price_list
train_df["PC1_Max_Purchase"] = max_price_list
train_df["PC1_Mean_Purchase"] = mean_price_list
train_df["PC1_25Per_Purchase"] = twentyfive_price_list
train_df["PC1_50Per_Purchase"] = fifty_price_list
train_df["PC1_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Product_Category_2")
train_df["PC2_Min_Purchase"] = min_price_list
train_df["PC2_Max_Purchase"] = max_price_list
train_df["PC2_Mean_Purchase"] = mean_price_list
train_df["PC2_25Per_Purchase"] = twentyfive_price_list
train_df["PC2_50Per_Purchase"] = fifty_price_list
train_df["PC2_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(train_df,train_df, "Product_Category_3")
train_df["PC3_Min_Purchase"] = min_price_list
train_df["PC3_Max_Purchase"] = max_price_list
train_df["PC3_Mean_Purchase"] = mean_price_list
train_df["PC3_25Per_Purchase"] = twentyfive_price_list
train_df["PC3_50Per_Purchase"] = fifty_price_list
train_df["PC3_75Per_Purchase"] = seventyfive_price_list
#define X
X = train_df.drop(columns=['User_ID','Product_ID','Purchase'],axis=1)
#define y
y = train_df["Purchase"]
#baseline model
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=300, max_depth = 10, learning_rate = 0.05, objective = "reg:squarederror", min_child_weight  = 10)
xgb.fit(X, y)
#we will take features that have importance more than 0.002
from sklearn.feature_selection import SelectFromModel
selection = SelectFromModel(xgb, threshold=0.002, prefit=True)
#define X_train after selecting the top few features
train_selection = selection.transform(X_train)
xgb_selected_features = XGBRegressor(n_estimators=300, max_depth = 10, learning_rate = 0.05, objective = "reg:squarederror", min_child_weight  = 10)
# train model
xgb_selected_features.fit(train_selection, y_train)
#Load the test data
test_df = pd.read_csv(r'test.csv')
#preprocessing of categorical features
test_df["Gender"] = test_df["Gender"].apply(lambda x: gender_dict[x])
test_df["Age"] = test_df["Age"].apply(lambda x: age_dict[x])
test_df["City_Category"] = test_df["City_Category"].apply(lambda x: city_dict[x])
test_df["Stay_In_Current_City_Years"] = test_df["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])
test_df["User_ID_Count"] = getCountofVar(test_df,"User_ID")
test_df["Product_ID_Count"] = getCountofVar(test_df,"Product_ID")
test_df["Gender_Count"] = getCountofVar(test_df,"Gender")
test_df["Age_Count"] = getCountofVar(test_df,"Age")
test_df["Occupation_Count"] = getCountofVar(test_df,"Occupation")
test_df["City_Count"] = getCountofVar(test_df,"City_Category")
test_df["Stay_Count"] = getCountofVar(test_df,"Stay_In_Current_City_Years")
test_df["Marital_Status_Count"] = getCountofVar(test_df,"Marital_Status")
test_df["PC1_Count"] = getCountofVar(test_df,"Product_Category_1")
test_df["PC2_Count"] = getCountofVar(test_df,"Product_Category_2")
test_df["PC3_Count"] = getCountofVar(test_df,"Product_Category_3")
#impute blank values
test_df.fillna(0, inplace=True)
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "User_ID")
test_df["User_ID_Min_Purchase"] = min_price_list
test_df["User_ID_Max_Purchase"] = max_price_list
test_df["User_ID_Mean_Purchase"] = mean_price_list
test_df["User_ID_25Per_Purchase"] = twentyfive_price_list
test_df["User_ID_50Per_Purchase"] = fifty_price_list
test_df["User_ID_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Product_ID")
test_df["Product_ID_Min_Purchase"] = min_price_list
test_df["Product_ID_Max_Purchase"] = max_price_list
test_df["Product_ID_Mean_Purchase"] = mean_price_list
test_df["Product_ID_25Per_Purchase"] = twentyfive_price_list
test_df["Product_ID_50Per_Purchase"] = fifty_price_list
test_df["Product_ID_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df, train_df, "Gender")
test_df["Gender_Min_Purchase"] = min_price_list
test_df["Gender_Max_Purchase"] = max_price_list
test_df["Gender_Mean_Purchase"] = mean_price_list
test_df["Gender_25Per_Purchase"] = twentyfive_price_list
test_df["Gender_50Per_Purchase"] = fifty_price_list
test_df["Gender_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Age")
test_df["Age_Min_Purchase"] = min_price_list
test_df["Age_Max_Purchase"] = max_price_list
test_df["Age_Mean_Purchase"] = mean_price_list
test_df["Age_25Per_Purchase"] = twentyfive_price_list
test_df["Age_50Per_Purchase"] = fifty_price_list
test_df["Age_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Occupation")
test_df["Occupation_Min_Purchase"] = min_price_list
test_df["Occupation_Max_Purchase"] = max_price_list
test_df["Occupation_Mean_Purchase"] = mean_price_list
test_df["Occupation_25Per_Purchase"] = twentyfive_price_list
test_df["Occupation_50Per_Purchase"] = fifty_price_list
test_df["Occupation_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "City_Category")
test_df["City_Min_Purchase"] = min_price_list
test_df["City_Max_Purchase"] = max_price_list
test_df["City_Mean_Purchase"] = mean_price_list
test_df["City_25Per_Purchase"] = twentyfive_price_list
test_df["City_50Per_Purchase"] = fifty_price_list
test_df["City_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Stay_In_Current_City_Years")
test_df["Stay_Min_Purchase"] = min_price_list
test_df["Stay_Max_Purchase"] = max_price_list
test_df["Stay_Mean_Purchase"] = mean_price_list
test_df["Stay_25Per_Purchase"] = twentyfive_price_list
test_df["Stay_50Per_Purchase"] = fifty_price_list
test_df["Stay_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Marital_Status")
test_df["Marital_Min_Purchase"] = min_price_list
test_df["Marital_Max_Purchase"] = max_price_list
test_df["Marital_Mean_Purchase"] = mean_price_list
test_df["Marital_25Per_Purchase"] = twentyfive_price_list
test_df["Marital_50Per_Purchase"] = fifty_price_list
test_df["Marital_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Product_Category_1")
test_df["PC1_Min_Purchase"] = min_price_list
test_df["PC1_Max_Purchase"] = max_price_list
test_df["PC1_Mean_Purchase"] = mean_price_list
test_df["PC1_25Per_Purchase"] = twentyfive_price_list
test_df["PC1_50Per_Purchase"] = fifty_price_list
test_df["PC1_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Product_Category_2")
test_df["PC2_Min_Purchase"] = min_price_list
test_df["PC2_Max_Purchase"] = max_price_list
test_df["PC2_Mean_Purchase"] = mean_price_list
test_df["PC2_25Per_Purchase"] = twentyfive_price_list
test_df["PC2_50Per_Purchase"] = fifty_price_list
test_df["PC2_75Per_Purchase"] = seventyfive_price_list
min_price_list, max_price_list, mean_price_list, twentyfive_price_list,fifty_price_list, seventyfive_price_list = getPurchaseStats(test_df,train_df, "Product_Category_3")
test_df["PC3_Min_Purchase"] = min_price_list
test_df["PC3_Max_Purchase"] = max_price_list
test_df["PC3_Mean_Purchase"] = mean_price_list
test_df["PC3_25Per_Purchase"] = twentyfive_price_list
test_df["PC3_50Per_Purchase"] = fifty_price_list
test_df["PC3_75Per_Purchase"] = seventyfive_price_list
#store the data in a file
test_df.to_csv(r'test_full_feature.csv',index=False)
#test_df = pd.read_csv(r'test_full_feature.csv')
#define test data
test_data = test_df.drop(columns=['User_ID','Product_ID'],axis=1)
#test data based on the features selected for the model
test_data_selection = selection.transform(test_data)
#predict the test data
test_df["Purchase"] = xgb_selected_features.predict(test_data_selection)
IDcol = ['User_ID','Product_ID']
IDcol.append("Purchase")
submission = pd.DataFrame({ x: test_df[x] for x in IDcol})
submission.to_csv(r"submission_xgb4.csv", index=False)