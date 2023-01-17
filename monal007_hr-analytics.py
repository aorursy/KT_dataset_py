import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import missingno as msg

from sklearn.model_selection import RandomizedSearchCV , train_test_split , cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder , StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , ExtraTreesClassifier
import lightgbm as lgbm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
df = pd.read_csv(r"../input/dataset/train.csv")
df1 = pd.read_csv(r"../input/dataset/test.csv")
sub = pd.read_csv(r"../input/dataset/sub.csv")

df
#describe
df.describe()
#info
df.info()
#check nan
msg.matrix(df)
msg.matrix(df1)
#employee id

df["employee_id"]
print(df["employee_id"].unique())
print(df["employee_id"].nunique())

print(df1["employee_id"].unique())
print(df1["employee_id"].nunique())
df.drop(["employee_id"] ,axis =1 , inplace = True)
df1.drop(["employee_id"] ,axis =1 , inplace = True)

#department
df["department"]
sb.countplot(df["department"])
plt.xticks(rotation = 90)
plt.title("train")
plt.show()
sb.countplot(df1["department"])
plt.xticks(rotation = 90)
plt.title("test")
plt.show()
df.groupby('department')["is_promoted"].mean().sort_values(ascending = False)
dep = {
    "Technology":9 , "Procurement":8,"Analytics":7,"Operations":6,"Finance":5,"Sales & Marketing":4,"R&D":3,"HR":2,"Legal":1
}
df.loc[: , 'department'] = df.loc[: , 'department'].map(dep)
df1.loc[: , 'department'] = df1.loc[: , 'department'].map(dep)

plt.figure(figsize = (8,5))
sb.heatmap(df.corr() , annot =True , cmap = "coolwarm")
#region
df["region"]
print(df["region"].unique())
print(df["region"].nunique())

print(df.groupby('region')["is_promoted"].mean().sort_values(ascending =False))
print(df.groupby('region')["age"].mean().sort_values(ascending =False))
plt.figure(figsize = (10,10))
sb.countplot(df["region"] , hue = df["is_promoted"])
plt.xticks(rotation = 90)
plt.show()
print(df.groupby('region')["is_promoted"].mean().sort_values(ascending =False))
df["region"] = df["region"].replace("region_4" , 5)
df["region"] = df["region"].replace("region_17",5)
df["region"] = df["region"].replace("region_25",5)
df["region"] = df["region"].replace("region_28",5)
df["region"] = df["region"].replace("region_23",5)
df["region"] = df["region"].replace("region_22",5)
df["region"] = df["region"].replace("region_3",5)
df["region"] = df["region"].replace("region_7",4)
df["region"] = df["region"].replace("region_1",4)
df["region"] = df["region"].replace("region_30",4)
df["region"] = df["region"].replace("region_13",4)
df["region"] = df["region"].replace("region_8",4)
df["region"] = df["region"].replace("region_2",4)
df["region"] = df["region"].replace("region_15",4)
df["region"] = df["region"].replace("region_27",3)
df["region"] = df["region"].replace("region_10",3)
df["region"] = df["region"].replace("region_14",3)
df["region"] = df["region"].replace("region_16",3)
df["region"] = df["region"].replace("region_12",3)
df["region"] = df["region"].replace("region_26",3)
df["region"] = df["region"].replace("region_19",3)
df["region"] = df["region"].replace("region_20",2)
df["region"] = df["region"].replace("region_31",2)
df["region"] = df["region"].replace("region_11",2)
df["region"] = df["region"].replace("region_6",2)
df["region"] = df["region"].replace("region_5",2)
df["region"] = df["region"].replace("region_21",2)
df["region"] = df["region"].replace("region_29",2)
df["region"] = df["region"].replace("region_32",1)
df["region"] = df["region"].replace("region_33",1)
df["region"] = df["region"].replace("region_24",1)
df["region"] = df["region"].replace("region_18",1)
df["region"] = df["region"].replace("region_34",1)
df["region"] = df["region"].replace("region_9",1)



df1["region"] = df1["region"].replace("region_4" , 5)
df1["region"] = df1["region"].replace("region_17",5)
df1["region"] = df1["region"].replace("region_25",5)
df1["region"] = df1["region"].replace("region_28",5)
df1["region"] = df1["region"].replace("region_23",5)
df1["region"] = df1["region"].replace("region_22",5)
df1["region"] = df1["region"].replace("region_3",5)
df1["region"] = df1["region"].replace("region_7",4)
df1["region"] = df1["region"].replace("region_1",4)
df1["region"] = df1["region"].replace("region_30",4)
df1["region"] = df1["region"].replace("region_13",4)
df1["region"] = df1["region"].replace("region_8",4)
df1["region"] = df1["region"].replace("region_2",4)
df1["region"] = df1["region"].replace("region_15",4)
df1["region"] = df1["region"].replace("region_27",3)
df1["region"] = df1["region"].replace("region_10",3)
df1["region"] = df1["region"].replace("region_14",3)
df1["region"] = df1["region"].replace("region_16",3)
df1["region"] = df1["region"].replace("region_12",3)
df1["region"] = df1["region"].replace("region_26",3)
df1["region"] = df1["region"].replace("region_19",3)
df1["region"] = df1["region"].replace("region_20",2)
df1["region"] = df1["region"].replace("region_31",2)
df1["region"] = df1["region"].replace("region_11",2)
df1["region"] = df1["region"].replace("region_6",2)
df1["region"] = df1["region"].replace("region_5",2)
df1["region"] = df1["region"].replace("region_21",2)
df1["region"] = df1["region"].replace("region_29",2)
df1["region"] = df1["region"].replace("region_32",1)
df1["region"] = df1["region"].replace("region_33",1)
df1["region"] = df1["region"].replace("region_24",1)
df1["region"] = df1["region"].replace("region_18",1)
df1["region"] = df1["region"].replace("region_34",1)
df1["region"] = df1["region"].replace("region_9",1)



df

#bins (size= 10)
category = [1,2,3,4,5]
labels = ["low","medium","high","very high"]

df["select_region"] = pd.cut(df["region"] , bins = category , labels = labels)
#bins (size= 10)
category = [1,2,3,4,5]
labels = ["low","medium","high","very high"]

df1["select_region"] = pd.cut(df1["region"] , bins = category , labels = labels)
df
df["select_region"] = df["select_region"].replace("very high",4)
df["select_region"] = df["select_region"].replace("high",3)
df["select_region"] = df["select_region"].replace("medium",2)
df["select_region"] = df["select_region"].replace("low",1)

df1["select_region"] = df1["select_region"].replace("very high",4)
df1["select_region"] = df1["select_region"].replace("high",3)
df1["select_region"] = df1["select_region"].replace("medium",2)
df1["select_region"] = df1["select_region"].replace("low",1)

sb.countplot(df["select_region"])
df["select_region"].isna().sum()
df1["select_region"].isna().sum()
df["select_region"].fillna(df["select_region"].mode()[0] , inplace =True)
df1["select_region"].fillna(df1["select_region"].mode()[0] , inplace =True)

df1["select_region"].isna().sum()
df["select_region"].isna().sum()
df.drop(["region"] ,axis =1 , inplace =True)
df1.drop(["region"] ,axis =1 , inplace =True)

#education
df["education"]
print(df["education"].unique())
print(df["education"].nunique())

sb.countplot(df["education"])
df.groupby('education')["is_promoted"].mean().sort_values(ascending =False)
df.loc[: , "education"] = df.loc[: , "education"].map({"Master's & above":3 , "Bachelor's":2,"Below Secondary ":1})
df1.loc[: , "education"] = df1.loc[: , "education"].map({"Master's & above":3 , "Bachelor's":2,"Below Secondary ":1})

df["education"].fillna(df["education"].mode()[0] , inplace = True)
df1["education"].fillna(df1["education"].mode()[0] , inplace = True)

#gender
df["gender"]
print(df["gender"].unique())
print(df["gender"].nunique())

sb.countplot(df["gender"])
df.loc[: , "gender"] = df.loc[: , "gender"].map({"f":0 , "m":1})
df1.loc[: , "gender"] = df1.loc[: , "gender"].map({"f":0 , "m":1})

df.groupby('gender')["is_promoted"].mean()
#recruitment channel
df["recruitment_channel"]
print(df["recruitment_channel"].unique())
print(df["recruitment_channel"].nunique())

df.groupby('recruitment_channel')["is_promoted"].mean()
sb.countplot(df["recruitment_channel"] , hue = df["is_promoted"])
# since recruit has no relation in promotion -- drop
df.drop('recruitment_channel' , axis =1 , inplace =True)
df1.drop('recruitment_channel' , axis =1 , inplace =True)

#no of train
df["no_of_trainings"]
print(df["no_of_trainings"].unique())
print(df["no_of_trainings"].nunique())

print(df["no_of_trainings"].value_counts())
df["total_train_time"] = round(df["no_of_trainings"]*df["avg_training_score"]) // 60
df1["total_train_time"] = round(df1["no_of_trainings"]*df1["avg_training_score"]) // 60

df.drop(["no_of_trainings"] ,axis =1 , inplace =True)
df1.drop(["no_of_trainings"] ,axis =1 , inplace =True)

#age
df["age"]
sb.distplot(df["age"])
df.groupby('age')["is_promoted"].mean().sort_values(ascending = False)
#bins
print(df["age"].min() , df["age"].max())
bins = [20,30,40,50,60]
labels = ['adult','middle_life','old','retire']
df["age_group"] = pd.cut(df["age"] , bins = bins , labels = labels)
bins = [20,30,40,50,60]
labels = ['adult','middle_life','old','retire']
df1["age_group"] = pd.cut(df1["age"] , bins = bins , labels = labels)
print(df["age_group"].isna().sum())
print(df1["age_group"].isna().sum())

df["age_group"].fillna(df['age_group'].mode()[0] , inplace =True)
df1["age_group"].fillna(df1['age_group'].mode()[0] , inplace =True)

df
df.groupby('age_group')["is_promoted"].mean().sort_values(ascending = False)
df.loc[: , "age_group"] = df.loc[: , "age_group"].map({"middle_life":4,"adult":3,"old":2,"retire":1})
df1.loc[: , "age_group"] = df1.loc[: , "age_group"].map({"middle_life":4,"adult":3,"old":2,"retire":1})

df.drop(["age"],axis =1 , inplace =True)
df1.drop(["age"],axis =1 , inplace =True)

#previous year rating
df["previous_year_rating"]
print(df["previous_year_rating"].unique())
print(df["previous_year_rating"].nunique())

df.groupby('previous_year_rating')["is_promoted"].mean().sort_values(ascending =False)
sb.countplot(df["previous_year_rating"])
df
import math

for i ,row in df.iterrows():
    if math.isnan(row['previous_year_rating']):
        if df.loc[i , "awards_won?"] == 1:
            df.loc[i , "previous_year_rating"] = 5.0
        else:
            df.loc[i , "previous_year_rating"] = np.random.randint(1,5)
        
print("done")



for i ,row in df1.iterrows():
    if math.isnan(row['previous_year_rating']):
        if df1.loc[i , "awards_won?"] == 1:
            df1.loc[i , "previous_year_rating"] = 5.0
        else:
            df1.loc[i , "previous_year_rating"] = np.random.randint(1,5)
        
print("done")


print(df["previous_year_rating"].isna().sum())
print(df1["previous_year_rating"].isna().sum())

#length of service
df["length_of_service"]
print(df["length_of_service"].unique())
print(df["length_of_service"].nunique())

df.groupby('length_of_service')["is_promoted"].mean().sort_values(ascending =False)
plt.hist(df["length_of_service"])
sb.boxplot(df["length_of_service"])
print(df["length_of_service"].value_counts())
#bins
bins = [1,10,20,30,40]
labels = [1,2,3,4]
df["promotion_chance_based_on_service"] = pd.cut(df["length_of_service"] , bins= bins , labels = labels)
#bins
bins = [1,10,20,30,40]
labels = [1,2,3,4]
df1["promotion_chance_based_on_service"] = pd.cut(df1["length_of_service"] , bins= bins , labels = labels)
df
#kpi  
df["KPIs_met >80%"]
print(df["KPIs_met >80%"].unique())
print(df["KPIs_met >80%"].nunique())
print(df["KPIs_met >80%"].value_counts())

#awards
df["awards_won?"]
print(df["awards_won?"].unique())
print(df["awards_won?"].nunique())
print(df["awards_won?"].value_counts())

for i in range(len(df)):
    if(( df.loc[i , "awards_won?"] == 1) and (df.loc[i , "KPIs_met >80%"] == 1)):
        df.loc[i , 'promotion_prize'] = 1
    else:
        df.loc[i , "promotion_prize"] = 0
print("done")
for i in range(len(df1)):
    if(( df1.loc[i , "awards_won?"] == 1) and (df1.loc[i , "KPIs_met >80%"] == 1)):
        df1.loc[i , 'promotion_prize'] = 1
    else:
        df1.loc[i , "promotion_prize"] = 0
print("done")
#avg train score
df["avg_training_score"]
sb.distplot(df["avg_training_score"])
df.groupby('avg_training_score')["is_promoted"].mean().sort_values(ascending =False)
#bins
bins = [39,59,79,99]
labels = [1,2,3]
df["promote_train_score"] = pd.cut(df["avg_training_score"] , bins = bins , labels = labels)
df1["promote_train_score"] = pd.cut(df1["avg_training_score"] , bins = bins , labels = labels)

print(df["promote_train_score"].value_counts())
df.groupby('promote_train_score')["is_promoted"].mean()
df.isna().sum()
df["promotion_chance_based_on_service"].fillna(df["promotion_chance_based_on_service"].mode()[0] , inplace = True)
df["promote_train_score"].fillna(df["promote_train_score"].mode()[0] , inplace = True)


df.groupby('select_region')["is_promoted"].mean().sort_values(ascending = False)
df.groupby('age_group')["is_promoted"].mean().sort_values(ascending = False)
df.groupby('promotion_chance_based_on_service')["is_promoted"].mean().sort_values(ascending = False)
df.groupby('promote_train_score')["is_promoted"].mean().sort_values(ascending = False)
df["age_group"] = df["age_group"].astype("int")
df["promote_train_score"] = df["promote_train_score"].astype("int")
df["promotion_chance_based_on_service"] = df["promotion_chance_based_on_service"].astype("int")
df["select_region"] = df["select_region"].astype("int")


df1["promotion_chance_based_on_service"].fillna(df1["promotion_chance_based_on_service"].mode()[0] , inplace = True)
df1["promote_train_score"].fillna(df1["promote_train_score"].mode()[0] , inplace = True)

df1["age_group"] = df1["age_group"].astype("int")
df1["promote_train_score"] = df1["promote_train_score"].astype("int")
df1["promotion_chance_based_on_service"] = df1["promotion_chance_based_on_service"].astype("int")
df1["select_region"] = df1["select_region"].astype("int")

df1.isna().sum()
df["is_promoted"]
df["is_promoted"].value_counts()
# use up sampling
x = df.drop(["is_promoted"] , axis = 1)
y = df["is_promoted"]
r = RandomOverSampler(sampling_strategy=0.7)
print(x.shape , y.shape)
x_new , y_new = r.fit_resample(x,y)
print(x_new.shape , y_new.shape)
x_train , x_test , y_train , y_test = train_test_split(x_new , y_new , test_size = 0.2 , random_state = 100)
lr = LogisticRegression(max_iter = 5000)
dt =DecisionTreeClassifier()
rf = RandomForestClassifier()
ada = AdaBoostClassifier()
xgb = XGBClassifier()
extra = ExtraTreesClassifier()
lgbm = lgbm.LGBMClassifier()
cat = CatBoostClassifier()
print(lr.fit(x_train , y_train))
print(dt.fit(x_train , y_train))
print(rf.fit(x_train , y_train))
print(ada.fit(x_train , y_train))
print(xgb.fit(x_train , y_train))
print(extra.fit(x_train , y_train))
print(lgbm.fit(x_train , y_train))
print(cat.fit(x_train , y_train))


print(cross_val_score(lr , x_new,y_new , cv = 5 ,scoring = "f1").mean())
print(cross_val_score(dt , x_new,y_new , cv = 5 ,scoring = "f1").mean())
print(cross_val_score(rf , x_new,y_new , cv = 5 ,scoring = "f1").mean())
print(cross_val_score(ada , x_new,y_new , cv = 5 ,scoring = "f1").mean())
print(cross_val_score(xgb , x_new,y_new , cv = 5 ,scoring = "f1").mean())
print(cross_val_score(extra , x_new,y_new , cv = 5 ,scoring = "f1").mean())
print(cross_val_score(cat , x_new,y_new , cv = 5 ,scoring = "f1").mean())

feat_importances = pd.Series(rf.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
feat_importances = pd.Series(extra.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
feat_importances = pd.Series(xgb.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
sub
sub["is_promoted"] = xgb.predict(df1).astype("int")
sub.to_csv(r"C:\Users\MANISH SHARMA\Desktop\datascience_av\hr\night_xgb1.csv" , index = False)
sub["is_promoted"].value_counts()
##########################################################################################################################################################