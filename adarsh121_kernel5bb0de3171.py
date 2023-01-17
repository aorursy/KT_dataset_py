import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.shape
test.shape
combined=pd.concat([train,test],ignore_index=True,sort=True)
combined_backup=combined.copy()
combined.head()
combined.info()
combined.isnull().sum()
combined.employee_id.nunique()
combined.shape
sns.boxplot(combined.age)
sns.distplot(combined.age)

#Highly right skewed 
combined["KPIs_met >80%"].value_counts().plot(kind="bar")
sns.boxplot(combined["avg_training_score"])
sns.distplot(combined["avg_training_score"])
combined.columns
sns.countplot(combined["awards_won?"])



#maximum poeple has won no awards. so obviously maximum people has got no awrds,



# we can figure out which are the poeple who got awards and were they promoted or  

combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==1.0)]["age"]
sns.distplot(combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==1.0)]["age"])
combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==0.0)]
sns.distplot(combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==0.0)]["age"])
combined[(combined["awards_won?"]==0) & (combined["is_promoted"]==1.0)]
#0 awards won and is getting promoted is 4109
sns.countplot(combined.department)

plt.xticks(rotation=90)
combined.department.value_counts().plot(kind="bar")
combined.education.value_counts().plot(kind="bar")
#most people are bachelors
combined.gender.value_counts().plot(kind="bar")
plt.figure(figsize=(10,5))

combined.length_of_service.value_counts().plot(kind="bar")

plt.xticks(rotation=90)
sns.countplot(combined["no_of_trainings"])
sns.countplot(combined.previous_year_rating)
sns.countplot(combined.recruitment_channel)
combined.region.value_counts().plot(kind="bar",figsize=(17,6))
combined.head()
#Going for Boxplots
sns.boxplot(combined["awards_won?"],combined.age)
sns.boxplot(combined["awards_won?"],combined.length_of_service)

sns.boxplot(combined["awards_won?"],combined.no_of_trainings)
sns.boxplot(combined["awards_won?"],combined.previous_year_rating)
sns.boxplot(combined["awards_won?"],combined.recruitment_channel.value_counts())
sns.boxplot(combined["is_promoted"],combined.age)
sns.boxplot(combined["is_promoted"],combined.length_of_service)
sns.boxplot(combined["is_promoted"],combined.previous_year_rating)
sns.boxplot(combined["is_promoted"],combined.no_of_trainings)
#Numerical vs Numerical
plt.scatter(combined.age,combined.avg_training_score)
plt.scatter(combined.age,combined.length_of_service)

# we can clearly see with age length of service is increasing
#categorical vs categorical analysis

combined.head()
combined.groupby(["education","department"]).describe().plot(kind="bar",figsize=(20,10))
combined.groupby(["education","department"])["age"].describe().plot(kind="bar",figsize=(20,10))
combined.groupby(["education","department","gender"]).describe().plot(kind="bar",figsize=(20,10))
combined.groupby(["education","department","gender"])["age"].describe().plot(kind="bar",figsize=(20,10))
combined.groupby(["department","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))
combined.groupby(["department","education","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))
combined.groupby(["department","awards_won?","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))
combined.head()
pd.DataFrame(combined.groupby(["no_of_trainings","KPIs_met >80%","is_promoted"])["age"].describe())
combined.groupby(["KPIs_met >80%","is_promoted"])["age"].describe()
sns.boxplot(combined["KPIs_met >80%"],combined["is_promoted"])

combined.groupby(["KPIs_met >80%","is_promoted"])["age"].describe().plot(kind="bar")
combined.groupby(["length_of_service","is_promoted"])["age"].describe()
#Perfroming Feature Engineering
combined.head()
train.shape
combined.iloc[54807]
combined.isnull().sum()
combined.previous_year_rating.mean()
combined.previous_year_rating.mode()
combined.previous_year_rating.median()
combined.previous_year_rating.skew()
combined.previous_year_rating.kurt()
combined.loc[combined.previous_year_rating.isnull(),"previous_year_rating"]=3.0
combined.isnull().sum()
combined.education.mode()
combined.education.dropna(inplace=True)
train=combined[:54808]
test=combined[54808:]
train.head()
train.corr()
plt.figure(figsize=(10,6))

sns.heatmap(train.corr(),annot=True)
train.drop("region",axis=1,inplace=True)

test.drop("region",axis=1,inplace=True)
train.drop("employee_id",axis=1,inplace=True)

test.drop("employee_id",axis=1,inplace=True)
d={"m":1,"f":0}

train.gender=train.gender.map(d)
test.gender=test.gender.map(d)
train.head()
dummytrain=pd.get_dummies(train).drop("recruitment_channel_other",axis=1)

dummytest=pd.get_dummies(test).drop("recruitment_channel_other",axis=1)
X=dummytrain.drop(["is_promoted"],axis=1)
y=dummytrain.is_promoted
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=123,test_size=0.2)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

scaled_train=pd.DataFrame(sc.fit_transform(X_train,y_train),columns=X_train.columns)

scaled_test=pd.DataFrame(sc.transform(X_test),columns=X_test.columns)
scaled_train.shape
scaled_test.shape
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

model_rf=rf.fit(scaled_train,y_train).predict(scaled_test)

from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error

print("The R sqaure of the model is ",r2_score(y_test,model_rf))

print("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_rf)))
features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,

            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "red")
from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier()

model_gb=gb.fit(scaled_train,y_train).predict(scaled_test)

print("The R sqaure of the model is ",r2_score(y_test,model_gb))

print("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_gb)))
from xgboost import XGBRFRegressor

xg=XGBRFRegressor()

model_xg=xg.fit(scaled_train,y_train).predict(scaled_test)

print("The RMSE IS ",np.sqrt(mean_squared_error(y_test,model_xg)))

print("tHE R SQAURE IS ",r2_score(y_test,model_xg))
from sklearn.ensemble import AdaBoostClassifier

ad=AdaBoostClassifier(random_state=123)

model_ad=ad.fit(scaled_train,y_train).predict(scaled_test)

print("The R sqaure of the model is ",r2_score(y_test,model_ad))

print("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_ad)))
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.min_rows', 1000)
combined.head()
#https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
combined.age.head()
combined.head()
combined.drop(["employee_id","region"],axis=1,inplace=True)
sns.distplot(np.sqrt(combined.age))
sns.distplot(np.log1p(combined.age))
sns.distplot(np.log1p(combined.length_of_service))
combined.length_of_service=np.log1p(combined.length_of_service)
combined.age=np.log1p(combined.age)
combined.head()
d={"f":0,"m":1}

combined.gender=combined.gender.map(d)
combined.head()
train.shape
test.shape
newtrain=combined[:54808]

newtest=combined[54808:]
newtest=combined[54808:]
newtest.drop("is_promoted",axis=1,inplace=True)
dummytrain=pd.get_dummies(newtrain)

dummytest=pd.get_dummies(newtest)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

cols = dummytrain.columns[dummytrain.columns!="is_promoted"]

scaled_train = pd.DataFrame(sc.fit_transform(dummytrain.drop("is_promoted", axis = 1)), 

             columns=cols)

scaled_test = pd.DataFrame(sc.transform(dummytest),

                          columns = dummytest.columns)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

model = rf.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,

            columns = ["Features"])

features.sort_values(by = "Features").plot(kind = "barh", color = "red")
from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error

print("The R sqaure of the model is ",r2_score(dummytrain.is_promoted[:23490],model))

print("The RMSE IS", np.sqrt(mean_squared_error(dummytrain.is_promoted[:23490],model)))
y_test.shape
model.shape
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 

                        "is_promoted":model})

solution.to_csv("RF MODEL.csv", index =False)
test.head()
x=pd.read_csv("RF MODEL.csv")
x.is_promoted=x.is_promoted.astype("int64")
x.head()
solution = pd.DataFrame({"employee_id":x.employee_id, 

                        "is_promoted":x.is_promoted})

solution.to_csv("RF MODEL2.csv", index =False)
from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier()

model_gb = gb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 

                        "is_promoted":model_gb})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("GB MODEL.csv", index =False)
from xgboost import XGBRFClassifier

xg=XGBRFClassifier(n_estimators=3,max_depth=500)

model_xg = xg.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 

                        "is_promoted":model_xg})

solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("xg MODEL.csv", index =False)
from sklearn.ensemble import AdaBoostRegressor

ad=AdaBoostRegressor(random_state=123)

model_ada = ad.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 

                        "is_promoted":model_xg})

solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("ADA MODEL.csv", index =False)
features = pd.DataFrame(ad.feature_importances_, index = scaled_test.columns,

            columns = ["Features"])

features.sort_values(by = "Features").plot(kind = "barh", color = "blue")
features.sort_values(by = "Features")
combined.head()
combined.head()
combined.groupby(["recruitment_channel","is_promoted"]).describe().plot(kind="bar",figsize=(10,5))
combined.drop(["recruitment_channel"],axis=1,inplace=True)
combined.head()
combined["total_hours"]=combined.avg_training_score*combined.no_of_trainings
combined.head()
combined["total_sum"]=combined["KPIs_met >80%"]+combined["awards_won?"]+combined["no_of_trainings"]
combined.head()
plt.figure(figsize=(10,5))

sns.heatmap(combined.corr(),annot=True,cmap="ocean")
combined.drop(["total_score"],axis=1,inplace=True)
combined.head()
combined.education.unique()
combined.isnull().sum()
combined.education.mode()

combined[combined.education.isnull()]["education"]=combined.education.mode()
combined.loc[combined.education.isnull(),"education"]="Bachelor's"
combined[combined.education.isnull()]["education"]
combined.isnull().sum()
combined.head()
newtrain=combined[:54808]

newtest=combined[54808:]

newtest.drop("is_promoted",axis=1,inplace=True)

dummytrain=pd.get_dummies(newtrain)

dummytest=pd.get_dummies(newtest)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

cols = dummytrain.columns[dummytrain.columns!="is_promoted"]

scaled_train = pd.DataFrame(sc.fit_transform(dummytrain.drop("is_promoted", axis = 1)), 

             columns=cols)

scaled_test = pd.DataFrame(sc.transform(dummytest),

                          columns = dummytest.columns)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

model = rf.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)

features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,

            columns = ["Features"])

features.sort_values(by = "Features").plot(kind = "barh", color = "red")
from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error

print("The R sqaure of the model is ",r2_score(dummytrain.is_promoted[:23490],model))

print("The RMSE IS", np.sqrt(mean_squared_error(dummytrain.is_promoted[:23490],model)))

solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 

                        "is_promoted":model})

solution.is_promoted=solution.is_promoted.astype("int64")

solution.to_csv("RF MODEL4_FEATURE ENG.csv", index =False)
from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier()

model_gb = gb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)

solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 

                        "is_promoted":model_gb})

solution.is_promoted=solution.is_promoted.astype("int64")

solution.to_csv("GB MODEL_feature reng.csv", index =False)
features = pd.DataFrame(gb.feature_importances_, index = scaled_test.columns,

            columns = ["Features"])

features.sort_values(by = "Features").plot(kind = "barh", color = "green")
!pip install catboost
from catboost import CatBoostClassifier

cb=CatBoostClassifier()
model_cb = cb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)

solution = pd.DataFrame({"employee_id":pd.read_csv("test.csv").employee_id, 

                        "is_promoted":model_gb})

solution.is_promoted=solution.is_promoted.astype("int64")

solution.to_csv("CAT BOOST MODEL_feature reng.csv", index =False)
features = pd.DataFrame(cb.feature_importances_, index = scaled_test.columns,

            columns = ["Features"])

features.sort_values(by = "Features").plot(kind = "barh", color = "magenta")
import pandas as pd

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")