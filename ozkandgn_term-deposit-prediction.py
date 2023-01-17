'''
### Connect to google drive

from google.colab import drive

drive.mount("/content/gdrive")
'''
'''
### Google drive folder

%cd /content/gdrive/"My Drive"/'Colab Notebooks'/"hangikredi_case"
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/termdepositmarketing2020/term-deposit-marketing-2020.csv")
data.sample(5)
### Unknown to np.nan transformation

data = data.replace({"unknown":np.nan})
### General information about data

data.info()
data.job.value_counts()
data.marital.value_counts()
data.education.value_counts()
data.contact.value_counts()
### 3 column(job,education,contact) have some Nan values 
### Relation between nan values

import missingno as msno

msno.matrix(data[["job","education","contact"]])
### Nan value correlation


msno.heatmap(data)
### Contact feature have very much null value

data_con = data[["contact","y"]].dropna()
data_con["contact"] = data_con["contact"].apply(lambda x: 1 if x=="cellular" else 0)
data_con["y"] = data_con["y"].apply(lambda x: 1 if x=="yes" else 0)

data_con.corr()
data = data.drop(["contact"],axis=1)
def binary_transformation(df,col):
    df[col] = df[col].apply(lambda x: 1 if x=="yes" else 0)
### !!! Run just one time !!!
binary_transformation(data,"default")
binary_transformation(data,"housing")
binary_transformation(data,"loan")
binary_transformation(data,"y")
data
sns.catplot(x="month",y="y",data=data,kind="bar")

### There is a relation between some month and y value
sns.catplot(x="day",y="y",data=data,kind="bar",height=8)

## There is a good relation between day and y
sns.catplot(x="housing",y="y",data=data,kind="bar",height=4)
sns.catplot(x="loan",y="y",data=data,kind="bar",height=4)
sns.catplot(x="default",y="y",data=data,kind="bar",height=4)

### There is a relation between this values and y but not much
data[["age","y"]].groupby("age").mean().plot()
### data size is low at high ages, but there is a little relation age and y
plt.figure(figsize=(20,10))
sns.barplot(x="job",y="y",data=data)

### There is a relation but student data is surprising
sns.catplot(x="marital",y="y",data=data,kind="bar")
### There is a little relation
sns.catplot(x="education",y="y",data=data,kind="bar")
### There is a normal relation
data[["duration","y"]].groupby("y").mean().plot.barh()

### There is a big relation between duration and y
data[["balance","y"]].groupby("y").mean().plot.barh()

### There is not a relation between balance and y
sns.catplot(x="education",y="balance",data=data,kind="bar",hue="y")
###There isn't a relation between education - balance - y
sns.catplot(x="marital",y="balance",data=data,kind="bar",hue="y")
###There isn't a relation between marital - balance - y
df_balance = data["balance"]

Q1 = df_balance.quantile(0.25)
Q3 = df_balance.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data["balance"]>upper_bound)|(data["balance"]<lower_bound)]
outliers
int_data = data[["age","balance","duration","housing","loan"]]
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors = 20)

clf.fit_predict(int_data)

df_scores = clf.negative_outlier_factor_
df_scores
sns.distplot(np.sort(df_scores))
### Multiple outlier values

data[df_scores < -1.75]
### Drop values

outlier_values = data[df_scores < -1.75]

data = data.drop(outlier_values.index).reset_index(drop=True)
### factorize string values to int and -1 - nan transformation.

def string_transformation(df,col):
    col_val, col_index = pd.factorize(df[col])
    df[col] = col_val
    df[col] = df[col].replace({-1:np.nan})
    return col_index
    
### !!! Run just one time !!!

job_index = string_transformation(data,"job")
marital_index = string_transformation(data,"marital")
education_index = string_transformation(data,"education")
month_index = string_transformation(data,"month")
data
!pip install ycimpute

from ycimpute.imputer import knnimput
nan_data = data[data.isna().any(axis=1)]
non_nan_data = data.drop(nan_data.index)
from sklearn.utils import shuffle
small_data = pd.concat((nan_data,non_nan_data.iloc[:20000,:]))
small_data = shuffle(small_data).reset_index(drop=True)
small_data
var_names = small_data.columns
var_names
var_values = small_data.values
var_values
dff = knnimput.KNN(k=3).complete(var_values)
new_small_data = pd.DataFrame(dff,columns = var_names)
new_small_data
new_small_data["job"] = new_small_data["job"].apply(lambda x: round(x))
new_small_data["education"] = new_small_data["education"].apply(lambda x: round(x))
remaining_data = non_nan_data.iloc[20000:,:]
remaining_data
data = pd.concat((new_small_data,remaining_data))
data.isna().any()
#Int to String

### Run just one time !!!

def int_transformation(df,val,val_index):
  df[val] = df[val].apply(lambda x: val_index[int(x)])

int_transformation(data,"job",job_index)
int_transformation(data,"marital",marital_index)
int_transformation(data,"education",education_index)
int_transformation(data,"month",month_index)
### Non-Nan and clear data

data
### Copy old data

pure_data = pd.DataFrame(data)
#for job-material-education-month
encoded_data = pd.get_dummies(data, columns = ["job","marital","education","month"])
encoded_data
#Drop dummy values

data = encoded_data.drop(["job_admin","marital_divorced","education_primary","month_apr"],axis=1)
X = data.drop("y",axis=1)
y = data["y"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier()

mlp_params = {"alpha":[0.1,0.02,0.001,0.005],
             "hidden_layer_sizes":[(5,5),(15,5,3),(20,10),(100,100)],
              "activation":["logistic"]}
from sklearn.model_selection import GridSearchCV

### Using Small Data !!
mlp_cv_model = GridSearchCV(mlp_model,mlp_params,cv=5,verbose=2).fit(X_train_scaled[:2000],y_train[:2000])
mlp_cv_model.best_params_
model = MLPClassifier(activation="logistic",
                      hidden_layer_sizes=(100,100),
                      alpha =0.1).fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error

### RMSE
np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=model,X=X,y=y,cv=5)
accuracies
### Neural Network Accuracy

accuracies.mean()
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

rf_params = {"max_depth":[5,7,9],
            "max_features":[2,4,8],
            "n_estimators":[200,500,2000],
            "min_samples_split":[2,10,80]}
rf_cv_model = GridSearchCV(rf_model,rf_params,cv=5,verbose = 2).fit(X_train[:2000],y_train[:2000])
rf_cv_model.best_params_
model = RandomForestClassifier(max_depth=9,
                       max_features=4,
                       min_samples_split=2,
                       n_estimators=2000).fit(X_train,y_train)
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
accuracies=cross_val_score(estimator=model,X=X,y=y,cv=5)
accuracies
### Random Forest Accuracy

accuracies.mean()
feature_imp = pd.Series(model.feature_importances_,
                index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(20,10))

sns.barplot(feature_imp,feature_imp.index)
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate":[0.005,0.01,0.04,0.1],
            "max_depth":[3,5,8,12],
            "n_estimators":[20,50,100,300]}
### İt's very fast and we don't need small data :)

lgbm_cv_model = GridSearchCV(lgbm_model,lgbm_params,cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train)
lgbm_cv_model.best_params_
lgbm_model = LGBMClassifier(learning_rate=0.1,
                       max_depth=5,
                       n_estimators=300).fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
accuracies=cross_val_score(estimator=model,X=X,y=y,cv=5)
accuracies
accuracies.mean()
feature_imp = pd.Series(lgbm_model.feature_importances_,
                index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(20,10))

sns.barplot(feature_imp,feature_imp.index)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=5).fit(X, y)
### If he/she don't have a home he/she can be take term deposit

plot_partial_dependence(model, X, [3])
# Take term deposit rate by balance

plot_partial_dependence(model, X, [2]) 
# Take term deposit rate by age

plot_partial_dependence(model, X, [0]) 
!pip install pdpbox
from pdpbox import pdp

def plot_partial_dep(model,data,feature):
    pdp_dist = pdp.pdp_isolate(model=model, dataset=data
                               , model_features=data.columns
                               , feature=feature)
    pdp.pdp_plot(pdp_isolate_out=pdp_dist, feature_name=feature);
X_train
plot_partial_dep(lgbm_model,X_train,"balance")

plot_partial_dep(lgbm_model,X_train,"age")

plot_partial_dep(lgbm_model,X_train,"housing")

plot_partial_dep(lgbm_model,X_train,"duration")

plot_partial_dep(lgbm_model,X_train,"campaign")
