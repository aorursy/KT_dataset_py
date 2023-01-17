import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/hr-analytics-analytics-vidya/train_LZdllcl.csv")
test= pd.read_csv("../input/hr-analytics-analytics-vidya/test_2umaH9m.csv")
test1 = test.copy()
train.head()

train.nunique()
train.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (13,8)
sns.countplot(train.department,hue = train.department,palette = 'dark')
plt.legend(loc = "best")
plt.show()
plt.rcParams["figure.figsize"] = (14,8)
sns.countplot(train.education,hue = train.education,palette = 'dark')
plt.legend(loc = "best")
plt.show()
plt.rcParams["figure.figsize"] = (5,5)
sns.countplot(train.gender,hue = train.gender,palette = 'dark')
plt.legend(loc = "best")
plt.show()
plt.rcParams["figure.figsize"] = (14,8)
sns.countplot(train.recruitment_channel,hue = train.recruitment_channel,palette = 'dark')
plt.legend(loc = "best")
plt.show()
train.no_of_trainings.value_counts()
train.no_of_trainings.unique()
labels = [1,2,3,4,5,6,7,8,9,10]
sizes = train.no_of_trainings.value_counts()
color = colors = plt.cm.copper(np.linspace(0, 1, 10))
explode = explode = [0.1, 0.1, 0.2, 0.3, 0.5,0.6,0.7,0.8,0.9,1.0]
plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(sizes, labels = labels, colors = color, explode = explode, shadow = True)
plt.title('International Repuatation for the Football Players', fontsize = 20)
plt.legend()
plt.show()


plt.rcParams["figure.figsize"] = (14,8)
sns.countplot(train.no_of_trainings,hue = train.no_of_trainings,palette = 'dark')
plt.legend(loc = "best")
plt.show()
plt.rcParams['figure.figsize'] = (15,8)
sns.distplot(train.age,color = 'blue')
plt.xlabel(xlabel = "Ages",fontsize = 10)
plt.ylabel(ylabel = "Count distribution",fontsize = 10)
plt.title("Distribution PLot",fontsize =20)
plt.legend()
plt.xticks(rotation = 0)
plt.show()
labels = ['5.0', '4.0', '3.0', '2.0', '1.0'] 
sizes = train.previous_year_rating.value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0, 0, 0, 0, 0.1]

# plt.rcParams["figure.figsize"] = (10,10)
plt.pie(sizes,labels = labels,colors = colors,explode = explode,shadow = True,startangle = 90)
plt.title("Rating Analysis")
plt.legend()
plt.show()
plt.rcParams['figure.figsize'] = (15,8)
sns.distplot(train.length_of_service,color = 'blue')
plt.xlabel(xlabel = "Service Duration",fontsize = 10)
plt.ylabel(ylabel = "Count distribution",fontsize = 10)
plt.title("Distribution PLot",fontsize =20)
plt.legend()
plt.xticks(rotation = 0)
plt.show()
# avg_training_score	
plt.rcParams['figure.figsize'] = (15,8)
sns.distplot(train.avg_training_score,color = 'blue')
plt.xlabel(xlabel = "Service Duration",fontsize = 10)
plt.ylabel(ylabel = "Count distribution",fontsize = 10)
plt.title("Distribution PLot",fontsize =20)
plt.legend()
plt.xticks(rotation = 0)
plt.show()
plt.style.use("dark_background")
train.department.value_counts().plot.bar(color = 'orange',figsize = (10,10))
plt.title("Department Distribution",fontsize = 20)
plt.xlabel("Department",fontsize = 16)
plt.ylabel("Count")
plt.show()

plt.rcParams['figure.figsize'] = (10,8)
plt.style.use("Solarize_Light2")
sns.scatterplot(x=train["age"],y = train["length_of_service"],palette = "Reds")
plt.legend()
plt.show()
print(train.shape,test.shape)
train.head()
train.drop(labels = ["employee_id","education","previous_year_rating","region"],axis=1,inplace = True)
test.drop(labels = ["employee_id","education","previous_year_rating","region"],axis=1,inplace = True)
train.head()
feat = ["department","gender","recruitment_channel"]
X = pd.get_dummies(train[feat])
X_test = pd.get_dummies(test[feat])
X.head()
train = pd.concat([train,X],axis=1)
test = pd.concat([test,X_test],axis=1)
train.head()
train.drop(labels = ["department","gender","recruitment_channel"],axis=1,inplace = True)
test.drop(labels = ["department","gender","recruitment_channel"],axis=1,inplace = True)
train.head()
y = train.is_promoted
X = train.drop(labels = ["is_promoted"],axis=1)
X.head()
from sklearn import preprocessing

x = X.values #returns a numpy array
test_s = test.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_train = min_max_scaler.fit_transform(x)
x_scaled_test = min_max_scaler.fit_transform(test_s)
df_train = pd.DataFrame(x_scaled_train)
df_test = pd.DataFrame(x_scaled_test)
from sklearn.model_selection import train_test_split,cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.4,stratify = y)
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgb


def metric(y,y0):
    return log_loss(y,y0)
def cross_valid(model,train,y_train,cv):
    results = cross_val_predict(model, train, y_train, method="predict_proba",cv=cv)
    return metric(y_train,results)
models = [lgb.LGBMClassifier(), xgb.XGBClassifier(), GradientBoostingClassifier(), LogisticRegression(), 
              RandomForestClassifier(), AdaBoostClassifier()
             ]
for i in models:
    error =  cross_valid(i,df_train,y,5)
    print(str(i).split("(")[0], error)
model = lgb.LGBMClassifier(booster= 'dart',
        objective= 'binary',
        learning_rate= 0.05,
        max_depth= 8)
model.fit(X_train,y_train)
preds = model.predict(X_test)
print(log_loss(y_test,preds))
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

predictions = model.predict(df_test)
final = pd.DataFrame({"employee_id":test1.employee_id,
                     "is_promoted":predictions})
final.head()
final.to_csv("submiss.csv",index = False)
xgb1 = xgb.XGBClassifier(
    booster='dart',
    objective='multi:softprob',
    learning_rate= 0.01,
#     num_round= 775,
    max_depth=8,
    seed=25,
    nthread=3,
    eval_metric='mlogloss',
    num_class=5

)
model = xgb1
model.fit(X_train,y_train)
preds = model.predict(X_test)
print(log_loss(preds,y_test))
predictions = model.predict(df_test)
final = pd.DataFrame({"employee_id":test1.employee_id,
                     "is_promoted":predictions})
final.head()
final.to_csv("submiss.csv",index = False)
print(classification_report(y_test, preds))

