import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.set_option("display.max_columns",999)
pd.set_option("display.max_rows", 999)
import warnings
warnings.simplefilter(action = "ignore")
df = pd.read_csv("../input/titanic/train.csv")
df.head(5)
df.info()
df.describe()
#total 891 observations, our dependent variable is Survived
#to find null values
null_list = []
null_count = []
null_percen = []
for i in df.columns:
    val = df[i].isnull().sum()
    if  val > 0:
        null_list.append(i)
        null_count.append(val)
        null_percen.append(val/len(df))
null_dic = {"null_columns" : null_list, "null_count" : null_count, "null_percentage" : null_percen}
null_df = pd.DataFrame.from_dict(null_dic)
null_df
fig = plt.figure()
sns.barplot(x = "null_columns", y = "null_percentage", data = null_df)

plt.show()
#for cabin, we need to replace the nan values with the another category
df["Cabin"] = df['Cabin'].fillna("miss")
#and also take the first letter of the cabin to specify the classes.
cabin_first = df['Cabin'].astype(str).str[0]
df['Cabin'] = df['Cabin'].astype(str).str[0]
cabin_dict = cabin_first.value_counts().to_dict()
cabin_dict
df['Cabin'] = df['Cabin'].map(cabin_dict)
#now focus on age column
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'].value_counts()
#missed values is only two so replace that with Highest count 'S'
df['Embarked'] = df['Embarked'].fillna('S')
df.head(3)
#name and ticket is not required
df.drop(['Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)
df.head(5)
sns.barplot(y = 'Age', x = 'Survived', hue = 'Sex', data = df)
# more female are survived than male
sns.stripplot(x = "Sex", y = "Fare", hue = "Embarked", data = df)
for i in df.columns:
    try:
        plt.figure()
        colors = "#" + str(np.random.randint(100000,999999))
        sns.distplot(df[i], color = colors)
        plt.figure()
    except:
        pass
sex = pd.get_dummies(df['Sex'], drop_first = True)
embarked = pd.get_dummies(df['Embarked'], drop_first = True)
df.drop(['Sex', 'Embarked'],axis = 1, inplace = True)
df = pd.concat([df, sex, embarked], axis = 1)
df.head(1)
Y = df[['Survived']]
df.drop('Survived', axis = 1, inplace = True)
X= df
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_X = scaler.fit_transform(X)
X = pd.DataFrame(scale_X, columns = X.columns)
X.head(5)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
x_train, x_test, y_train,y_test = train_test_split(X,Y, test_size = 0.3)
#for logisticregression
LOR = LogisticRegression()
LOR.fit(x_train,y_train)
y_pred = LOR.predict(x_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
#for randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
1e-4
params = {
    'penalty' : ['l1', 'l2', 'elasticnet', None],
    'C': [0.2,0.5,0.7,1.0,1,4,1,5],
    'class_weight': [None,1,2,3,4,5,8],
    'solver' : ['liblinear', 'newton-cg', 'lbfgs'],
    'max_iter' : np.linspace(0,501,50),
    'l1_ratio' : [0.2,0.4,0.6,0.8,1]
}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
m1_rn = RandomizedSearchCV(LOR, param_distributions = params, n_jobs = -1, verbose = 3, n_iter = 100, cv = 5, random_state = 0)
m1_rn.fit(x_train,y_train)
m1_rn.best_params_
params = { 'C': [0,0.2,0.3],
    'class_weight': [None,1,2],
    'solver' : ['newton-cg', 'lbfgs'],
    'max_iter' : [300,340,380],
    'l1_ratio' : [0,0.1,0.2,0.3],
        'penalty' : ["l2"]
         }
m2_grid = GridSearchCV(LOR, param_grid = params, n_jobs = -1, cv = 4, verbose = 4)
m2_grid.fit(x_train,y_train)
m2_grid.best_estimator_
m2_grid.best_params_
m3_lor = LogisticRegression(C= 0.2,
 class_weight= None,
 l1_ratio= 0,
 max_iter= 300,
 penalty= 'l2',
 solver='newton-cg')
m3_lor.fit(x_train,y_train)
y_pred = m3_lor.predict(x_test)
print(accuracy_score(y_pred,y_test))
#perform hyper parameter tuning xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
params = {
    "max_depth" : [1,2,3,4,5,6,7,8,9],
    "learning_rate" : [0.005,0.010,0.030,0.070,0.010,0.03,0.05,0.09,0.10,0.20,0.30,0.40,0.50,0.60],
    "booster" : ['gbtree', 'gblinear', 'dart'],
    "gamma" : [0,0.05,0.1,0.2,0.3,0.4,0.5],
    "min_child_weight" : [1,2,3,4,5,6,7],
    "subsample" : [0.2,0.3,0.4,0.5,0.6,0.7],
    "colsample_bytree": [0.1,0.2,0.3,0.4,0.5]
}
m4_xgb = RandomizedSearchCV(xgb, param_distributions = params, n_iter = 200, cv = 5, n_jobs = -1, verbose = 3, random_state = 0)
m4_xgb.fit(x_train,y_train)
m4_xgb.best_params_
y_pred = m4_xgb.predict(x_test)
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
params = {
    "max_depth" : [6,7,8,9],
    "learning_rate" : [0.08,0.07,.09,0.10,0.11],
    "booster" : ['dart'],
    "gamma" : [0,0.09,0.02],
    "min_child_weight" : [3,4,5],
    "subsample" : [0.6,0.7,0.8,0.9],
    "colsample_bytree": [0.4,0.5,0.6,0.7]
}
m5_grid = GridSearchCV(xgb, param_grid = params, n_jobs = -1, cv = 4,verbose = 3)
m5_grid.fit(x_train,y_train)
y_pred = m5_grid.predict(x_test)
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
m5_grid.best_estimator_
test_data = pd.read_csv("../input/titanic/test.csv")
test_data.drop(['PassengerId', 'Ticket', 'Name'], axis = 1, inplace = True)
test_data.head(5)
#to find null values
null_list = []
null_count = []
null_percen = []
for i in test_data.columns:
    val = test_data[i].isnull().sum()
    if  val > 0:
        null_list.append(i)
        null_count.append(val)
        null_percen.append(val/len(df))
null_list
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data['Cabin'] = test_data['Cabin'].fillna('miss')
test_data['Cabin'] = test_data['Cabin'].astype(str).str[0]
cabin_dict = test_data['Cabin'].value_counts().to_dict()
test_data['Cabin'] = test_data['Cabin'].map(cabin_dict)
test_data.head()
sex = pd.get_dummies(test_data['Sex'], drop_first = True)
embarked = pd.get_dummies(test_data['Embarked'], drop_first = True)
test_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
test_data = pd.concat([test_data, sex, embarked], axis = 1)
scaled_data = scaler.fit_transform(test_data)
test_data = pd.DataFrame(scaled_data, columns = test_data.columns)
test_data.head()
# fitting the train data with the best model
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ("xgb", XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.09, max_delta_step=0, max_depth=9,
              min_child_weight=4, missing=np.nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.7,
              tree_method='exact', validate_parameters=1, verbosity=None))
])

pipe.fit(X,Y)
pipe.predict(test_data)
import joblib
joblib.dump(pipe, 'TitanicModel.pkl')
submission = pd.read_csv("../input/titanic/gender_submission.csv")
y_pred = pipe.predict(test_data)
print(confusion_matrix(y_pred,submission['Survived']))
print(accuracy_score(y_pred,submission['Survived']))
print(classification_report(y_pred,submission['Survived']))
submission['Survived'] = y_pred
submission.to_csv("test_result.csv", index = False)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
params = {
    "n_estimators" : [100,200,300,400,500],
    "criterion" : ['entropy', 'gini'],
    "max_depth" : [1,3,5,7,9],
    "min_samples_split" : [1,3,5,6,7,8],
    "min_samples_leaf" : [1,2,3,4],
    "max_leaf_nodes" : [None, 1,2,3,4],

}
model_new1 = RandomizedSearchCV(rfc, param_distributions = params, n_jobs = -1, verbose = 3, n_iter = 300, cv = 4, random_state = 0)
model_new1.fit(X,Y)
y_pred = model_new1.predict(test_data)
print(confusion_matrix(y_pred,submission['Survived']))
print(accuracy_score(y_pred,submission['Survived']))
print(classification_report(y_pred,submission['Survived']))
y_pred
submission['Survived'] = y_pred
submission.to_csv("./test_result.csv", index = False)