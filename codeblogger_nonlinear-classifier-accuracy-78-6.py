import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("seaborn-whitegrid")

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.head(10)
data
data.columns.values.reshape(-1,1)
data.describe().T
data.info()
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(data[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.show()

    

    

columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',

       'Insulin','BMI', 'DiabetesPedigreeFunction','Age','Outcome']



for col in columns:

    plot_hist(col)
data.isnull().sum()
# Outlier Detection 

for col in columns:

    sns.boxplot(x = data[col])

    plt.show()
data['Glucose'].fillna(data['Glucose'].mean(), inplace = True)

data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace = True)

data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace = True)

data['Insulin'].fillna(data['Insulin'].median(), inplace = True)

data['BMI'].fillna(data['BMI'].median(), inplace = True)
data.shape
import missingno as msno

p=msno.bar(data)
sns.countplot(x=data.Outcome,data=data)

plt.show()
g = sns.factorplot(x="Outcome",y="BMI",data=data,kind="bar")

g.set_ylabels("BMI")

plt.show()
g = sns.factorplot(x="Outcome",y="Pregnancies",data=data,kind="bar")

g.set_ylabels("Pregnancies")

plt.show()
g = sns.factorplot(x="Outcome",y="Glucose",data=data,kind="bar")

g.set_ylabels("Glucose")

plt.show()
g = sns.factorplot(x="Outcome",y="BloodPressure",data=data,kind="bar")

g.set_ylabels("BloodPressure")

plt.show()
g = sns.factorplot(x="Outcome",y="SkinThickness",data=data,kind="bar")

g.set_ylabels("SkinThickness")

plt.show()
g = sns.factorplot(x="Outcome",y="Insulin",data=data,kind="bar")

g.set_ylabels("Insulin")

plt.show()
g = sns.factorplot(x="Outcome",y="BMI",data=data,kind="bar")

g.set_ylabels("BMI")

plt.show()
g = sns.factorplot(x="Outcome",y="DiabetesPedigreeFunction",data=data,kind="bar")

g.set_ylabels("DiabetesPedigreeFunction")

plt.show()
g = sns.factorplot(x="Outcome",y="Age",data=data,kind="bar")

g.set_ylabels("Age")

plt.show()
f,ax = plt.subplots(figsize=(10,8))

corr = data.corr()

sns.heatmap(

    corr,

    mask= np.zeros_like(corr, dtype=np.bool),

    cmap= sns.diverging_palette(240,10,as_cmap=True),

    square= True,

    ax=ax

    )

plt.show()
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import neighbors

from sklearn import metrics

from sklearn.svm import SVR
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



print("X_train: ", len(X_train))

print("X_test: ", len(X_test))

print("y_train: ", len(y_train))

print("y_test: ", len(y_test))
knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

y_pred = knn_model.predict(X_test)

y_pred
print("mean of error squares: ", np.sqrt(mean_squared_error(y_test,y_pred)))

print("Score: ", knn_model.score(X_test,y_test)*100)
# find k value

score_list = []



for each in range(1,100):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test,y_test))

    

plt.plot(range(1,100),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
knn_model = KNeighborsClassifier(n_neighbors=43).fit(X_train,y_train)

y_pred = knn_model.predict(X_test)

print("mean of error squares: ", np.sqrt(mean_squared_error(y_test,y_pred)))

print("Accuracy: ", knn_model.score(X_test,y_test)*100)
# Another way

param_grid = {'n_neighbors':np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train)

y_pred = knn_cv.predict(X_test)
print("Accuracy: ", knn_cv.best_score_*100)
zero = data[data["Outcome"] == 0]

one = data[data["Outcome"] == 1]

# scatter plot

plt.scatter(zero.BMI,zero.DiabetesPedigreeFunction,color="red",label="zero",alpha= 0.3)

plt.scatter(one.BMI,one.DiabetesPedigreeFunction,color="green",label="one",alpha= 0.3)

plt.legend()

plt.show()
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



svr_model = SVR("linear").fit(X_train,y_train)

svr_model
y_pred = svr_model.predict(X_test)

print("mean of error squares: ", np.sqrt(mean_squared_error(y_test,y_pred)))

print("Accuracy: ", svr_model.score(X_test,y_test)*100)
# GRID SEARCH METHOD

svr_model = SVR('linear')

svr_params = {

    "C": [0.1, 0.5, 1, 3]  # penalty coefficient values

}

svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, verbose=2, n_jobs=-1).fit(X_train,y_train)
svr_cv_model.best_params_
print("Accuracy: ",svr_cv_model.best_score_*100)
data.head()
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



mlp_model = MLPClassifier().fit(X_train,y_train)

mlp_model
y_pred = mlp_model.predict(X_test)

print("mean of error squares: ", np.sqrt(mean_squared_error(y_test,y_pred)))

print("Accuracy: ", mlp_model.score(X_test,y_test)*100)
# another way

mlp_params = {

    "alpha": [0.1, 0.01, 0.02, 0.001, 0.0001],

    "hidden_layer_sizes": [(10,20),(5,5),(100,100)]

}



mlp_cv_model = GridSearchCV(mlp_model,mlp_params,cv=10,verbose=2,n_jobs=-1).fit(X_train,y_train) 
print("Accuracy: ", mlp_cv_model.best_score_*100)
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



card_model = DecisionTreeClassifier(random_state=42)

card_model.fit(X_train,y_train)

y_pred = card_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
print("mean of error squares: ", np.sqrt(mean_squared_error(y_test,y_pred)))

print("Accuracy: ",card_model.score(X_test,y_test)*100)
# another way

card_params = {

    "max_depth": [2,3,4,5,10,20],

    "min_samples_split": [2,10,5,30,50,10]

}

card_model = DecisionTreeClassifier()

card_cv_model = GridSearchCV(card_model,card_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print("Accuracy: ",card_cv_model.best_score_*100)
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



rf_model = RandomForestClassifier(random_state=42).fit(X_train,y_train)

y_pred = rf_model.predict(X_test)

print("Accuracy: ", rf_model.score(X_test,y_test))
# another way

rf_params = {

    "max_depth": [5,8,10],

    "max_features": [2,5,10],

    "n_estimators": [200,500,1000,2000],

    "min_samples_split": [2,10,80,100]

}



rf_cv_model = GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print("Accuracy: ", rf_cv_model.best_score_*100)
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



gbm_model = GradientBoostingClassifier().fit(X_train,y_train)

y_pred = gbm_model.predict(X_test)

print("Accuracy: ", gbm_model.score(X_test,y_test)*100)
# another way

gbm_model = GradientBoostingClassifier().fit(X_train,y_train)

gbm_params = {

    "learning_rate": [0.001,0.1,0.001],

    "max_depth": [3,5,8],

    "n_estimators": [100,200,500],

    "min_samples_split": [1,0.5,0.8],

}



gbm_cv_tuned = GridSearchCV(gbm_model,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print("Accuracy: ", gbm_cv_tuned.best_score_*100)
import xgboost

from xgboost import XGBClassifier



y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



xgb = XGBClassifier().fit(X_train,y_train)

print("Accuracy: ", xgb.score(X_test,y_test)*100)
xgb = XGBClassifier()



xgb_params = {

    "learning_rate": [0.01,0.01,0.5],

    "max_depth": [2,3,4,5,8],

    "n_estimators": [100,200,500,1000],

    "colsample_bytree": [0.4,0.7,1]

}



xgb_cv_model = GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=1).fit(X_train,y_train)
print("Accuracy: ", xgb_cv_model.best_score_*100)
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



from lightgbm import LGBMClassifier



lgb_model = LGBMClassifier().fit(X_train,y_train)

print("Accuracy: ", lgb_model.score(X_test,y_test)*100)
lgbm_params = {

    "learning_rate": [0.01,0.1,0.5,1],

    "n_estimators": [20,40,100,200,500,1000],

    "max_depth": [1,2,3,4,5,6,7,8,9,10]

}



lgbm_cv_model = GridSearchCV(lgb_model,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print("Accuracy: ", lgbm_cv_model.best_score_*100)
y = data["Outcome"].values

x_data = data.drop(["Outcome"],axis=1)

X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



from catboost import CatBoostClassifier

catb_model = CatBoostClassifier(random_state=42).fit(X_train,y_train)

print("Accuracy: ", catb_model.score(X_test,y_test)*100)
# another way

catb_params = {

    "iterations": [200,500,100],

    "learning_rate": [0.01,0.1],

    "depth": [3,6,8]

}

catb_model = CatBoostClassifier()

catb_cv_model = GridSearchCV(catb_model,catb_params,cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train)
print("Accuracy: ", catb_cv_model.best_score_*100)
def compML(df,alg):

    y = data["Outcome"].values

    x_data = data.drop(["Outcome"],axis=1)

    X = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)



    model = alg().fit(X_train,y_train)

    model_name = alg.__name__

    print(model_name,": ", model.score(X_test,y_test)*100)
models = [

    LGBMClassifier,

    XGBClassifier,

    GradientBoostingClassifier,

    RandomForestClassifier,

    DecisionTreeClassifier,

    MLPClassifier,

    KNeighborsClassifier,

    SVR

]



for i in models:

    compML(data,i)