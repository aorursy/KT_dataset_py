import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pd.set_option("display.max_columns", 100)
%matplotlib inline

import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.dtypes
df.SeniorCitizen.describe()
df.SeniorCitizen.replace([0, 1], ["No", "Yes"], inplace= True)
df.TotalCharges.describe()
df.TotalCharges.unique()
for charge in df.TotalCharges:
    try:
        charge = float(charge)
    except:
        print("charge:", charge, "length", len(charge))
charges = [float(charge) if charge != " " else np.nan for charge in df.TotalCharges]
df.TotalCharges = charges
df.describe()
df.describe(include=object)
df.hist(figsize=(15, 5), layout=(1, 3))
plt.show()
sns.countplot(y="MultipleLines", data= df)
plt.show()
sns.barplot(y="MultipleLines", x="TotalCharges", data= df)
plt.show()
sns.barplot(y="MultipleLines", x="MonthlyCharges", data= df)
plt.show()
sns.lmplot("MonthlyCharges", "TotalCharges", hue="InternetService", data= df, fit_reg= False)
sns.lmplot("MonthlyCharges", "TotalCharges", hue="Contract", data= df, fit_reg= False)
sns.lmplot("MonthlyCharges", "MonthlyCharges", hue="InternetService", data= df, fit_reg= False)
sns.lmplot("tenure", "TotalCharges", data= df, hue="Churn", fit_reg= False)
for col in df.dtypes[df.dtypes == object].index:
    print(col, df[col].unique())
df["ProtectedCustomer"] = ["Yes" if df.OnlineBackup[i]=="Yes" and df.OnlineSecurity[i]=="Yes" else "No" for i in range(len(df))]
df["StreamerCustomer"] = ["Yes" if df.StreamingMovies[i]=="Yes" and df.StreamingTV[i]=="Yes" else "No" for i in range(len(df))]
df["FamilyCustomer"] = ["Yes" if df.Partner[i]=="Yes" or df.Dependents[i]=="Yes" else "No" for i in range(len(df))]
df["OldFashioned"] = ["Yes" if df.PaperlessBilling[i]=="No" and df.PaymentMethod[i]=="Mailed check" \
                      else "No" for i in range(len(df))]
df["PowerUser"] = ["Yes" if df.ProtectedCustomer[i]=="Yes" and df.StreamerCustomer[i]=="Yes" \
                   and df.DeviceProtection[i]=="Yes" and df.TechSupport[i]=="Yes" else "No" for i in range(len(df))]
df["FamilyMultiple"] = ["Yes" if df.FamilyCustomer[i]=="Yes" and df.MultipleLines[i]=="Yes" else "No" for i in range(len(df))]
df.describe()
df["FullCharges"] = df.tenure * df.MonthlyCharges
df["Discount"] = df.FullCharges - df.TotalCharges
df.head()
df.isna().sum()
df.TotalCharges.fillna(df.TotalCharges.median(), inplace= True)
df.Discount.fillna(df.Discount.median(), inplace= True)
df.to_csv("cleaned.csv", index= False)
df = pd.read_csv("cleaned.csv")
df.shape
df.head()
df.drop("customerID", axis= 1, inplace= True)
df.Churn.replace(["Yes", "No"], [1, 0], inplace= True)
df = pd.get_dummies(df)
X = df.drop("Churn", axis= 1)
y = df.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)
print( len(X_train), len(X_test), len(y_train), len(y_test) )
X_train.shape, y_train.shape
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=123)),
    "nb": make_pipeline(StandardScaler(), GaussianNB()),
    "kn": make_pipeline(StandardScaler(), KNeighborsClassifier())
}
pipelines["nb"].get_params()
rf_hyperparameters = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingclassifier__n_estimators": [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}
kn_hyperparameters = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 10]
}
nb_hyperparameters = {
    'gaussiannb__priors': [None]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters,
                   "nb": nb_hyperparameters,
                   "kn": kn_hyperparameters}
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('ACC:', accuracy_score(y_test, pred))
    print("ROC:", roc_auc_score(y_test, pred))
    print("CoM:\n", confusion_matrix(y_test, pred))
with open('final_model_churn.pkl', 'wb') as f:
    pickle.dump(fitted_models['gb'].best_estimator_, f)
df = pd.read_csv("cleaned.csv")
df.shape
df.head()
df.drop("customerID", axis= 1, inplace= True)
for col in df.dtypes[df.dtypes==object].index:
    if set(df[col].unique().tolist()) == set(["Yes", "No"]):
        df[col].replace(["Yes", "No"], [1, 0], inplace= True)
df.head()
df.gender.replace(["Male", "Female"], [1, 0], inplace=True)
df = pd.get_dummies(df)
X = df.drop(["TotalCharges", "FullCharges"], axis= 1)
y = df.TotalCharges
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)
print( len(X_train), len(X_test), len(y_train), len(y_test) )
X_train.shape, y_train.shape
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}
rf_hyperparameters = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingregressor__n_estimators": [100, 200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    
print("\nMean:", np.mean(y_test))
plt.scatter(y, fitted_models["rf"].predict(X))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()
with open('final_model_total_price.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)
df = pd.read_csv("cleaned.csv")
df.shape
df.head()
df.drop(["customerID", "FullCharges", "Discount", "TotalCharges"], axis= 1, inplace= True)
for col in df.dtypes[df.dtypes==object].index:
    if set(df[col].unique().tolist()) == set(["Yes", "No"]):
        df[col].replace(["Yes", "No"], [1, 0], inplace= True)
df.head()
df.gender.replace(["Male", "Female"], [1, 0], inplace=True)
df = pd.get_dummies(df)
X = df.drop("MonthlyCharges", axis= 1)
y = df.MonthlyCharges
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)
print( len(X_train), len(X_test), len(y_train), len(y_test) )
X_train.shape, y_train.shape
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}
rf_hyperparameters = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingregressor__n_estimators": [100, 200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    
print("\nMean:", np.mean(y_test))
plt.scatter(y, fitted_models["rf"].predict(X))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()
with open('final_model_monthly_price.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)