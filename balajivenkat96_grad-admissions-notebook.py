# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
admissions_data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv", index_col = "Serial No.")
admissions_data.head()
([any(pd.isnull(admissions_data[c])) for c in admissions_data.columns])
sns.heatmap(admissions_data.drop("Chance of Admit ", axis = 1).corr(), annot = True)
sns.heatmap(admissions_data.corr(),annot = True) #With the target variable included
#sns.scatterplot(x = admissions_data["GRE Score"], y = admissions_data["CGPA"])

#plt.figure(8,6)

sns.lmplot(data = admissions_data, x = "GRE Score", y = "CGPA")
sns.lmplot(data = admissions_data,x = "TOEFL Score", y = "SOP")
sns.distplot(admissions_data.loc[(admissions_data["Research"] == 1) & (admissions_data["University Rating"] == 5),"Chance of Admit "])

plt.title("University Rating 5, With research")
sns.distplot(admissions_data.loc[(admissions_data["Research"] == 0) & (admissions_data["University Rating"] == 5),"Chance of Admit "])

plt.title("University rating 5, without research")
sns.distplot(admissions_data.loc[(admissions_data["Research"] == 1) & (admissions_data["University Rating"] == 1),"Chance of Admit "])

plt.title("University rating 1, with research")
sns.distplot(admissions_data.loc[(admissions_data["Research"] == 0) & (admissions_data["University Rating"] == 1),"Chance of Admit "])

plt.title("University rating 1, without research")
sns.lmplot(data = admissions_data.loc[admissions_data["University Rating"] == 5], x = "CGPA", y = "Chance of Admit ")#y = admissions_data.loc[admissions_data["University Rating"] == 5, "Chance of Admit "])

plt.title("University rating = 5")


sns.lmplot(data = admissions_data.loc[(admissions_data["University Rating"] == 1) | (admissions_data["University Rating"] == 5)], x = "CGPA", y = "Chance of Admit ", hue = "University Rating")
admissions_data.head()
p = sns.lmplot(data = admissions_data.loc[admissions_data["University Rating"] >= 4] ,x = "CGPA", y = "Chance of Admit ", hue = "Research")

plt.title("University rating 4 and 5")
#With research

from sklearn.linear_model import LinearRegression

with_research = LinearRegression()

with_research.fit(admissions_data.loc[(admissions_data["University Rating"] >= 4) & (admissions_data["Research"] == 1), "CGPA"].values.reshape(-1,1),admissions_data.loc[(admissions_data["University Rating"] >= 4) & (admissions_data["Research"] == 1), "Chance of Admit "])

with_research.coef_
#Without research

without_research = LinearRegression()

without_research.fit(admissions_data.loc[(admissions_data["University Rating"] >= 4) & (admissions_data["Research"] == 0), "CGPA"].values.reshape(-1,1),admissions_data.loc[(admissions_data["University Rating"] >= 4) & (admissions_data["Research"] == 0), "Chance of Admit "])

without_research.coef_
p = sns.lmplot(data = admissions_data.loc[(admissions_data["University Rating"] >= 4) & ((admissions_data["LOR "] == 5) | (admissions_data["LOR "] < 3))] ,x = "CGPA", y = "Chance of Admit ", hue = "LOR ")

plt.title("University rating 4 and 5")
sns.regplot(data = admissions_data.loc[(admissions_data["University Rating"] >= 4) & (admissions_data["SOP"] == 5)], x = "CGPA", y = "Chance of Admit ", label = "SOP = 5")

sns.regplot(data = admissions_data.loc[(admissions_data["University Rating"] >= 4) & (admissions_data["SOP"] == 3)], x = "CGPA", y = "Chance of Admit ", label = "SOP = 3")

plt.legend()

plt.title("University rating 4 and 5")
sns.lmplot(data = admissions_data.loc[(admissions_data["University Rating"] >= 4)], x = "LOR ", y = "SOP")

plt.title("University rating 4 and 5")
sns.distplot(admissions_data.loc[admissions_data["University Rating"] == 5, "Chance of Admit "])

plt.title("University rating = 5")
sns.distplot(admissions_data.loc[admissions_data["University Rating"] == 4, "Chance of Admit "])

plt.title("University rating = 4")
sns.distplot(admissions_data.loc[admissions_data["University Rating"] == 3, "Chance of Admit "])

plt.title("University rating = 3")
sns.distplot(admissions_data.loc[admissions_data["University Rating"] == 2, "Chance of Admit "])

plt.title("University rating = 2")
sns.distplot(admissions_data.loc[admissions_data["University Rating"] == 1, "Chance of Admit "])

plt.title("University rating = 1")
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Ridge

scale = StandardScaler()

preprocessor = ColumnTransformer(transformers = [('standard_scaler',scale,["GRE Score","TOEFL Score","CGPA"])],remainder = "passthrough")

lr_model = LinearRegression()

lr_pipeline = Pipeline([("preprocess",preprocessor),("model",lr_model)])

admissions_data.columns
from sklearn.model_selection import train_test_split
X = admissions_data.copy()

y = X["Chance of Admit "]

X.drop("Chance of Admit ",axis = 1, inplace = True)
X_train,X_val,y_train,y_val = train_test_split(X,y,train_size = 0.8)
lr_pipeline.fit(X_train,y_train)
preds = lr_pipeline.predict(X_val)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_val,preds)
lr_model.coef_,lr_model.intercept_
X_train_processed = pd.DataFrame(preprocessor.transform(X_train))

X_train_processed.head()

#X_train_processed.head()

X_train_processed.columns = ["GRE Score","TOEFL Score","CGPA","University Rating", "SOP", "LOR","Research"]
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 100)
from sklearn.model_selection import GridSearchCV
#param_grid = {"rf_model__n_estimators":np.arange(100,1100,100)}

rf_pipeline = Pipeline([("preprocess",preprocessor),("model",rf_model)])
rf_search = GridSearchCV(rf_pipeline,param_grid = {"model__n_estimators":np.arange(100,1100,100)},n_jobs = -1)
rf_search.fit(X_train,y_train)
rf_search.best_params_
rf_search.best_score_
rf_model = RandomForestRegressor(n_estimators = rf_search.best_params_["model__n_estimators"])
rf_pipeline = Pipeline([("preprocess",preprocessor),("model",rf_model)])

rf_pipeline.fit(X_train,y_train)
mean_absolute_error(rf_pipeline.predict(X_val),y_val)