import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("../input/Admission_Predict.csv")
df1 = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df = df.merge(df1, how = "outer") ## To get a universal single dataframe
df = df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
df.shape
df.head(20).T
plt.figure(figsize = [10, 5]) ## Plotting the chance of admit to cgpa
sns.scatterplot(df["Chance of Admit"], df["CGPA"])
plt.show()
plt.figure(figsize = [10, 5]) ## Plotting the cgpa to gre scores
sns.scatterplot(df["CGPA"], df["GRE Score"])
plt.show()
## Feature Engineering extra parameters
df["Ratio_CGPA_GRE"] = (df["CGPA"] / df["GRE Score"]) * 100
df["Ratio_CGPA_TOEFL"] = (df["CGPA"] / df["TOEFL Score"]) * 100
df["Ratio_CGPA_GRE"].head()
df["Ratio_CGPA_TOEFL"].head()
## Dropping the serial number column
df.drop("Serial No.", inplace = True, axis = 1)

## Checking the correlation matrix
plt.figure(figsize = [10, 5])
sns.heatmap(df.corr(), linewidths = 0.2, annot = True)
plt.show()
y = df["Chance of Admit"] ## Get the dependent variable
df.drop("Chance of Admit", axis = 1, inplace = True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
## Cross - validate the dataset
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 42)
forest_params = {
    "n_estimators" : [50, 100, 150, 200],
    "min_samples_leaf" : [5, 10, 20, 30],
    "max_features" : [1, 2, 3, 5]
}
rfr = RandomForestRegressor(oob_score = True)
opti_rfr = GridSearchCV(rfr, forest_params, cv = 5, n_jobs = -1)
opti_rfr.fit(X_train, y_train)
print(opti_rfr.best_estimator_)
print(opti_rfr.best_score_)
print("Test MAE: ", mean_absolute_error(opti_rfr.predict(X_test), y_test))
print("Train MAE: ", mean_absolute_error(opti_rfr.predict(X_train), y_train))
print("Test Score: ", opti_rfr.score(X_test, y_test))
print("Train Score: ", opti_rfr.score(X_train, y_train))
feat_importances = pd.Series(opti_rfr.best_estimator_.feature_importances_, index = df.columns)
feat_importances.nlargest(20).plot(kind='barh')
