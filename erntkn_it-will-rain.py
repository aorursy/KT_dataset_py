# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
df.head()
df.info()
df["Date"] = pd.to_datetime(df["Date"])
df.drop("RISK_MM",axis=1,inplace = True)
df["RainTomorrow"] = [1 if each == "Yes" else 0 for each in df["RainTomorrow"]]
df["RainToday"] = [1 if each == "Yes" else 0 for each in df["RainToday"]]
df.describe()
cat_cols = []
num_cols = []
other_cols = []

for each in df.columns:
    if df[each].dtype == "object":
        cat_cols.append(each)
    elif df[each].dtype == "float64":
        num_cols.append(each)
    else:
        other_cols.append(each)
print("Categorical Columns: ",cat_cols)
print("Numerical Columns: ",num_cols)
print("Other Columns: ",other_cols)
def ctgplt(variable,to):
    
    "Function for visualization of categorical variables."
    
    var = df[variable]
    values=var.value_counts()
    
    f, ax = plt.subplots(figsize = (8,8))
    g = sns.barplot(x = variable, y = to, data = df)
    g.set_xticklabels(g.get_xticklabels(),rotation = 90)
    plt.show()
    
    print("{}:\n{}".format(variable,values))

def numplt(data,variable,to):
  
  "Function for visualization of numerical variables."

  c = sns.FacetGrid(data,col=to,height=6)
  c.map(sns.distplot,variable,bins=25)
  plt.show()

for i in cat_cols:
    ctgplt(i, "RainTomorrow")
for k in num_cols:
    numplt(df, k, "RainTomorrow")
sns.boxplot(x = df["Rainfall"])
plt.show()
sns.boxplot(x= df["Evaporation"])
plt.show()
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax)
plt.show()
df.drop(columns = ["Temp3pm", "Temp9am", "Pressure9am"], axis=1, inplace = True)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax)
plt.show()
# I removed the columns that i just deleted from dataframe.
to_remove = ("Temp3pm", "Temp9am", "Pressure9am")
num_cols = [each for each in num_cols if each not in to_remove]
Q3 = df["Rainfall"].quantile(0.75)
Q1 = df["Rainfall"].quantile(0.25)

IQR = Q3 - Q1
step = IQR * 3

maxm = Q3 + step
minm = Q1 - step

df = df[df["Rainfall"].fillna(1) < (maxm)]

Q3 = df["Evaporation"].quantile(0.75)
Q1 = df["Evaporation"].quantile(0.25)

IQR = Q3 - Q1
step = IQR * 3

maxm = Q3 + step
minm = Q1 - step

df = df[df["Evaporation"].fillna(1) < (maxm)]
sns.distplot(df["Evaporation"])
plt.show()
df["RainTomorrow"].value_counts()
sns.countplot(x = "RainTomorrow", data=df, palette = "RdBu")
plt.show()
def missing_values_table(data):
        # Total missing values
        mis_val = data.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * data.isnull().sum() / len(data)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_table(df)
for i in cat_cols:
    df[i].fillna(value=df[i].mode()[0],inplace=True)

for k in num_cols:
    df[k].fillna(value=df[k].median(),inplace=True)
df.isnull().sum()
df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Day"] = df["Date"].dt.day

df.drop("Date",axis=1,inplace=True)
df.head()
le = LabelEncoder()
mms = MinMaxScaler()

for each in cat_cols:
    df[each] = le.fit_transform(df[each])

df[df.columns] = mms.fit_transform(df[df.columns])
df.head()
df.describe()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
X = df.drop("RainTomorrow",axis=1)
y = df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
lr = LogisticRegression()

lr.fit(X_train, y_train)
preds = lr.predict(X_test)
print("train_score",lr.score(X_train, y_train))
print("test_score",lr.score(X_test,y_test))
cf_matrix = confusion_matrix(y_test, preds)
sns.heatmap(cf_matrix,annot = True, fmt="g",cmap="Greens")
plt.show()
xgb = XGBClassifier(objective = "binary:logistic")
xgb.fit(X_train,y_train)
pred = xgb.predict(X_test)
print(xgb.score(X_train,y_train))
print(xgb.score(X_test,y_test))
# I will not use a huge parameter grid because it will took to long to train, so here few parameters that could be useful.
params = {
  'min_child_weight':[1,2],
  'max_depth': [3,5],
  'n_estimators':[200,300],
  'colsample_bytree':[0.7,0.8],
  'scale_pos_weight':[1.1,1.2]  
}

model = GridSearchCV(estimator=XGBClassifier(objective="binary:logistic"), param_grid=params, cv=StratifiedKFold(n_splits=5), scoring="f1_macro", n_jobs=-1, verbose=3)
model.fit(X_train, y_train)

print("Best Score: ",model.best_score_)
print("Best Estimator: ",model.best_estimator_)
mat = confusion_matrix(y_test,model.predict(X_test))
sns.heatmap(mat,annot=True,cmap="Greens", fmt="g")
plt.show()
print(classification_report(y_test,model.predict(X_test)))
importances = pd.Series(data=xgb.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize=(8,8))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
from imblearn.over_sampling import SMOTE

method = SMOTE()

X_resampled, y_resampled = method.fit_sample(X_train, y_train)
xgb.fit(X_resampled, y_resampled)
pred1 = xgb.predict(X_test)
print("Train Score: ", xgb.score(X_resampled,y_resampled))
print("Test Score: ", xgb.score(X_test,y_test))
mat = confusion_matrix(y_test,pred1)
sns.heatmap(mat,annot=True,cmap="Greens", fmt="g")
plt.show()
print(classification_report(y_test,pred1))