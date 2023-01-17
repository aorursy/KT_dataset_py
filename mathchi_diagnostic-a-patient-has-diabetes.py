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



import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning) 

warnings.filterwarnings("ignore", category=UserWarning) 



%config InlineBackend.figure_format = 'retina'



# to display all columns and rows:

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);  # to display all columns and rows

pd.set_option('display.float_format', lambda x: '%.2f' % x) # The number of numbers that will be shown after the comma.

#Reading the dataset

df = pd.read_csv("../input/diabetes-data-set/diabetes.csv")
df.shape
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
df["Outcome"].value_counts()*100/len(df)
df.Outcome.value_counts()
df["Age"].hist(edgecolor = "black");
df.isnull().sum()
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
df.isnull().sum()
import missingno as msno

msno.bar(df);
def carp(x,y):

    

    z = x*y

    

    return z

carp(4,5)
# The missing values will be filled with the median values of each variable.



def median_target(var):   

    

    temp = df[df[var].notnull()]

    

    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()

    

    return temp
columns = df.columns



columns = columns.drop("Outcome")
columns
median_target('Glucose')
# The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick.



columns = df.columns



columns = columns.drop("Outcome")



for col in columns:

    

    df.loc[(df['Outcome'] == 0 ) & (df[col].isnull()), col] = median_target(col)[col][0]

    df.loc[(df['Outcome'] == 1 ) & (df[col].isnull()), col] = median_target(col)[col][1]
df.loc[(df['Outcome'] == 0 ) & (df["Pregnancies"].isnull()), "Pregnancies"]
df[(df['Outcome'] == 0 ) & (df["BloodPressure"].isnull())]
Q1 = df["BloodPressure"].quantile(0.25)

Q3 = df["BloodPressure"].quantile(0.75)

IQR = Q3-Q1

lower = Q1 - 1.5*IQR

upper = Q3 + 1.5*IQR
lower
upper
df[(df["BloodPressure"] > upper)].any(axis=None)
for feature in df:

    print(feature)
for feature in df:

    

    Q1 = df[feature].quantile(0.05)

    Q3 = df[feature].quantile(0.95)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR

    

    if df[(df[feature] > upper)].any(axis=None):

        print(feature,"yes")

    else:

        print(feature, "no")
df.head()
df.shape
# According to BMI, some ranges were determined and categorical variables were assigned.

NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")



df["NewBMI"] = NewBMI



df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]



df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]

df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]

df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]

df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]

df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]
df.head()
def set_insulin(row):

    if row["Insulin"] >= 16 and row["Insulin"] <= 166:

        return "Normal"

    else:

        return "Abnormal"     
df.head()
df["NewInsulinScore"] = df.apply(set_insulin, axis=1)
df.head()
#df.drop("NewInsulinScore", inplace = True, axis = 1)

#df.head()
# Some intervals were determined according to the glucose variable and these were assigned categorical variables.

NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")



df["NewGlucose"] = NewGlucose



df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]



df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]



df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]



df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]
df.head()
df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)
df.head()
categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',

                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]
y = df["Outcome"]

X = df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',

                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)

cols = X.columns

index = X.index
y.head()
X.head()
cols
index
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(X)

X = transformer.transform(X)

X = pd.DataFrame(X, columns = cols, index = index)
X.head()
X = pd.concat([X, categorical_df], axis = 1)
X.head()
models = []

models.append(('LR', LogisticRegression(random_state = 12345)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier(random_state = 12345)))

models.append(('RF', RandomForestClassifier(random_state = 12345)))

models.append(('SVM', SVC(gamma='auto', random_state = 12345)))

models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))

models.append(("LightGBM", LGBMClassifier(random_state = 12345)))



# evaluate each model in turn

results = []

names = []



for name, model in models:

    

        kfold = KFold(n_splits = 10, random_state = 12345)

        

        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

# boxplot algorithm comparison

fig = plt.figure(figsize=(15,10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()



rf_params = {"n_estimators" :[100,200,500,1000], 

             "max_features": [3,5,7], 

             "min_samples_split": [2,5,10,30],

            "max_depth": [3,5,8,None]}



rf_model = RandomForestClassifier(random_state = 12345)
gs_cv = GridSearchCV(rf_model, 

                    rf_params,

                    cv = 10,

                    n_jobs = -1,

                    verbose = 2).fit(X, y)

gs_cv.best_params_
rf_tuned = RandomForestClassifier(**gs_cv.best_params_)
rf_tuned = rf_tuned.fit(X,y)
cross_val_score(rf_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(rf_tuned.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Significance Score Of Variables')

plt.ylabel('Variables')

plt.title("Variable Severity Levels")

plt.show()
lgbm = LGBMClassifier(random_state = 12345)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],

              "n_estimators": [500, 1000, 1500],

              "max_depth":[3,5,8]}

gs_cv = GridSearchCV(lgbm, 

                     lgbm_params, 

                     cv = 10, 

                     n_jobs = -1, 

                     verbose = 2).fit(X, y)
gs_cv.best_params_
lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X,y)
cross_val_score(lgbm_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(lgbm_tuned.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Significance Score Of Variables')

plt.ylabel('Variables')

plt.title("Variable Severity Levels")

plt.show()
xgb = GradientBoostingClassifier(random_state = 12345)
xgb_params = {

    "learning_rate": [0.01, 0.1, 0.2, 1],

    "min_samples_split": np.linspace(0.1, 0.5, 10),

    "max_depth":[3,5,8],

    "subsample":[0.5, 0.9, 1.0],

    "n_estimators": [100,1000]}
xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X, y)
xgb_cv_model.best_params_
xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X,y)
cross_val_score(xgb_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(xgb_tuned.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Significance Score Of Variables')

plt.ylabel('Variables')

plt.title("Variable Severity Levels")

plt.show()
models = []



models.append(('RF', RandomForestClassifier(random_state = 12345, max_depth = 8, max_features = 7, min_samples_split = 2, n_estimators = 500)))

models.append(('XGB', GradientBoostingClassifier(random_state = 12345, learning_rate = 0.1, max_depth = 5, min_samples_split = 0.1, n_estimators = 100, subsample = 1.0)))

models.append(("LightGBM", LGBMClassifier(random_state = 12345, learning_rate = 0.01,  max_depth = 3, n_estimators = 1000)))



# evaluate each model in turn

results = []

names = []
for name, model in models:

    

        kfold = KFold(n_splits = 10, random_state = 12345)

        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

# boxplot algorithm comparison

fig = plt.figure(figsize=(15,10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()