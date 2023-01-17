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



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

pd.set_option("display.max_columns", None)
train = pd.read_csv("../input/summeranalytics2020/train.csv", index_col= "Id")

test = pd.read_csv("../input/summeranalytics2020/test.csv")

train.head()
train.isnull().sum()
train.duplicated().sum()
train.drop_duplicates(inplace= True)
train.dtypes
train.describe().T
train.nunique()
train.drop("Behaviour", axis= "columns", inplace= True)

test.drop("Behaviour", axis= "columns", inplace= True)
train.shape
sns.countplot(x= "Attrition", data= train)
train.columns
nominal_variables = train.select_dtypes(include= "object").columns.to_list()

continuous_variables = ['Age','DistanceFromHome','EmployeeNumber','MonthlyIncome','NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

ordinal_variables = ['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','Education','CommunicationSkill','PerformanceRating','StockOptionLevel']



print (len(nominal_variables), len(continuous_variables), len(ordinal_variables))

len(nominal_variables + continuous_variables + ordinal_variables)
fig, ax = plt.subplots(4, 3, figsize= (15, 15))

for feature, ax in zip(continuous_variables, fig.axes):

    sns.distplot(train[feature], ax= ax)
plt.figure(figsize= (10, 10))

sns.heatmap(train[continuous_variables].corr(), vmin= -1, vmax= 1, annot= True, square= True, cmap= "viridis")
def cont_cols_plot(feature):

    fig, ax = plt.subplots(1, 3, figsize= (15, 4))

    sns.distplot(train[train.Attrition == 0][feature], color = "blue", ax= ax[0])

    sns.distplot(train[train.Attrition == 1][feature], color = "darkred", ax= ax[0])

    

    sns.boxplot(x= "Attrition", y= feature, data= train, ax= ax[1])

    sns.pointplot(x= "Attrition", y= feature, data= train, ax= ax[2])

    

    plt.show()
for i in continuous_variables:

    cont_cols_plot(i)

    print ("-"*50)

from scipy.stats import ttest_ind



ttest_pvalue = pd.DataFrame(index= continuous_variables, columns= ["p_value"])



for feature in continuous_variables:

    ttest = ttest_ind(train[train.Attrition ==0][feature], train[train.Attrition == 1][feature], equal_var= False)

    ttest_pvalue.loc[feature, "p_value"] = ttest[1]

    

ttest_pvalue["p_value < 0.05"] = ttest_pvalue.apply(lambda x: x < 0.05)

ttest_pvalue
import seaborn as sns

plt.figure(figsize= (8, 8))

sns.heatmap(train[ordinal_variables].corr(method= "kendall"), vmin= -1, vmax= 1, square= True, cmap= "viridis", annot= True) 
train[ordinal_variables].nunique()
fig, ax = plt.subplots(2, 4, figsize= (18, 8))

for feature, ax in zip(ordinal_variables, fig.axes):

    sns.countplot(x= feature, data= train, ax= ax)
def ordinal_cols_plot(feature):

    fig, ax = plt.subplots(1, 2, figsize = (10, 4))

    sns.countplot(x= feature, hue= "Attrition", data= train, ax= ax[0])

    

    cross = pd.crosstab(index= train[feature], columns= train.Attrition, normalize= "index")

    sns.pointplot(x= cross.index, y= cross[1]*100, ax= ax[1])

    plt.ylabel("Attrition Percentage")

    

    print (cross)

    plt.show()
for i in ordinal_variables:

    ordinal_cols_plot(i)

    print ("--"*50)
train[nominal_variables].nunique()
fig, ax = plt.subplots(2, 4, figsize= (18, 8))

for feature, ax in zip(nominal_variables, fig.axes):

    sns.countplot(x= feature, data= train, ax= ax, order= train[feature].value_counts().index)
def nominal_cols_plot(feature):

    fig, ax = plt.subplots(1, 2, figsize = (10, 4))

    sns.countplot(x= feature, hue= "Attrition", data= train, ax= ax[0], order = train[feature].value_counts().index)

    

    cross = pd.crosstab(index= train[feature], columns= train.Attrition, normalize= "index")

    sns.pointplot(x= cross.index, y= cross[1]*100, ax= ax[1], order = train[feature].value_counts().index)

    plt.ylabel("Attrition Percentage")

    

    print (cross)

    plt.show()
for i in nominal_variables:

    nominal_cols_plot(i)

    print ("-"*50)
fig, ax = plt.subplots(2, 1, figsize = (10, 9))

sns.countplot(x= "EducationField", hue= "Attrition", data= train, ax= ax[0], order= train.EducationField.value_counts().index)

    

cross = pd.crosstab(index= train.EducationField, columns= train.Attrition, normalize= "index")

sns.pointplot(x= cross.index, y= cross[1]*100, ax= ax[1], order= train.EducationField.value_counts().index)

plt.ylabel("Attrition Percentage")



print (cross)

plt.show()
fig, ax = plt.subplots(2, 1, figsize = (18, 9))

sns.countplot(x= "JobRole", hue= "Attrition", data= train, ax= ax[0], order = train.JobRole.value_counts().index)

    

cross = pd.crosstab(index= train.JobRole, columns= train.Attrition, normalize= "index")

sns.pointplot(x= cross.index, y= cross[1]*100, ax= ax[1], order = train.JobRole.value_counts().index)

plt.ylabel("Attrition Percentage")

plt.xticks(rotation= 90)



print (cross)

plt.show()
a = ordinal_variables + nominal_variables

from scipy.stats import chi2_contingency



chi2_pvalue = pd.DataFrame(index= a, columns= ["p_value"])

for feature in a:

    cross = pd.crosstab(index= train[feature], columns= train.Attrition)

    chi2 = chi2_contingency(cross)

    

    chi2_pvalue.loc[feature, "p_value"] = chi2[1]



chi2_pvalue["p_value < 0.05"] = chi2_pvalue.apply(lambda x: x < 0.05)   

chi2_pvalue    
features_to_drop = ["EmployeeNumber", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", 

                    "YearsInCurrentRole", "YearsWithCurrManager", "YearsSinceLastPromotion", 

                    "Education", "CommunicationSkill", "PerformanceRating", "Gender"]

len(features_to_drop)
train_dropped = train.drop(features_to_drop, axis= "columns").copy()

test_dropped = test.drop(features_to_drop, axis= "columns").copy()
Id = test_dropped.pop("Id")

y = train_dropped.pop("Attrition")
train_dropped.columns
nominal_cols = train_dropped.select_dtypes(include= "object").columns.tolist()

ordinal_cols = [i for i in ordinal_variables if i in train_dropped.columns]

nominal_cols + ordinal_cols
cat_index = [train_dropped.columns.get_loc(i) for i in nominal_cols + ordinal_cols]

cat_index
train_dropped[nominal_cols].head()
le = LabelEncoder()



for i in nominal_cols:

    train_dropped[i] = le.fit_transform(train_dropped[i])

    print (le.classes_)

    test_dropped[i] = le.transform(test_dropped[i])

    
train_dropped[nominal_cols].head()
y.value_counts()
from imblearn.over_sampling import SMOTENC
sm = SMOTENC(categorical_features = np.asarray(cat_index), random_state= 5)

X_train_res, y_train_res = sm.fit_sample(train_dropped, y)

y_train_res.value_counts()
X_train_res.shape
skf= StratifiedKFold(n_splits=5, shuffle= False, random_state= 5)
def model_accuracy(model, param_grid):

    classifier = Pipeline(steps=[

        ("scaler", StandardScaler()),

        ("model", model)

    ])

    grid= GridSearchCV(estimator= classifier, param_grid= param_grid, cv= skf, scoring= "roc_auc")

    grid.fit(X_train_res, y_train_res)

    

    print ("Best_score: ", grid.best_score_, "\n", "Best_params: ", grid.best_params_)

    return (pd.DataFrame(grid.cv_results_)

            .sort_values(["rank_test_score"])[["params", "mean_test_score", "std_test_score"]].head(10).T)
model_accuracy(SVC(), param_grid= { "model__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], "model__gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]})
model_accuracy(LogisticRegression(penalty= "l1", solver= "liblinear"), param_grid= {"model__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]})
final_svm = Pipeline(steps= [

    ("scaler", StandardScaler()),

    ("model", SVC(C= 5, gamma= 0.1, probability= True))

])



final_svm.fit(X_train_res, y_train_res)
predictions = final_svm.predict_proba(test_dropped)[:, 1]

predictions[:5]
output = pd.DataFrame({"Id": Id,

                       "Attrition": predictions})

output.head()
output.to_csv("submission.csv", index= False)