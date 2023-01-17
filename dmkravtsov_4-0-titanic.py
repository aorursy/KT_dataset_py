# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing libraries



import warnings

warnings.filterwarnings("ignore")

import matplotlib  

import statsmodels.formula.api as smf    

import statsmodels.api as sm  

from sklearn.preprocessing import robust_scale

from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

## for explainer    

from lime import lime_tabular

from mlxtend.preprocessing import minmax_scaling

from scipy import stats

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing.data import QuantileTransformer

from scipy.stats import skew

pd.set_option('display.max_rows', 1000)

## for data

import pandas as pd

import numpy as np

from pylab import rcParams

rcParams['figure.figsize'] = 8, 5

## for plotting

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits = 5, random_state = 2)



from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



## for statistical tests

import scipy

import statsmodels.formula.api as smf

import statsmodels.api as sm

## for machine learning

from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

## for explainer

from lime import lime_tabular

## for machine learning

from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition



from IPython.core.interactiveshell import InteractiveShell 

InteractiveShell.ast_node_interactivity = "all"
df_train_main = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test_main = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test_main.shape
# outlier imputation 

df_train_main.loc[df_train_main['PassengerId'] == 631, 'Age'] = 48
# Outlier detector (make worse for our algo)

from collections import Counter



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



#detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(df_train_main,2,["Age","SibSp","Parch","Fare"])

df_train_main.loc[Outliers_to_drop] # Show the outliers rows

# Drop outliers

df_train_main = df_train_main.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

# # Installing and loading the library

# !pip install dabl



# import dabl
# dabl.plot(df_train_main, target_col="Survived")
# ec = dabl.SimpleClassifier(random_state=0).fit(df_train_main, target_col="Survived") 
df = pd.concat((df_train_main.loc[:,'Pclass':'Embarked'], df_test_main.loc[:,'Pclass':'Embarked']))
#import pandas_profiling 

#profile = df.profile_report(title='Profiling Report')

#profile.to_file(output_file="Data profiling.html")
# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 

# One of the best notebooks on getting started with a ML problem.



# def missing_values_table(df):

#         # Total missing values

#         mis_val = df.isnull().sum()

        

#         # Percentage of missing values

#         mis_val_percent = 100 * df.isnull().sum() / len(df)

        

#         # Make a table with the results

#         mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

#         # Rename the columns

#         mis_val_table_ren_columns = mis_val_table.rename(

#         columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

#         # Sort the table by percentage of missing descending

#         mis_val_table_ren_columns = mis_val_table_ren_columns[

#             mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

#         '% of Total Values', ascending=False).round(1)

        

#         # Print some summary information

#         print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

#             "There are " + str(mis_val_table_ren_columns.shape[0]) +

#               " columns that have missing values.")

        

#         # Return the dataframe with missing information

#         return mis_val_table_ren_columns
# missing= missing_values_table(df)

# missing
# !pip install missingno

# import missingno as msno
# msno.matrix(df)
## some feature engeneering



# NaNs:

# # Handling missing values



# from sklearn.impute import SimpleImputer

# #setting strategy to 'most frequent' to impute by the mean

# imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 

# df.iloc[:,:] = imputer.fit_transform(df)



## Age tuning:

df['Age'] = df.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))

df["Age"] = df["Age"].astype(int)



## Fare tuning:

df['Fare'] = df.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

#df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0) #worse



## Fare Zero tuning:

df['Fare'] = df['Fare'].replace(0, df['Fare'].median())



# new Fare_cat feature:



def fare_category(fr): ## worse

    if fr <= 18:

        return 1

    elif fr <= 33 and fr > 18:

        return 2

    elif fr <= 48 and fr > 33:

        return 3

    elif fr <= 63 and fr > 48:

        return 4

    elif fr <= 78 and fr > 63:

        return 5

    elif fr <= 93 and fr > 78:

        return 6



    return 7









df['Fare_cat'] = df['Fare'].apply(fare_category) 

df["Fare_cat"] = df["Fare_cat"].astype(int)



# Embarked tuning

df["Embarked"] = df["Embarked"].fillna("C")

df["Embarked"][df["Embarked"] == "S"] = 1

df["Embarked"][df["Embarked"] == "C"] = 2

df["Embarked"][df["Embarked"] == "Q"] = 2

df["Embarked"] = df["Embarked"].astype(int)



# New 'familySize' feature & dripping 2 features:

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['FamilySize'] = [1 if i < 5 else 0 for i in df['FamilySize']]



# def family_category(n):

#     if n == 1:

#         return 1

#     elif n == 2 :

#         return 2

#     elif n == 3  :

#         return 3

#     elif n >= 4  :

#         return 3

    

# df['FamilySize_cat'] = df['FamilySize'].apply(family_category) 



df = df.drop(['Parch'], axis=1)

df = df.drop(['SibSp'], axis=1)



# Other

#df = df.drop(['Ticket'], axis=1)



tickets = []

for i in list(df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

df["Ticket"] = tickets



df = pd.get_dummies(df, columns= ["Ticket"], prefix = "T")



# Convert 'Sex' variable to integer form!

df["Sex"][df["Sex"] == "male"] = 0

df["Sex"][df["Sex"] == "female"] = 1

df["Sex"] = df["Sex"].astype(int)



# New Title feature

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

# Bundle rare salutations: 'Other' category

df['Title'] = df['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':4}

# Mapping 'Title' to group

df['Title'] = df['Title'].map(title_category)

##dropping Name feature

df = df.drop(['Name'], axis=1)



# Cabin feature tuning:



# Replace missing values with 'U' for Cabin

df['Cabin'] = df['Cabin'].fillna('U')

import re

# Extract first letter

df['Cabin'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

cabin_category = {'A':9, 'B':8, 'C':7, 'D':6, 'E':5, 'F':4, 'G':3, 'T':2, 'U':1}

# Mapping 'Cabin' to group

df['Cabin'] = df['Cabin'].map(cabin_category)

#df = df.drop(['Cabin'], axis=1) ## worse



df["Pclass"][df["Pclass"] == 3] = 4

df["Pclass"][df["Pclass"] == 2] = 5

df["Pclass"][df["Pclass"] == 1] = 6

df["Pclass"] = df["Pclass"].astype(int)



## new features creation



df['FareCat_Sex'] = df['Fare_cat']*df['Sex']

#df['Fare_Sex'] = df['Fare']*df['Sex']

df['Pcl_Sex'] = df['Pclass']*df['Sex']

#df['Pcl_Age'] = df['Pclass']*df['Age']

#df['Pcl_FareCat'] = df['Pclass']*df['Fare_cat']

df['Pcl_Title'] = df['Pclass']*df['Title']

#df['Age_Sex'] = df['Age']*df['Sex']

# df['FareCat_3'] = df['Fare_cat']**3

# df['FareCat_2'] = df['Fare_cat']**2

# df['FareCat_Sq'] = np.sqrt(df['Fare_cat'])

# df['FareCat_Log'] = np.log1p(df['Fare_cat'])

# df['Age_3'] = df['Age']**3

# df['Age_2'] = df['Age']**2

# df['Age_Sq'] = np.sqrt(df['Age'])

#df['Age_Log'] = - np.log1p(df['Age'])

# df['Cab_Title'] = df['Cabin']*df['Title']

# df['Cab_Pcl'] = df['Cabin']*df['Pclass']

# df['Cab_Emb'] = df['Cabin']*df['Embarked']

# df['Cab_Fare'] = df['Cabin']*df['Fare']





# def age_category(age): ## worse

#     if age <= 19:

#         return 5

#     elif age <= 26 and age > 19:

#         return 4

#     elif age <= 38 and age > 26:

#         return 3

#     elif age <= 57 and age > 38:

#         return 2

#     elif age <= 76 and age > 57:

#         return 1



def age_category(age):

    if age <= 6:

        return 4

    elif age <= 15 and age > 6:

        return 3



    elif age <= 80 and age > 15:

        return 2

#     elif age <= 80 and age > 50:

#         return 1

    return 1

df['Age_cat'] = df['Age'].apply(age_category) 



df['Age_cat_Sex'] = df['Age_cat']*df['Sex']

df['Age_cat_Pclass'] = df['Age_cat']*df['Pclass']





#df = df.drop(['FamilySize'], axis=1)

#df = df.drop(['Age'], axis=1) ##worse

#df = df.drop(['Fare'], axis=1) ##worse

#df['Emb_Sex'] = df['Embarked']*df['Sex'] #worse

#df['Emb_Fare_cat'] = df['Embarked']*df['Fare_cat'] #worse

df['Title_Sex'] = df['Title']*df['Sex']

#df['Fare_cat_Sex'] = df['Sex']*df['Fare_cat']

#df['Age_Fare'] = df['Age_cat']*df['Fare_cat'] #worse

df['Age_Fare'] = df['Age_cat']*df['Fare_cat']

#df['Title_Age_cat'] = df['Title']*df['Age_cat'] #worse

#df['FamilySize_Title'] = df['FamilySize']*df['Title'] #worse



# Create new feature of family size

#df['NotAlone'] = df['FamilySize'].map(lambda s: 0 if s == 1 else 1)

# df['SmallF'] = df['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

# df['MedF'] = df['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

# df['LargeF'] = df['FamilySize'].map(lambda s: 0 if s >= 5 else 1)



df = df.drop(['Age'], axis=1)

# df = df.drop(['FamilySize'], axis=1)

# df = df.drop(['Fare'], axis=1)
#df['Fare'].describe()
#df['Fare'].value_counts(bins=5)
#df[df['Fare']<=105]['Fare'].value_counts(bins=6)
#df[df['Fare']<=18]['Fare'].value_counts(bins=3)
#df['Age'].value_counts(bins=4)
#df[(df['Age']>19) & (df['Age']<=38)]['Age'].value_counts(bins=3)
df.describe(include='all').transpose()
df.info()
df.isnull().sum().sort_values(ascending = False).head()
for column in df.columns:

    print(f"{column}: {df[column].nunique()}")

    if df[column].nunique() < 10:

        print(f"{df[column].value_counts()}")

    print("====================================")


# numerical_features = df.select_dtypes(exclude = ["object"]).columns

# num = df[numerical_features]

# # Log transform of the skewed numerical features to lessen impact of outliers

# # Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

# # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed

# skewness = num.apply(lambda x: skew(x))

# skewness = skewness[abs(skewness) > 0.5]

# print(str(skewness.shape[0]) + " skewed numerical features to log transform")

# skewed_features = skewness.index

# df[skewed_features] = np.log1p(df[skewed_features])



# ## scale for numerical features

# scaler = StandardScaler()

# df[skewed_features] = scaler.fit_transform(df[skewed_features])



#creating matrices for feature selection:

X_train = df[:df_train_main.shape[0]]

X_test_fin = df[df_train_main.shape[0]:]

y = df_train_main.Survived

X_train['Y'] = y

df = X_train

df.head(20) ## DF for Model training
def bar_plot(variable):

    """

        input: variable ex: "Sex"

        output: bar plot & value count

    """

    # get feature

    var = df[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1 = ["Y","Sex","Pclass","Embarked"]

for c in category1:

    bar_plot(c)
def plot_hist(variable):

    # for numerical features

    plt.figure(figsize = (9,3))

    plt.hist(df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare"]

for n in numericVar:

    plot_hist(n)
# Plcass vs Survived

df[["Pclass","Y"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Y",ascending = False)
categorical_val = []

continous_val = []

for column in df.columns:

    print('==============================')

    print(f"{column} : {df[column].unique()}")

    if len(df[column].unique()) <= 10:

        categorical_val.append(column)

    else:

        continous_val.append(column)
# categorical_val.remove('Y')

# df = pd.get_dummies(df, columns = categorical_val)
# ## sample of interactive plots

# #importing plotly and cufflinks in offline mode

# import cufflinks as cf

# import plotly.offline

# cf.go_offline()

# cf.set_config_file(offline=False, world_readable=True)

# df[['Pclass', 'Y']].groupby(['Pclass'], as_index=False).mean().iplot(kind='bar')

#Correlation with output variable

cor = df.corr()

cor_target = (cor['Y'])

#Selecting highly correlated features (8% level)

relevant_features = cor_target[(cor_target<=-0.08) | (cor_target>=0.08) ]

relevant_features.sort_values(ascending = False).head(60)
##feature selection according to its correlation to key feature

features = relevant_features.keys().tolist()

df = df[features]

df.head()


# Import module for dataset splitting

from sklearn.model_selection import train_test_split



X = df.drop('Y', axis=1)

y = df.Y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X = scaler.transform(X)



#Here is out local validation scheme!

df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=1)

## print info    

print("X_train shape:", df_train.drop("Y",axis=1).shape, "| X_test shape:", df_test.drop("Y",axis=1).shape)    

print("y_train mean:", round(np.mean(df_train["Y"]),2), "| y_test mean:", round(np.mean(df_test["Y"]),2))    

print(df_train.shape[1], "features:", df_train.drop("Y",axis=1).columns.to_list())
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) 

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
# ## optimized params below



# from sklearn.model_selection import GridSearchCV

# # set model. max_iter - Maximum number of iterations taken for the solvers to converge.

# lr = LogisticRegression(random_state = 64, max_iter = 1000)



# # set parameters values we are going to check

# optimization_dict = {'class_weight':['balanced', None],

#                      'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

#                      'C': [0.01, 0.05, 0.07, 0.1, 0.5, 1, 2, 4, 5, 10, 15, 20, 50, 100, 200, 500, 1000]

#                      }

# # set GridSearchCV parameters

# GS = GridSearchCV(lr, optimization_dict, 

#                      scoring='accuracy', n_jobs = -1, cv = 10)



# # use training features

# GS.fit(X_train, y_train)



# # print result

# print(GS.best_score_)

# print(GS.best_params_)
from sklearn.linear_model import LogisticRegression

# set best parameters to the model (as per optimized earlier)

lr_tuned_model =  LogisticRegression(solver = 'newton-cg',

                                     C = 200,

                                     random_state = 64,

                                     class_weight = None,

                                     n_jobs = -1)
# train our model with training data

import numpy as np

lr_tuned_model.fit(X_train, y_train)



# calculate importances based on coefficients.

importances = abs(lr_tuned_model.coef_[0])

importances = 100.0 * (importances / importances.max())

# sort 

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [df.columns[i] for i in indices]



# visualize

plt.figure(figsize = (12, 5))

sns.set_style("whitegrid")

chart = sns.barplot(x = names, y = importances[indices])

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.title('Logistic regression. Feature importance')

plt.tight_layout()
## USE W/O SCALER

# # Calculating and Displaying importance using the eli5 library

# import eli5

# from eli5.sklearn import PermutationImportance



# perm = PermutationImportance(lr_tuned_model, random_state=1).fit(X_test,y_test)

# eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# from sklearn.model_selection import GridSearchCV



# # set model

# rf = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 64)

# # create a dictionary of parameters values we want to try

# optimization_dict = {'criterion':['gini', 'entropy'],

#                      'n_estimators': [50,100, 200, 500],

#                      'max_depth': [3, 7, 10, 15],

#                      'min_samples_split': [6, 7, 8],

#                      'min_samples_leaf': [2, 3, 4, 6]

#                      }



# # set GridSearchCV parameters

# GS = GridSearchCV(rf, optimization_dict, 

#                      scoring='accuracy', verbose = 1, n_jobs = -1, cv = 5)



# # use training data

# GS.fit(X_train, y_train)



# # print best score and best parameters combination

# print(GS.best_score_)

# print(GS.best_params_)

# set best parameters to the model

rf_tuned_model =  RandomForestClassifier(criterion = 'gini',

                                       n_estimators = 50,

                                       max_depth = 15,

                                       min_samples_split = 6,

                                       min_samples_leaf = 3,

                                       max_features = 'auto',

                                       oob_score = True,

                                       random_state = 64,

                                       n_jobs = -1)

# train model using training dataset

rf_tuned_model.fit(X_train, y_train)



# Calculate feature importances

importances = rf_tuned_model.feature_importances_



# Visualize Feature Importance

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [df.columns[i] for i in indices]



plt.figure(figsize = (12, 5))

sns.set_style("whitegrid")

chart = sns.barplot(x = names, y=importances[indices])

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.title('Random forest. Feature importance')

plt.tight_layout()


# from sklearn.model_selection import GridSearchCV

# # set model

# xgb_model = XGBClassifier(random_state = 64)

# # create a dictionary of parameters values we want to try

# optimization_dict = {'n_estimators': [1000, 1500, 2000],

#                      'max_depth': [4, 6, 8],

#                      'learning_rate': [0.1, 0.5, 0.9],

#                      'gamma': [1, 5, 7],

#                      'min_child_weight':[5, 7, 9],

#                      'subsample': [0.8, 0.9, 1.0]

#                      }

# # set GridSearchCV parameters

# GS = GridSearchCV(xgb_model, optimization_dict, 

#                      scoring='accuracy', verbose = 1, n_jobs = -1, cv = 5)



# # use training data

# GS.fit(X_train, y_train)

# print(GS.best_score_)

# print(GS.best_params_)

# set model with best parameters

xgb =  XGBClassifier(n_estimators = 1000,

                               max_depth = 6,

                               learning_rate = 0.1,

                               gamma = 1,

                               min_child_weight = 5,

                               subsample = 1.0,

                               random_state = 64)

# train model with training dataset

xgb.fit(X_train, y_train)



# Calculate feature importances

importances = xgb.feature_importances_



# Visualize Feature Importance

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [df.columns[i] for i in indices]



plt.figure(figsize = (12, 5))

sns.set_style("whitegrid")

chart = sns.barplot(x = names, y=importances[indices])

plt.xticks(rotation=45, horizontalalignment='right', fontweight='light')

plt.title('XGBoost. Feature importance')

plt.tight_layout()
# ## call model RandomizedSearchCV

# gb = ensemble.GradientBoostingClassifier()

# ## define hyperparameters combinations to try

# optimization_dict = {'learning_rate':[0.1,0.05,0.01],      #weighting factor for the corrections by new trees when added to the model

# 'n_estimators':[750,1000,1250],  #number of trees added to the model

# 'max_depth':[5,6,7],    #maximum depth of the tree

# 'min_samples_split':[8,10,20],    #sets the minimum number of samples to split

# 'min_samples_leaf':[5,7,9],     #the minimum number of samples to form a leaf

# 'max_features':[4,5,6],     #square root of features is usually a good starting point

# 'subsample':[0.85,0.9,0.95]}       #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.

# ## random search

# RS = model_selection.RandomizedSearchCV(gb, 

#        param_distributions = optimization_dict, n_iter=1000, 

#        scoring="accuracy", verbose = 1, n_jobs = -1, cv = 5).fit(X_train, y_train)

# print("Best Model parameters:", RS.best_params_)

# print("Best Model mean accuracy:", RS.best_score_)

# model = RS.best_estimator_

gb = ensemble.GradientBoostingClassifier(subsample= 0.95, n_estimators= 1000, min_samples_split= 10, min_samples_leaf= 5, max_features= 6, max_depth= 5, learning_rate= 0.1)

# train model with training dataset

gb.fit(X_train, y_train)



# Calculate feature importances

importances = gb.feature_importances_



# Visualize Feature Importance

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [df.columns[i] for i in indices]



plt.figure(figsize = (12, 5))

sns.set_style("whitegrid")

chart = sns.barplot(x = names, y=importances[indices])

plt.xticks(rotation=45, horizontalalignment='right', fontweight='light')

plt.title('GBoost. Feature importance')

plt.tight_layout()
models = []

# add our tuned models into list

models.append(('Logistic Regression', lr_tuned_model))

models.append(('Random Forest', rf_tuned_model))

models.append(('XGBoost', xgb))

models.append(('GBoost', gb))

#models.append(('KNN', sq))





results = []

names = []



# evaluate each model in turn

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state = 64)

    cv_results = model_selection.cross_val_score(model, X, 

                                                 y, 

                                                 cv = 10, scoring = 'accuracy')

    results.append(cv_results)

    names.append(name)

    # print mean accuracy and standard deviation

    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, BatchNormalization

import keras

from keras.optimizers import SGD

import graphviz

from numpy.random import seed



import tensorflow
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
# Initialising the NN

sq = Sequential()



# layers

sq.add(Dense(units = 8, kernel_initializer = 'he_normal', activation = 'relu', input_dim = 18))

sq.add(Dropout(0.50))

# sq.add(BatchNormalization())

sq.add(Dense(units = 32, kernel_initializer = 'he_normal', activation = 'relu'))

sq.add(Dropout(0.50))

# sq.add(BatchNormalization())

sq.add(Dense(units = 64, kernel_initializer = 'he_normal', activation = 'relu'))

sq.add(Dropout(0.50))

sq.add(Dense(units = 1, kernel_initializer = 'he_normal', activation = 'sigmoid'))



#optimizers list

#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']



# Compiling the ANN

sq.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])







# Train the ANN

r = sq.fit(X_train, y_train, batch_size = 9, epochs = 50, verbose=0,  validation_data=(X_test, y_test))

scores = sq.evaluate(X_test, y_test, batch_size=100)

print("%s: %.2f%%" % (sq.metrics_names[1], scores[1]*100))
# def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):

    

#     # set random seed for reproducibility

#     seed(42)

#     tensorflow.random.set_seed(42)

    

#     model = Sequential()

    

#     # create first hidden layer

#     model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    

#     # create additional hidden layers

#     for i in range(1,len(lyrs)):

#         model.add(Dense(lyrs[i], activation=act))

    

#     # add dropout, default is none

#     model.add(Dropout(dr))

    

#     # create output layer

#     model.add(Dense(1, activation='sigmoid'))  # output layer

    

#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    

#     return model
# model = create_model()

# print(model.summary())
# r = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# val_acc = np.mean(r.history['accuracy'])

# print("\n%s: %.2f%%" % ('accuracy', val_acc*100))
plt.figure(figsize=(10, 6))

plt.plot(r.history['loss'], label='loss')

plt.plot(r.history['val_loss'], label='val_loss')

plt.legend()
plt.plot(r.history['accuracy'])

plt.plot(r.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# from keras.wrappers.scikit_learn import KerasClassifier

# from sklearn.model_selection import GridSearchCV

# # create model

# model = KerasClassifier(build_fn=create_model, verbose=0)



# # define the grid search parameters

# batch_size = [16, 32, 64, 128]

# epochs = [25, 50, 100, 150, 200, 250, 300]

# param_grid = dict(batch_size=batch_size, epochs=epochs)



# # search the grid

# grid = GridSearchCV(estimator=model, 

#                     param_grid=param_grid,

#                     cv=3,

#                     verbose=2,

#                    n_jobs=-1)  # include n_jobs=-1 if you are using CPU



# grid_result = grid.fit(X_train, y_train)
# # summarize results

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))
# # create model

# model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=16, verbose=0)



# # define the grid search parameters

# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']

# param_grid = dict(opt=optimizer)



# # search the grid

# grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

# grid_result = grid.fit(X_train, y_train)
# # summarize results

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))
# seed(42)

# tensorflow.random.set_seed(42)



# # create model

# model = KerasClassifier(build_fn=create_model, 

#                         epochs=150, batch_size=16, verbose=0)



# # define the grid search parameters

# layers = [(8),(10),(10,5),(12,6),(12,8,4)]

# param_grid = dict(lyrs=layers)



# # search the grid

# grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

# grid_result = grid.fit(X_train, y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))
# # create model

# model = KerasClassifier(build_fn=create_model, 

#                         epochs=150, batch_size=16, verbose=0)



# # define the grid search parameters

# drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

# param_grid = dict(dr=drops)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

# grid_result = grid.fit(X_train, y_train)
# # summarize results

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))
# # create final model

# model = create_model(lyrs=[10,5], dr=0.0, opt='RMSprop')



# print(model.summary())
# # train model on full train set, with 80/20 CV split

# training = model.fit(X_train, y_train, epochs=150, batch_size=16, 

#                      validation_split=0.2, verbose=0)



# # evaluate the model

# scores = model.evaluate(X_train, y_train)

# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# #save result

# df_test_main['Survived'] = sq.predict_classes(X_test_fin)

# df_test_main[['PassengerId', 'Survived']].to_csv('submission.csv', index = False)

# print("Your submission was successfully saved!")

# #save result for neuro model

# y_pred = sq.predict_classes(X_test_fin)

# output = pd.DataFrame({'PassengerId': df_test_main.PassengerId, 'Survived': y_pred[:, 0]})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")

from keras.wrappers.scikit_learn import KerasClassifier

n_folds = 5

cv_score_lg = cross_val_score(estimator=lr_tuned_model, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)

cv_score_xgb = cross_val_score(estimator=xgb, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)

cv_score_gb = cross_val_score(estimator=gb, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)

cv_score_rf = cross_val_score(estimator=rf_tuned_model, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)

cv_score_sq = cross_val_score(estimator=KerasClassifier(build_fn=sq, batch_size=16, epochs=50, verbose=0),

                                 X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_result = {'lr': cv_score_lg, 'xgb': cv_score_xgb, 'gb': cv_score_gb, 'rf': cv_score_rf, 'sq': cv_score_sq}

cv_data = {model: [score.mean(), score.std()] for model, score in cv_result.items()}

cv_df = pd.DataFrame(cv_data, index=['Mean_accuracy', 'Variance'])

cv_df
fig = plt.figure(figsize=(6,4))

plt.boxplot(results)

plt.title('Algorithm Comparison')

plt.xticks([1,2,3,4], names)

plt.show()
## for best model xgb

xgb.fit(X_train, y_train)

predicted_prob = xgb.predict_proba(X_test)[:,1]

predicted = xgb.predict(X_test)

## Accuracy  AUC

accuracy = metrics.accuracy_score(y_test, predicted)

auc = metrics.roc_auc_score(y_test, predicted_prob)

print("Accuracy (overall correct predictions):",  round(accuracy,2))

print("Auc:", round(auc,2))

    

## Precision e Recall

recall = metrics.recall_score(y_test, predicted)

precision = metrics.precision_score(y_test, predicted)

print("Recall (all 1s predicted right):", round(recall,2))

print("Precision (confidence when predicting a 1):", round(precision,2))

print("Detail:")

print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))



classes = np.unique(y_test)

fig, ax = plt.subplots()

cm = metrics.confusion_matrix(y_test, predicted, labels=classes)

sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)

ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")

ax.set_yticklabels(labels=classes, rotation=0)

plt.show()
cross_val_score(gb, X_test, y_test, cv = kf).mean()
gb = ensemble.GradientBoostingClassifier(n_estimators=1000, learning_rate = 0.5, min_samples_split = 8, min_samples_leaf = 4,   max_features=6, max_depth = 6, random_state = 0)

gb.fit(X_train, y_train)

predicted_prob = gb.predict_proba(X_test)[:,1]

predicted = gb.predict(X_test)

## Accuray e AUC

accuracy = metrics.accuracy_score(y_test, predicted)

auc = metrics.roc_auc_score(y_test, predicted_prob)

print("Accuracy (overall correct predictions):",  round(accuracy,2))

print("Auc:", round(auc,2))

    

## Precision e Recall

recall = metrics.recall_score(y_test, predicted)

precision = metrics.precision_score(y_test, predicted)

print("Recall (all 1s predicted right):", round(recall,2))

print("Precision (confidence when predicting a 1):", round(precision,2))

print("Detail:")

print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))
cross_val_score(gb, X_test, y_test, cv = kf).mean()