# installation of libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

%config InlineBackend.figure_format = 'retina'

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

df = pd.read_csv("../input/predicting-churn-for-bank-customers/Churn_Modelling.csv", index_col=0)
df.head()
df.shape
# dataframe's index dtype and column dtypes, non-null values and memory usage information
df.info()
# explanatory statistics values of the observation units corresponding to the specified percentages
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
# transposition of the df table was taken to facilitate the evaluation
# seeing the distribution of age of people who have an account in the bank
sns.distplot(df.Age, bins = 10)
plt.show()
sns.distplot(df.Balance, bins = 10)
plt.show()
sns.boxplot(data = df, x= 'Geography', y = 'Age')
plt.show()
df.groupby("Gender")['Gender'].count()
sns.barplot(x="Geography", y="Exited", hue = 'Gender', data=df)
plt.show()
df['Exited'].value_counts() # mostly not exited customers
sns.countplot(df['Exited'], palette='Set1')
plt.title('Counts of Two Types of Customers')
# as expected, most customers did not churn
f, ax = plt.subplots(1,1, figsize=(8,8))

colors = ["darkturquoise", "red"]
labels ="Did not exit", "Exit"

plt.suptitle('Information on Customer Churn', fontsize=20)

df["Exited"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax, shadow=True, colors=colors, labels=labels, fontsize=12, startangle=25)
#Create figure
f, ax = plt.subplots(figsize = (12,12))

#Create and plot correlation matrix
corr = df.corr()
sns.heatmap(corr, ax=ax, linewidths= 1, linecolor='white',annot = True, cmap = 'coolwarm',center = 0);
# create figure
f, ax = plt.subplots(figsize = (10,7))

# plot target
df.groupby(['Geography','Gender'])['Exited'].agg({'count','sum'}).plot(kind = 'bar', ax = ax, color = ['orange', 'grey'])

cats = ['France\nWomen','France\nMen','Germany\nWomen','Germany\nMen','Spain\nWomen','Spain\nMen']
# ax.set_xticks(cats)
ax.set_xticklabels(cats)
plt.xticks(rotation=0)

# set plot aesthetics
ax.set_title('Churn Distributions', style = 'italic')
ax.set_xlabel('')
ax.set_ylabel('Count', style = 'italic')
ax.legend(['Exited', 'Stayed'], shadow = True, frameon = True)
ax.grid(axis = 'x',b=False)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# no missing data in the data set
df.isnull().sum()
# Outlier Observation Analysis
for feature in df[['CreditScore','Tenure', 'Balance','EstimatedSalary']]:
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")
df["NewAGT"] = df["Age"] - df["Tenure"]
df["New_CreditsScore"] = pd.qcut(df['CreditScore'], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df["AgeScore"] = pd.qcut(df['Age'], 8, labels = [1, 2, 3, 4, 5, 6, 7, 8])
df["BalanceScore"] = pd.qcut(df['Balance'].rank(method="first"), 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df["EstSalaryScore"] = pd.qcut(df['EstimatedSalary'], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df["NewEstimatedSalary"] = df["EstimatedSalary"] / 12 
df.head()
df = pd.get_dummies(df, columns =["Geography", "Gender"], drop_first = True)
df.head()
df = df.drop(["CustomerId","Surname"], axis = 1)
df.head()
cat_df = df[["Geography_Germany", "Geography_Spain", "Gender_Male", "HasCrCard","IsActiveMember"]]
cat_df.head()
y = df["Exited"]
X = df.drop(["Exited","Geography_Germany", "Geography_Spain", "Gender_Male", "HasCrCard","IsActiveMember"], axis = 1)
cols = X.columns
index = X.index
X.head()    
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)
X = pd.concat([X,cat_df], axis = 1)
X.head()
# Splitting the dataset into Training and Testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
models = []
models.append(('LR', LogisticRegression(random_state = 12345)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 12345)))
models.append(('RF', RandomForestClassifier(random_state = 12345)))
models.append(('SVM', SVC(gamma='auto', random_state = 12345)))
models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))
models.append(("LightGBM", LGBMClassifier(random_state = 12345)))
models.append(("CatBoost", CatBoostClassifier(random_state = 12345, verbose = False)))

# evaluate each model in turn
results = []
names = []

for name, model in models:
        
        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
# comparison of algorithms with boxplot
fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results,
            vert=True, # vertical box alignment
            patch_artist=True) # fill with color
                         
ax.set_xticklabels(names)
plt.show()
rf_params = {"n_estimators" :[100,200], 
             "max_features": [3,5], 
            "max_depth": [3,5]}
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

sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Severity Levels")
plt.show()
xgb = GradientBoostingClassifier(random_state = 12345)
xgb_params = {
    "learning_rate": [0.01, 0.1, 1],
    "max_depth":[3,5],
    "subsample":[0.5, 0.9],
    "n_estimators": [100,200]}
xgb_cv = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X, y)
xgb_cv.best_params_
xgb_tuned = GradientBoostingClassifier(**xgb_cv.best_params_).fit(X,y)
cross_val_score(xgb_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(xgb_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Severity Levels")
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

sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Severity Levels")
plt.show()
catboost = LGBMClassifier(random_state = 12345)
catboost_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}
gs_cv = GridSearchCV(catboost, 
                     catboost_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X, y)
gs_cv.best_params_
catboost_tuned = CatBoostClassifier(**gs_cv.best_params_).fit(X,y)
cross_val_score(catboost_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(catboost_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Severity Levels")
plt.show()
models = []

models.append(('RF', RandomForestClassifier(random_state = 12345, max_depth = 8,max_features = 7, min_samples_split = 10,n_estimators = 500))) 
models.append(('XGB', GradientBoostingClassifier(random_state = 12345,learning_rate = 0.1, max_depth = 3, min_samples_split = 0.1, n_estimators = 500, subsample = 0.9))) 
models.append(("LightGBM", LGBMClassifier(random_state = 12345, learning_rate = 0.01, max_depth = 5, n_estimators = 1000))) 
models.append(("CatBoost", CatBoostClassifier(random_state = 12345, learning_rate = 0.01, max_depth = 5, n_estimators = 1000)))

results = [] 
names = []
for name, model in models:

    cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# comparison of algorithms with boxplot
fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results, 
            vert=True, # vertical box alignment
            patch_artist=True) # fill with color

ax.set_xticklabels(names) 
plt.show()