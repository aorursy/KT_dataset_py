import numpy as np
import pandas as pd 
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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.simplefilter(action = "ignore")
#Reading data set and the first 5 observation
data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.head()
#Data set consists of 768 observation units and 9 variables.
data.shape
#Descriptive statistics of the data set accessed.
data.describe([0.10,0.25,0.50,0.75,0.90,0.99]).T
#The distribution of the Outcome variable
data["Outcome"].value_counts()*100/len(data)
#The classes of the Outcome variable
sns.countplot(x = 'Outcome', data = data);
data.Outcome.value_counts()
#The histogram of the Age variable
data["Age"].hist(edgecolor = "red");
#Pregnancy and age averages by Outcome classes
print(data.groupby("Outcome").agg({"Pregnancies":"mean"}))
print(data.groupby("Outcome").agg({"Age":"mean"}))
print(data.groupby("Outcome").agg({"Insulin": "mean"}))
print(data.groupby("Outcome").agg({"Glucose": "mean"}))
print(data.groupby("Outcome").agg({"BMI": "mean"}))
print("Maximum Age: " + str(data["Age"].max()))
print("Minimum Age: " + str(data["Age"].min()))
#Histogram and density graphs of all variables
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(data.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(data.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(data.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(data.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(data.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(data.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(data.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(data.BMI, bins = 20, ax=ax[3,1])
my_colors = ['lightblue','lightsteelblue','silver']
ax = data["Outcome"]
ax.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=my_colors)
plt.show()
#Access to the correlation of the data set was provided. What kind of relationship is examined between the variables. 
#If the correlation value is> 0, there is a positive correlation. While the value of one variable increases, the value of the other variable also increases.
#Correlation = 0 means no correlation.
#If the correlation is <0, there is a negative correlation. While one variable increases, the other variable decreases. 
#When the correlations are examined, there are 2 variables that act as a positive correlation to the Salary dependent variable.
#These variables are Glucose. As these increase, Outcome variable increases.
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
#We fill cells with a value of 0 as NAN
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#Now, we can look at where are missing values
data.isnull().sum()
#Have been visualized using the missingno library for the visualization of missing observations
import missingno as msno
msno.bar(data);
#The missing values will be filled with the median values of each variable
def median_target(variable):   
    temp = data[data[variable].notnull()]
    temp = temp[[variable, 'Outcome']].groupby(['Outcome'])[[variable]].median().reset_index()
    return temp
#The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick
columns = data.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    data.loc[(data['Outcome'] == 0 ) & (data[i].isnull()), i] = median_target(i)[i][0]
    data.loc[(data['Outcome'] == 1 ) & (data[i].isnull()), i] = median_target(i)[i][1]
#Missing values were filled
data.isnull().sum()
#In the data set, there were asked whether there were any outlier observations compared to the 15% and 85% quarters.

for feature in data:
    Q1 = data[feature].quantile(0.15)
    Q3 = data[feature].quantile(0.85)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    if data[(data[feature] > upper) | (data[feature] < lower)].any(axis=None):
        print(feature,"→ YES")
        print(data[(data[feature] > upper) | (data[feature] < lower)].shape[0])
    else:
        print(feature,"→ NO")
#The process of visualizing the Insulin variable with boxplot method was done. We find the outlier observations on the chart.
sns.boxplot(x = data["Insulin"]);
#We suppress contradictory values
Q1 = data.Insulin.quantile(0.20)
Q3 = data.Insulin.quantile(0.80)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
data.loc[data["Insulin"] > upper,"Insulin"] = upper

sns.boxplot(x = data["Insulin"]);
Q1 = data.DiabetesPedigreeFunction.quantile(0.25)
Q3 = data.DiabetesPedigreeFunction.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
data.loc[data["DiabetesPedigreeFunction"] > upper,"DiabetesPedigreeFunction"] = upper

sns.boxplot(x = data["DiabetesPedigreeFunction"]);
#We determine outliers between all variables with the LOF method
from sklearn.neighbors import LocalOutlierFactor
lof=LocalOutlierFactor(n_neighbors= 10)
lof.fit_predict(data)
data_scores = lof.negative_outlier_factor_
np.sort(data_scores)[0:30]
#We choose the threshold value according to lof scores
threshold = np.sort(data_scores)[5]
threshold
#We delete those that are higher than the threshold
outlier = data_scores > threshold
data = data[outlier]
data.shape
#According to BMI, some ranges were determined and categorical variables were assigned.
New_BMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
data["New_BMI"] = New_BMI
data.loc[data["BMI"] < 18.5, "New_BMI"] = New_BMI[0]
data.loc[(data["BMI"] > 18.5) & (data["BMI"] <= 24.9), "New_BMI"] = New_BMI[1]
data.loc[(data["BMI"] > 24.9) & (data["BMI"] <= 29.9), "New_BMI"] = New_BMI[2]
data.loc[(data["BMI"] > 29.9) & (data["BMI"] <= 34.9), "New_BMI"] = New_BMI[3]
data.loc[(data["BMI"] > 34.9) & (data["BMI"] <= 39.9), "New_BMI"] = New_BMI[4]
data.loc[data["BMI"] > 39.9 ,"New_BMI"] = New_BMI[5]
data.head()
#A categorical variable creation process is performed according to the insulin value.
def set_insulin(row): 
    if row["Insulin"] >= 100 and row["Insulin"] <= 126:
        return "Normal"
    else:
        return "Abnormal"
#The operation performed was added to the dataframe.
data = data.assign(New_Insulin_Score=data.apply(set_insulin, axis=1))
data.head()
#Some intervals were determined according to the glucose variable and these were assigned categorical variables.
New_Glucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")
data["New_Glucose"] = New_Glucose
data.loc[data["Glucose"] <= 70, "New_Glucose"] = New_Glucose[0]
data.loc[(data["Glucose"] > 70) & (data["Glucose"] <= 99), "New_Glucose"] = New_Glucose[1]
data.loc[(data["Glucose"] > 99) & (data["Glucose"] <= 126), "New_Glucose"] = New_Glucose[2]
data.loc[data["Glucose"] > 126 ,"New_Glucose"] = New_Glucose[3]
data.head()
#Let's look at the breaking of newly created variables.
data.groupby(["New_BMI","New_Insulin_Score", "New_Glucose"]).agg({"Outcome": "count"})
#The new "New_diabetes" variable defines whether the probability of having diabetes is high or normal or low.
data.loc[data["New_BMI"] == "Underweight", "New_Diabet"] = "Dusuk"
data.head()
data[data["New_BMI"] == "Underweight"]
data.loc[data["New_Glucose"] == "Low", "New_Diabet"] = "Dusuk"
data[data["New_Glucose"] == "Low"]
data.loc[data["New_Glucose"] == "High", "New_Diabet"] = "Yuksek"
data[data["New_Glucose"] == "High"]
data.loc[data["New_Glucose"] == "Secret", "New_Diabet"] = "Yuksek"
data[data["New_Glucose"] == "Secret"]
data.loc[(data["New_BMI"] == "Obesity 3") & (data["New_Insulin_Score"] == "Normal") & (data["New_Glucose"] == "Secret"), "New_Diabet"] = "Dusuk"
data[(data["New_BMI"] == "Obesity 3") & (data["New_Insulin_Score"] == "Normal") & (data["New_Glucose"] == "Secret")]
data["New_Diabet"].fillna("Normal", inplace = True)
data["New_Diabet"].isnull().sum()
data.head()
#Here, by making One Hot Encoding transformation, categorical variables were converted into numerical values. It is also protected from the Dummy variable trap.
data = pd.get_dummies(data, columns =["New_BMI","New_Insulin_Score", "New_Glucose", "New_Diabet"], drop_first = True)
data.head()
categorical_data = data[['New_BMI_Obesity 1','New_BMI_Obesity 2', 'New_BMI_Obesity 3', 'New_BMI_Overweight','New_BMI_Underweight',
                     'New_Insulin_Score_Normal','New_Glucose_Low','New_Glucose_Normal', 'New_Glucose_Overweight', 'New_Glucose_Secret','New_Diabet_Normal','New_Diabet_Yuksek']]
categorical_data.head()
y = data["Outcome"]
X = data.drop(["Outcome",'New_BMI_Obesity 1','New_BMI_Obesity 2', 'New_BMI_Obesity 3', 'New_BMI_Overweight','New_BMI_Underweight',
                     'New_Insulin_Score_Normal','New_Glucose_Low','New_Glucose_Normal', 'New_Glucose_Overweight', 'New_Glucose_Secret','New_Diabet_Normal','New_Diabet_Yuksek'], axis = 1)
cols = X.columns
index = X.index
X.head()
#The variables in the data set are an effective factor in increasing the performance of the models by standardization.  
#There are multiple standardization methods. These are methods such as" Normalize"," MinMax"," Robust" and "Scale".
from sklearn import preprocessing
normalizer = preprocessing.Normalizer().fit(X)
X = normalizer.transform(X) 
X = pd.DataFrame(X, columns = cols, index = index)
X = pd.concat([X,categorical_data], axis = 1)
X.head()
y.head()
#Validation scores of all base models

models = []
models.append(('LR', LogisticRegression(random_state = 123456)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 123456)))
models.append(('RF', RandomForestClassifier(random_state = 123456)))
models.append(('SVM', SVC(gamma='auto', random_state = 123456)))
models.append(('XGB', GradientBoostingClassifier(random_state = 123456)))
models.append(("LightGBM", LGBMClassifier(random_state = 123456)))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    
        kfold = KFold(n_splits = 10, random_state = 1234)
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
rf_model = RandomForestClassifier(random_state = 123456)
gs_cv = GridSearchCV(rf_model, 
                    rf_params,
                    cv = 15,
                    n_jobs = -1,
                    verbose = 2).fit(X, y)
gs_cv.best_params_
rf_tuned = RandomForestClassifier(**gs_cv.best_params_)
rf_tuned = rf_tuned.fit(X,y)
cross_val_score(rf_tuned, X, y, cv = 15).mean()
feature_imp = pd.Series(rf_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()
lgbm = LGBMClassifier(random_state = 123456)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}
gs_cv = GridSearchCV(lgbm, 
                     lgbm_params, 
                     cv = 15, 
                     n_jobs = -1, 
                     verbose = 2).fit(X, y)
gs_cv.best_params_
lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X,y)
cross_val_score(lgbm_tuned, X, y, cv = 15).mean()
feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()
xgb = GradientBoostingClassifier(random_state = 123456)
xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2, 1],
    "min_samples_split": np.linspace(0.1, 0.5, 10),
    "max_depth":[3,5,8],
    "subsample":[0.5, 0.9, 1.0],
    "n_estimators": [100,1000]}
xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 15, n_jobs = -1, verbose = 2).fit(X, y)
xgb_cv_model.best_params_
xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X,y)
cross_val_score(xgb_tuned, X, y, cv = 15).mean()
feature_imp = pd.Series(xgb_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()
models = []

models.append(('RF', RandomForestClassifier(random_state = 123456, max_depth = 8, max_features = 3, min_samples_split = 2, n_estimators = 500)))
models.append(("LightGBM", LGBMClassifier(random_state = 123456, learning_rate = 0.01,  max_depth = 5, n_estimators = 1500)))
models.append(('XGB', GradientBoostingClassifier(random_state = 123456, learning_rate = 1, max_depth = 5, min_samples_split = 0.2777777777777778, n_estimators = 1000, subsample = 0.9)))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    
        kfold = KFold(n_splits = 10, random_state = 123456)
        cv_results = cross_val_score(model, X, y, cv = 15, scoring= "accuracy")
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