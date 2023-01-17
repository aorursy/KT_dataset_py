## importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#reading dataset
df = pd.read_csv(r'../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.info()
df.shape
df.columns
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
len(categorical_features)
numerical_features = [feature for feature in df.columns if feature not in categorical_features]
len(numerical_features)
#Checking value counts for categorical variables
for col in categorical_features:
    print(df[col].value_counts())   
#Drop column Over18 as all values are Yes
df.drop('Over18', axis = 1, inplace = True)
#Checking number of distinct values for numerical variables
for col in numerical_features:
    print(col, df[col].nunique())
#Drop these as there are only 1 type of value in whole variable
df.drop(['EmployeeCount','StandardHours'], axis = 1, inplace = True)
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
len(categorical_features)
numerical_features = [feature for feature in df.columns if feature not in categorical_features]
len(numerical_features)
# getting list of discrete numerical features
discrete_numerical_features = []
for col in numerical_features:
    if (df[col].nunique()<11):
        discrete_numerical_features.append(col)
len(discrete_numerical_features)
numerical_features = [feature for feature in numerical_features if feature not in discrete_numerical_features]
len(numerical_features)
df.describe()
df.describe(include = ['O'])
Attrition_mapping = {"Yes": 1, "No": 0}
df['Attrition'] = df['Attrition'].map(Attrition_mapping)
sns.countplot(df['Attrition'])
attrition = df[(df['Attrition'] != 0)]
no_attrition = df[(df['Attrition'] == 0)]
print('Percentage of Attrition: {}'.format(len(attrition)/len(df)))
df[['Gender', 'Attrition']].groupby(['Gender'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['BusinessTravel', 'Attrition']].groupby(['BusinessTravel'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['Department', 'Attrition']].groupby(['Department'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['EducationField', 'Attrition']].groupby(['EducationField'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['JobRole', 'Attrition']].groupby(['JobRole'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['MaritalStatus', 'Attrition']].groupby(['MaritalStatus'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['OverTime', 'Attrition']].groupby(['OverTime'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
sns.set_style('whitegrid')
sns.distplot(df['Age'], bins = 10)
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'Age', bins=15)
sns.distplot(df['MonthlyIncome'], bins = 15)
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'MonthlyIncome', bins=15)
sns.distplot(df['DistanceFromHome'], bins = 15)
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'DistanceFromHome', bins=15)
sns.distplot(df['DailyRate'])
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'DailyRate', bins=15)
sns.boxplot(df['Attrition'],df['DailyRate'])
sns.distplot(df['MonthlyRate'])
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'MonthlyRate', bins=15)
sns.boxplot(df['Attrition'],df['MonthlyRate'])
sns.distplot(df['HourlyRate'])
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'HourlyRate', bins=15)
sns.boxplot(df['Attrition'],df['HourlyRate'])
sns.distplot(df['PercentSalaryHike'])
sns.boxplot(df['Attrition'],df['PercentSalaryHike'])
numerical_features
sns.distplot(df['YearsAtCompany'])
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'YearsAtCompany', bins=15)
sns.boxplot(df['Attrition'],df['YearsAtCompany'])
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'YearsInCurrentRole', bins=15)
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'YearsSinceLastPromotion', bins=15)
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'YearsWithCurrManager', bins=15)
sns.set_style('whitegrid')
g = sns.FacetGrid(df, col='Attrition')
g.map(plt.hist, 'TotalWorkingYears', bins=15)
grid = sns.FacetGrid(df, col='Attrition', row='MaritalStatus', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.8, bins=15)
grid.add_legend();
df[['Education', 'Attrition']].groupby(['Education'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['EnvironmentSatisfaction', 'Attrition']].groupby(['EnvironmentSatisfaction'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['JobInvolvement', 'Attrition']].groupby(['JobInvolvement'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['JobLevel', 'Attrition']].groupby(['JobLevel'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['JobSatisfaction', 'Attrition']].groupby(['JobSatisfaction'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['NumCompaniesWorked', 'Attrition']].groupby(['NumCompaniesWorked'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
sns.countplot(df['NumCompaniesWorked'], hue = df['Attrition'])
df[['PerformanceRating', 'Attrition']].groupby(['PerformanceRating'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['RelationshipSatisfaction', 'Attrition']].groupby(['RelationshipSatisfaction'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['StockOptionLevel', 'Attrition']].groupby(['StockOptionLevel'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['TrainingTimesLastYear', 'Attrition']].groupby(['TrainingTimesLastYear'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
df[['WorkLifeBalance', 'Attrition']].groupby(['WorkLifeBalance'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
numerical_features
plt.figure(figsize = (12,6))
sns.countplot(df['TotalWorkingYears'], hue = df['Attrition'])
#Drop these as there are only 1 type of value in whole variable
df.drop(['HourlyRate', 'MonthlyRate','DailyRate','PerformanceRating'], axis = 1, inplace = True)
from scipy.stats import norm, skew
numerical_features.remove('EmployeeNumber')
numerical_features.remove('HourlyRate')
numerical_features.remove('MonthlyRate')
numerical_features.remove('DailyRate')
skewed_feat = df[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame({'Skew' :skewed_feat})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
lamda = 0.15
for feat in skewed_features:
    df[feat] = boxcox1p(df[feat],lamda)
df['New_feature'] = (df['Gender'].astype(str) + '_' + df['MaritalStatus'].astype(str))
df.drop(['Gender', 'MaritalStatus'], axis = 1, inplace = True)
df.info()
df.shape
df.head()
#Checking correaltions between variables
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
k_corr_matrix1 =df.corr()
plt.figure(figsize=(20,14))
sns.heatmap(k_corr_matrix1, annot=True, cmap=plt.cm.RdBu_r)
plt.title('Heatmap for Correlation between Features')
# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_features = correlation(df, 0.7)
len(set(corr_features))
corr_features
corr_features.remove('MonthlyIncome')
corr_features.remove('TotalWorkingYears')
corr_features.update(['JobLevel'])
corr_features
df.drop(corr_features, axis=1, inplace = True)
df.shape
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
len(categorical_features)
#Label-Encoding ordinal categorical features 
from sklearn.preprocessing import LabelEncoder
for c in categorical_features:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))

# shape        
print('Shape all_data: {}'.format(df.shape))
from sklearn.model_selection import train_test_split
Id_train = df['EmployeeNumber']
X = df.drop(['Attrition', 'EmployeeNumber'], axis = 1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
cv = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)
# Using random forest on balanced dataset
rf = RandomForestClassifier()
param_grid=dict(n_estimators= [120, 300, 500, 800, 1200],max_depth=range(1,20), min_samples_split = [1, 2, 5, 10, 15, 100],
               min_samples_leaf = [1,2,5,10], max_features = ['log2', 'sqrt', None])
grid_rf = RandomizedSearchCV(rf, param_grid, cv=cv, scoring = 'f1_macro')
grid_rf.fit(X_train,y_train)
# Check out best parameters and best score
print(grid_rf.best_score_)
print(grid_rf.best_params_)
# Using random forest on balanced dataset
rf = RandomForestClassifier(class_weight={0:1,1:5}, random_state = 42)
param_grid=dict(n_estimators= [120, 300, 500, 800, 1200],max_depth=range(1,20), min_samples_split = [1, 2, 5, 10, 15, 100],
               min_samples_leaf = [1,2,5,10], max_features = ['log2', 'sqrt', None])
grid_rf = RandomizedSearchCV(rf, param_grid, cv=cv, scoring = 'f1_macro')
grid_rf.fit(X_train,y_train)
# Check out best parameters and best score
print(grid_rf.best_score_)
print(grid_rf.best_params_)
import scikitplot as skplt
y_test_pred = grid_rf.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_test_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_test_pred)
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(y_test, y_test_pred)
rou_auc=metrics.auc(fpr,tpr)
plt.title("Reciever Operating Characteristic")
plt.plot(fpr,tpr,"orange",label="AUC-0.4f" % rou_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],color="darkblue",linestyle="--")
plt.ylabel("tpr")
plt.xlabel("fpr")
plt.show()
import xgboost
# Using xgboost on balanced dataset
xg = xgboost.XGBClassifier(scale_pos_weight = 5, random_state = 2) #scale_pos_weight for balancing dataset internally
# Hyper-parameters to be tuned
param_grid = dict(eta = [0.01,0.015, 0.025, 0.05, 0.1], learning_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                  max_depth = [3,5,7,9,12,15,17,25], min_child_weight = [1,3,5,7], gamma = [0.05,0.1,0.3,0.5,0.7,0.9,1.0], 
                  colsample_bytree = [0.6, 0.7, 0.8, 0.9, 1.0], subsample = [0.6, 0.7, 0.8, 0.9, 1.0],
                  alpha = [0, 0.1, 0.5, 1.0])
grid_xg = RandomizedSearchCV(xg, param_grid, cv=cv, scoring = 'f1_macro')
grid_xg.fit(X_train,y_train)
# Check out best parameters and best score
print(grid_xg.best_score_)
print(grid_xg.best_params_)
y_test_pred = grid_xg.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))
roc_auc_score(y_test, y_test_pred)
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(y_test, y_test_pred)
rou_auc=metrics.auc(fpr,tpr)
plt.title("Reciever Operating Characteristic")
plt.plot(fpr,tpr,"orange",label="AUC-0.4f" % rou_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],color="darkblue",linestyle="--")
plt.ylabel("tpr")
plt.xlabel("fpr")
plt.show()
from imblearn.ensemble import EasyEnsembleClassifier 
eec = EasyEnsembleClassifier(base_estimator = xgboost.XGBClassifier(), random_state=42)
eec.fit(X_train, y_train)
y_pred = eec.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(y_test, y_pred)
rou_auc=metrics.auc(fpr,tpr)
plt.title("Reciever Operating Characteristic")
plt.plot(fpr,tpr,"orange",label="AUC-0.4f" % rou_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],color="darkblue",linestyle="--")
plt.ylabel("tpr")
plt.xlabel("fpr")
plt.show()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
param_grid = dict(C = [0.001, 0.01,1, 10, 100], penalty = ['l1', 'l2'])
grid_lr = RandomizedSearchCV(lr, param_grid, cv=cv, scoring = 'f1_macro')
grid_lr.fit(X_train,y_train)
# Check out best parameters and best score
print(grid_lr.best_score_)
print(grid_lr.best_params_)
skplt.metrics.plot_confusion_matrix(y_test, y_pred)
y_pred = grid_lr.predict(X_test)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(y_test, y_pred)
rou_auc=metrics.auc(fpr,tpr)
plt.title("Reciever Operating Characteristic")
plt.plot(fpr,tpr,"orange",label="AUC-0.4f" % rou_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],color="darkblue",linestyle="--")
plt.ylabel("tpr")
plt.xlabel("fpr")
plt.show()