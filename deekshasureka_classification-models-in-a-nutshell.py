import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score
import scipy.stats as stats
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
dataset.head(5)
dataset.shape
dataset.isnull().sum()
dataset.info()
dataset.describe(include = 'object')
dataset.describe()
dataset.nunique()
cat_cols = dataset.select_dtypes(include = 'object')
for x in cat_cols.columns:
    plt.figure(figsize = (10,6))
    sns.countplot(dataset[x],hue = dataset['Attrition'],palette = 'viridis')
    plt.xticks(rotation = 90)
    plt.title(x,fontweight='bold',size=15)
    plt.show()
dataset['Attrition'] = dataset['Attrition'].replace({'Yes':1,'No':0})
cat_cols = dataset.select_dtypes(include = 'object')
for x in cat_cols.columns:
    z = pd.crosstab(dataset[x],dataset['Attrition'])
    plt.figure(figsize = (10,6))
    plt.pie(z[1], labels = z.index, autopct='%1.0f%%')
    plt.title(x,fontweight='bold',size=15)
    plt.show()
cat_cols = dataset.select_dtypes(include = 'object')
for x in cat_cols.columns:
    z = pd.crosstab(dataset[x],dataset['Attrition'])
    z['Sum'] = z.T.sum().values
    for i in z.columns:
        z[i] = (z[i]/z['Sum'])*100
    z.drop('Sum',1,inplace = True)
    z.plot(kind = 'bar', stacked = True, color = ['teal','gold'])
    plt.title(x,fontweight='bold',size=15)
    plt.show()
num_col = dataset.select_dtypes(exclude = 'object')
for y in num_col.columns:
    sns.boxplot(dataset['Attrition'],dataset[y],palette = 'tab20c_r')
    plt.title(y,fontweight='bold',size=15)
    plt.show()
correlation_matrix = dataset.corr()
plt.figure(figsize = (18,14))
sns.heatmap(correlation_matrix,cmap = 'tab20c_r')
plt.title('Correlation Matrix')
plt.show()
dataset.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber'],1,inplace = True)
num_col = dataset.select_dtypes(exclude = 'object')
features_to_drop = []
features_to_transform = []
for y in num_col.columns:
    value_0 = dataset[dataset['Attrition']==0][y]
    value_1 = dataset[dataset['Attrition']==1][y]
    ttest_ind = stats.ttest_ind(value_1,value_0)
    mann_whitney_u = stats.mannwhitneyu(value_1,value_0)
    if ttest_ind[1]>0.05 and mann_whitney_u[1]>0.05:
        features_to_drop.append(y)
    elif ttest_ind[1]>0.05 and mann_whitney_u[1]<0.05:
        features_to_transform.append(y)
    else:
        continue
cat_col = dataset.select_dtypes(include = 'object')
for y in cat_col.columns:
    crosstab_matrix = pd.crosstab(dataset[y],dataset['Attrition'])
    test_stat, pvalue, DOF, expected_value = stats.chi2_contingency(crosstab_matrix)
    if pvalue>0.05:
        features_to_drop.append(y)
    else:
        continue
dataset.drop(features_to_drop,1,inplace = True)
dataset['YearsSinceLastPromotion']= np.power(dataset[features_to_transform],0.3)
X = dataset.drop('Attrition',1)
y = dataset['Attrition']
categorical_X = X.select_dtypes(include = 'object')
numerical_X = X.select_dtypes(exclude = 'object')
lb = LabelEncoder()
X1 = pd.DataFrame()
for x in range(categorical_X.shape[1]):
    label_X = lb.fit_transform(categorical_X.iloc[:,x])
    X1[categorical_X.columns[x]] = label_X
combined_X = pd.concat([X1,numerical_X],1)
combined_X
sc = StandardScaler()
transformed_numeric = sc.fit_transform(combined_X)
transformed_numeric_df = pd.DataFrame(transformed_numeric)
transformed_numeric_df.columns = combined_X.columns
final_dataset = pd.concat([transformed_numeric_df,y],1)
final_dataset.head(5)
X = final_dataset.drop('Attrition',1)
y = final_dataset['Attrition']
logr=LogisticRegression()
rfe=RFECV(logr, scoring = 'f1')
rfe_fe=rfe.fit(X,y)

rfe_rank=pd.DataFrame()
rfe_rank['Feature']=X.columns
rfe_rank['Rank']=rfe_fe.ranking_
rfe_feature=rfe_rank[rfe_rank['Rank']==1]
features = rfe_feature['Feature'].values
final_X = X[features]
xtr,xte,ytr,yte = train_test_split(final_X,y,test_size = 0.3, random_state = 46)
# Logistic Regression
logr = LogisticRegression(random_state = 46)
logr.fit(xtr,ytr)
y_pred = logr.predict(xte)
print(classification_report(yte,y_pred))
math_logr = matthews_corrcoef(yte,y_pred)
math_logr
mcc_scores = []
mcc_scores.append(math_logr)
confusion_matrix(yte,y_pred)
# Regularized Decision Tree
dt = DecisionTreeClassifier(random_state = 46)
params = {'max_depth': np.arange(1,15),'criterion':['entropy','gini']}
gs = GridSearchCV(dt,params,cv = 5)
gs.fit(xtr,ytr)
gs.best_params_
dt_reg = DecisionTreeClassifier(max_depth = 2, criterion = 'gini')
dt_reg.fit(xtr,ytr)
y_pred = dt_reg.predict(xte)
print(classification_report(yte,y_pred))
math_dt_reg = matthews_corrcoef(yte,y_pred)
math_dt_reg
mcc_scores.append(math_dt_reg)
confusion_matrix(yte,y_pred)
# Random Forest
rf = RandomForestClassifier(random_state = 46)
params = {'n_estimators': np.arange(1,15)}
gs = GridSearchCV(rf,params,cv = 5)
gs.fit(xtr,ytr)
gs.best_params_
rf_reg = RandomForestClassifier(n_estimators = 12)
rf_reg.fit(xtr,ytr)

y_pred = rf_reg.predict(xte)
print(classification_report(yte,y_pred))
math_rf = matthews_corrcoef(yte,y_pred)
math_rf
mcc_scores.append(math_rf)
confusion_matrix(yte,y_pred)
# K Nearest Neighbors
knn = KNeighborsClassifier()
params = {'n_neighbors':np.arange(1,100), 'weights': ['distance','uniform']}
gs = GridSearchCV(knn,params,cv = 5)
gs.fit(xtr,ytr)
gs.best_params_
knn_tuned = KNeighborsClassifier(n_neighbors = 11, weights = 'distance')
knn_tuned.fit(xtr,ytr)
y_pred = knn_tuned.predict(xte)
print(classification_report(yte,y_pred))
math_knn = matthews_corrcoef(yte,y_pred)
math_knn
mcc_scores.append(math_knn)
confusion_matrix(yte,y_pred)
#Bagged Logistic Regression
logr = LogisticRegression(random_state = 20)
bagged_logr = BaggingClassifier(logr,random_state = 20)
bagged_logr.fit(xtr,ytr)
y_pred = bagged_logr.predict(xte)
print(classification_report(yte,y_pred))
math_bagged_logr = matthews_corrcoef(yte,y_pred)
math_bagged_logr
mcc_scores.append(math_bagged_logr)
confusion_matrix(yte,y_pred)
# Boosted Logistic Regression
boosted_logr = AdaBoostClassifier(logr,random_state = 46)
boosted_logr.fit(xtr,ytr)
y_pred = boosted_logr.predict(xte)
print(classification_report(yte,y_pred))
math_boosted_logr = matthews_corrcoef(yte,y_pred)
math_boosted_logr
mcc_scores.append(math_boosted_logr)
confusion_matrix(yte,y_pred)
# Gradient Boosting
gb = GradientBoostingClassifier(random_state = 46)
gb.fit(xtr,ytr)
y_pred = gb.predict(xte)
print(classification_report(yte,y_pred))
math_gboost = matthews_corrcoef(yte,y_pred)
math_gboost
mcc_scores.append(math_gboost)
confusion_matrix(yte,y_pred)
models = ['Logistic Regression','Regularized Decision Tree','Random Forest','K Nearest Neighbors','Bagged Logistic Regression','Boosted Logistic Regression','Gradient Boosting']
results = pd.DataFrame()
results['Matthews correlation coefficient Scores'] = mcc_scores
results.index = models
results