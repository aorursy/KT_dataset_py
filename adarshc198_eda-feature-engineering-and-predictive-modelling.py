import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
bank_data = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
bank_data.head()


bank_data.shape
bank_data.isnull().sum()
labels = 'Exited', 'Retained'
sizes = [bank_data.Exited[bank_data['Exited']==1].count(), bank_data.Exited[bank_data['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()
Bank_data_geography_wise = bank_data.groupby('Geography')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None, axis =1).reset_index()
Bank_data_geography_wise.head()
Bank_data_geography_wise.columns =['Geography', 'Not_Exited', 'Exited']
Bank_data_geography_wise.head()
Bank_data_geography_wise["Total"] = Bank_data_geography_wise["Not_Exited"] + Bank_data_geography_wise["Exited"]
Bank_data_geography_wise["Percentage_Exited"] = (Bank_data_geography_wise["Exited"] / Bank_data_geography_wise["Total"]) * 100
Bank_data_geography_wise["Percentage_Not_Exited"] = (Bank_data_geography_wise["Not_Exited"] / Bank_data_geography_wise["Total"]) * 100
Bank_data_geography_wise.head()
Bank_data_geography_wise.plot(x="Geography", y=["Percentage_Exited", "Percentage_Not_Exited"], kind="bar")
plt.xlabel('Geography')
plt.ylabel('Percentage')
plt.title('Percentage of customers exited and not exited')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
bank_data_gender_wise = bank_data.groupby('Gender')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None, axis =1).reset_index()
bank_data_gender_wise.head()
bank_data_gender_wise.columns = ['Gender','Not_Exited','Exited']
bank_data_gender_wise["Total"] = bank_data_gender_wise['Not_Exited'] + bank_data_gender_wise['Exited']
bank_data_gender_wise.head()
bank_data_gender_wise["Percentage_Exited"] = (bank_data_gender_wise["Exited"] / bank_data_gender_wise["Total"]) * 100
bank_data_gender_wise["Percentage_Not_Exited"] = (bank_data_gender_wise["Not_Exited"] / bank_data_gender_wise["Total"]) * 100
bank_data_gender_wise.head()
bank_data_gender_wise.plot(x = "Gender", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited (Gender wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
print("Min Age in dataset", min(bank_data["Age"].unique()))
print("Max Tenure in dataset", max(bank_data["Age"].unique()))
bank_data["Age_Group"] = pd.cut(x=bank_data['Age'], bins = [0, 20 , 35, 50, 100], 
                                labels = ["Teenager","Younger","Elder","Older"])
bank_data.head()
bank_data.tail()
bank_data_age_group_wise = bank_data.groupby('Age_Group')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None , axis = 1).reset_index()
bank_data_age_group_wise.head()
bank_data_age_group_wise.columns = ["Age_Group","Not_Exited","Exited"]
bank_data_age_group_wise.head()
bank_data_age_group_wise["Total"] = bank_data_age_group_wise["Not_Exited"] + bank_data_age_group_wise["Exited"]
bank_data_age_group_wise["Percentage_Exited"] = (bank_data_age_group_wise["Exited"] / bank_data_age_group_wise["Total"]) * 100
bank_data_age_group_wise["Percentage_Not_Exited"] = (bank_data_age_group_wise["Not_Exited"] / bank_data_age_group_wise["Total"]) * 100
bank_data_age_group_wise.head()
bank_data_age_group_wise.plot(x = "Age_Group", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('Age_Group')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited (Age Group Wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
g = sns.FacetGrid(bank_data, col = "Exited")
g.map(sns.distplot, "Age", bins = 25)
plt.show()
plt.figure(figsize=(12,5))
plt.title("Box plot for credit score and customer exited")
sns.boxplot(y="Exited", x="CreditScore", data = bank_data, orient="h", palette = 'magma')
plt.xlabel('Credit Score')
plt.ylabel('Customer _Exited (1 = True, 0 = False) ')
g = sns.FacetGrid(bank_data, col = "Exited")
g.map(sns.distplot, "CreditScore", bins = 25)
plt.show()
plt.figure(figsize=(12,5))
plt.title("Box plot for salary and customer exited")
sns.boxplot(y="Exited", x="EstimatedSalary", data = bank_data, orient="h", palette = 'magma')
plt.xlabel('Estimated Salary')
plt.ylabel('Customer _Exited (1 = True, 0 = False) ')
g = sns.FacetGrid(bank_data, col = "Exited")
g.map(sns.distplot, "EstimatedSalary", bins = 25)
plt.show()
g = sns.FacetGrid(bank_data, col = "Exited")
g.map(sns.distplot, "Balance", bins = 25)
plt.show()
Bank_data_No_of_products = bank_data.groupby('NumOfProducts')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None, axis =1).reset_index()
Bank_data_No_of_products.head()
Bank_data_No_of_products.columns = ['Product_count','Not_Exited','Exited']
Bank_data_No_of_products.head(10)
Bank_data_No_of_products["Not_Exited"] = Bank_data_No_of_products["Not_Exited"].fillna(0)

Bank_data_No_of_products.head()
Bank_data_No_of_products["Total"] = Bank_data_No_of_products["Not_Exited"] + Bank_data_No_of_products["Exited"]
Bank_data_No_of_products["Percentage_Exited"] = (Bank_data_No_of_products["Exited"] / Bank_data_No_of_products["Total"]) * 100
Bank_data_No_of_products["Percentage_Not_Exited"] = (Bank_data_No_of_products["Not_Exited"] / Bank_data_No_of_products["Total"]) *100
Bank_data_No_of_products.head()
Bank_data_No_of_products.plot(x = "Product_count", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('Product_count')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( Product_count wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
Bank_data_Tenure = bank_data.groupby('Tenure')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None, axis =1).reset_index()
Bank_data_Tenure.head(10)
Bank_data_Tenure.columns = ['Tenure','Not_Exited','Exited']
Bank_data_Tenure.head(10)
Bank_data_Tenure["Total"] = Bank_data_Tenure["Not_Exited"] + Bank_data_Tenure["Exited"]
Bank_data_Tenure["Percentage_Exited"] = (Bank_data_Tenure["Exited"] / Bank_data_Tenure["Total"]) * 100
Bank_data_Tenure["Percentage_Not_Exited"] = (Bank_data_Tenure["Not_Exited"] / Bank_data_Tenure["Total"]) *100
Bank_data_Tenure.head(10)
Bank_data_Tenure.plot(x = "Tenure", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('Tenure')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( Tenure wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
Bank_data_IsActiveMember = bank_data.groupby('IsActiveMember')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None, axis =1).reset_index()
Bank_data_IsActiveMember.columns = ['IsActiveMember','Not_Exited','Exited']
Bank_data_IsActiveMember.head(10)
Bank_data_IsActiveMember["Total"] = Bank_data_IsActiveMember["Not_Exited"] + Bank_data_IsActiveMember["Exited"]
Bank_data_IsActiveMember["Percentage_Exited"] = (Bank_data_IsActiveMember["Exited"] / Bank_data_IsActiveMember["Total"]) * 100
Bank_data_IsActiveMember["Percentage_Not_Exited"] = (Bank_data_IsActiveMember["Not_Exited"] / Bank_data_IsActiveMember["Total"]) *100
Bank_data_IsActiveMember.head(10)
Bank_data_IsActiveMember.plot(x = "IsActiveMember", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('IsActiveMember')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( Active Member wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
Bank_data_HasCrCard = bank_data.groupby('HasCrCard')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None, axis =1).reset_index()
Bank_data_HasCrCard.columns = ['HasCrCard','Not_Exited','Exited']
Bank_data_HasCrCard.head(10)
Bank_data_HasCrCard["Total"] = Bank_data_HasCrCard["Not_Exited"] + Bank_data_HasCrCard["Exited"]
Bank_data_HasCrCard["Percentage_Exited"] = (Bank_data_HasCrCard["Exited"] / Bank_data_HasCrCard["Total"]) * 100
Bank_data_HasCrCard["Percentage_Not_Exited"] = (Bank_data_HasCrCard["Not_Exited"] / Bank_data_HasCrCard["Total"]) *100
Bank_data_HasCrCard.head(10)
Bank_data_HasCrCard.plot(x = "HasCrCard", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('HasCrCard')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( HasCrCard wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
print("Min Balance in dataset", min(bank_data["Balance"].unique()))
print("Max Balance in dataset", max(bank_data["Balance"].unique()))
bank_data["Balance_1"] = pd.cut(x=bank_data['Balance'], bins = [-1, 10000, 100000, 200000,300000 ], 
                                labels = ["Low balance","Medium balance","High balance","Highest balance"])
bank_data.head()
bank_data_balance = bank_data.groupby('Balance_1')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None , axis = 1).reset_index()
bank_data_balance.columns = ['Balance_1','Not_Exited','Exited']
bank_data_balance.head(10)
bank_data_balance["Not_Exited"] = bank_data_balance["Not_Exited"].fillna(0)
bank_data_balance["Total"] = bank_data_balance["Not_Exited"] + bank_data_balance["Exited"]
bank_data_balance["Percentage_Exited"] = (bank_data_balance["Exited"] / bank_data_balance["Total"]) * 100
bank_data_balance["Percentage_Not_Exited"] = (bank_data_balance["Not_Exited"] / bank_data_balance["Total"]) *100
bank_data_balance.head(10)
bank_data_balance.plot(x = "Balance_1", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('Balance_1')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( Balance wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
print("Min CreditScore", min(bank_data["CreditScore"].unique()))
print("Max CreditScore", max(bank_data["CreditScore"].unique()))
bank_data["CreditScore_1"] = pd.cut(x=bank_data['CreditScore'], bins = [249, 500, 650, 851 ], 
                                labels = ["Low score","Medium score","High score"])
bank_data.head()
bank_data_creditscore = bank_data.groupby('CreditScore_1')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None , axis = 1).reset_index()
bank_data_creditscore.columns = ['CreditScore_1','Not_Exited','Exited']
bank_data_creditscore.head(10)
bank_data_creditscore["Total"] = bank_data_creditscore["Not_Exited"] + bank_data_creditscore["Exited"]
bank_data_creditscore["Percentage_Exited"] = (bank_data_creditscore["Exited"] / bank_data_creditscore["Total"]) * 100
bank_data_creditscore["Percentage_Not_Exited"] = (bank_data_creditscore["Not_Exited"] / bank_data_creditscore["Total"]) *100
bank_data_creditscore.head(10)
bank_data_creditscore.plot(x = "CreditScore_1", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('CreditScore_1')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( CreditScore wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
print("Min Estimated Salary in dataset", min(bank_data["EstimatedSalary"].unique()))
print("Max Estimated Salary in dataset", max(bank_data["EstimatedSalary"].unique()))
bank_data["EstimatedSalary_1"] = pd.cut(x=bank_data['EstimatedSalary'], bins = [10, 1000, 10000, 50000, 100000, 200000 ], 
                                labels = ["Extremly_low","Low","Average","Above Average","High"])
bank_data.head()
bank_data_EstimatedSalary = bank_data.groupby('EstimatedSalary_1')['Exited'].value_counts().unstack().add_prefix('').rename_axis(None , axis = 1).reset_index()
bank_data_EstimatedSalary.columns = ['EstimatedSalary_1','Not_Exited','Exited']
bank_data_creditscore.head(10)
bank_data_EstimatedSalary["Total"] = bank_data_EstimatedSalary["Not_Exited"] + bank_data_EstimatedSalary["Exited"]
bank_data_EstimatedSalary["Percentage_Exited"] = (bank_data_EstimatedSalary["Exited"] / bank_data_EstimatedSalary["Total"]) * 100
bank_data_EstimatedSalary["Percentage_Not_Exited"] = (bank_data_EstimatedSalary["Not_Exited"] / bank_data_EstimatedSalary["Total"]) *100
bank_data_EstimatedSalary.head(10)
bank_data_EstimatedSalary.plot(x = "EstimatedSalary_1", y = ["Percentage_Exited", "Percentage_Not_Exited"], kind = "bar")
plt.xlabel('EstimatedSalary')
plt.ylabel('Percentage')
plt.title('Percentage of customers Exited and Not Exited ( EstimatedSalary wise)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
bank_data.head()
bank_data.drop(['RowNumber','CustomerId','Surname','CreditScore','Age','Tenure','Balance','HasCrCard','EstimatedSalary'], axis = 1, inplace = True)
bank_data.head()
bank_data = pd.get_dummies(bank_data, columns=["Geography","Gender","NumOfProducts"])
bank_data.head()
replace_nums = {"Age_Group":     {"Teenager": 0, "Younger": 1, "Elder":2, "Older":3},
                "EstimatedSalary_1": {"Extremly_low": 0, "Low": 1, "Average": 2, "Above Average": 3,
                                  "High": 4},
                "CreditScore_1": {"Low score":0, "Medium score":1, "High score":2},
                "Balance_1":{"Low balance":0, "Medium balance":1, "High balance":2, "Highest balance":3}
               }
bank_data.replace(replace_nums, inplace=True)
bank_data.head()
corrmat = bank_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(bank_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
bank_data.head()
Predictors = bank_data.drop(['Exited'], axis = 1)
Target = bank_data['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Predictors, Target, test_size = 0.20, random_state = 42, stratify = Target)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X_train,y_train)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
plt.plot(model.feature_importances_)
plt.xticks(np.arange(X_train.shape[1]), X_train.columns.tolist(), rotation=90);
X_train.head()
y_train.head()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)
# Fit logistic regression
param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [100,150,200], 'fit_intercept':[True],'intercept_scaling':[1],
              'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}
lr = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid, cv=10, refit=True, verbose=0)
lr.fit(X_train,y_train)
best_model(lr)
# Fit best logistic regression
lr_best = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
lr_best.fit(X_train,y_train)
y_test.head(10)
X_test.head(10)
y_pred = lr_best.predict(X_test) 
print(':',"%.3f" % accuracy_score(y_pred, y_test))
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
r_probs = [0 for _ in range(len(y_test))]
lr_probs = lr_best.predict_proba(X_test)
lr_probs
r_probs
lr_probs_p = lr_probs[:, 1]
lr_probs_p
r_auc = roc_auc_score(y_test, r_probs)
lr_auc = roc_auc_score(y_test, lr_probs_p)
print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Logistic Regression: AUROC = %.3f' % (lr_auc))
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs_p)
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % lr_auc)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()
# Fit logistic regression with degree 2 polynomial
param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
              'tol':[0.0001,0.000001]}
lr_2 = PolynomialFeatures(degree=2)
lr_2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
lr_2_Grid.fit(X_train,y_train)
best_model(lr_2_Grid)
# Fit best_logistic regression with pol 2 kernel
lr_2 = PolynomialFeatures(degree=2)
lr_2d = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='ovr', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
lr_2d.fit(X_train,y_train)
y_pred_1 = lr_2d.predict(X_test) 
print(':',"%.3f" % accuracy_score(y_pred_1, y_test))
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_1)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
lr2_probs = lr_2d.predict_proba(X_test)
lr2_probs = lr2_probs[:, 1]
r_auc = roc_auc_score(y_test, r_probs)
lr2_auc = roc_auc_score(y_test, lr2_probs)
print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Logistic Regression Degree 2: AUROC = %.3f' % (lr2_auc))
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
lr_2_fpr, lr_2_tpr, _ = roc_curve(y_test, lr2_probs)
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(lr_2_fpr, lr_2_tpr, marker='.', label='Logistic Regression degree 2(AUROC = %0.3f)' % lr2_auc)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()
# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(X_train, y_train)
best_model(RanFor_grid)
# Fit best_Random Forest classifier
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(X_train, y_train)
y_pred_4 = RF.predict(X_test) 
print(':',"%.3f" % accuracy_score(y_pred_4, y_test))
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_4)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
RF_probs = RF.predict_proba(X_test)
RF_probs = RF_probs[:, 1]
r_auc = roc_auc_score(y_test, r_probs)
RF_auc = roc_auc_score(y_test, RF_probs)
print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (RF_auc))
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, RF_probs)
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest(AUROC = %0.3f)' % RF_auc)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()
# Fit Extreme Gradient boosting classifier
param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(X_train, y_train)
best_model(xgb_grid)
# Fit best_Extreme Gradient Boost Classifier
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.2, max_delta_step=0,max_depth=5,
                    min_child_weight=10, missing=None, n_estimators=20,n_jobs=0, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
XGB.fit(X_train, y_train)
y_pred_5 = XGB.predict(X_test) 
print(':',"%.3f" % accuracy_score(y_pred_5, y_test))
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_5)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
XGB_probs = XGB.predict_proba(X_test)
XGB_probs = XGB_probs[:, 1]
r_auc = roc_auc_score(y_test, r_probs)
XGB_auc = roc_auc_score(y_test, XGB_probs)
RF_auc = roc_auc_score(y_test, RF_probs)
lr2_auc = roc_auc_score(y_test, lr2_probs)
lr_auc = roc_auc_score(y_test, lr_probs_p)
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
XGB_fpr, XGB_tpr, _ = roc_curve(y_test, XGB_probs)
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression(AUROC = %0.3f)' % lr_auc)
plt.plot(lr_2_fpr, lr_2_tpr, marker='.', label='Logistic Regression degree 2(AUROC = %0.3f)' % lr2_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest(AUROC = %0.3f)' % RF_auc)
plt.plot(XGB_fpr, XGB_tpr, marker='.', label='XGB (AUROC = %0.3f)' % XGB_auc)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()