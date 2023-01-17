# importing libararies

# Warning
import warnings
warnings.filterwarnings("ignore")

# for Data
import pandas as pd
import numpy as np
import math

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='whitegrid')
sns.set(font_scale=1.5);
%matplotlib inline


# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler; std_scaler = StandardScaler()

# Splitting
from sklearn.model_selection import train_test_split


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Metrics
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# reading dataset
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
# information about the dataset structure
data.info()
data['TotalCharges'] = data['TotalCharges'].convert_objects(convert_numeric=True)
data['TotalCharges'].dtype
# Numerical features stats
data.describe()
data.describe(include=['object'])
for dataset in [data]:
    dataset['MultipleLines'] = dataset['MultipleLines'].replace({'No phone service':'No'})
    dataset['OnlineSecurity'] = dataset['OnlineSecurity'].replace({'No internet service':'No'})
    dataset['DeviceProtection'] = dataset['DeviceProtection'].replace({'No internet service':'No'})
    dataset['TechSupport'] = dataset['TechSupport'].replace({'No internet service':'No'})
    dataset['StreamingTV'] = dataset['StreamingTV'].replace({'No internet service':'No'})
    dataset['OnlineBackup'] = dataset['OnlineBackup'].replace({'No internet service':'No'})
    dataset['StreamingMovies'] = dataset['StreamingMovies'].replace({'No internet service':'No'})
    
print ("Number of unique values in each column\n")
for col_name in data.columns:
 print(col_name,": " ,data[col_name].nunique())
# count of customers churn

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
ax = data['Churn'].value_counts().plot(kind='pie',autopct='%.1f%%', ax=axes[0])
ax.set_title('Number of customer churn', fontsize=15)


ax = sns.countplot(y='Churn', data=data, ax=axes[1]);
for i,j in enumerate(data["Churn"].value_counts().values) : 
    ax.text(.1,i,j,fontsize = 20,color = "k")

ax.set_title('Number of customer churn', fontsize=15)

ig, ax = plt.subplots(figsize=(15,9))
sns.violinplot(x="gender", y="tenure", hue='Churn', data=data, split=True, bw=0.05 , palette='husl', ax=ax)
plt.title('Churn by gender ')
plt.show()
g = sns.factorplot(x="InternetService", y="tenure", hue="Churn", col="gender", data=data, kind="swarm", dodge=True, palette='husl', size=8, aspect=.9, s=8)
fig, ax = plt.subplots(figsize=(12, 5))

ax = sns.distplot(data[data['Churn']=='Yes']['MonthlyCharges'],label='Churn', bins= 10, kde=True)
#ax = sns.distplot(data[data['Churn']=='No']['MonthlyCharges'],label='Not Churn', bins= 18, kde=False)

ax.legend()
ax.set_title('Monthly Charges distrobution for Churn cutomers')
plt.show()
fig, ax = plt.subplots(figsize=(20,12))
ax =  sns.stripplot('InternetService', 'MonthlyCharges', 'Churn', data=data,
                        palette="husl", size=15, marker="D",
                        edgecolor="red", alpha=.30)
FacetGrid = sns.FacetGrid(data, hue='Churn', aspect=4)
FacetGrid.map(sns.kdeplot, 'tenure', shade=True)
FacetGrid.set(xlim=(0, data['tenure'].max()))
FacetGrid.add_legend()
FacetGrid = sns.FacetGrid(data, hue='Churn', aspect=4)
FacetGrid.map(sns.kdeplot, 'TotalCharges', shade=True)
FacetGrid.set(xlim=(0, data['TotalCharges'].max()))
FacetGrid.add_legend()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

# count of churn customers by payment method
ax = sns.countplot(y='PaymentMethod', hue='Churn', data=data, palette="husl", ax=axes[0]);

ax.set_yticklabels(ax.get_yticklabels(), rotation=30, ha="right")
ax.set_xlabel('Payment Method', fontsize = 12)
ax.set_ylabel('Number of Customers', fontsize = 12)

ax.set_title('Count of churn customers by Payment Method', fontsize=15)





# count of customers by payment method
ax = sns.countplot(x='PaymentMethod', data=data, palette="husl", ax=axes[1]);

ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_xlabel('Payment Method', fontsize = 12)
ax.set_ylabel('Number of Customers', fontsize = 12)

ax.set_title('Count customers by Payment Method', fontsize=15)

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(26, 24))

plot = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'TechSupport',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'Contract',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
index = 0
for i in range(5):
    for j in range(3):

        ax = sns.countplot(x=plot[index], hue='Churn', data=data, palette="husl", ax=axes[i,j]);
        index+=1


df = data.copy()
df = df.drop('customerID', axis=1)
# Getting dummy variables for these columns
# using drop_first=True in order to avoid the dummy variables trap

df = pd.get_dummies(data = df,columns = ['InternetService', 'Contract', 'PaymentMethod'], drop_first=True )
enc = LabelEncoder()
df = df.apply(enc.fit_transform)
df.head()
X = df.drop('Churn', axis=1)
y= df['Churn']
X = std_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
acc_log_reg = log_reg.score(X_test, y_test)*100
print("{:.2f}".format(acc_log_reg))
print (classification_report(y_test, y_pred))
lr_conf = confusion_matrix(y_test, y_pred)
sns.heatmap(lr_conf, annot=True, fmt="d")
plt.show
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)

# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
lr_auc = roc_auc_score(y_test, y_pred)
print("Roc score: ", round(lr_auc,2)*100, "%")
#Random Forest
rf = RandomForest(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = rf.score(X_test, y_test) * 100
print("{:.2f}".format(acc_rf))
RF_conf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(RF_conf, annot=True, fmt="d")
plt.show
# Ada Boost
AdaBoost = AdaBoostClassifier(random_state=42)
AdaBoost.fit(X_train, y_train)
y_pred = AdaBoost.predict(X_test)
acc_AdaBoost = AdaBoost.score(X_test, y_test) * 100
print("{:.2f}".format(acc_AdaBoost))
Ada_conf = confusion_matrix(y_test, y_pred)
sns.heatmap(Ada_conf, annot=True, fmt="d")
plt.show
# LGBM Classifier
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
acc_lgbm = lgbm.score(X_test, y_test) * 100
print("{:.2f}".format(acc_lgbm))
LGBM_conf = confusion_matrix(y_test, y_pred)
sns.heatmap(LGBM_conf, annot=True, fmt="d")
plt.show
# XGBoost
xg = XGBClassifier(random_state=42)
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
acc_xg = xg.score(X_test, y_test) * 100
print("{:.2f}".format(acc_xg))
XG_conf = confusion_matrix(y_test, y_pred)
sns.heatmap(XG_conf, annot=True, fmt="d")
plt.show
clf = AdaBoostClassifier(random_state=42)

param_grid = {"n_estimators": [50, 100, 300, 500],\
              "learning_rate" : [1, 0.0001, 0.5]}

grid_obj = GridSearchCV(clf, param_grid=param_grid, cv=10)


grid_fit = grid_obj.fit(X_train, y_train)

print("Best parameter: ", grid_obj.best_params_)

# Get the estimator/ clf
best_clf = grid_fit.best_estimator_

grid_y_pred = best_clf.predict(X_test)

print("Optimal accuracy score on the testing data: {:.2f}".format(accuracy_score(y_test, grid_y_pred)*100))

print (classification_report(y_test, grid_y_pred))
Grid_conf = confusion_matrix(y_test, grid_y_pred)
sns.heatmap(Grid_conf, annot=True, fmt="d")
plt.show
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, grid_y_pred)

# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
grid_auc = roc_auc_score(y_test, grid_y_pred)
print("Roc score: {:.2f}".format((grid_auc)*100), "%")
