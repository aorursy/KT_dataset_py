import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline

plt.style.use("fivethirtyeight")

plt.rcParams['xtick.labelsize']=8

plt.rcParams['ytick.labelsize']=8



from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix, roc_curve, auc,classification_report, recall_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,RandomizedSearchCV



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")
train_data.drop("Unnamed: 0", axis=1, inplace=True)

train_data.describe()
plt.figure(figsize=(8,3))

train_data['SeriousDlqin2yrs'].value_counts().plot(kind='bar')
class0 = train_data['SeriousDlqin2yrs'].value_counts()[0]

class1 = train_data['SeriousDlqin2yrs'].value_counts()[1]

print("class 0 : {}".format(class0))

print("class 1 : {}".format(class1))

print("delinquency rate: {}".format(class1/(class0+class1)))
train_data[train_data['age'] < 18]
plt.figure(figsize=(8,3))

sns.boxplot(train_data['age'])
train_data.loc[train_data['age'] < 18, 'age'] = train_data['age'].median()
cols =["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse","NumberOfTimes90DaysLate", "DebtRatio","NumberOfOpenCreditLinesAndLoans","NumberRealEstateLoansOrLines", "RevolvingUtilizationOfUnsecuredLines"]

fig, axes = plt.subplots(len(cols),1, figsize=(10,10))

i = 0

for c in cols:

    ax = sns.boxplot(train_data[c], ax = axes[i])

    ax.set_ylabel(c, rotation=0,labelpad=150)

    ax.set_xlabel("Number of Times")

    i +=1

plt.show()
debtratio_q = train_data["DebtRatio"].quantile(0.86)

print("Debt Ratio: {}".format(debtratio_q))



colormap = {0:'blue', 1:'red'}



fig, (ax1, ax2) = plt.subplots(1,2,  figsize=(15,4))

for delinquency, color in colormap.items():

    tmp = train_data[(train_data['DebtRatio'] > debtratio_q) & (train_data['SeriousDlqin2yrs']==delinquency)][['DebtRatio','MonthlyIncome']]

    ax1.scatter((tmp['DebtRatio']), (tmp['MonthlyIncome']), c=color, alpha=0.8, label= str(delinquency) + ":{}".format(tmp.shape[0]))

ax1.legend()

ax1.set_title("Debt Ratio 86% Quantile against Monthly Income",fontsize=10)

ax1.set_xlabel("DebtRatio")

ax1.set_ylabel("Monthly Income")



for delinquency, color in colormap.items():

    tmp = train_data[(train_data['SeriousDlqin2yrs']==delinquency)][['DebtRatio','MonthlyIncome']]

    ax2.scatter(np.log(tmp['DebtRatio']), np.log(tmp['MonthlyIncome']), c=color, alpha=0.8, label= str(delinquency) + ":{}".format(tmp.shape[0]))

ax2.legend()

ax2.set_title("Log of Debt Ratio against log of Monthly Income",fontsize=10)

ax2.set_xlabel("log(DebtRatio)")

ax2.set_ylabel("log(Monthly Income)")

plt.show()
print("Number of records with monthly income equals 1: {}".format(train_data[(train_data['MonthlyIncome'] == 1) ].shape[0]))
train_data["DebtRatio"] = np.log(train_data["DebtRatio"])

removedmonthincome_1 = train_data[train_data["MonthlyIncome"] != 1]

removedmonthincome_1["DebtRatio"].replace([np.inf, -np.inf], -10, inplace=True)



removedmonthincome_1.describe()
u_list = []

d_list = []

for u in range(int(train_data['RevolvingUtilizationOfUnsecuredLines'].max())):

    default_rate = train_data[train_data['RevolvingUtilizationOfUnsecuredLines'] > u]['SeriousDlqin2yrs'].mean()

    u_list.append(u)

    d_list.append(default_rate)
fig, (ax1,ax2)= plt.subplots(1,2, figsize=(15,4))

df = pd.DataFrame({"utilization":u_list, "default":d_list})

df.plot("utilization","default",ax=ax1)

ax1.set_ylabel("Default Rate")

ax1.set_xlabel("Utilization Ratio")



utilization_outlier = df[df.default==0]['utilization'].min()

print("Remove Outliers at point UtilizatoinRation={}".format(utilization_outlier))



dftmp = df[df < utilization_outlier]

dftmp.plot("utilization","default",ax=ax2)

ax2.plot([3500 for i in range(dftmp.shape[0])], np.linspace(0,0.35,dftmp.shape[0]), 'r--')

ax2.plot([6200 for i in range(dftmp.shape[0])], np.linspace(0,0.35,dftmp.shape[0]), 'r--')

ax2.set_ylabel("Default Rate")

ax2.set_xlabel("Utilization Ratio")
removedUtilization = train_data[train_data.RevolvingUtilizationOfUnsecuredLines <= utilization_outlier]



def categorize_utilization(u):

    if u < 3500:

        return 0

    elif (u >= 3500) & (u < 6200):

        return 1

    else:

        return 2

    

removedUtilization["UtlizationCategory"] = removedUtilization["RevolvingUtilizationOfUnsecuredLines"].apply(categorize_utilization)
cols = train_data.columns

nullcounts = []

value_counts = []

for col in cols:

    nullcounts.append(train_data[col].isnull().sum())

    value_counts.append(train_data[col].shape[0] - train_data[col].isnull().sum())



fig, ax = plt.subplots(figsize=(10,3))

ax.barh(cols, value_counts, label='not missing')

ax.barh(cols, nullcounts, label='missing', left=value_counts)

ax.set_xlabel('Value Count')

ax.set_ylabel('Labels')

plt.show()
imputedf = train_data[['age','NumberOfDependents','MonthlyIncome']].copy()



def categorizeAge(age):

    if (age < 35):

        return 'junior'

    elif (age >= 35) & (age < 60):

        return'senior'

    else:

        return 'mature'



imputedf['seniority'] = imputedf['age'].apply(categorizeAge)

income_dict = imputedf.groupby('seniority')['MonthlyIncome'].mean().to_dict()

income_dict
#Impute Monthly Income by median of seniority

for k, v in income_dict.items():

    imputedf["MonthlyIncome"] = np.where((imputedf["MonthlyIncome"].isnull()) & (imputedf['seniority'] == k), int(v), imputedf["MonthlyIncome"])

train_data['MonthlyIncome'] = imputedf["MonthlyIncome"]
# Impute NumberOfDependents with Mode

print(train_data['NumberOfDependents'].mode())

# Fill na with mode 

train_data['NumberOfDependents'].fillna(0, inplace=True)
corr = train_data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True, fmt=".2f")
print(train_data["NumberOfTime30-59DaysPastDueNotWorse"].sort_values().unique())

print(train_data["NumberOfTime60-89DaysPastDueNotWorse"].sort_values().unique())

print(train_data["NumberOfTimes90DaysLate"].sort_values().unique())
tmpdf = train_data[(train_data["NumberOfTime30-59DaysPastDueNotWorse"] == 98) & (train_data["NumberOfTime60-89DaysPastDueNotWorse"] == train_data["NumberOfTime30-59DaysPastDueNotWorse"]) & (train_data["NumberOfTimes90DaysLate"] == train_data["NumberOfTime60-89DaysPastDueNotWorse"])][cols]

print("98 times past due where 3 columns have same value: {}".format(tmpdf.shape[0]))
# Impute outliers with max value

times_map = {"NumberOfTime30-59DaysPastDueNotWorse": 13, "NumberOfTime60-89DaysPastDueNotWorse":11, "NumberOfTimes90DaysLate":17}

for col, v in times_map.items():

    train_data.loc[train_data[col] >= 96, col] = times_map[col]
train_data["CombinedPastDue"] = train_data["NumberOfTime30-59DaysPastDueNotWorse"] + train_data["NumberOfTime60-89DaysPastDueNotWorse"] + train_data["NumberOfTimes90DaysLate"]
train_data["CombinedCreditLoans"] = train_data["NumberOfOpenCreditLinesAndLoans"] + train_data["NumberRealEstateLoansOrLines"]
income_dict
def categorizeAge(age):

    if (age < 35):

        return 'junior'

    elif (age >= 35) & (age < 60):

        return'senior'

    else:

        return 'mature'

    

def data_preprocess(df, is_submission=False):

    print("Shape before: {}".format(df.shape))

    df.loc[df['age'] < 18, 'age'] = df['age'].median()

    

    df["DebtRatio"] = np.log(df["DebtRatio"])

    df["DebtRatio"].replace([np.inf, -np.inf], -10, inplace=True)

    if not is_submission:

        df = df[df["MonthlyIncome"] != 1]

    

    utilization_outlier = 8328

    if not is_submission:

        df = df[df.RevolvingUtilizationOfUnsecuredLines <= utilization_outlier]

#     df["UtlizationCategory"] = df["UtilisationRatio"].apply(categorize_utilization)

    

    imputedf = df[['age','NumberOfDependents','MonthlyIncome']].copy()

    imputedf['seniority'] = imputedf['age'].apply(categorizeAge)

    income_dict = imputedf.groupby('seniority')['MonthlyIncome'].mean().to_dict()

    for k, v in income_dict.items(): 

        imputedf["MonthlyIncome"] = np.where((imputedf["MonthlyIncome"].isnull()) & (imputedf['seniority'] == k), int(v), imputedf["MonthlyIncome"])

    df['MonthlyIncome'] = imputedf["MonthlyIncome"]

    

    df['NumberOfDependents'].fillna(0, inplace=True)

    

    times_map = {"NumberOfTime30-59DaysPastDueNotWorse": 13, "NumberOfTime60-89DaysPastDueNotWorse":11, "NumberOfTimes90DaysLate":17}

    for col, v in times_map.items():

        df.loc[df[col] >= 96, col] = times_map[col]

    

    df["CombinedPastDue"] = df["NumberOfTime30-59DaysPastDueNotWorse"] + df["NumberOfTime60-89DaysPastDueNotWorse"] + df["NumberOfTimes90DaysLate"]

    

    df["CombinedCreditLoans"] = df["NumberOfOpenCreditLinesAndLoans"] + df["NumberRealEstateLoansOrLines"]

    

    print("Shape after: {}".format(df.shape))

    return df



train_data = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")

train_data.drop("Unnamed: 0", axis=1, inplace=True)



train_data = data_preprocess(train_data)
corr = train_data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True, fmt=".2f")
train_data.columns
cols_drop = ["NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse","NumberOfOpenCreditLinesAndLoans","NumberRealEstateLoansOrLines"]

train_data.drop(cols_drop, axis=1, inplace=True)
train_data.sample(5)
X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:,1:], train_data.iloc[:,0], random_state=42)
scaler = StandardScaler().fit(X_train)



X_train_scaled = scaler.transform(X_train) 

X_test_scaled = scaler.transform(X_test)
logit = LogisticRegression(random_state=42)

l_model = logit.fit(X_train_scaled, y_train)
logit_scores_proba  = l_model.predict_proba(X_test_scaled)

logit_scores = logit_scores_proba[:,1]
def plot_roc(y_test, y_predict):

    fpr, tpr, _ = roc_curve(y_test, y_predict)

    roc_auc = auc(fpr,tpr)

    print(roc_auc)

    plt.figure(figsize=(10,8))

    plt.title("ROC curve")

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0,1], [0,1],'r--')

    plt.legend(loc="lower right")

plot_roc(y_test, logit_scores)
random_forest = RandomForestClassifier()

param_grid={

    "n_estimators":[9,18,27,36,100],

    "max_depth":[5,7,9],

    "min_samples_leaf":[2,4,6,8]

}

rf_model = RandomizedSearchCV(random_forest, param_distributions = param_grid, cv=5)

rf_model.fit(X_train, y_train)
rf_model.best_params_
best_est_rf = rf_model.best_estimator_

best_est_rf.fit(X_train, y_train)

y_pred_rf = best_est_rf.predict_proba(X_test)[:,1]
plot_roc(y_test, y_pred_rf)
def plot_feature_importances(model):

    plt.figure(figsize=(10,8))

    n_features = X_train.shape[1]

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), X_train.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)



plot_feature_importances(best_est_rf)
test_data = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv")

test_data.drop("Unnamed: 0", axis=1, inplace=True)



test_data = data_preprocess(test_data, True)



cols_drop = ["SeriousDlqin2yrs","NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse","NumberOfOpenCreditLinesAndLoans","NumberRealEstateLoansOrLines"]

test_data.drop(cols_drop, axis=1, inplace=True)

test_data.shape
submission_score = best_est_rf.predict_proba(test_data)[:,1]



ids = np.arange(1,101504)

submission = pd.DataFrame( {'Id': ids, 'Probability': submission_score})

submission.to_csv("/kaggle/working/credit_score_submision.csv", index=False)