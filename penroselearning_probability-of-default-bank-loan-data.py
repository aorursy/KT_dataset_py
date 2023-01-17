import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Does Float Values with 2 Decimal Places 
pd.options.display.float_format = "{:,.2f}".format
data = pd.read_csv('../input/sample-data-bank-loans/bank_loan_data.csv')
data.shape
data.info()
data.head()
data.describe()
data["Years in current job"].unique()
data["Home Ownership"].unique()
data["Purpose"].unique()
data.isnull().sum()
data = data.iloc[:,2:]
data.drop('Months since last delinquent',axis=1,inplace=True)
data["Years in current job"].fillna(data["Years in current job"].mode()[0],inplace = True)
cols = list(data.select_dtypes(include=['int64','float64']).columns)

for col in cols:
    data[col].fillna(data[col].median(),inplace = True)
data.insert(3,"Defaulter",np.where(data['Loan Status'] == "Fully Paid", 0,1))
cols = ['Term', 'Years in current job', 'Home Ownership',
       'Purpose']

for col in cols:
    data[col] = pd.Categorical(data[col])
    data[col] = data[col].cat.codes
cols = ['Current Loan Amount','Annual Income','Monthly Debt',
        'Number of Credit Problems','Current Credit Balance']

plt.figure(figsize=(12,10))
for sno,col in enumerate(cols,321):
    ax = plt.subplot(sno)
    ax.boxplot(data[col])
    ax.set_ylabel(col)
cols = ['Current Loan Amount','Annual Income','Monthly Debt',
        'Number of Credit Problems','Current Credit Balance']

data_zscore = np.abs(data[cols].apply(stats.zscore))
outliers = list(data_zscore[data_zscore > 3].dropna(thresh=1).index)
data.drop(data.index[outliers],inplace=True)
plt.figure(figsize=(16,6))
sns.heatmap(data.corr().abs(),xticklabels=data.corr().abs().columns, yticklabels=data.corr().abs().columns, annot=True)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

cols = list(data.select_dtypes(include=['int64','float64']).columns)

X = data[['Current Loan Amount', 'Term', 'Credit Score',
       'Annual Income', 'Years in current job', 'Home Ownership', 'Purpose',
       'Monthly Debt', 'Years of Credit History', 'Number of Open Accounts',
       'Number of Credit Problems', 'Current Credit Balance',
       'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']]


y = data[['Defaulter']]

model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print(f"Num Features: {fit.n_features_}")
print(f"Feature Ranking: {fit.ranking_}")
ranking = pd.Series(fit.ranking_)

ranked_cols= list(ranking.where(ranking==1).dropna().index)
selected_cols = list(X.iloc[:,ranked_cols].columns)

print("Selected Columns using RFE")
print("-" * 30)
for c in selected_cols:
    print(c)
X = data[["Term","Home Ownership","Number of Credit Problems","Credit Score"]]

y = data[['Defaulter']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=100)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)
y_predict = model.predict(X_test)
y_predict
f'{round(model.score(X_test,y_test) * 100,2)}%'
f'{round(model.score(X_train,y_train) * 100,2)}%'
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_score(y_test, y_pred_dt)
rf = RandomForestClassifier(n_estimators=1000, max_features="auto",random_state=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_score(y_test, y_pred_rf)
ab = AdaBoostClassifier(n_estimators=1000, random_state=100)
ab.fit(X_train, y_train)
y_pred_ab = ab.predict(X_test)
accuracy_score(y_test, y_pred_ab)
gb = GradientBoostingClassifier(n_estimators=1000,random_state=100)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
accuracy_score(y_test, y_pred_gb)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predict))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
tn_fp,fn_tp = confusion_matrix(y_test, y_pred_dt)
classification_report(y_test, y_pred_dt)

print(f"True Negative: {tn_fp[0]:,}")
print(f"False Positive: {tn_fp[1]:,}")
print(f"False Negative: {fn_tp[0]:,}")
print(f"True Positive: {fn_tp[1]:,}")
tn_fp,fn_tp = confusion_matrix(y_test, y_pred_rf)
classification_report(y_test, y_pred_rf)

print(f"True Negative: {tn_fp[0]:,}")
print(f"False Positive: {tn_fp[1]:,}")
print(f"False Negative: {fn_tp[0]:,}")
print(f"True Positive: {fn_tp[1]:,}")
tn_fp,fn_tp = confusion_matrix(y_test, y_pred_ab)
classification_report(y_test, y_pred_ab)

print(f"True Negative: {tn_fp[0]:,}")
print(f"False Positive: {tn_fp[1]:,}")
print(f"False Negative: {fn_tp[0]:,}")
print(f"True Positive: {fn_tp[1]:,}")
tn_fp,fn_tp = confusion_matrix(y_test, y_pred_gb)
classification_report(y_test, y_pred_gb)

print(f"True Negative: {tn_fp[0]:,}")
print(f"False Positive: {tn_fp[1]:,}")
print(f"False Negative: {fn_tp[0]:,}")
print(f"True Positive: {fn_tp[1]:,}")
pred = y_test.copy()
pred.insert(1,"Prediction",y_pred_gb)
pred.rename(columns={"Defaulter":"Actual"},inplace=True)
fn = list(pred[(pred["Actual"]==1) & (pred["Prediction"]==0)].index)
fp = list(pred[(pred["Actual"]==0) & (pred["Prediction"]==1)].index)
data[data.index.isin(fn)].to_csv("fn.csv")
data[data.index.isin(fp)].to_csv("fp.csv")
"Non-Defaulter" if ab.predict([[1,5,0,500]])[0] == 0 else "Defaulter"