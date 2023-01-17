import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
plt.style.use('fivethirtyeight')
data = pd.read_csv('../input/loan.csv', low_memory=False)
data.drop(484446, inplace = True)
data.drop(531886, inplace = True)
data.drop(475046, inplace = True)
data.drop(532701, inplace = True)
data.drop(540456, inplace = True)
sns.set(font_scale=1.5)
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
sns.regplot(x='dti', y='annual_inc', data= data, line_kws={'color':'red'}, ax=ax)
data.boxplot(column='int_rate', by='grade', figsize=(12,6))
data.home_ownership.value_counts()
data=data.drop(data[data.home_ownership=='OTHER'].index)
data=data.drop(data[data.home_ownership=='ANY'].index)
data=data.drop(data[data.home_ownership=='NONE'].index)
data.home_ownership.replace('OWN','MORTGAGE', inplace=True)
data.home_ownership.value_counts()
data.loan_status.value_counts().plot(kind='barh', figsize=(7,5), title = "Loan Status", fontsize = 15)
matureLoan = data[(data.loan_status=='Fully Paid') | (data.loan_status=='Charged Off')].copy()
matureLoan.loan_status.value_counts()
possibleFeatures=matureLoan[['emp_length', 'home_ownership', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                                'mths_since_last_delinq','mths_since_last_record', 'open_acc', 'pub_rec',
                                'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med',
                                'mths_since_last_delinq', 'open_acc_6m', 'open_il_6m','open_il_12m',
                                'open_il_24m', 'mths_since_rcnt_il', 'il_util','open_rv_12m', 'open_rv_24m','max_bal_bc',
                                'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m','tot_coll_amt','tot_cur_bal','loan_status']]
possibleFeatures.isnull().sum()
matureLoan.emp_length.replace({'10+ years':10, '< 1 year':1, '1 year':1, '3 years':3, '8 years':8, '9 years':9, '4 years':4, '5 years':5, '6 years':6, '2 years':2, '7 years':7}, inplace=True)
matureLoan.emp_length.value_counts(dropna=False)
features = pd.get_dummies(matureLoan[['emp_length', 'home_ownership', 'annual_inc', 'dti', 'delinq_2yrs', 
                                       'inq_last_6mths', 'open_acc', 'pub_rec','revol_bal', 'revol_util', 'total_acc',
                                       'collections_12_mths_ex_med','tot_coll_amt','tot_cur_bal','loan_status']],
                                    drop_first = True)
features.isnull().sum()
features.dropna(inplace=True)
features.isnull().sum()
features.describe()
X=features.drop('loan_status_Fully Paid', axis=1)
y=features['loan_status_Fully Paid']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
name = features.columns

coef = logreg.coef_[0]

pd.DataFrame([name,coef],index = ['Name','Coef']).transpose()
features1 = pd.get_dummies(matureLoan[['annual_inc', 'dti','inq_last_6mths', 'revol_util', 'total_acc','tot_cur_bal','loan_status']],
                                    drop_first = True)
features1.dropna(inplace=True)
features1.isnull().sum()
X1=features1.drop('loan_status_Fully Paid', axis=1)
y1=features1['loan_status_Fully Paid']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)
logreg1= LogisticRegression()
logreg1.fit(X1_train,y1_train)
name = features1.columns

coef = logreg1.coef_[0]

pd.DataFrame([name,coef],index = ['Name','Coef']).transpose()
y_pred1 = logreg1.predict(X1_test)
metrics.accuracy_score(y1_test,y_pred1)
y1_test.mean()
from sklearn.tree import DecisionTreeClassifier

treeclf = DecisionTreeClassifier(max_depth=4, random_state=42)
treeclf.fit(X, y)
pd.DataFrame({'feature':X.columns, 'importance':treeclf.feature_importances_})
cm = metrics.confusion_matrix(y1_test,y_pred1)
plt.clf()
plt.rcParams["figure.figsize"] = [6,6]
plt.imshow(cm, cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Loan Status Fully Paid')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
y_pred_prob = logreg1.predict_proba(X1_test)[:, 1]
plt.rcParams['font.size'] = 14
plt.rcParams["figure.figsize"] = [7,7]
plt.hist(y_pred_prob)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')
plt.rcParams["figure.figsize"] = [7,7]
plt.hist(y_pred_prob, label='prediction')
plt.hist(y1_test, label='test')
plt.xlim(0, 1)
plt.title('Histogram of test data vs. prediction')
plt.xlabel('Actual data vs. predicted probability')
plt.ylabel('Frequency')
plt.legend()
y_pred_class6 = np.where(y_pred_prob > 0.6, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class6)
y_pred_class7 = np.where(y_pred_prob > 0.7, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class7)
y_pred_class8 = np.where(y_pred_prob > 0.8, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class8)
y_pred_class9 = np.where(y_pred_prob > 0.9, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class9)
matureLoan['costChargeOff'] = matureLoan.loan_amnt - matureLoan.total_pymnt
cost=matureLoan.costChargeOff[matureLoan.loan_status=='Charged Off']
cost.mean()
benefit = matureLoan.total_rec_int[matureLoan.loan_status=='Fully Paid']
benefit.mean()
# Benefit of a True Positive = $1902
BTP = 1902
# Benefit of a True Negative = $0 since they don't qualify for the loan
BTN = 0
# Cost of a False Positive = $8188
CFP = -8188
# Cost of a False Negative = $0 since they don't qualify for the loan
CFN = 0

# Calculate the probabilities for each confusion matrix entry
TP = 46168/56718
TN = 3/56718
FP = 10537/56718
FN = 10/56718

TP, TN, FP, FN
EV = BTP * TP + BTN * TN + CFP * FP + CFN * FN
EV
thresholds=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
MSE=[27, 37, 127, 229, 308, 113]

plt.plot(thresholds, MSE)
plt.xlabel('Threshold')
plt.ylabel('Expected Value ($)')
