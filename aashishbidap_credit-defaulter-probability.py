import pandas as pd
import numpy as py
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
#test data
test = pd.read_csv("../input/GiveMeSomeCredit/cs-test.csv")
#train data
training = pd.read_csv("../input/GiveMeSomeCredit/cs-training.csv")
#data dictionary
data_dictionary = pd.read_excel("../input/GiveMeSomeCredit/Data Dictionary.xls")
training.head(10)
training.isna().sum()
test.isna().sum()
sns.barplot(x=training['SeriousDlqin2yrs'].value_counts().index,y=training['SeriousDlqin2yrs'].value_counts())
plt.title("Distribution of the Defaulters in the data")
print("Minimum Age",training.age.min())
print("Maximum Age",training.age.max())
print("Median Age",training.age.median())
print("Mean Age",training.age.mean())
print("Mode Age",training.age.mode()[0])

sns.distplot(training['age'],bins=10)
training.loc[training['age'] == 0, 'age']
training.loc[training['age'] == 0, 'age'] = training.age.mode()[0]
default_0 = training[training['SeriousDlqin2yrs'] == 0]
sns.distplot(default_0['age'],bins=7)
default_1 = training[training['SeriousDlqin2yrs'] == 1]
sns.distplot(default_1['age'],bins=7)
sns.scatterplot(x=training['DebtRatio'],y=training['age'])
sns.scatterplot(x=training['DebtRatio'],y=training['RevolvingUtilizationOfUnsecuredLines'])
training['MonthlyIncome'].describe()
monthly_income_less_10000 = training[training['MonthlyIncome'] < training['MonthlyIncome'].quantile(0.99)]
sns.scatterplot(x=monthly_income_less_10000['MonthlyIncome'],y=monthly_income_less_10000['NumberOfOpenCreditLinesAndLoans'])
f, ax = plt.subplots(figsize=(10, 8))
corr = training.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
print("Minimum MonthlyIncome",training.MonthlyIncome.min())
print("Maximum MonthlyIncome",training.MonthlyIncome.max())
print("Median MonthlyIncome",training.MonthlyIncome.median())
print("Mean MonthlyIncome",training.MonthlyIncome.mean())
print("Mode MonthlyIncome",training.MonthlyIncome.mode()[0])
print("Null Values",training.MonthlyIncome.isna().sum())

sns.distplot(training['MonthlyIncome'])
training.loc[training.MonthlyIncome.isna(),'MonthlyIncome'] = training.MonthlyIncome.median()
print("Minimum NumberOfDependents",training.NumberOfDependents.min())
print("Maximum NumberOfDependents",training.NumberOfDependents.max())
print("Median NumberOfDependents",training.NumberOfDependents.median())
print("Mean NumberOfDependents",training.NumberOfDependents.mean())
print("Mode NumberOfDependents",training.NumberOfDependents.mode()[0])
print("Null Values",training.NumberOfDependents.isna().sum())

sns.distplot(training['NumberOfDependents'])
training.loc[training.NumberOfDependents.isna(),'NumberOfDependents'] = training.NumberOfDependents.median()
sns.barplot(x=training['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().index,y=training['NumberOfTime30-59DaysPastDueNotWorse'].value_counts())
training.isna().sum()
#Spliting of Data:
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,roc_curve
y = training.loc[:,training.columns.isin(['SeriousDlqin2yrs'])]
X_attributes=[
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines',
       'NumberOfDependents'] # Excluding NumberOfTime60-89DaysPastDueNotWorse' because of strong collinearity
X = training.loc[:,training.columns.isin(X_attributes)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
xgb_model = xgb.XGBClassifier(objective="binary:logistic",random_state=42)
xgb_model.fit(X_train,y_train.values.ravel())
y_pred = xgb_model.predict(X_test)
y_probab = xgb_model.predict_proba(X_test)
accuracy_score(y_pred,y_test)
#Feature Importance Plot
feature_important = xgb_model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.plot(kind='barh')
def plot_roc(y_test,probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()
plot_roc(y_test,y_probab[:,1])
test.isna().sum()
test.loc[test['MonthlyIncome'].isna(),'MonthlyIncome'] = test['MonthlyIncome'].dropna().median()
test.loc[test['NumberOfDependents'].isna(),'NumberOfDependents'] = test['NumberOfDependents'].dropna().mode()
test_proba = xgb_model.predict_proba(test.loc[:,test.columns.isin(X_attributes)])
len(np.arange(1,len(test_proba)+1))
len(test_proba)
df = pd.DataFrame({'Id':np.arange(1,len(test_proba)+1),'Probability':test_proba[:,1]})
#Test data predicitions
df
!pwd
df.to_csv('submission.csv', index = False)
import shap

mybooster=xgb_model.get_booster()

model_bytearray = mybooster.save_raw()[4:]

def myfun(self=None):
    return model_bytearray

mybooster.save_raw = myfun
explainerXGB = shap.TreeExplainer(mybooster)
shap_values = explainerXGB.shap_values(X_train.loc[:,X_train.columns.isin(feature_important)])
shap.summary_plot(
    shap_values,
    X_train.loc[:,X_train.columns.isin(feature_important)],
    max_display=110,
    show=True,
)