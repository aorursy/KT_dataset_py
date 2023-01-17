import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



plt.style.use('ggplot')

sns.set(style="ticks", context = 'talk', palette = 'bright', rc={'figure.figsize':(11.7,8.27)})
df = pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')
df.head()
print('Dataset has {} rows and {} columns.'.format(df.shape[0],df.shape[1]))
df.dtypes
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']

num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
df.describe()
df.isnull().sum()
df['Gender'].value_counts()
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].value_counts()
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].value_counts()
sns.countplot(x='Dependents', hue='Married', data=df)
df['Dependents'].fillna(df['Married'], inplace=True)

df['Dependents'] = df['Dependents'].apply(lambda x : {'No' : 0, 'Yes' : 1, '0' : 0, '1' : 1, '2' : 2, '3+' : 3}[x])
df.loc[df['Dependents'].isna() & (df['Married'] == 'Yes')]['Dependents'].fillna('1', inplace=True)
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].describe()
sns.distplot(df['LoanAmount'], rug = True, color = 'r')
df[~df['LoanAmount'].isnull()].groupby('Loan_Status').describe().T.loc['LoanAmount']
for row in range(df.shape[0]):

        if pd.isnull(df.loc[row, 'LoanAmount']):

            if df.loc[row, 'Loan_Status'] == 'Y':

                df.loc[row, 'LoanAmount'] = 151.22

            elif df.loc[row, 'Loan_Status'] == 'N':

                df.loc[row, 'LoanAmount'] = 144.29

            else:

                pass
df['Loan_Amount_Term'].describe()
df['Loan_Amount_Term'].value_counts()
df[~df['Loan_Amount_Term'].isnull()].groupby('Loan_Status').describe().T.loc['Loan_Amount_Term']
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].describe()
df['Credit_History'].value_counts()
df[~df['Credit_History'].isnull()].groupby('Loan_Status').describe().T.loc['Credit_History']
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df.isna().sum()
df['Loan_Status'].value_counts()
df['Loan_Status'].value_counts(normalize=True)
sns.countplot(x = 'Loan_Status', data = df)
plt.subplot(231)

df['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')



plt.subplot(232)

df['Married'].value_counts(normalize=True).plot.bar(title= 'Married')



plt.subplot(233)

df['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')



plt.subplot(234)

df['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')



plt.subplot(235)

df['Education'].value_counts(normalize=True).plot.bar(title= 'Education')



plt.show()
plt.subplot(121)

df['Dependents'].value_counts(normalize=True).plot.bar(figsize=(12,4), title= 'Dependents')



plt.subplot(122)

df['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')



plt.show()
plt.subplot(121)

sns.distplot(df['ApplicantIncome']);



plt.subplot(122)

df['ApplicantIncome'].plot.box(figsize=(16,5))



plt.show()
df.boxplot(column='ApplicantIncome', by = 'Education')

plt.suptitle("")
plt.subplot(121)

sns.distplot(df['CoapplicantIncome']);



plt.subplot(122)

df['CoapplicantIncome'].plot.box(figsize=(16,5))



plt.show()
plt.subplot(121)

sns.distplot(df['LoanAmount']);



plt.subplot(122)

df['LoanAmount'].plot.box(figsize=(16,5))



plt.show()
df['Loan_Amount_Term'].value_counts(normalize=True).plot.bar(title='Loan Amount Term')
Gender = pd.crosstab(df['Gender'], df['Loan_Status'])

print(Gender)

Gender.div(Gender.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
Married = pd.crosstab(df['Married'], df['Loan_Status'])

print(Married)

Married.div(Married.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
Edu = pd.crosstab(df['Education'], df['Loan_Status'])

print(Edu)

Edu.div(Edu.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
SE = pd.crosstab(df['Self_Employed'], df['Loan_Status'])

print(SE)

SE.div(SE.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
CH = pd.crosstab(df['Credit_History'], df['Loan_Status'])

print(CH)

CH.div(CH.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
prop = pd.crosstab(df['Property_Area'], df['Loan_Status'])

print(prop)

prop.div(prop.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
dep = pd.crosstab(df['Dependents'], df['Loan_Status'])

print(dep)

dep.div(dep.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
term = pd.crosstab(df['Loan_Amount_Term'], df['Loan_Status'])

print(term)

term.div(term.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked=True)

plt.ylabel('Percentage')
print(df.groupby('Loan_Status')['ApplicantIncome'].mean())



df.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

plt.ylabel('Applicant Income')
bins = [0,2500,4000,6000,81000]

group = ['Low','Average','High', 'Very high']

df['Income_bins'] = pd.cut(df['ApplicantIncome'],bins,labels=group)



Income_bin = pd.crosstab(df['Income_bins'],df['Loan_Status'])

print(Income_bin)

Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('ApplicantIncome')

P = plt.ylabel('Percentage')
bins = [0,1000,3000,42000]

group = ['Low','Average','High']

df['Co_Income_bins'] = pd.cut(df['CoapplicantIncome'],bins,labels=group)



CoIncome_bin = pd.crosstab(df['Co_Income_bins'],df['Loan_Status'])

print(CoIncome_bin)

CoIncome_bin.div(CoIncome_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Co-Applicant Income')

P = plt.ylabel('Percentage')
print("{:.2f}% of Co-applicant's income is 0".format(len(df[df['CoapplicantIncome'] == 0])/len(df)*100))
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']



bins = [0,2500,4000,6000,81000]

group = ['Low','Average','High', 'Very high']

df['Total_Income_bins'] = pd.cut(df['Total_Income'],bins,labels=group)



Total_Income_bin = pd.crosstab(df['Total_Income_bins'],df['Loan_Status'])

print(Total_Income_bin)

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Total Income')

P = plt.ylabel('Percentage')
bins = [0,100,300,700]

group = ['Low','Average','High']

df['LoanAmount_bins'] = pd.cut(df['LoanAmount'],bins,labels=group)



Loan_bin = pd.crosstab(df['LoanAmount_bins'],df['Loan_Status'])

print(Loan_bin)

Loan_bin.div(Loan_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Loan Amount')

P = plt.ylabel('Percentage')
df.drop(['Income_bins', 'Co_Income_bins', 'Total_Income', 'Total_Income_bins', 'LoanAmount_bins'], axis=1, inplace=True)
df.columns
df['Loan_Status'] = df['Loan_Status'].apply(lambda x : {'N' : 0, 'Y' : 1}[x])
sns.heatmap(df.corr(), square=True, cmap='BuPu', annot=True)
ax1 = plt.subplot(121)

df['LoanAmount'].hist(bins=20, figsize=(12,4))

ax1.set_title("Loan Amount")
df['LoanAmount'] = np.log(df['LoanAmount'])
ax1 = plt.subplot(121)

df['LoanAmount'].hist(bins=20, figsize=(12,4))

ax1.set_title("Loan Amount")
df.drop('Loan_ID', axis=1, inplace=True)
df['Property_Area'].unique()
df['Gender'] = df['Gender'].apply(lambda x : {'Male' : 1, 'Female' : 0}[x])

df['Married'] = df['Married'].apply(lambda x : {'Yes' : 1, 'No' : 0}[x])

df['Education'] = df['Education'].apply(lambda x : {'Graduate' : 1, 'Not Graduate' : 0}[x])

df['Self_Employed'] = df['Self_Employed'].apply(lambda x : {'Yes' : 1, 'No' : 0}[x])

df['Property_Area'] = df['Property_Area'].apply(lambda x : {'Semiurban': 2, 'Urban' : 1, 'Rural' : 0}[x])

df.dtypes
df.head()
df.describe()
X = df.drop('Loan_Status', axis=1)

Y = df['Loan_Status']
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost Classifier']

accs = []

f1 = []

aucs = []
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score



model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accs.append(np.mean(y_pred==y_test)*100)

f1.append(f1_score(y_test, y_pred))



print(classification_report(y_test, y_pred))



fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)



aucs.append(auc)



plt.figure(figsize=(12,8))

plt.plot(fpr,tpr,label="validation, auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accs.append(np.mean(y_pred==y_test)*100)

f1.append(f1_score(y_test, y_pred))



print(classification_report(y_test, y_pred))



fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(12,8))



aucs.append(auc)



plt.plot(fpr,tpr,label="validation, auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accs.append(np.mean(y_pred==y_test)*100)

f1.append(f1_score(y_test, y_pred))



print(classification_report(y_test, y_pred))



fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)



aucs.append(auc)



plt.figure(figsize=(12,8))

plt.plot(fpr,tpr,label="validation, auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
from xgboost import XGBClassifier



model = XGBClassifier(random_state=1, n_estimators=50, max_depth=4)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accs.append(np.mean(y_pred==y_test)*100)

f1.append(f1_score(y_test, y_pred))



print(classification_report(y_test, y_pred))



fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)



aucs.append(auc)



plt.figure(figsize=(12,8))

plt.plot(fpr,tpr,label="validation, auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
report = pd.DataFrame({'Models': models, 'Accuracy': accs, 'F1-Score': f1, 'AUC': aucs})

report.sort_values('F1-Score', inplace=True, ascending=False)
report