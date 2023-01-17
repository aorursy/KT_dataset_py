import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/Train.csv')
df.head()
df.info()
x_categories = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
for element in x_categories:
    cat_plot = pd.crosstab(df[element], df['Loan_Status'])
    cat_plot.plot(kind='bar', stacked=True, color=['red', 'green'])
for category in x_categories:
    df[category] = df[category].astype('category')
df.info()
df.describe()
numeric_categories = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for element in numeric_categories:
    x = df[element]
    x.plot(kind='hist', bins = 30, fontsize = 20, figsize= (20,10), edgecolor='black', linewidth=1.2)
    plt.xlabel(element, fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.show()
total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
df['total_income_log'] = np.log(total_income)
df['LoanAmount_log'] = np.log(df['LoanAmount'])

df['total_income_log'].plot(kind='hist', bins = 30, fontsize = 20, figsize= (20,10), edgecolor='black', linewidth=1.2)
plt.xlabel(element, fontsize=30)
plt.ylabel('Frequency', fontsize=30)
plt.show()

df['LoanAmount_log'].plot(kind='hist', bins = 30, fontsize = 20, figsize= (20,10), edgecolor='black', linewidth=1.2)
plt.xlabel(element, fontsize=30)
plt.ylabel('Frequency', fontsize=30)
plt.show()
df.drop(['Loan_ID', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'], axis=1, inplace=True)
df['Dependents'].replace('0', 'none', inplace=True)
df['Dependents'].replace('1', 'one', inplace=True)
df['Dependents'].replace('3+', 'three+', inplace=True)
df.head()
df.info()
df_dummies = pd.get_dummies(df, columns = ['Gender', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
df_dummies.Married.replace(('Yes', 'No'), (1,0), inplace=True)
df_dummies.Loan_Status.replace(('Y', 'N'), (1,0), inplace=True)
df_dummies.head()
df_dummies['total_income_log'].fillna(df_dummies['total_income_log'].mean(), inplace=True)
df_dummies['LoanAmount_log'].fillna(df_dummies['LoanAmount_log'].mean(), inplace=True)
df_dummies['Credit_History'].fillna(df_dummies['Credit_History'].value_counts().index[0], inplace=True)
df_dummies['Married'].fillna(df_dummies['Married'].value_counts().index[0], inplace=True)
df_dummies['Credit_History'] = df_dummies['Credit_History'].astype('float64')
df_dummies.info()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
X = df_dummies.drop('Loan_Status', axis=1).values
y = df_dummies['Loan_Status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X_train, y_train)
print("Tuned log reg parameter {}".format(logreg_cv.best_params_))
print("Highest score {}".format(logreg_cv.best_score_))

