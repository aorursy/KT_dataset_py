# Data Processing

import numpy as np 

import pandas as pd 



# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='whitegrid')



# Modeling

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold



from sklearn.linear_model import SGDClassifier



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score





from sklearn.model_selection import RandomizedSearchCV
df_train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
df_train.head()
df_test.head()
df_train = df_train.drop(['id'], axis=1)

df_test = df_test.drop(['id'], axis=1)
df_train[['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']].describe()
df_train.isna().sum()
df_test.isna().sum()
df_train['Response'].value_counts()
b = sns.countplot(x='Response', data=df_train)

b.set_title("Response Distribution")
b = sns.countplot(x='Gender', data=df_train)

b.set_title("Gender Distribution");
pd.crosstab(df_train['Response'], df_train['Gender']).plot(kind="bar", figsize=(10,6))



plt.title("Response distribution for Gender")

plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")

plt.ylabel("Amount")

plt.legend(["Female", "Male"])

plt.xticks(rotation=0);
b = sns.distplot(df_train['Age'])

b.set_title("Age Distribution");
b = sns.boxplot(y = 'Age', data = df_train)

b.set_title("Age Distribution");
b = sns.boxplot(y='Age', x='Response', data=df_train);

b.set_title("Age Distribution for each Response");
df_train['Driving_License'].value_counts()
df_train = df_train.drop("Driving_License", axis=1)

df_test = df_test.drop("Driving_License", axis=1)
df_train['Region_Code'].value_counts().head(30).plot(kind='barh', figsize=(20,10), title="Region_Code distribution in df_train");
df_train['Previously_Insured'].value_counts()
pd.crosstab(df_train['Response'], df_train['Previously_Insured'])
pd.crosstab(df_train['Response'], df_train['Previously_Insured']).plot(kind="bar", figsize=(10,6))



plt.title("Response distribution for Previously_Insured")

plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")

plt.ylabel("Amount")

plt.legend(["Customer doesn't have Vehicle Insurance", "Customer already has Vehicle Insurance"])

plt.xticks(rotation=0);
df_train['Vehicle_Age'].value_counts()
pd.crosstab(df_train['Response'], df_train['Vehicle_Age']).plot(kind="bar", figsize=(10,6))



plt.title("Response distribution for Vehicle_Age")

plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")

plt.ylabel("Amount")

plt.legend(["1-2 Year", "< 1 Year", "> 2 Years"])

plt.xticks(rotation=0);
df_train['Vehicle_Damage'].value_counts()
pd.crosstab(df_train['Response'], df_train['Vehicle_Damage'])
pd.crosstab(df_train['Response'], df_train['Vehicle_Damage']).plot(kind="bar", figsize=(10,6))



plt.title("Response distribution for Vehicle_Damage")

plt.xlabel("0 = Customer is Not interested, 1 = Customer is interested")

plt.ylabel("Amount")

plt.legend(["Vehicle Damage", "No Vehicle Damage"])

plt.xticks(rotation=0);
df_train['Annual_Premium'].describe()
b = sns.boxplot(y='Annual_Premium', x='Response', data=df_train);

b.set_title("Annual_Premium Distribution for each Response");
df_train['Policy_Sales_Channel'].describe()
b = sns.boxplot(y='Policy_Sales_Channel', x='Response', data=df_train);

b.set_title("Policy_Sales_Channel Distribution for each Response");
df_train['Vintage'].describe()
b = sns.boxplot(y='Vintage', x='Response', data=df_train);

b.set_title("Vintage Distribution for each Response");
df_train.head()
df_train['Gender'] = pd.Categorical(df_train['Gender'])

df_train['Previously_Insured'] = pd.Categorical(df_train['Previously_Insured'])

df_train['Vehicle_Age'] = pd.Categorical(df_train['Vehicle_Age'])

df_train['Vehicle_Damage'] = pd.Categorical(df_train['Vehicle_Damage'])

df_train['Response'] = pd.Categorical(df_train['Response'])

df_train['Region_Code'] = pd.Categorical(df_train['Region_Code'])



df_train = pd.concat([df_train[['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']],

           pd.get_dummies(df_train[['Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']])], axis=1)
df_train.head()
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df_train.corr()

sns.heatmap(cor, annot=True)

plt.show()
X = df_train.drop(["Response"], axis=1).to_numpy()

y = df_train['Response'].values
np.random.seed(42)



# Defining a dictionary of models

models = {"Logistic Regression": LogisticRegression(max_iter=10000), 

          "Random Forest": RandomForestClassifier(),

          "GradientBoostingClassifier" : GradientBoostingClassifier()}





# Initialize StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True)









for name, model in models.items():

    

    # Create list for ROC AUC scores

    roc_auc_score_list = []

    

    for train_index, test_index in skf.split(X,y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        model.fit(X_train, y_train)



        roc_auc_score_list.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

        plt.plot(fpr, tpr)



        #print(f"ROC AUC Score for the fold no. {i} on the test set: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")





    print(f'Mean roc_auc_score {name} : {np.mean(roc_auc_score_list)}')