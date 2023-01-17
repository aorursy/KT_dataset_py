# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# Importing dataset

data = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

data.head(10)
data.isnull().sum()
from sklearn.impute import SimpleImputer as Imputer

imputer = Imputer(missing_values=np.nan, strategy='most_frequent', verbose=0)

for i in range(len(data.columns)):

    data[data.columns[i]] = imputer.fit_transform(data[data.columns[i]].values.reshape(-1, 1))
data.isnull().sum().sum()

cat_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig, ax =plt.subplots(1, len(cat_columns), figsize=(15, 6), sharey=True)

for i in range(len(cat_columns)): 

    cat = cat_columns[i]

    axx = ax[i]

    data.boxplot(column=cat, by='Loan_Status', ax=axx)

    plt.sca(axx)

    plt.yscale("log")
cat_columns = ['Gender', 'Married','Education', 'Self_Employed', 'Dependents', 'Credit_History', 'Property_Area']

fig, ax =plt.subplots(1, len(cat_columns), figsize=(20, 4), sharey=True)

for i in range(len(cat_columns)): 

    cat = cat_columns[i]

    axx = ax[i]

    data.groupby([cat, 'Loan_Status'])[cat].count().unstack().plot(kind='bar', ax=axx)
cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Dependents', 'Credit_History', 'Property_Area']

fig, ax =plt.subplots(1, len(cat_columns), figsize=(20, 4), sharey=True)

for i in range(len(cat_columns)): 

    cat = cat_columns[i]

    categories = list(set(data[cat]))

    axx = ax[i]

    df = []

    for x in categories:

        data_x = data.loc[data[cat] == x]

        data_loc = data_x.loc[data_x['Loan_Status']=='Y']

        df.append(data_loc.Loan_Status.value_counts().values[0]/len(data_x))

    

    plt.sca(axx)

    plt.suptitle('Percentage of accorded Loan respect to different categories')

    plt.title(cat)

    plt.bar(categories, df)

X = data.iloc[:, 1:-1].values

y = data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
# find categorical data indexes

cat_list = []

for i in range(len(data.columns)):

    col =data.columns[i]

    if data[col].dtypes == 'O':

        cat_list.append(i)

# rewrite indexes for X

cat_list.pop(0)

cat_list.pop(-1)

cat_list = [cat_list[x]-1 for x in range(len(cat_list))]

cat_list
for x in cat_list:

    X[:, x] = LabelEncoder().fit_transform(X[:, x])
# %% Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_sc = sc_X.fit_transform(X)

# Transform in dataframe only to have a look

data_enc = pd.DataFrame(X_sc, columns=data.columns[1:-1])

data_enc['Status'] = y

#import seaborn as sns

corr = data_enc.corr()

corr['Status'].abs().sort_values(ascending= False)

#plt.figure(figsize=(10,7))

#ax = sns.heatmap(corr, annot=True)

#Similarity are not OK ( loan amount and applicanr income)



# we can use only the first 4
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,

                                                    random_state=0)



# %% Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# we will use 4 different models for training



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



models = {

    'LogisticRegression': LogisticRegression(random_state=42, solver='lbfgs'),

    'KNeighborsClassifier': KNeighborsClassifier(),

    'SVC': SVC(random_state=42, gamma='auto'),

    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)

}



# %% Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score



def train_eval_kfold(models, X, y):

    for name, model in models.items():

        print(name,':')

        accuracies = cross_val_score(estimator=model,

                                 X=X, y=y, cv=10)

        print('- mean accuracies:', accuracies.mean())

        print('- std accuracies:', accuracies.std())

        print('-'*30)

        

train_eval_kfold(models, X_train, y_train)