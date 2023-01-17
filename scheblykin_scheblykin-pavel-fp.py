import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sb



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



%matplotlib inline

%time

df = pd.read_csv("../input/datacrtr/credit_train.csv")
print('Кол-во строк: ', df.shape[0])

print('Кол-во столбцов: ', df.shape[1])
df = df.fillna(0)
df.info
df.describe()
df['Loan Status'].value_counts().plot(kind='bar', label='Loan Status')

plt.legend()

plt.title('1.0 - Fully Paid, 0.0 - Charged Off')
cor = df.corr()

plt.figure(figsize=(12,12))

sb.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,annot=True)

plt.show

#Создаем тепловую карту для определения наиболее значимых факторов
features = list(set(df.columns) - set(['Loan Status','Credit Score','Annual Income',

                                       'Monthly Debt','Number of Credit Problems',

                                       'Current Credit Balance','Bankruptcies','Tax Liens']))



df[features].hist(figsize=(20,12))
sb.pairplot(df[features + ['Loan Status']], hue='Loan Status')
sb.lmplot(x='Annual Income', y='Credit Score', size=4, aspect=2, data=df, )
sb.jointplot(x='Annual Income', y='Credit Score', data=df, kind='reg')
credit_model = df[['Loan Status','Credit Score','Annual Income',

                   'Monthly Debt','Number of Credit Problems','Current Credit Balance',

                   'Bankruptcies','Tax Liens']]

credit_model.head()
credit_model.info
credit_model.describe()
credit_model.hist(figsize = (20,20), bins = 80)
X = credit_model[['Credit Score','Annual Income','Monthly Debt',

                  'Number of Credit Problems','Current Credit Balance',

                  'Bankruptcies','Tax Liens']]

y = credit_model[['Loan Status']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2019)

X_train = X_train.fillna(0)
models = []

accuracy_score_t = []
def model_performance(model,model_name,features = X_train,target = y_train, test_features = X_test, true_values = y_test):

    models.append(model_name)

    model.fit(features,target)

    y_pred = model.predict(test_features)

    accuracy = accuracy_score(y_pred, true_values)

    accuracy_score_t.append(accuracy)

    cm = confusion_matrix(y_pred, true_values)

    print(accuracy)

    print(cm)
reg = LogisticRegression()

model_performance(reg, 'Logistic Regression')
reg = DecisionTreeClassifier()

model_performance(reg, 'Decision Tree Classifier')
reg = KNeighborsClassifier()

model_performance(reg, 'KNN Classifier')
reg = GaussianNB()

model_performance(reg, 'Naive Bayes')
reg = RandomForestClassifier()

model_performance(reg, 'Random Forest Classifier')
reg = AdaBoostClassifier()

model_performance(reg, 'AdaBoost Classifier')
reg = GradientBoostingClassifier()

model_performance(reg, 'GradientBoosting Classifier')
model_comparision_df = pd.DataFrame({'Model': models, 'Accuracy Score': accuracy_score_t})

model_comparision_df = model_comparision_df.sort_values(by = 'Accuracy Score', ascending= False)

sb.barplot(x = 'Accuracy Score', y='Model', data = model_comparision_df)