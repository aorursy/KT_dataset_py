import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
bank_customers = pd.read_csv("../input/bank-customer-churn-modeling/Churn_Modelling.csv")
bank_customers.head(10)
bank_customers.info()
# Drop RowNumber and CustomerId because it won't be useful in predictive task
bank_customers.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
# Convert object dtype into category
cat_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Tenure']
for colname in cat_features:
    bank_customers[colname] = bank_customers[colname].astype('category')
bank_customers[cat_features].nunique()
sns.countplot(x='Geography', data=bank_customers)
sns.countplot(x='Gender', data=bank_customers)
sns.countplot(x='HasCrCard', data=bank_customers)
sns.countplot(x='IsActiveMember', data=bank_customers)
sns.countplot(x='NumOfProducts', data=bank_customers)
sns.countplot(x='Tenure', data=bank_customers, color='blue')
bank_customers.groupby('Geography').mean()
bank_customers.groupby('Gender').mean()
bank_customers.groupby('HasCrCard').mean()
bank_customers.groupby('IsActiveMember').mean()
bank_customers.groupby("NumOfProducts").mean()
bank_customers.groupby('Tenure').mean()
bank_customers.groupby("Exited").mean()
sns.countplot(x='NumOfProducts', hue='Exited', data=bank_customers)
sns.countplot(x=bank_customers['IsActiveMember'], hue=bank_customers['Exited'])
sns.countplot(x=bank_customers["Gender"], hue=bank_customers['Exited'])
sns.boxplot(x=bank_customers['Geography'], y=bank_customers['Balance'])
sns.boxplot(x=bank_customers['Geography'], y=bank_customers['Balance'], hue=bank_customers['Gender'])
bank_customers.columns
num_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
bank_customers[num_features].nunique()
bank_customers['CreditScore'].hist(bins=30, grid=False)
bank_customers[bank_customers['CreditScore'] == bank_customers['CreditScore'].max()]['Exited'].value_counts()
bank_customers['Age'].hist(bins=30, grid=False)
bank_customers['Balance'].hist(bins=30, grid=False)
bank_customers[bank_customers['Balance'] == 0]['Exited'].value_counts()
bank_customers['EstimatedSalary'].hist(bins=30, grid=False)
#num_df = bank_customers[num_features + ['Exited']] 
sns.pairplot(bank_customers[num_features + ['Exited']] , hue='Exited')
from sklearn.model_selection import train_test_split
X = bank_customers.drop(['Exited'], axis = 1)
y = bank_customers['Exited']
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.25, random_state=42)
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
target_names = ['Did not Exit', "Exited"]
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
pred = dummy_clf.predict(X_train)
print(classification_report(y_train, pred))
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
standardized_df = pd.DataFrame(ss.fit_transform(bank_customers[num_features]), columns = num_features)
standardized_df['Exited'] = bank_customers['Exited']
from sklearn.preprocessing import LabelEncoder
le_geography = LabelEncoder()
le_gender = LabelEncoder()
le_HasCrCard = LabelEncoder()
le_IsActiveMember = LabelEncoder()
le_NumOfProducts = LabelEncoder()
le_Tenure = LabelEncoder()

le_df = pd.DataFrame()

le_df['Geography'] = le_geography.fit_transform(bank_customers['Geography'])
le_df['Gender'] = le_gender.fit_transform(bank_customers['Gender'])
le_df['HasCrCard'] = le_HasCrCard.fit_transform(bank_customers['HasCrCard'])
le_df['IsActiveMember'] = le_IsActiveMember.fit_transform(bank_customers['IsActiveMember'])
le_df['NumOfProducts'] = le_NumOfProducts.fit_transform(bank_customers['NumOfProducts'])
le_df['Tenure'] = le_Tenure.fit_transform(bank_customers['Tenure'])

le_df.head()
model_df = pd.concat([standardized_df,le_df],axis=1)
model_df.head()

X = model_df.drop(['Exited'],axis=1)
y = model_df['Exited']
X_train, x_test, y_train, y_test = train_test_split(X,y)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error

dt_clf = DecisionTreeClassifier(max_depth=5,max_features='sqrt')

dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(x_test)
print(classification_report(y_test, pred))
fig, ax = plt.subplots(figsize=(20, 20))
tree.plot_tree(dt_clf.fit(X_train, y_train), ax=ax, filled=True)
plt.show()
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
pred = neigh.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1800, min_samples_split=2, 
                             min_samples_leaf=1, max_features='auto', 
                             max_depth=10, bootstrap=True)
rf.fit(X_train,y_train)
pred = rf.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(n_estimators=1800)
ada_clf.fit(X_train, y_train)
pred = ada_clf.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
 
gpc = GaussianProcessClassifier().fit(X_train, y_train)
gpc.score(X_train, y_train)
from sklearn.naive_bayes import GaussianNB
GNB_clf = GaussianNB()
GNB_clf.fit(X_train, y_train)
pred = GNB_clf.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
svc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_clf.fit(X_train, y_train)
pred = svc_clf.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.neural_network import MLPClassifier
MLP_clf = MLPClassifier(max_iter=3000).fit(X_train, y_train)
pred = MLP_clf.predict(x_test)
print(classification_report(y_test, pred))
rf = RandomForestClassifier(n_estimators=1800, min_samples_split=2, 
                             min_samples_leaf=1, max_features='auto', 
                             max_depth=10, bootstrap=True)
rf.fit(X_train,y_train)
pred = rf.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


clf = RandomizedSearchCV(rf, random_grid)
search = clf.fit(X_train,y_train)
search.best_params_
X_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
rf = RandomForestClassifier(n_estimators=2000, min_samples_split=2, 
                             min_samples_leaf=4, max_features='auto', 
                             max_depth=50, bootstrap=True)
rf.fit(X_train,y_train)
pred = rf.predict(x_test)
print(classification_report(y_test, pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)