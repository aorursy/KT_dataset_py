import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pickle
# loading the dataset

df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
df.isna().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values =np.NaN,strategy='median')

imputer.fit(df)

x = imputer.transform(df)
df_new = pd.DataFrame(x,columns=df.columns)
df_new.head()
df_new.info()
corr_matrix=df_new.corr()
corr_matrix
corr_matrix['Outcome'].sort_values(ascending=False)
sns.heatmap(corr_matrix,annot=True)
df_new.hist(bins=50,figsize=(20,15))
df_new.plot(kind='box',figsize=(20,15))
df_new.plot(kind='scatter',y='Pregnancies',x='Age',figsize=(20,15))
X = df_new.iloc[:,:-1].values

Y = df_new.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=42)
# import all the algorithm we want to test

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVC', SVC()))

models.append(('RFC', RandomForestClassifier()))

models.append(('DTR', DecisionTreeClassifier()))
from sklearn.model_selection import KFold,cross_val_score

names = []

results = []



for name,model in models:

    kfold = KFold(n_splits=10,random_state=7)

    cv_results = cross_val_score(model,X,Y,cv=kfold,scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (

        name, cv_results.mean(), cv_results.std()

    )

    print(msg)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,Y_train)
filename = 'model.pkl'

pickle.dump(model,open(filename, 'wb'))
pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(Y_test,pred)

print(cm)
accuracy_score(Y_test,pred)