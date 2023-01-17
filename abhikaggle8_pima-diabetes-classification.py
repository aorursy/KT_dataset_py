import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_pima=pd.read_csv('../input/diabetes.csv')

df_pima.head(8)
df_pima.describe()
df_pima.isnull().sum()
df_pima['Glucose'] = df_pima['Glucose'].replace('0', np.nan)

df_pima['BloodPressure'] = df_pima['BloodPressure'].replace('0', np.nan) 

df_pima['SkinThickness'] = df_pima['SkinThickness'].replace('0', np.nan) 

df_pima['Insulin'] = df_pima['Insulin'].replace('0', np.nan)        

df_pima['BMI'] = df_pima['BMI'].replace('0', np.nan) 

df_pima['DiabetesPedigreeFunction'] = df_pima['DiabetesPedigreeFunction'].replace('0', np.nan) 

df_pima['Age'] = df_pima['Age'].replace('0', np.nan) 



df_pima.head(8)
df_pima.isnull().sum()
df_pima['BMI'].fillna(df_pima['BMI'].median(), inplace=True)

df_pima['Glucose'].fillna(df_pima['Glucose'].median(), inplace=True)

df_pima['BloodPressure'].fillna(df_pima['BloodPressure'].median(), inplace=True)

df_pima['SkinThickness'].fillna(df_pima['SkinThickness'].median(), inplace=True)

df_pima['Insulin'].fillna(df_pima['Insulin'].median(), inplace=True)



df_pima.describe()
corr = df_pima[df_pima.columns].corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, annot = True)
X_features = pd.DataFrame(data = df_pima, columns = ["Glucose","BMI","Age"])

X_features.head(2)

#Considering the 3 features showing the max correlation. 
Y = df_pima.iloc[:,8]

Y.head(3)
scaler = StandardScaler()

X_features = scaler.fit_transform(X_features)

X_features
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.25, random_state=10)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

models = []



models.append(("Logistic Regression:",LogisticRegression()))

models.append(("Naive Bayes:",GaussianNB()))

models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=3)))

models.append(("Decision Tree:",DecisionTreeClassifier()))

models.append(("Support Vector Machine-linear:",SVC(kernel="linear",C=0.2)))

models.append(("Support Vector Machine-rbf:",SVC(kernel="rbf")))

models.append(("Ranom Forest:",RandomForestClassifier(n_estimators=5)))

models.append(("eXtreme Gradient Boost:",XGBClassifier()))



print('Models appended...')

results = []

names = []

for name,model in models:

    kfold = KFold(n_splits=5, random_state=3)

    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")

    names.append(name)

    results.append(cv_result)

for i in range(len(names)):

    print(names[i],results[i].mean()*100)