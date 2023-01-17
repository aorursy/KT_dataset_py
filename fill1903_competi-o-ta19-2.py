import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('train.csv')

df.info()
df.head()
df.tail()
df.loc[:,'PAY_0':'PAY_6'] #Vamos analisar uma boa parte do dataset, para ver como estão, de fato, apresentados os valores faltantes
#Para imputar utilizaremos a biblioteca fancyimpute, dela importaremos o objeto KNN

from fancyimpute import KNN 



df.loc[:,'PAY_0':'PAY_6'] = KNN(k=3).fit_transform(df.loc[:,'PAY_0':'PAY_6'])

df.loc[:,'PAY_0':'PAY_6'] = df.loc[:,'PAY_0':'PAY_6'].round(0) #Aqui arredondamos os valores pois o KNN pode gerar valores deci-

#mais, mas nossas features são todas inteiras.
df.loc[:,'PAY_0':'PAY_6']
df.loc[:,'BILL_AMT1':'PAY_AMT6'].head()
df.loc[:,'BILL_AMT1':'PAY_AMT6'] = KNN(k=6).fit_transform(df.loc[:,'BILL_AMT1':'PAY_AMT6'])
df.loc[:,'BILL_AMT1':'PAY_AMT6'].head()
df.loc[:,'BILL_AMT1':'PAY_AMT6'].isna().sum()
#df = df.drop('ID', axis = 1)

df2 = df.drop(['SEX', 'EDUCATION', 'AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'MARRIAGE', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default.payment.next.month'], axis = 1)

df2
df2 = KNN(k=6).fit_transform(df2)
pd.DataFrame(df2)
df.LIMIT_BAL = df2[:, 1]

df = df.drop(columns = 'ID')
df.tail()
df.AGE = df.AGE.replace(-1e+07, np.nan)

df.AGE = df.AGE.replace(1.8e+2, np.nan)

df['AGE'] = df['AGE'].fillna(df['AGE'].median())

df.AGE = df.AGE.replace(0, df.AGE.median())
df = df.replace('f', 'female') 

df = df.replace('m', 'male')
print(df['MARRIAGE'].value_counts())
df = df.replace('OTHERS', 'others') 

df = df.replace('MARRIED', 'married')

df = df.replace('SINGLE', 'single') 

df['MARRIAGE'] = df['MARRIAGE'].replace('others', 'others_marriage')

print(df['MARRIAGE'].value_counts())
print(df['EDUCATION'].value_counts())
df['SEX'] = df['SEX'].astype('category').cat.codes

df['MARRIAGE'] = df['MARRIAGE'].astype('category').cat.codes

df['EDUCATION'] = df['EDUCATION'].astype('category').cat.codes
df['SEX'] = df['SEX'].replace(-1, np.nan)

df['EDUCATION'] = df['EDUCATION'].replace(-1, np.nan)

df['MARRIAGE'] = df['MARRIAGE'].replace(-1, np.nan)

df.head()
df.loc[:,'SEX':'MARRIAGE'] = KNN(k=5).fit_transform(df.loc[:,'SEX':'MARRIAGE'])
df.loc[:,'EDUCATION':'MARRIAGE'] = df.loc[:,'EDUCATION':'MARRIAGE'].round(0) #Aqui arredondamos os valores pois o KNN pode gerar valores deci-

#mais, mas nossas features são todas inteiras.
df2 = df

df2 = pd.get_dummies(df2, columns = ['EDUCATION', 'MARRIAGE'], prefix = ['edu', 'marriage'], dummy_na = True)

df2.loc[df2.edu_nan == 1, ["edu_0.0", "edu_1.0", "edu_3.0", "edu_2.0"]] = np.nan

del df2["edu_nan"]

df2.loc[df2.marriage_nan == 1, ["marriage_0.0", "marriage_1.0", "marriage_2.0"]] = np.nan

del df2["marriage_nan"]

df2.head()

df = df2
df.head()
df.isna().sum()
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split
y = df['default.payment.next.month']

X = df.drop('default.payment.next.month', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
rf = RandomForestClassifier(random_state = 0, n_estimators = 300, min_samples_split= 5, min_samples_leaf= 7, max_features= 'auto', max_depth= 15)



rf.fit(X_train, y_train)



acc = accuracy_score(y_test, rf.predict(X_test))

f1 = f1_score(y_test, rf.predict(X_test))

matrix = confusion_matrix(y_test, rf.predict(X_test))

print(dict(zip(X.columns, rf.feature_importances_.round(2))))

print("{0:.3%} accuracy on test set.".format(acc))

print("{0:.3%} F1_score on test set.".format(f1))
rf.feature_importances_.sum()
matrix
rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 18, step = 1, verbose = 1)

rfe.fit(X_train, y_train)

mask = rfe.support_

reduced_X = X_train.loc[:, mask]

reduced_X_test = X_test.loc[:, mask]
rf2 = RandomForestClassifier(random_state = 0, n_estimators = 300, min_samples_split= 5, min_samples_leaf= 7, max_features= 'auto', max_depth= 15)

rf2.fit(reduced_X, y_train)



acc = accuracy_score(y_test, rf2.predict(reduced_X_test))

f1 = f1_score(rf2.predict(reduced_X_test), y_test)

matrix = confusion_matrix(y_test, rf2.predict(reduced_X_test))

print(dict(zip(X.columns, rf2.feature_importances_.round(2))))

print("{0:.3%} accuracy on test set.".format(acc))

print("{0:.3%} F1_score on test set.".format(f1))
df_test = pd.read_csv('Teste.csv')
df_test = df_test.drop('Unnamed: 0', axis = 1)

df_test_reduzido = df_test.loc[:, mask]
pred = rf2.predict(df_test_reduzido)
Id = pd.read_csv('test.csv')



output = pd.DataFrame({'ID': Id["ID"], 'default.payment.next.month': pred})

output.to_csv("resultados11.csv", index=False)