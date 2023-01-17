# Primeiro daremos import nos pacotes que serão necessários

import pandas  as pd

import numpy  as np

import matplotlib.pyplot  as plt

import seaborn  as sns

from sklearn.model_selection import train_test_split

%matplotlib inline
df = pd.read_csv('../input/ta192/train.csv')
df.info()
df.rename(columns = {'default.payment.next.month':'NEXT_MONTH'}, inplace = True)

df.rename(columns = {'PAY_0':'PAY_1'}, inplace = True)

df.drop('ID', axis = 1,inplace = True)
df['AGE'].describe()
df = df[df['AGE'] > 0]

df = df[df['AGE'] <= 110]

df.info()
plt.boxplot(df['AGE'])

plt.show()
plt.figure(figsize = (15, 10))

sns.boxplot(y=df['AGE'], x=df['NEXT_MONTH'])

plt.show()
df['MARRIAGE'] = df['MARRIAGE'].astype('category')

df['SEX'] = df['SEX'].astype('category')

df['EDUCATION'] = df['EDUCATION'].astype('category')

df['NEXT_MONTH'] = df['NEXT_MONTH'].astype('category')
df['MARRIAGE'].value_counts()
df['MARRIAGE'] = df['MARRIAGE'].replace(['MARRIED', 'SINGLE', 'OTHERS'],['married', 'single', 'others'])

df['MARRIAGE'].fillna('others', inplace = True)

df['MARRIAGE'].value_counts()

df['MARRIAGE'] = df['MARRIAGE'].replace(['married', 'single', 'others'], [1, -1, 0])
df['SEX'].value_counts()
df['SEX'] = df['SEX'].replace(['male', 'female'],['m', 'f'])

df['SEX'].value_counts()
df['SEX'] = df['SEX'].replace(['m', 'f'],['-1', '1'])

df['SEX'].fillna(0, inplace = True)
df['SEX'].value_counts()
df['EDUCATION'].value_counts()
df['EDUCATION'].fillna('others', inplace = True)

df['EDUCATION'] = df['EDUCATION'].replace(['others', 'high school', 'graduate school', 'university'], [0, 1, 2, 3])
for column in list(df.columns.values)[5:23]:

    df[column].fillna(0, inplace = True)
df['LIMIT_BAL'].fillna(0, inplace = True)
df.info()
df['MARRIAGE'] = df['MARRIAGE'].astype('category')

df['SEX'] = df['SEX'].astype('category')

df['EDUCATION'] = df['EDUCATION'].astype('category')

df['NEXT_MONTH'] = df['NEXT_MONTH'].astype('category')
from sklearn.preprocessing import robust_scale





X = df.iloc[:, 0:23]

Y = df.loc[:, 'NEXT_MONTH']
for column in list(X.columns.values)[4:23]:

    X[column] = robust_scale(X[column])



X['LIMIT_BAL'] = robust_scale(X['LIMIT_BAL'])
X_n = pd.get_dummies(data = X, columns = ['MARRIAGE','EDUCATION','SEX'], drop_first = True)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
knn = KNeighborsClassifier(n_neighbors = 10, algorithm = 'ball_tree')

score_knn = cross_val_score(knn, X_n, Y, cv = 10)

score_knn
rf = RandomForestClassifier(n_estimators = 60)

score_rf = cross_val_score(rf, X_n, Y, cv = 10)

score_rf
from sklearn import svm



svm_cl = svm.SVC()

score_svm = cross_val_score(svm_cl, X_n, Y, cv = 10)

score_svm
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

score_ada = cross_val_score(ada, X_n, Y, cv = 10)

score_ada
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(loss = 'exponential')

score_gb = cross_val_score(gb, X_n, Y, cv = 10)

score_gb
gb.fit(X_n, Y)
df_test = pd.read_csv('../input/ta192/test.csv')
df_test.rename(columns = {'default.payment.next.month':'NEXT_MONTH'}, inplace = True)

df_test.rename(columns = {'PAY_0':'PAY_1'}, inplace = True)

df_test.drop('ID', axis = 1,inplace = True)



df_test['MARRIAGE'] = df_test['MARRIAGE'].replace(['MARRIED', 'SINGLE', 'OTHERS'],['married', 'single', 'others'])

df_test['MARRIAGE'].fillna('others', inplace = True)

df_test['MARRIAGE'] = df_test['MARRIAGE'].replace(['married', 'single', 'others'], [1, -1, 0])



df_test['SEX'] = df_test['SEX'].replace(['male', 'female'],['m', 'f'])

df_test['SEX'] = df_test['SEX'].replace(['m', 'f'],['-1', '1'])

df_test['SEX'].fillna(0, inplace = True)



df_test['EDUCATION'].fillna('others', inplace = True)

df_test['EDUCATION'] = df_test['EDUCATION'].replace(['others', 'high school', 'graduate school', 'university'], [0, 1, 2, 3])
for column in list(df_test.columns.values)[5:23]:

    df_test[column].fillna(0, inplace = True)

    

df_test['LIMIT_BAL'].fillna(0, inplace = True)



# Como não podemos jogar observações de teste fora, substituí valores estranhos pela mediana (resiste à outliers)

df_test.loc[df_test['AGE'] > 110, 'AGE'] = df_test['AGE'].median()

df_test.loc[df_test['AGE'] < 10, 'AGE'] = df_test['AGE'].median()

df_test['AGE'].fillna(df_test['AGE'].median(), inplace = True)



df_test['MARRIAGE'] = df_test['MARRIAGE'].astype('category')

df_test['SEX'] = df_test['SEX'].astype('category')

df_test['EDUCATION'] = df_test['EDUCATION'].astype('category')





for column in list(df_test.columns.values)[4:23]:

    df_test[column] = robust_scale(df_test[column])



df_test['LIMIT_BAL'] = robust_scale(df_test['LIMIT_BAL'])



df_test = pd.get_dummies(data = df_test, columns = ['MARRIAGE','EDUCATION','SEX'], drop_first = True)
predict = gb.predict(df_test)
# Essa parte copiei de forma vergonhosa do Ariel



df_com_id = pd.read_csv('../input/ta192/test.csv') #dataframe cujo papel é fornecer o "ID" para o CSV



output = pd.DataFrame({'ID': df_com_id["ID"], 'default.payment.next.month': predict})



output.to_csv("foda.csv", index=False)
