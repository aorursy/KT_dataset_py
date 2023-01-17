import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.despine(left=True, bottom=True)
cores_genero = ['#8181F7','#F781D8']

cores_fumante = ['#8FBC8F', '#008080']

paleta_genero = sns.color_palette(cores_genero)

paleta_fumante = sns.color_palette(cores_fumante)
df = pd.read_csv('../input/insurance/insurance.csv')

df.head()
df.describe()
df.info()
def bmi_cat(i):

    if 18.5 > i:

      return 'underweith'

    elif 18.5 <= i <= 25:

      return 'normalweith'

    elif 25 < i <= 30:

      return 'overweith'

    elif i > 30:

      return 'obese'







df['obes'] = df['bmi'].apply(lambda i: bmi_cat(i))
def kids(x):

  if x > 0:

    return 'yes'

  else:

    return 'no'

  

df['kids'] = df['children'].apply(lambda x: kids(x))
df.isnull().sum()
df.describe(include='O')
df['sex'].value_counts()
fig = plt.figure(figsize=(8,8))

plt.pie(df['sex'].value_counts(), labels=['MASCULINO', 'FEMININO'], colors=cores_genero);
print(f'Median:     {df.age.median()}')

print(f'{df.age.describe()}')
fig = plt.figure(figsize=(15,5))

sns.distplot(df['age'])

sns.despine()
df.smoker.value_counts()
fig = plt.figure(figsize=(8,8))

plt.pie(df['smoker'].value_counts(), labels=['NÃO-FUMANTE', 'FUMANTE'], colors=cores_fumante);
df['children'].value_counts()
fig = plt.figure(figsize=(12,5))

sns.countplot(x='children',data=df)

sns.despine()
fig = plt.figure(figsize=(15,5))

sns.distplot(df['bmi'])

sns.despine()
df['obes'].value_counts()
fig = plt.figure(figsize=(12,5))

sns.countplot(x='obes',order = df.obes.value_counts().index,data=df, palette=sns.cubehelix_palette(4, reverse=True));
fig = plt.figure(figsize=(12,5))

sns.violinplot(y='bmi', x='obes',data=df, palette=sns.color_palette("Set2"));
fig = plt.figure(figsize=(12,5))

sns.countplot(x='region', data=df, palette=sns.color_palette("Set3"))

sns.despine()
fig = plt.figure(figsize=(15,5))

sns.distplot(df['charges'])

sns.despine()
fig = plt.figure(figsize=(12,5))



ax = fig.add_subplot(121)

sns.distplot(df[df['sex']=='male']['age'])

sns.despine()

ax = fig.add_subplot(122)

sns.distplot(df[df['sex']=='female']['age'], color='#F781D8')

sns.despine()
fig2 = plt.figure(figsize=(15,5))

sns.distplot(df[df['sex']=='male']['age'], color='#8181F7')

sns.distplot(df[df['sex']=='female']['age'], color='#F781D8')

sns.despine()
fig = plt.figure(figsize=(11,5))

ax = fig.add_subplot(121)

plt.hist(df[df['sex']=='male']['bmi'], bins=15)

sns.despine()

ax = fig.add_subplot(122)

plt.hist(df[df['sex']=='female']['bmi'], color='pink', bins=15)

sns.despine();
fig = plt.figure(figsize=(11,7))

plt.hist(df[df['sex']=='male']['bmi'], bins=15)

plt.hist(df[df['sex']=='female']['bmi'], color='pink', bins=15)

sns.despine();
fig = plt.figure(figsize=(12,5))

sns.countplot(x='smoker', hue_order=df['sex'].value_counts().index, hue='sex',data=df, palette=paleta_genero)

sns.despine()
fig = plt.figure(figsize=(15,5))

sns.distplot(df['charges'])

sns.despine()
sns.lmplot('age', 'charges', data=df)

sns.despine()
sns.pairplot(df);
fig = plt.figure(figsize=(12,5))

sns.boxplot(x='smoker', y='charges', palette=paleta_fumante, data=df)

sns.despine()
fig = plt.figure(figsize=(12,5))



ax = fig.add_subplot(121)

sns.distplot(df[df['smoker']=='yes']['charges'],color='#8FBC8F' , ax=ax)

sns.despine()

ax = fig.add_subplot(122)

sns.distplot(df[df['smoker']=='no']['charges'],color='#008080', ax=ax)

sns.despine();
fig = plt.figure(figsize=(15,5))

sns.distplot(df[df['smoker']=='yes']['charges'], color= '#8FBC8F')

sns.distplot(df[df['smoker']=='no']['charges'], color= '#008080')

sns.despine()
fig = plt.figure(figsize=(12,5))

sns.boxplot(x='smoker', y='charges',hue='sex',hue_order=df['sex'].value_counts().index, data=df, palette=paleta_genero)

sns.despine()
fig = plt.figure(figsize=(10,5))

sns.violinplot(x='smoker', y='age', data=df, pallete=paleta_fumante);
fig = plt.figure(figsize=(15,5))

sns.lmplot('age', 'charges',hue='smoker', data=df, palette=paleta_fumante)
sns.jointplot(x="age", y="charges", data=df[df.smoker=='yes'],color='#F781D8', kind="kde");
sns.jointplot(x="age", y="charges", data=df[df.smoker=='no'],color='#F781D8', kind="kde");
fig = plt.figure(figsize=(15,5))

sns.lmplot('age', 'charges', col='obes', data=df, palette=paleta_fumante)
fig=plt.figure(figsize=(12,5))

sns.scatterplot(x="bmi", y="charges",hue= 'obes',data=df,palette=sns.cubehelix_palette(4));
sns.lmplot(x="bmi", y="charges",hue= 'obes',col='sex', data=df,palette=sns.cubehelix_palette(4));
sns.lmplot(x="bmi", y="charges",hue= 'obes',col = 'sex', data=df,palette=sns.cubehelix_palette(4));
sns.lmplot(x="bmi", y="charges", hue="smoker", data=df, palette=paleta_fumante);
sns.lmplot(x="bmi", y="charges", hue="sex", data=df, palette=paleta_genero);
sns.lmplot(x="bmi", y="charges", hue="sex",col='kids' ,data=df, palette=paleta_genero);
sns.lmplot('bmi', 'charges',col='children', data=df)
sns.heatmap(df.corr());
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df.smoker.drop_duplicates()) 

df.smoker = le.transform(df.smoker)



le.fit(df.sex.drop_duplicates())

df.sex = le.transform(df.sex)



le.fit(df.region.drop_duplicates()) 

df.region = le.transform(df.region)



le.fit(df.obes.drop_duplicates())

df.obes = le.transform(df.obes)



le.fit(df.kids.drop_duplicates())

df.kids = le.transform(df.kids)
from sklearn.model_selection import train_test_split

X = df.drop(['charges', 'region'],axis=1)

Y = df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression



linear = LinearRegression()

linear.fit(X_train, Y_train)

linear.score(X_test, Y_test)
linear_coefs = linear.coef_

linear_intercept = linear.intercept_

linear_intercept
df_linear_coef = pd.DataFrame(linear_coefs, index=X.columns, columns=['Coefficients'])

df_linear_coef.T
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline



degree=3

polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())

polyreg.fit(X_train,Y_train)



polyreg.score(X_test, Y_test)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



dt_regressor = DecisionTreeRegressor(random_state=0)

dt_regressor.fit(X_train, Y_train)

cross_val_score(dt_regressor,X_test, Y_test, cv=10).mean()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error



randomF = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 1, n_jobs = -1)

randomF.fit(X_train, Y_train)



randomF_train_pred = randomF.predict(X_train)

randomF_test_pred = randomF.predict(X_test)





r2_score(Y_test,randomF_test_pred)