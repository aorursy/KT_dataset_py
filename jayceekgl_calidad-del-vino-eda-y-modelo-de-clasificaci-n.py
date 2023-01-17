import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from statistics import mean, median

from scipy import stats

import math



sns.set_style('darkgrid')



from sklearn.preprocessing import LabelEncoder, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report

from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.linear_model import LogisticRegression
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
df.isnull().any()
df.info()
fig = plt.figure(figsize=(20,12))



cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']



count = 1



for col in cols:

    fig.add_subplot(3,4,count)

    sns.distplot(df[col])



    count += 1



plt.show()
fig = plt.figure(figsize=(20,12))



cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']



count = 1



for col in cols:

    fig.add_subplot(3,4,count)

    sns.boxplot(df[col])



    count += 1



plt.show()
df.loc[df['citric acid'] == df['citric acid'].max()]
df.loc[df['residual sugar'] == df['residual sugar'].max()]
df.nlargest(2, 'total sulfur dioxide')
df.nlargest(2, 'sulphates')
df.loc[df['pH'] == df['pH'].max()]
df.loc[df['alcohol'] == df['alcohol'].max()]
df.loc[df['chlorides'] == df['chlorides'].max()]
sns.distplot(df['quality'], kde=False)

plt.show()
df.quality.unique()
x = np.sort(df['quality'])

y = np.arange(1, len(x)+1) / len(x)



plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Wine quality')

plt.ylabel('CDF')

plt.margins(0.02)



plt.show()

df.describe(include='all')
g = sns.PairGrid(df)

g.map(plt.scatter)

plt.show()
pearson_corr = df.corr(method='pearson')

pearson_corr
spearman_corr = df.corr(method='spearman')

spearman_corr
plt.figure(figsize=(16,12))

sns.heatmap(spearman_corr, xticklabels=spearman_corr.columns, yticklabels=spearman_corr.columns, center=0, annot=True)



plt.show()
bins = [0, 4, 7, 10]

labels = ['bad', 'normal', 'good']

df['quality'] = pd.cut(df['quality'], bins=bins, labels=labels)

df.head()
sns.countplot(df['quality'])

plt.show()
sns.violinplot(x=df['quality'], y=df['volatile acidity'])

plt.show()
sns.violinplot(x=df['quality'], y=df['sulphates'])

plt.show()
sns.violinplot(x=df['quality'], y=df['pH'])

plt.show()
sns.violinplot(x=df['quality'], y=df['alcohol'])

plt.show()
sns.violinplot(x=df['quality'], y=df['density'])

plt.show()
bad_wines = df[df['quality'] == 'bad']

good_wines = df[df['quality'] == 'good']
print('La media de alcohol de vinos de baja calidad es de:', bad_wines['alcohol'].mean())

print('El tamaño de la muestra es de:', len(bad_wines))

print('La varianza de la muestra es:', bad_wines['alcohol'].var())



print('La media de alcohol de vinos de alta calidad es de:', good_wines['alcohol'].mean())

print('El tamaño de la muestra es de:', len(good_wines))

print('La varianza de la muestra es:', good_wines['alcohol'].var())
fig = plt.figure(figsize=(10,6))



fig.add_subplot(121)

sns.distplot(bad_wines['alcohol'])



fig.add_subplot(122)

sns.distplot(good_wines['alcohol'])



plt.show()
stats.ttest_ind(bad_wines['alcohol'], good_wines['alcohol'], equal_var=False)
se = math.sqrt((bad_wines['alcohol'].var() / len(bad_wines)) + (good_wines['alcohol'].var() / len(good_wines)))

print('El error estándar de la diferencia de medias entre los dos grupos es de:', se)
y = df['quality']



X = df.drop('quality', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.33)
dtc = DecisionTreeClassifier(max_leaf_nodes=20, random_state=1)
dtc.fit(X_train, y_train)
dt_pred = dtc.predict(X_valid)
print('Algunas predicciones son:', list(dt_pred[:5]))

print('Comparadas con el objetivo:', list(y_valid[:5]))
confusion_matrix(y_valid, dt_pred)
accuracy_score(y_valid, dt_pred)
dt_report = classification_report(y_valid, dt_pred, output_dict=True)



df_dt_report = pd.DataFrame(dt_report).transpose()

df_dt_report
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_valid)
confusion_matrix(y_valid, rf_pred)
accuracy_score(y_valid, rf_pred)
rf_report = classification_report(y_valid, rf_pred, output_dict=True)



df_rf_report = pd.DataFrame(rf_report).transpose()

df_rf_report
scaler = RobustScaler()



X_train = scaler.fit_transform(X_train)

X_valid = scaler.transform(X_valid)
rf.fit(X_train, y_train)



rf_pred = rf.predict(X_valid)
confusion_matrix(y_valid, rf_pred)
rf_report = classification_report(y_valid, rf_pred, output_dict=True)



df_rf_report = pd.DataFrame(rf_report).transpose()

df_rf_report
df2 = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
bins = [0, 4, 5, 6, 10]

labels = ['bad', 'normal', 'good', 'very good']

df2['quality'] = pd.cut(df2['quality'], bins=bins, labels=labels)

df2.head()
sns.countplot(df2['quality'])

plt.show()
y = df2['quality']

X = df2.drop('quality', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.33)
rf2 = RandomForestClassifier()



rf2.fit(X_train, y_train)



rf2_pred = rf2.predict(X_valid)
confusion_matrix(y_valid, rf2_pred)
rf2_report = classification_report(y_valid, rf2_pred, output_dict=True)



df_rf2_report = pd.DataFrame(rf2_report).transpose()

df_rf2_report
df2['alcohol squared'] = df2['alcohol'] ** 2
y = df2['quality']

X = df2.drop('quality', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.33)
rf2.fit(X_train, y_train)



rf2_pred = rf2.predict(X_valid)
confusion_matrix(y_valid, rf2_pred)
rf2_report = classification_report(y_valid, rf2_pred, output_dict=True)



df_rf2_report = pd.DataFrame(rf2_report).transpose()

df_rf2_report
df3 = df2.copy()
df3.columns
y = df3['quality']

X = df3.drop('quality', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.33)
ros = RandomOverSampler(random_state=0)

X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
sns.countplot(y_train_resampled)

plt.show()
rf2.fit(X_train_resampled, y_train_resampled)



rf2_pred = rf2.predict(X_valid)
confusion_matrix(y_valid, rf2_pred)
rf2_report = classification_report(y_valid, rf2_pred, output_dict=True)



df_rf2_report = pd.DataFrame(rf2_report).transpose()

df_rf2_report
sm = SMOTE(random_state=42, k_neighbors=8)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
y_train_res.value_counts()
rf2.fit(X_train_res, y_train_res)



rf2_pred = rf2.predict(X_valid)
confusion_matrix(y_valid, rf2_pred)
rf2_report = classification_report(y_valid, rf2_pred, output_dict=True)



df_rf2_report = pd.DataFrame(rf2_report).transpose()

df_rf2_report
data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data['quality'] = data.quality.apply(lambda x: 1 if x > 5 else 0)
data['quality'].head()
sns.countplot(data['quality'])

plt.show()
y = data['quality']

X = data.drop('quality', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
logit = LogisticRegression()
logit.fit(X_train, y_train)



logit_pred = logit.predict(X_valid)
confusion_matrix(y_valid, logit_pred)
logit_report = classification_report(y_valid, logit_pred, output_dict=True)



df_logit_report = pd.DataFrame(logit_report).transpose()

df_logit_report
roc_auc_score(y_valid, logit_pred)
data['alcohol squared'] = data.alcohol ** 2
y = data.quality

X = data.drop('quality', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
logit2 = LogisticRegression()
logit2.fit(X_train, y_train)



logit2_pred = logit2.predict(X_valid)
confusion_matrix(y_valid, logit2_pred)
logit2_report = classification_report(y_valid, logit2_pred, output_dict=True)



df_logit2_report = pd.DataFrame(logit2_report).transpose()

df_logit2_report
roc_auc_score(y_valid, logit2_pred)