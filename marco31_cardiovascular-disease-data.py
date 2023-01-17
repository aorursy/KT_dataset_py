import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm             import SVC
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.metrics         import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from IPython.display         import Image
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble        import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import LogisticRegression

df = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv", sep=";")
df.head()
df.shape
df.info()
df['age'] = df['age'].apply( lambda x: round(x/365))
df.describe().T
df.isnull().sum()
fig, ax = plt.subplots(1, 2, figsize=(18,4))

df1 = df[(df['ap_hi'] > 40) & (df['ap_hi'] < 200)]
df1 = df1[(df1['ap_lo'] > 40) & (df1['ap_lo'] < 130)]

sns.boxplot(x=df1['ap_hi'], ax=ax[0])
sns.boxplot(x=df1['ap_lo'], ax=ax[1])
fig, ax = plt.subplots(1, 2, figsize=(18,4))

sns.distplot(df1['weight'], ax=ax[0])
sns.distplot(df1['height'], ax=ax[1])
corr = df1.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr, annot=True);
aux1 = df1[(df['gluc']!= 1)] 

sns.countplot(x = 'gluc', hue = 'cardio', data=aux1, palette='deep')
plt.xlabel('Glicose')
plt.ylabel('Quantidade');
plt.figure(figsize=(8,5))
sns.scatterplot(x = 'ap_hi', y = 'ap_lo', hue = 'cardio', data=df1, palette='deep')
plt.title('Gráfico pressão alta x baixa')
plt.xlabel('Sistólica - Alta')
plt.ylabel('Diastólica - Baixa')
plt.legend(loc='upper left');
sns.scatterplot(x = 'weight', y = 'height', hue = 'cardio', data=df1, palette='deep')
sns.countplot(x = 'smoke', hue = 'cardio', data=df1, palette='deep')
plt.xlabel('Fumantes')
plt.ylabel('Quantidade');
sns.countplot(x = 'active', hue = 'cardio', data=df1, palette='deep')
plt.xlabel('Atividade Física')
plt.ylabel('Quantidade');
dfage = df1.copy()
dfage['age'] = dfage['age'].apply(lambda x: 'Jovem' if x<45 else ('meia idade' if x>=45 and x<60 else 'idoso'))


plt.figure(figsize=(8,5))
sns.countplot(x = 'age', hue = 'cardio', data=dfage, palette='deep')
sns.countplot(x = 'cholesterol', hue = 'cardio', data=df1,palette='deep')
plt.xlabel('Colesterol')
plt.ylabel('Quantidade');
sns.countplot(x = 'gender', hue = 'cardio', data=df1, palette='deep')
plt.xlabel('Gênero')
plt.ylabel('Quantidade');
mm = MinMaxScaler()
rb = RobustScaler()
df1['age'] = mm.fit_transform(df1[['age']].values)

df1['ap_hi'] = mm.fit_transform(df1[['ap_hi']].values)

df1['ap_lo'] = mm.fit_transform(df1[['ap_lo']].values)

df1['height'] = rb.fit_transform(df1[['height']].values)

df1['weight'] = rb.fit_transform(df1[['weight']].values)
X = df1.drop(['cardio', 'id'], axis=1)
y = df1['cardio']
y.value_counts()
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
X.hist(ax=ax);
kfold = KFold(n_splits=10)
lr = LogisticRegression(max_iter=10000)
rf = RandomForestClassifier()
xb = GradientBoostingClassifier()
et = ExtraTreesClassifier()
knc = KNeighborsClassifier()
scores_lr = cross_val_score(lr, X, y, cv = kfold)
scores_rf = cross_val_score(rf, X, y, cv = kfold)
scores_xb = cross_val_score(xb, X, y, cv = kfold)
scores_et = cross_val_score(et, X, y, cv = kfold)
scores_knc = cross_val_score(knc, X, y, cv = kfold)
print ("Regressão Logistica:", scores_lr.mean())
print ("Random Forest:", scores_rf.mean())
print ("GradientBoostingClassifier:", scores_xb.mean())
print ("Extra Tree:", scores_et.mean())
print ("KNeighborsClassifier:", scores_knc.mean())
param_grid = [{'n_estimators':[3,10,50,100],'max_features':[2,4,6,8], 'ccp_alpha':[0.0, 0.5, 0.7], 
               'learning_rate':[0.0, 0.5, 0.7], 'loss':['deviance', 'exponential']}]
grid = RandomizedSearchCV(xb, param_grid, cv=20)
grid.fit(X,y)
grid.best_params_
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify=y)
xb = GradientBoostingClassifier(n_estimators=100, max_features= 4, learning_rate=0.5, loss='exponential')
xb.fit(X_train, y_train)
y_pred = xb.predict(X_test)
accuracy_score(y_test, y_pred)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.xlabel('Previsões')
plt.ylabel('Real');
