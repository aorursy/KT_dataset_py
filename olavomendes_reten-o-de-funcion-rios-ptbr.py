import pandas as pd

import numpy as np

import matplotlib as matplot

import matplotlib.pyplot as plt

import seaborn as sns



import scipy.stats as stats

import statsmodels.api as sm



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import RobustScaler



from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')

df.head()
# Valores nulos

df.isnull().sum()
# Renomear alguns colunas para melhor leitura

df = df.rename(columns={

    'Work_accident': 'work_accident',

    'promotion_last_5years': 'promotion_last_5_years',

    'Department': 'department'

})
# Mover a coluna "left" (classe) para a primeira coluna da tabela

front = df['left']

df.drop(labels=['left'], axis=1, inplace=True)

df.insert(0, 'left', front)

df.head()
# Dimensões

df.shape
# Tipos

df.dtypes
# Percentual de demissões

left_rate = df.left.value_counts() / len(df)

left_rate
# Alguns dados sobre as demissões

left_summary = df.groupby('left')

left_summary.mean()
# Descrição estatística dos dados

df.describe().T
# Matriz de correlação

corr = df.corr()

corr = (corr)



plt.figure(figsize=(8, 6))



sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='coolwarm_r')



plt.show()
# Populaçao dos funcionários

emp_population = df['satisfaction_level'][df['left'] == 0].mean()



# População dos funcionários demitidos

emp_left_population = df['satisfaction_level'][df['left'] == 1].mean()



print('Satisfação média dos funcionários que NÃO FORAM demitidos:',emp_population)

print('Satisfação média dos funcionários que FORAM demitidos: ', emp_left_population)
stats.ttest_1samp(a=df[df['left'] == 1]['satisfaction_level'], popmean=emp_population)
# Grau de confiança

degree_freedom = len(df[df['left'] == 1])

# Quartil esquerdo

LQ = stats.t.ppf(0.025, degree_freedom)

# Quartil direito

RQ = stats.t.ppf(0.975, degree_freedom)



print('Distrubuição t LQ: ', LQ)

print('Distrubuição t RQ: ', RQ)
f, ax = plt.subplots(ncols=3, figsize=(15, 5))



# satisfaction_level

sns.distplot(df.satisfaction_level, color='g', ax=ax[0]).set_title('Distribuição do nível de satisfação')

ax[0].set_ylabel('Quantidade')



# last_evaluation

sns.distplot(df.last_evaluation, color='r', ax=ax[1]).set_title('Distribuição da avaliação')



# average_montly_hours

sns.distplot(df.average_montly_hours, color='b', ax=ax[2]).set_title('Distribuição da média de horas trabalhadas mensalmente')



plt.tight_layout()

plt.show()
plt.figure(figsize=(10, 4))



sns.countplot(y='salary', hue='left', palette=['b', 'r'], data=df)

plt.title('Salário vs Demissão')

plt.grid(axis='x')



plt.show()
plt.figure(figsize=(12, 6))



sns.countplot(x='department', data=df)

plt.title('Departamentos')

plt.xlabel('Departamentos')

plt.xticks(rotation=-45)

plt.grid(axis='y')



plt.show()
plt.figure(figsize=(12, 5))



sns.countplot(y='department', hue='left', palette=['b', 'r'], data=df)

plt.title('Demissões por departamento')



plt.show()
plt.figure(figsize=(10, 6))



sns.barplot(x='number_project', y='number_project', hue='left', palette=['blue', 'red'], data=df, estimator=lambda x: len(x) / len(df) * 100)

plt.ylabel('Percentual')



plt.show()
plt.figure(figsize=(15, 4))



sns.kdeplot(df.loc[(df['left'] == 0), 'last_evaluation'], shade=True, label='Não demitido')

sns.kdeplot(df.loc[(df['left'] == 1), 'last_evaluation'], color='r', shade=True, label='Demitido')

plt.title('Distribuição dos funcionários \nNão demitido vs Demitido')

plt.show()
plt.figure(figsize=(12, 5))



sns.kdeplot(df.loc[(df['left'] == 0), 'average_montly_hours'], color='b', shade=True, label='Não demitido')

sns.kdeplot(df.loc[(df['left'] == 1), 'average_montly_hours'], color='r', shade=True, label='Demitido')

plt.xlabel('Média de horas mensais')

plt.title('Distribuição de horas mensais trabalhadas \nNão demitido vs Demitido')



plt.show()
plt.figure(figsize=(12, 5))



sns.kdeplot(df.loc[(df['left'] == 0), 'satisfaction_level'], color='b', shade=True, label='Não demitido')

sns.kdeplot(df.loc[(df['left'] == 1), 'satisfaction_level'], color='r', shade=True, label='Demitido')

plt.xlabel('Satisfação')

plt.title('Distribuição da satisfação \nNão demitido vs Demitido')



plt.show()
plt.figure(figsize=(10, 5))



sns.boxplot(x='number_project', y='average_montly_hours', hue='left', palette=['lightblue','r'], data=df)



plt.show()
plt.figure(figsize=(10, 5))



sns.boxplot(x='number_project', y='last_evaluation', hue='left', palette=['lightblue','r'], data=df)



plt.show()
plt.figure(figsize=(10, 6))



sns.lmplot(x='satisfaction_level', y='last_evaluation', palette=['lightblue','r'], data=df, fit_reg=False, hue='left')



plt.show()
plt.figure(figsize=(10, 6))

sns.barplot(x='time_spend_company', y='time_spend_company', hue='left', palette=['b', 'r'], data=df, estimator=lambda x: len(x) / len(df) * 100)

plt.ylabel('Porcentagem')



plt.show()
# Graph and create 3 clusters of Employee Turnover

kmeans = KMeans(n_clusters=3,random_state=2)



kmeans.fit(df[df.left==1][["satisfaction_level","last_evaluation"]])



kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]



fig = plt.figure(figsize=(10, 6))



plt.scatter(x="satisfaction_level",y="last_evaluation", data=df[df.left==1], alpha=0.25,color = kmeans_colors)

plt.xlabel("Nível de satisfação")

plt.ylabel("Avaliação")

plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)

plt.title("Clusters dos funcionários demitidos")



plt.show()
plt.rcParams['figure.figsize'] = (12,6)



df['department'] = df['department'].astype('category').cat.codes

df['salary'] = df['salary'].astype('category').cat.codes



X=df.drop('left', 1)

y=df['left']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



tree = DecisionTreeClassifier(class_weight='balanced', min_weight_fraction_leaf=0.01)



model = tree.fit(X_train, y_train)



importances = model.feature_importances_

feat_names = df.drop('left', 1).columns



indices = np.argsort(importances)[::-1]



plt.title('Importância das "features"')

plt.bar(range(len(indices)), importances[indices], color='blue', align='center')

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulativo')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical', fontsize=14)

plt.xlim([-1, len(indices)])

plt.grid(True)



plt.show()
df['int'] = 1

indep_var = ['satisfaction_level', 'last_evaluation', 'time_spend_company', 'int', 'left']

df = df[indep_var]
X = df.drop('left', 1)

y = df.left



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
iv = ['satisfaction_level', 'last_evaluation', 'time_spend_company', 'int']



log_reg = sm.Logit(y_train, X_train[iv])

answer = log_reg.fit()



answer.summary()

answer.params
coef = answer.params



def y(coef, satisfaction_level, last_evaluation, time_spend_company):

    return coef[3] + coef[0] * satisfaction_level + coef[1] * last_evaluation + coef[2] * time_spend_company



y1 = y(coef, 0.7, 0.8, 3)

p = np.exp(y1) / (1 + np.exp(y1))

p
def base_model(X):

    y = np.zeros(X.shape[0])

    

    return y
X = df.drop('left', 1)

y = df.left



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=42, stratify=y)
y_base = base_model(X_test)

print('Acurácia base', round(accuracy_score(y_test, y_base), 3))
model = LogisticRegression(penalty='l2', C=1)



model.fit(X_train, y_train)



preds = model.predict(X_test)

print('Acurácia de Logistic Regression', round(accuracy_score(y_test, preds), 3))
kfold = KFold(n_splits=10, random_state=42)



cv_model = LogisticRegression(class_weight='balanced')

results = cross_val_score(cv_model, X_train, y_train, cv=kfold, scoring='roc_auc')



print('AUC: {:.3f} ({:.3f})'.format(results.mean(), results.std()))
print('----------- Base --------------')

base_roc_auc = roc_auc_score(y_test, base_model(X_test))

print('AUC base = {:.2f}'.format(base_roc_auc))

print(classification_report(y_test, base_model(X_test)))



logis = LogisticRegression(class_weight = "balanced")

logis.fit(X_train, y_train)

print ("\n\n ------ Logistic Regression ---------")

logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test))

print ("Logistic AUC = %2.2f" % logit_roc_auc)

print(classification_report(y_test, logis.predict(X_test)))





# Decision Tree 

dtree = DecisionTreeClassifier(class_weight="balanced", min_weight_fraction_leaf=0.01)

dtree = dtree.fit(X_train,y_train)

print ("\n\n ------ Decision Tree ------")

dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))

print ("Decision Tree AUC = %2.2f" % dt_roc_auc)

print(classification_report(y_test, dtree.predict(X_test)))





# Random Forest

rf = RandomForestClassifier(

    n_estimators=1000, 

    max_depth=None, 

    min_samples_split=10, 

    class_weight="balanced" 

    )

rf.fit(X_train, y_train)

print ("\n\n ------Random Forest ------")

rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))

print ("Random Forest AUC = %2.2f" % rf_roc_auc)

print(classification_report(y_test, rf.predict(X_test)))





# Ada Boost

ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)

ada.fit(X_train,y_train)

print ("\n\n ------- AdaBoost ------")

ada_roc_auc = roc_auc_score(y_test, ada.predict(X_test))

print ("AdaBoost AUC = %2.2f" % ada_roc_auc)

print(classification_report(y_test, ada.predict(X_test)))
plt.figure(figsize=(10, 6))



fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(X_test)[:,1])

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])

ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])



plt.figure()



# Plot Logistic Regression ROC

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)



# Plot Random Forest ROC

plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)



# Plot Decision Tree ROC

plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)



# Plot AdaBoost ROC

plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)



# Plot Base Rate ROC

plt.plot([0,1], [0,1],label='Base Rate')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Graph')

plt.legend(loc="lower right")

plt.show()
preds = logis.predict_proba(X_test)

pd.DataFrame(preds, columns=['Not left', 'Left']).sort_values(by=['Left'], ascending=False).head()