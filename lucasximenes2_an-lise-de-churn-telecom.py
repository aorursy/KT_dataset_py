# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sklearn
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns

sklearn.set_config(display="diagram")
#Em um primeiro momento, vi que o dataset possuia alguns valores nulos que não estavam sendo contabilizados como tal.
#Por isso, optei por, logo de início, descrever o nulos como " ".
df_url = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(df_url, na_values=" ")
df = df.dropna()
#Para fazer análises posteriores, criei uma coluna de target.
#Com isso, consigo fazer análises da média de ocorrência de target em determinadas combinações e análises de features.
def flag_target(row):
  if row['Churn'] == 'Yes':
    target = 1
  else: 
    target = 0 
  return target
df['target'] = df.apply(flag_target, axis=1)
df.head(5)
No = len(df.query('Churn == "No"')) / df.shape[0]

Yes = len(df.query('Churn == "Yes"')) / df.shape[0]

print(f'Os dados de No Churn representam {No:.2f}%\nSendo que os dados de Churn representam {Yes:.2f}%')

print()

labels = 'No Churn', 'Churn'
sizes = [No, Yes]
explode = (0, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
One = len(df.query('Contract == "One year"')) / df.shape[0]

Two = len(df.query('Contract == "Two year"')) / df.shape[0]

Month = len(df.query('Contract == "Month-to-month"')) / df.shape[0]


labels = 'One year', 'Two year', 'Month-to-month'
sizes = [One, Two, Month]
explode = (0, 0.1, 0.05)
colors = ['gold', 'yellowgreen', 'lightcoral','lightskyblue']
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,colors=colors)
ax1.axis('equal')
ax1.title.set_text('Divisão dos contratos')
plt.show()
One = len(df.query('Contract == "One year" and Churn == "Yes"')) / len(df.query('Churn == "Yes"'))

Two = len(df.query('Contract == "Two year" and Churn == "Yes"')) / len(df.query('Churn == "Yes"'))

Month = len(df.query('Contract == "Month-to-month" and Churn == "Yes"')) / len(df.query('Churn == "Yes"'))

colors = ['gold', 'yellowgreen', 'lightcoral','lightskyblue']
labels = 'One year', 'Two year', 'Month-to-month'
sizes = [One, Two, Month]
explode = (0.2, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1,colors = colors)
ax1.axis('equal')
ax1.title.set_text('Proporção dos tipos de contrato que deram Churn')
plt.show()
One = len(df.query('Contract == "One year" and Churn == "No"')) / len(df.query('Churn == "No"'))

Two = len(df.query('Contract == "Two year" and Churn == "No"')) / len(df.query('Churn == "No"'))

Month = len(df.query('Contract == "Month-to-month" and Churn == "No"')) / len(df.query('Churn == "No"'))

colors = ['gold', 'yellowgreen', 'lightcoral','lightskyblue']
labels = 'One year', 'Two year', 'Month-to-month'
sizes = [One, Two, Month]
explode = (0, 0, 0)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, labeldistance = 1.1,colors=colors)
ax1.axis('equal')
ax1.title.set_text('Proporção dos tipos de contrato que NÃO deram Churn')
plt.show()
# Distribuição de probabilidade dos scores em cada sub-população
plt.subplots(figsize=(15,5))
sns.distplot(df.query('target == 0')['tenure'], bins=20, color='blue',kde=False)
sns.distplot(df.query('target == 1')['tenure'], bins=20, color='red',kde=False)
plt.title('Histograma com o tempo de contrato')
plt.legend(['No Churn','Churn'])
dsl = len(df.query('InternetService == "DSL"')) / df.shape[0]

fiber_optic = len(df.query('InternetService == "Fiber optic"')) / df.shape[0]

no = len(df.query('InternetService == "No"')) / df.shape[0]

labels = 'DSL', 'fiber optic', 'No'
sizes = [dsl, fiber_optic, no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet')
plt.show()
dsl = len(df.query('InternetService == "DSL" and Churn == "Yes"')) / len(df.query('Churn == "Yes"'))

fiber_optic = len(df.query('InternetService == "Fiber optic" and Churn == "Yes"')) / len(df.query('Churn == "Yes"'))

no = len(df.query('InternetService == "No" and Churn == "Yes"')) / len(df.query('Churn == "Yes"'))

labels = 'DSL', 'fiber optic', 'No'
sizes = [dsl, fiber_optic, no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet que tiveram Churn')
plt.show()
dsl = len(df.query('InternetService == "DSL" and Churn == "No"')) / len(df.query('Churn == "No"'))

fiber_optic = len(df.query('InternetService == "Fiber optic" and Churn == "No"')) / len(df.query('Churn == "No"'))

no = len(df.query('InternetService == "No" and Churn == "No"')) / len(df.query('Churn == "No"'))

labels = 'DSL', 'fiber optic', 'No'
sizes = [dsl, fiber_optic, no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet que NÃO tiveram Churn')
plt.show()
mom_dsl = len(df.query('InternetService == "DSL" and Contract == "Month-to-month"')) / len(df.query('Contract == "Month-to-month"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "Month-to-month"')) / len(df.query('Contract == "Month-to-month"'))

mom_no = len(df.query('InternetService == "No" and Contract == "Month-to-month"')) / len(df.query('Contract == "Month-to-month"'))

colors = ['gold', 'magenta','cyan']
labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1,colors=colors)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato mensal')
plt.show()
mom_dsl = len(df.query('InternetService == "DSL" and Contract == "Month-to-month" and Churn =="Yes"')) / len(df.query('Contract == "Month-to-month" and Churn =="Yes"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "Month-to-month" and Churn =="Yes"')) / len(df.query('Contract == "Month-to-month" and Churn =="Yes"'))

mom_no = len(df.query('InternetService == "No" and Contract == "Month-to-month" and Churn =="Yes"')) / len(df.query('Contract == "Month-to-month" and Churn =="Yes"'))

colors = ['gold', 'magenta','cyan']
labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1,colors=colors)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato mensal que deram Churn')
plt.show()
mom_dsl = len(df.query('InternetService == "DSL" and Contract == "Month-to-month" and Churn =="No"')) / len(df.query('Contract == "Month-to-month" and Churn =="No"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "Month-to-month" and Churn =="No"')) / len(df.query('Contract == "Month-to-month" and Churn =="No"'))

mom_no = len(df.query('InternetService == "No" and Contract == "Month-to-month" and Churn =="No"')) / len(df.query('Contract == "Month-to-month" and Churn =="No"'))

colors = ['gold', 'magenta','cyan']
labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, labeldistance = 1.1,colors=colors)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato mensal que NÃO deram Churn')
plt.show()
mom = df.query('InternetService == "Fiber optic" and Contract == "Month-to-month"')['MonthlyCharges'].mean()

y = df.query('InternetService == "Fiber optic" and Contract == "One year"')['MonthlyCharges'].mean()

ty = df.query('InternetService == "Fiber optic" and Contract == "Two year"')['MonthlyCharges'].mean()

print(f'Média da fatura dos assinantes de fibra ótica por mês [Contrato Month-to-Month]: ${mom:.2f}')
print(f'Média da fatura dos assinantes de fibra ótica por mês [Contrato Anual]: ${y:.2f}')
print(f'Média da fatura dos assinantes de fibra ótica por mês [Contrato de Dois anos]: ${ty:.2f}')
#@title

mom_dsl = len(df.query('InternetService == "DSL" and Contract == "One year" and Churn =="Yes"')) / len(df.query('Contract == "One year" and Churn =="Yes"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "One year" and Churn =="Yes"')) / len(df.query('Contract == "One year" and Churn =="Yes"'))

mom_no = len(df.query('InternetService == "No" and Contract == "One year" and Churn =="Yes"')) / len(df.query('Contract == "One year" and Churn =="Yes"'))

labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato anual que deram Churn')
plt.show()
mom_dsl = len(df.query('InternetService == "DSL" and Contract == "One year" and Churn =="No"')) / len(df.query('Contract == "One year" and Churn =="No"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "One year" and Churn =="No"')) / len(df.query('Contract == "One year" and Churn =="No"'))

mom_no = len(df.query('InternetService == "No" and Contract == "One year" and Churn =="No"')) / len(df.query('Contract == "One year" and Churn =="No"'))

labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato anual que NÃO deram Churn')
plt.show()
mom_dsl = len(df.query('InternetService == "DSL" and Contract == "Two year" and Churn =="Yes"')) / len(df.query('Contract == "Two year" and Churn =="Yes"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "Two year" and Churn =="Yes"')) / len(df.query('Contract == "Two year" and Churn =="Yes"'))

mom_no = len(df.query('InternetService == "No" and Contract == "Two year" and Churn =="Yes"')) / len(df.query('Contract == "Two year" and Churn =="Yes"'))

labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato de dois anos que deram Churn')
plt.show()
mom_dsl = len(df.query('InternetService == "DSL" and Contract == "Two year" and Churn =="No"')) / len(df.query('Contract == "Two year" and Churn =="No"'))

mom_fibra = len(df.query('InternetService == "Fiber optic" and Contract == "Two year" and Churn =="No"')) / len(df.query('Contract == "Two year" and Churn =="No"'))

mom_no = len(df.query('InternetService == "No" and Contract == "Two year" and Churn =="No"')) / len(df.query('Contract == "Two year" and Churn =="No"'))

labels = 'DSL', 'fiber optic', 'No'
sizes = [mom_dsl, mom_fibra, mom_no]
explode = (0, 0.1, 0.1)

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=135, labeldistance = 1.1)
ax1.axis('equal')
ax1.title.set_text('Taxa de Distribuição da internet para contrato de dois anos que NÃO deram Churn')
plt.show()
print('A média de churn em cada serviço de internet por contrato é:')
df[['Contract','InternetService','target']].groupby(['Contract','InternetService']).mean().round(2).sort_values('target',ascending=False)
features = [
            'gender',
            'SeniorCitizen',
            'Partner',
            'Dependents',
            'tenure',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod',
            'MonthlyCharges',
            'TotalCharges',
]

df['target'] = df.apply(flag_target, axis=1)

X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    shuffle=True, 
    random_state=0
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
numerical_feats = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
categorical_feats = [
            'gender',
            'Partner',
            'Dependents',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod',
]


numerical_preproc = SimpleImputer(strategy='mean')

  # Pré-processamento de dados categóricos
categorical_preproc = make_pipeline(OrdinalEncoder())

  # Aplicar cada etapa de pré-processamento a colunas
  # específicas.
preprocessing_pipeline = make_column_transformer(
    (numerical_preproc, numerical_feats),
    (categorical_preproc, categorical_feats),
)
tree_clf = DecisionTreeClassifier(criterion="gini",
    splitter="best",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_leaf_nodes=None,
    random_state=0)

model = make_pipeline(
  preprocessing_pipeline,
  tree_clf,
)


model.fit(X_train, y_train)
  

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

rec_test = recall_score(y_test, y_test_pred)
  
auc_train = roc_auc_score(y_train, y_train_pred)
auc_test = roc_auc_score(y_test, y_test_pred)

print(f"Acurácia no treino: {acc_train:.2f}")
print(f"Acurácia no teste:  {acc_test:.2f}")
print(f"Curva ROC Treino:  {auc_train:.2f}")
print(f"Curva ROC Teste:  {auc_test:.2f}")
plot_roc_curve(model, X_train, y_train,name='Área sobre a curva ROC - TREINO')
plot_roc_curve(model, X_test, y_test,name='Área sobre a curva ROC - TESTE')
fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=200)

plot_tree(
    model[-1], 
    max_depth=2, 
    filled=True, 
    ax=ax,
    feature_names=numerical_feats + categorical_feats,
    class_names=['Not Churn', 'Churn'],
    node_ids= False);
tree = model[-1]

feat_importances = tree.feature_importances_

feat_importances_df = pd.DataFrame(
    feat_importances, 
    index=numerical_feats + categorical_feats, 
    columns=['importance']
)

feat_importances_df = feat_importances_df.sort_values('importance', ascending=False)

fig1, ax1 = plt.subplots(figsize=(15, 7))
ax1.bar(feat_importances_df.index, feat_importances_df['importance'])
plt.xticks(rotation=90)
ax1.title.set_text('Features mais importantes')

def print_decision_path(model, X_test, sample_id):
  preproc = model[:-1]
  X_test_transformed = preproc.transform(X_test)

  tree_model = model[-1]
  feature_names = numerical_feats + categorical_feats

  n_nodes = tree_model.tree_.node_count
  children_left = tree_model.tree_.children_left
  children_right = tree_model.tree_.children_right
  feature = tree_model.tree_.feature
  threshold = tree_model.tree_.threshold

  node_indicator = tree_model.decision_path(X_test_transformed)
  leaf_id = tree_model.apply(X_test_transformed)
  prediction = tree_model.predict(X_test_transformed)

  # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
  node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                      node_indicator.indptr[sample_id + 1]]

  print('Rules used to predict sample {id} as "{prediction}":\n'.format(
      id=sample_id, prediction=prediction[sample_id]))
  for node_id in node_index:
      # continue to the next node if it is a leaf node
      if leaf_id[sample_id] == node_id:
          continue

      # check if value of the split feature for sample 0 is below threshold
      if (X_test_transformed[sample_id, feature[node_id]] <= threshold[node_id]):
          threshold_sign = "<="
      else:
          threshold_sign = ">"

      print("decision node {node:4d} : {feat_name:10s}  "
            "{inequality:2s} {threshold:6.2f} ({value:6.2f})".format(
                node=node_id,
                # sample=sample_id,
                # feature=feature[node_id],
                feat_name=feature_names[feature[node_id]],
                value=X_test_transformed[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id]))
sample_id = int(input('Digite a posição do cliente: '))
print_decision_path(model, X_test, sample_id)
random_forest = RandomForestClassifier(n_estimators=10, 
                                       max_depth=5,
                                       min_samples_leaf= 1,
                                       oob_score=True,
                                       random_state=3)

model = make_pipeline(
  preprocessing_pipeline,
  random_forest
)


model.fit(X_train, y_train)
  

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

acc_oob = random_forest.oob_score

auc_train = roc_auc_score(y_train, y_train_pred)
auc_test = roc_auc_score(y_test, y_test_pred)


print(f"Acurácia no treino: {acc_train:.2f}")
print(f"Acurácia no teste:  {acc_test:.2f}")
print(f"Acurácia Out of bag:  {acc_oob:.2f}")
print(f"Curva ROC Treino:  {auc_train:.2f}")
print(f"Curva ROC Teste:  {auc_test:.2f}")
ada = AdaBoostClassifier(DecisionTreeClassifier(criterion="gini",
    splitter="best",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_leaf_nodes=None,
    random_state=0),
    n_estimators=5,
     random_state=5)

model = make_pipeline(
  preprocessing_pipeline,
  ada
)


model.fit(X_train, y_train)
  

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

#auc_train = roc_auc_score(y_train, y_train_pred)
#auc_test = roc_auc_score(y_test, y_test_pred)


print(f"Acurácia no treino: {acc_train:.2f}")
print(f"Acurácia no teste:  {acc_test:.2f}")
#print(f"Curva ROC Treino:  {auc_train:.2f}")
#print(f"Curva ROC Teste:  {auc_test:.2f}")
y_plot_train = []
y_plot_test = []

lista = np.arange(1,16)
for max_depth in lista:
  ada = AdaBoostClassifier(DecisionTreeClassifier(
  max_depth=max_depth),
  n_estimators=1,
  random_state=5)

  model = make_pipeline(
  preprocessing_pipeline,
  ada
  )


  model.fit(X_train, y_train)
  

  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)


  acc_train = accuracy_score(y_train, y_train_pred)
  acc_test = accuracy_score(y_test, y_test_pred)

  #print(f'{acc_train:.2f}, {acc_test:.2f}')
  y_plot_train.append(acc_train)
  y_plot_test.append(acc_test)

plt.plot(lista, y_plot_test,label='Teste')
plt.plot(lista, y_plot_train, label='Treino')
plt.legend()
plt.show()

