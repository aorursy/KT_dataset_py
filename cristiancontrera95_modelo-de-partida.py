import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt



pd.set_option('max_rows', None)  # para ver todas las filas de un df

sns.set()
!ls ../input/bankdscor9
# Descomentar las siguientes lineas para leer la descripcion de las columnas

# with open('bankdata/bank-columns-description.txt') as fp:

#     print(fp.read())
df = pd.read_csv('../input/bankdscor9/data_train.csv').reset_index(drop=True)

df.tail(3)
df.shape
df_test = pd.read_csv('../input/bankdscor9/data_test.csv').reset_index(drop=True)

df_test.tail(3)
df_test.head(5)
print('Columnas nulas:', sum(df.isnull().sum() > 0))

print('Columnas con valores categoricos:', sum([df[col].dtype == 'object' for col in df.columns]))

print('Columnas con valores reales:', sum([df[col].dtype == 'float' for col in df.columns]))

print('Columnas con valores enteras:', sum([df[col].dtype == 'int64' for col in df.columns]))
df.y = df.y.map({'no': 0, 'yes': 1})
df.isnull().sum()
df_test.isnull().sum()
df.describe()
chart = sns.countplot(df.y)

chart.set_title(f'Cantidad NO ({sum(df.y==0)}) vs SI ({sum(df.y==1)})',{'fontsize':13});
corr = df.corr().abs()  # vemos los valores absolutos para buscar relaciones entre las features

# mask = np.tril(np.ones_like(corr, dtype=np.bool))



plt.figure(figsize=(15,10))

chart = sns.heatmap(corr, cmap='hot_r', annot=True, fmt= '.2f', linewidths=0.4, vmin=0, vmax=1, annot_kws={'size': 10}) # add "mask=mask," for triangular matrix 

chart.set_title("Matriz de correlacion", {'fontsize':14});
# Esta comentado porque puede tardar un rato

# sns.pairplot(df, hue="y", diag_kind="kde");
# for col in df.drop(columns=['y', 'id']).columns.to_list():

#     if df[col].dtype != 'object':

#         print(col, df[col].dtype)

#         plt.figure(figsize=(12,4))

        

#         chart = sns.distplot(df[col], label='Train')

#         chart = sns.distplot(df_test[col], label='Test')

        

#         chart.set_title(f'Distribution of "{col}" feature')

#         plt.legend()

#         plt.show()
for col in df.columns:

    if df[col].dtype == 'object':

        print(f'- Column "{col}": \n\t{df[col].unique()}', end='\n\n') 
for col in df.drop(columns=['y']).columns.to_list():

    if df[col].dtype == 'object':

        df = pd.concat([df.drop(columns=[col]), pd.get_dummies(df[col], prefix=col)], axis=1)



df.tail(5)
for col in df_test.columns.to_list():

    if df_test[col].dtype == 'object':

        df_test = pd.concat([df_test.drop(columns=[col]), pd.get_dummies(df_test[col], prefix=col)], axis=1)



df_test.tail(5)
# la funcion de set de python nos devuelve un conjunto dada una lista, esto es, un grupo de valores unicos

common_features = list(set(df.columns).intersection(set(df_test.columns)))



df = df[common_features + ['y']]  # dejamos la columna target en el df de entrenamiento y validacion



df_test = df_test[common_features]
df.duplicated().sum()
# Borramos los duplicados

# df.drop_duplicates(inplace=True)
from sklearn.model_selection import train_test_split



from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import classification_report, roc_curve, roc_auc_score, f1_score
X = df.drop(columns=['y']).values

y = df.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X_train, y_train)
f1_score( y_test, tree.predict(X_test) )
print(classification_report(y_test, tree.predict(X_test)))
y_pred_proba = tree.predict_proba(X_test)[:,1]



fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)



auc = roc_auc_score(y_test, y_pred_proba)



plt.figure(figsize=(10,5))

plt.plot(fpr, tpr)



plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([-0.01, 1.05])

plt.title(f'ROC CURVE   auc={auc:.4f}')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
y_pred = tree.predict(df_test.values)
import pickle 



with open('model1.pkl', 'wb') as fp:

    pickle.dump(tree, fp)
pd.Series(y_pred, name='y').to_csv('sample_submit.csv', index_label='id')
!head -n 20 sample_submit.csv