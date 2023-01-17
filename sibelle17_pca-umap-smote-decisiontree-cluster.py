# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # algebra linear
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Quaisquer resultados que você gravar no diretório atual serão salvos como saída.
import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
#import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 999) # Verificar todas as colunas.
%time df_train=pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
df_train.head()
%time df_test=pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
df_test.head()
import numpy as np

seed = 43278
np.random.seed(seed)

amostra_train = df_train.sample(frac = 0.025)
amostra_train.shape
df_train.info()
df_test.info()
df_train['ID_code'].head()
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(df_train)
missing_data(df_test)
df_train.describe()
df_test.describe()
x = amostra_train.iloc[:, 2:].values
y = amostra_train.iloc[:, 1].values
test_df= df_test.iloc[:, 1:]
df_train.target.value_counts().plot(kind='bar',color=('blue','orange'))
plt.title('Desbalanceamento - target', fontsize=15);
fig, ax = plt.subplots()
colors = {0:'blue', 1:'orange'}

ax.scatter(df_train['var_0'],
          df_train['var_1'],
          c=df_train['target'].apply(lambda x: colors[x]), alpha=0.5
         )

# Etiquetas
ax.set_xlabel('var_0')
ax.set_ylabel('var_1')

# Legenda
import matplotlib.patches as mpatches
green_patch = mpatches.Patch(color='orange', label='Fraude')
blue_patch = mpatches.Patch(color='blue', label='Não Fraude')
plt.legend(handles=[green_patch, blue_patch])
plt.title('Desbalanceamento - target', fontsize=15);
#pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)
smote_df=pd.DataFrame(y_res,columns=['target'])
smote_df.target.value_counts().plot(kind='bar',color=('blue','orange'))
plt.title('Balanceamento - SMOTE', fontsize=15);
from sklearn.preprocessing import MinMaxScaler

mmscale = MinMaxScaler()  
X_treino = mmscale.fit_transform(x_res)  
#X_teste = mmscale.transform(Xtest) 
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)  
x_treino_pca = pca.fit_transform(x_res) 
pca_df=pd.DataFrame(x_treino_pca,columns=['pca_1','pca_2'])
pca_df['target'] = y_res
pca_df.head()
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
ax = sns.scatterplot(x="pca_1", y="pca_2", data=pca_df, hue='target',\
                     palette=['blue','orange'],alpha=0.5)
plt.title('PCA projection of the Santander dataset', fontsize=24);
#pip install -U umap-learn
import umap
reducer = umap.UMAP(n_components=2)
reducer = reducer.fit(X_treino.data)
x_treino_umap = reducer.transform(X_treino.data)
umap_df=pd.DataFrame(x_treino_umap,columns=['umap_1','umap_2'])
umap_df['target'] = y_res
umap_df.head()
fig = plt.figure(figsize=(15,10))
ax = sns.scatterplot(x="umap_1", y="umap_2", data=umap_df, hue='target',\
                     palette=['blue','orange'],alpha=0.5)
plt.title('UMAP projection of the Santander dataset', fontsize=24);
import sklearn.cluster as cluster
import time
%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
data = x_treino_umap
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
plot_clusters(data, cluster.KMeans, (), {'n_clusters':2})
plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all':False})
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.025})
#conda install -c conda-forge hdbscan
import hdbscan
plot_clusters(data, hdbscan.HDBSCAN,(), {'min_cluster_size':15})
x = amostra_train.iloc[:, 2:].values
y = amostra_train.iloc[:, 1].values
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import MinMaxScaler

mmscale = MinMaxScaler()  
X_train_sc = mmscale.fit_transform(X_train)  
X_test_sc = mmscale.transform(X_test) 
# importando as bibliotecas dos modelos classificadores
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# definindo uma lista com todos os modelos
classifiers = [
    KNeighborsClassifier(),
    GaussianNB(),
    LogisticRegression(dual=False,max_iter=5000),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier()]

# rotina para instanciar, predizer e medir os rasultados de todos os modelos
for clf in classifiers:
    # instanciando o modelo
    clf.fit(X_train_sc, y_train)
    # armazenando o nome do modelo na variável name
    name = clf.__class__.__name__
    # imprimindo o nome do modelo
    print("="*30)
    print(name)
    
    # imprimindo os resultados do modelo
    print('****Results****')
    y_pred = clf.predict(X_test_sc)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
x = amostra_train.iloc[:, 2:]
y = amostra_train.iloc[:, 1]
%%time
# importando GridSearchCV
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# criando uma lista com o Grid Searrch dos parâmetros
parameters = {
    'criterion': ('gini','entropy'),
    'max_depth': range(1,20,2), # Profundidade Maxima( range vai de 1 até 20,indo de 2 em 2)
    'min_samples_split': range(10,500,20),# Quantidade minima de sample por split(vai de 10 a 500, indo de 20 em 20)
    'min_impurity_decrease': [0.0, 0.05, 0.1],
    }

# instanciando o modelo
clf_tree = DecisionTreeClassifier()
# parametrizando o modelo
clf=GridSearchCV(clf_tree,parameters,verbose=1)
# ajustando o modelo
clf.fit(X_train_sc, y_train)

# imprimindo os melhores parâmetros
print("Best Parameters: " + str(clf.best_params_))# imprimir os melhores parâmetros 
# imprimindo os resultados
print('****Results****')
# fazendo predições
y_pred = clf.predict(X_test) # Predições 
# calculando e imprimindo as métricas
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier() 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots(figsize=(12,8))
#plt.figure(figsize=(12,12))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1,figsize=(12,8))

ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()
# importando a biblioteca
from sklearn import tree
from sklearn.metrics import confusion_matrix
# instanciando e ajustando o modelo
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=50,min_samples_split=80, ccp_alpha=0.002)
clf = clf.fit(X_train,y_train)
#fazendo predições
y_pred = clf.predict(X_test)
# calculando e imprimindo as métricas
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm_dtc = confusion_matrix(y_test,y_pred)
sns.heatmap(cm_dtc,annot=True,cmap=['blue','orange'],fmt="d",cbar=False, annot_kws={"size": 15})
plt.title("Decision Tree Classifier Confusion Matrix")
from sklearn import tree
x = amostra_train.iloc[:, 2:]
# importando o Graphviz
import graphviz 
# exportando a árvore de decisão
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Santander") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=x.columns.values,  
                                class_names=['Não Fraude', 'Fraude'],  
                                filled=True, 
                                rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
from sklearn import tree
# instanciando e ajustando o modelo
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=50,min_samples_split=80, ccp_alpha=0.002)
clf = clf.fit(X_train,y_train)
pred= clf.predict(test_df)
submission = pd.read_csv("C:/Users/santander-customer-transaction-prediction/sample_submission.csv")
submission['target'] = pred
submission.to_csv('si_submission_iter3.csv', index=False)