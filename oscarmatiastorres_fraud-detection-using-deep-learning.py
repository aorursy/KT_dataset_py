import pandas as pd

import numpy as np

import matplotlib as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import LocalOutlierFactor

import sklearn.metrics as metrics

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

import scikitplot as skplt

new_style = {'grid': False}

plt.rc('axes', **new_style)

import matplotlib.pyplot as plt
XY = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
print(u'- El número de filas en el dataset es: {}'.format(XY.shape[0]))

print(u'- El número de columnas en el dataset es: {}'.format(XY.shape[1]))

print(u'- Los nombres de las variables son: {}'.format(list(XY.columns)))

XY[:2]
#definicion de funciones que se usaran 



def NormalizeData(data):

    return (data - np.min(data)) / (np.max(data) - np.min(data))



def repre_matriz_confusion(matriz):

    df_matriz_confusion = pd.DataFrame(matriz,

                     ['True Normal','True Fraud'],

                     ['Pred Normal','Pred Fraud'])

    plt.figure(figsize = (8,4))

    sns.set(font_scale=1.4)

    plt.title(u'Matriz de confusión')

    _ = sns.heatmap(df_matriz_confusion, annot=True, annot_kws={"size": 16}, fmt='g')

    

def reporting_modelo(y_reales, y_clase):

    matriz_confusion = metrics.confusion_matrix(y_reales, y_clase)

    roc_auc = metrics.roc_auc_score(y_reales, y_clase)

    metrica_f1 = metrics.f1_score(y_reales, y_clase)

    print(u'La AUC de la ROC es de: {}'.format(round(roc_auc,2)))

    print(u'La F1 es de: {}'.format(round(metrica_f1,2)))

    print("\nAccuracy\t{}".format(round(metrics.accuracy_score(y_reales, y_clase),3)))  

    print("Sensitividad\t{}".format(round(metrics.recall_score(y_reales, y_clase),3)))

    print(u"Precisión\t{}".format(round(metrics.precision_score(y_reales, y_clase),3)))   

    repre_matriz_confusion(matriz_confusion)

    

def repres_doble_hist(y_prob_pos, y_prob_neg):

    

    fig = plt.figure(figsize=(20,10))

    ax = sns.distplot(y_prob_pos,norm_hist=True, bins=30, hist=False,

    label='', kde_kws={"color": "r", "lw": 5})  

    ax2 = ax.twinx()

    sns.distplot(y_prob_neg,norm_hist=True ,ax=ax2, bins=30, hist=False,

    label='', kde_kws={"color": "g", "lw": 2}) 

    sns.set_style("whitegrid", {'axes.grid' : False})

    ax.figure.legend(['Clase fraudulenta', 'Clase no fraudulenta'])

    new_style = {'grid': False}

    plt.rc('axes', **new_style)

    plt.title('Representación de las probabilidades asignadas a ambas clases')

    plt.show()
X = XY.drop('Class', axis=1)
y = XY['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#Grafico de fraudes vs no fraudes

XY['Class'].value_counts().plot(kind='pie', figsize=(7,7))

plt.title('Distribución de transacciones', fontsize=20)

#Como vemos, las clases están muy desbalanceadas ya que apenas hay casos fraudulentos.
df_plt=XY[XY['Class']==0].sample(4000)

df_plt_pos=XY[XY['Class']==1].sample(60)

df_plt=pd.concat([df_plt,df_plt_pos])

y_plt=df_plt['Class']

X_plt=df_plt.drop('Class',axis=1)
pca2 = PCA(n_components=3)

X_PCA = pca2.fit_transform(X_plt)
fig = plt.figure(figsize=(10,7))

ax = Axes3D(fig)

ax.scatter(X_PCA[:,0], X_PCA[:,1], X_PCA[:,2], c=y_plt, cmap=plt.cm.get_cmap("bwr"))

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

ax.set_zlabel('PC3')

ax.set_xticks([-75000,75000])

ax.set_title('Representación 3D de las tres primeras componentes en un PCA')

plt.show()
plt.figure(figsize=(12,8))

plt.scatter(X_PCA[:,0], X_PCA[:,1], c=y_plt, cmap=plt.cm.get_cmap("Paired", 2))

plt.colorbar(ticks=range(2))

plt.title('Representación 2D de las dos primeras componentes de un PCA')

plt.xlabel('PC1'); _=plt.ylabel('PC2')
clf=LocalOutlierFactor(n_neighbors=10, 

                        algorithm='auto', 

                        leaf_size=30,

                        metric='minkowski', 

                        p=2, 

                        metric_params=None, 

                        n_jobs=-1,

                        novelty=False)
%%time

clf.fit(X)
factores_lof = clf.negative_outlier_factor_

factores_lof
#Se pone el umbral en 2% como se mencionó 

Y_pred_clase = factores_lof.copy()

Y_pred_clase[factores_lof>=np.percentile(factores_lof,2.)] = 0

Y_pred_clase[factores_lof<np.percentile(factores_lof,2.)] = 1
#Se utiliza la funciona para ver como funciona el modelo ya definida

reporting_modelo(y, Y_pred_clase) 
Y_probs = NormalizeData(factores_lof)

Y_pred_prob_pos = NormalizeData(factores_lof)[np.where(y == 1)]

Y_pred_prob_neg = NormalizeData(factores_lof)[np.where(y == 0)]
repres_doble_hist(Y_pred_prob_pos, Y_pred_prob_neg)
Y_probs_1_0 = np.column_stack((Y_probs,list(map(lambda x: 1-x, Y_probs))))

Y_probs_1_0
skplt.metrics.plot_cumulative_gain(y, Y_probs_1_0, figsize=(7,7))

plt.show()
skplt.metrics.plot_lift_curve(y, Y_probs_1_0, figsize=(15,7))

plt.show()