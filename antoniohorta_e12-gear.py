import numpy as np # linear algebraimport numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# ML
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
# Grafico
from matplotlib import pyplot as plt
import seaborn as sns
# Counter groups
import collections
# Dendograma
import scipy.cluster.hierarchy as sch

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        dataframe = pd.read_csv(os.path.join(dirname, filename),';')
        print(dirname, filename, " ", dataframe.shape)
        
df_hard = pd.read_csv(os.path.join('/kaggle/input/', 'e12_hard_pattern.csv'),';')
df = pd.read_csv(os.path.join('/kaggle/input/', 'e12_full_pattern.csv'),';')

# Ajuste da talela de str para num
df = df.apply(lambda x: x.str.replace(',','.'))


test_size = df.shape[0] - df_hard.shape[0]
print(test_size)
# Visualização dos tipos
sns.countplot(y=df['type'], palette= ["red","gold","blue"]).set(title = 'Priori Mitre ATT&CK')
# Definição de X e Y
X = df.iloc[:, 2:].values
Y = df.iloc[:, 1].values
from sklearn import datasets
from sklearn.decomposition import PCA

# Redução de de variáveis com PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def resultado(model, confmtx=False):
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)
    pred = model.predict(X_test)

    min_reliability = 0
    n_test = len(pred)
    n_reliability = 0
    reliability_index = 0

    for p in prob:
      if p[0] > min_reliability:
        n_reliability = n_reliability + 1
        reliability_index = reliability_index + p[0] 

    accuracy = accuracy_score(y_test, pred)
    reliability = n_reliability/n_test
    reliability_index = reliability_index/n_reliability

    print("Accuracy.............................. " + str(accuracy))
    print("Accuracy with minimal reliability..... " + str(accuracy * reliability))
    print("Reliability index..................... " + str(reliability_index))


    if confmtx is True:
        import seaborn as sns
        c_matrix = confusion_matrix(pred, y_test)
        sns.set(color_codes=True, rc={"figure.figsize":(6,4)}, font_scale=1.5)
        sns.heatmap(c_matrix, annot=True, annot_kws={"size": 20})
    
    
def pipeline(titulo):
    print("######## ", titulo ," ##########")

    print("*** DecisionTreeClassifier ********************************")
    model = DecisionTreeClassifier(max_depth=45, random_state=0)
    resultado(model)

    print("*** MLPClassifier *****************************************")
    model = MLPClassifier(alpha=1, max_iter=1000)
    resultado(model)

    print("*** RandomForestClassifier ********************************")
    model = RandomForestClassifier(n_estimators=45, random_state=0)
    resultado(model, True)

    print("*** GaussianProcessClassifier *****************************")
    model = GaussianProcessClassifier(1.0 * RBF(1.0))
    resultado(model)

    print("*** SVC ***************************************************")
    model = SVC(kernel="linear", C=0.025, probability=True)
    resultado(model)

    print("*** MultinomialNB *****************************************")
    model = MultinomialNB()
    resultado(model)

    print("*** KNeighborsClassifier **********************************")
    model = KNeighborsClassifier(3)
    resultado(model)

    print("*** GaussianNB ********************************************")
    model = GaussianNB()
    resultado(model)

    print("*** AdaBoostClassifier ************************************")
    model = AdaBoostClassifier()
    resultado(model)

    print("*** QuadraticDiscriminantAnalysis *************************")
    model = QuadraticDiscriminantAnalysis()
    resultado(model)
# Verificação do Dendograma para achar o número de Clusters para o HAC

sch.set_link_color_palette(['gold', 'blue', 'red'])

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrograma')
plt.xlabel('Pseudo-classes (agrupamento)')
plt.ylabel('Distancias Euclidianas')
plt.show()


# Aplicação do algoritimo de HAC - Agrupamento de padões
clusters = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters.fit(X)
labels = clusters.labels_

# Definição de base de teste segundo novos agrupamento
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, shuffle=False)

# Plotagem da base
plt.scatter(X_reduced[labels==0, 0], X_reduced[labels==0, 1], s=25, marker='o', color='red', label='A (pseudo-malware)')
plt.scatter(X_reduced[labels==1, 0], X_reduced[labels==1, 1], s=25, marker='o', color='blue',label='B (pseudo-tool)')
plt.scatter(X_reduced[labels==2, 0], X_reduced[labels==2, 1], s=25, marker='o', color='gold', label='C (pseudo-intrusion-set)')

plt.legend( fontsize='xx-small')
plt.title('Categorias de ameaças agrupadas por HAC')
plt.show()
# Visualização PRIORI pós agrupamento
sns.countplot(y=labels, palette= ["red","gold","blue"]).set(title = 'Priori pós agrupamento')
# COM CLUSTERING HAC

pipeline("COM CLUSTERING HAC")
# Definição de base de teste SEM HAC
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)

labels = Y
labels_train = y_train
labels_test = y_test

# Definição de base de teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=414, shuffle=False)

labels = Y

# Plotagem da base
plt.scatter(X_reduced[labels=='malware', 0], X_reduced[labels=='malware', 1], s=25, marker='o', color='red',label='malware')
plt.scatter(X_reduced[labels=='intrusion-set', 0], X_reduced[labels=='intrusion-set', 1], s=25, marker='o', color='gold', label='intrusion-set')
plt.scatter(X_reduced[labels=='tool', 0], X_reduced[labels=='tool', 1], s=25, marker='o', color='blue', label='tool')

plt.legend( fontsize='xx-small')
plt.title('Categorias de Ameaças ATT&CK')
plt.show()

# SEM CLUSTERING HAC

pipeline("SEM CLUSTERING HAC")