import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



suicidiosMundo = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
suicidiosMundo.head()
suicidiosBrasil = suicidiosMundo.loc[suicidiosMundo.loc[:, 'country']=='Brazil',:]
suicidiosBrasil.head()
plt.figure(figsize=(16,7))

bar_age = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data = suicidiosBrasil)
cat_accord_year = sns.catplot('sex','suicides_no',hue='sex',col='year',data=suicidiosBrasil,kind='bar',col_wrap=3)
sex_suicides_percent = suicidiosBrasil.groupby('sex')['suicides_no'].sum()



colors_pie = ['#e86466', '#6bbce8']

plt.pie(sex_suicides_percent, 

        labels=sex_suicides_percent.index,

        autopct='%.1f%%',

        shadow=True,

        colors=colors_pie,

        explode=[0.1, 0]);
plt.figure(figsize=(35,16))

sns.heatmap(suicidiosBrasil.corr(),linewidths=.1, annot=True)

plt.yticks(rotation=0);
suicidiosBrasil.isnull().sum()
suicidiosBrasil.describe()
y=suicidiosBrasil['sex']

X=suicidiosBrasil.drop(['sex'],axis=1)

X.head()
from sklearn.preprocessing import LabelEncoder

Encoder_X = LabelEncoder() 

for col in X.columns:

    X[col] = Encoder_X.fit_transform(X[col])

Encoder_y=LabelEncoder()

y = Encoder_y.fit_transform(y)
X.head()
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_treino = sc.fit_transform(X_treino)

X_teste = sc.transform(X_teste)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)



X_treino = pca.fit_transform(X_treino)

X_teste = pca.transform(X_teste)
def visualizar_treino(modelo):

    sns.set_context(context='notebook',font_scale=2)

    plt.figure(figsize=(16,9))

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_treino, y_treino

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.6, cmap = ListedColormap(('#e86466','#6bbce8')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('#cc0000', '#004475'))(i), label = j)

    plt.title("%s Treino" %(modelo))

    plt.legend();

    

def visualizar_teste(modelo):

    sns.set_context(context='notebook',font_scale=2)

    plt.figure(figsize=(16,9))

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_teste, y_teste

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('#e86466','#6bbce8')))

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(('#cc0000', '#004475'))(i), label = j)

    plt.title("%s Teste " %(modelo))

    plt.legend();
from sklearn.metrics import accuracy_score
def printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag, name):

    if flag == True:

        print("Resultado do treino utilizando " + name + ":\n")

        print('Precisão do resultado: {0:.4f}\n'.format(accuracy_score(y_treino,classifier.predict(X_treino))))

    elif flag == False:

        print("Resultado do teste ultizando " + name + ":\n")

        print('Precisão do resultado: {0:.4f}\n'.format(accuracy_score(y_teste,classifier.predict(X_teste))))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



classifier.fit(X_treino,y_treino)
visualizar_treino('Regressão Logística')

printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=True,name="Regressão Logística")
visualizar_teste('Regressão Logística')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=False,name="Regressão Logística")
from sklearn.neighbors import KNeighborsClassifier as KNN



classifier = KNN()

classifier.fit(X_treino,y_treino)
visualizar_treino('K-Nearest Neighbors')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=True,name="K-Nearest Neighbors")
visualizar_teste('K-Nearest Neighbors')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=False,name="K-Nearest Neighbors")
from sklearn.tree import DecisionTreeClassifier as DT



classifier = DT(criterion='entropy',random_state=42)

classifier.fit(X_treino,y_treino)
visualizar_treino('Árvore de Decisão')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=True, name="Árvore de Decisão")
visualizar_teste('Árvore de Decisão')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=False, name="Árvore de Decisão ")
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)

classifier.fit(X_treino, y_treino)
visualizar_treino('Random Forest')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=True, name="Random Forest")
visualizar_teste('Random Forest')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=False, name="Random Forest")
from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=42)



classifier.fit(X_treino,y_treino)
visualizar_treino('SVC')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=True, name="SVC")
visualizar_teste('SVC')
printando_resultado(classifier,X_treino,y_treino,X_teste,y_teste,flag=False, name="SVC")