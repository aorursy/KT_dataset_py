import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB



sns.set(style="whitegrid")
# Leitura dos datasets:

dataset = pd.read_csv('../input/dataset_5secondWindow%5B1%5D.csv')

dataset.head()
# Verificação de valores nulos:

dataset.info()
# Verificação de linhas duplicadas:

dataset.duplicated().sum()
# Excluir as variáveis de desvio padrão:

to_drop = [c for c in dataset.columns if '#std' in c]

dataset.drop(to_drop, axis=1, inplace=True)

dataset.head()
# Renomear as colunas para nomes mais legíveis:

dataset.columns = dataset.columns.str.replace('android.sensor.','').str.replace('#','_')

dataset.head()
# Verificação da distribuição das variáveis restantes por sensor:



plt.figure(figsize=(10,15))



plt.subplot(4,1,1)

sns.distplot(dataset.iloc[:,0])

plt.xlabel('Time')



plt.subplot(4,1,2)

for i in range(1,4):

    sns.distplot(dataset.iloc[:,i])

plt.legend(dataset.iloc[:,1:4].columns)

plt.xlabel('Accelerometer')



plt.subplot(4,1,3)

for i in range(4,7):

    sns.distplot(dataset.iloc[:,i])

plt.legend(dataset.iloc[:,4:7].columns)

plt.xlabel('Gyroscope')



plt.subplot(4,1,4)

for i in range(7,10):

    sns.distplot(dataset.iloc[:,i])

plt.legend(dataset.iloc[:,7:10].columns)

plt.xlabel('Sound')



plt.show()
# Exclusão das variáveis sound_min e sound_max, que apresentaram correlação máxima com sound_mean:

dataset.drop(['sound_max','sound_min'],axis=1,inplace=True)

dataset.head()
# Verificação dos valores das variáveis restantes através de um boxplot:



plt.figure(figsize=(17,4))



plt.subplot(141)

plt.boxplot(dataset.iloc[:,0])

plt.xlabel('Time')



plt.subplot(142)

plt.boxplot([dataset.iloc[:,1],dataset.iloc[:,2],dataset.iloc[:,3]])

plt.xlabel('Accelerometer')



plt.subplot(143)

plt.boxplot([dataset.iloc[:,4],dataset.iloc[:,5],dataset.iloc[:,6]])

plt.xlabel('Gyroscope')



plt.subplot(144)

plt.boxplot(dataset.iloc[:,7]);

plt.xlabel('Sound')



plt.show()
# Verificação da representatividade da variável target no dataset:

sns.barplot(x=dataset.target.value_counts().index,y=dataset.target.value_counts(),color="salmon",saturation=.5);
# Separação do Dataset (também serão utilizados em todos os modelos posteriores a este):

X = dataset.drop(['target'],axis=1)

y = dataset.target



# Separação de datasets para treino e teste:

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
# Desempenho do algorítmo KNN, medido pela acurácia, em função do hiperparâmetro k:



plt.figure(figsize=(8,5))



# Valores de K

K = range(1,11)



# Teste dos modelos

scores = []

for k in K:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    scores.append(knn.score(X_test,y_test))



# Resultados

sns.lineplot(K, scores)

plt.ylabel('Score')

plt.xlabel('K')

plt.show()
# Classificador de Decisão Bayesiana:



class DecisaoBayesiana:

    

    def fit(self, train_x, train_y):

        from scipy.stats import multivariate_normal

        import numpy as np

        import pandas as pd

        

        if type(train_x) != np.ndarray:

            self.train_x = train_x.values

        if type(train_y) != np.ndarray:

            self.train_y = train_y.values

        

        return self

        

    def score(self, test_x, test_y):

        from sklearn.metrics import accuracy_score

        from scipy.stats import multivariate_normal

        

        if type(test_x) != np.ndarray:

            test_x = test_x.values

        if type(test_y) != np.ndarray:

            test_y = test_y.values

        

        classes = y_train.unique()

        train_x = self.train_x

        train_y = self.train_y

        P = pd.DataFrame(data=np.zeros((test_x.shape[0], len(classes))), columns=classes)

        

        for i in np.arange(0, len(classes)):

            elements = tuple(np.where(train_y == classes[i]))

            Z = train_x[elements,:][0]

            m = np.mean(Z, axis = 0)

            cv = np.cov(np.transpose(Z))

            for j in np.arange(0,test_x.shape[0]):

                x = test_x[j,:]

                pj = multivariate_normal.pdf(x, mean=m, cov=cv)

                P[classes[i]][j] = pj

        pred_y = []

        

        for i in np.arange(0, test_x.shape[0]):

            c = np.argmax(np.array(P.iloc[[i]]))

            pred_y.append(classes[c])



        return accuracy_score(test_y, pred_y)
# Seleção de modelos a serem testados:

models = [

    KNeighborsClassifier(n_neighbors=1),

    GaussianNB(),

    DecisaoBayesiana()

]
# Desempenho dos algoritmos testados, medido pela acurácia:

scores = []

for model in models:

    model.fit(X_train,y_train)

    scores.append(model.score(X_test,y_test))



# Resultados

sns.barplot(x=[type(m).__name__ for m in models], y=scores, color="salmon", saturation=.5);

plt.ylabel('Score')

plt.show()
# Desempenho dos modelos testados, medido pela acurácia, com o mesmo conjunto de dados com diferentes pré-processamentos:



# Normalização (min=0 max=1)

X_norm = MinMaxScaler().fit_transform(X)



# Padronização (mean=0 std=1)

X_padr = StandardScaler().fit_transform(X)



# Teste dos modelos

scores = []

for X_currrent,dataset_name in zip([X, X_norm, X_padr],['Sem processamento','Normalizados','Padronizados']):

    X_current_train, X_current_test, y_current_train, y_current_test = train_test_split(X_currrent,y,random_state=42)

    for model in models:

        model.fit(X_current_train,y_current_train)

        score = model.score(X_current_test,y_current_test)

        scores.append({'Model':type(model).__name__, 'dataset':dataset_name,'score':score})

scores_df = pd.DataFrame(scores)



# Resultados

plt.figure(figsize=(10,6))

sns.barplot(x=scores_df.dataset, y=scores_df.score.values, hue=scores_df.Model);

plt.xlabel('')

plt.show()
# Transformação de target para variável numérica:

dataset_ex6 = dataset.copy()

dataset_ex6.target = LabelEncoder().fit_transform(dataset_ex6.target)

dataset_ex6.sample(5)
# Correlações com target:

correlations = dataset_ex6.corr()

correlations[['target']]
# Matriz de correlação entre novas as variáveis:



X_low_corr = dataset_ex6.drop(['target'],axis=1)

y_low_corr = dataset_ex6.target



X_low_corr = X_low_corr[correlations[correlations.target<0.4].index]



plt.figure(figsize=(5,4))

plt.title('Matriz de variáveis de correlação < 0.5 com Target')

sns.heatmap(X_low_corr.corr(), cmap='PuBu',annot=True)



plt.show()
# Normalização (min=0 max=1)

X_low_corr_norm = MinMaxScaler().fit_transform(X_low_corr)



# Padronização (mean=0 std=1)

X_low_corr_padr = StandardScaler().fit_transform(X_low_corr)



# Teste dos modelos

scores = []

for X_currrent,base in zip([X_low_corr, X_low_corr_norm, X_low_corr_padr],

                           ['Sem processamento','Normalizados','Padronizados']):

    X_current_train, X_current_test, y_current_train, y_current_test = train_test_split(X_currrent,y,random_state=42)

    for model in models:

        model.fit(X_current_train,y_current_train)

        score = model.score(X_current_test,y_current_test)

        scores.append({'Model':type(model).__name__, 'dataset':base,'score':score})

scores_df_low_corr = pd.DataFrame(scores)



# Resultado

plt.figure(figsize=(20,6))



ax = plt.subplot(121)

sns.barplot(x=scores_df.dataset, y=scores_df.score.values, hue=scores_df.Model);

plt.xlabel('')

plt.ylabel('Score')

plt.title('Classificação com todas as variáveis')



plt.subplot(122, sharey=ax)

sns.barplot(x=scores_df_low_corr.dataset, y=scores_df_low_corr.score.values, hue=scores_df.Model);

plt.xlabel('')

plt.ylabel('Score')

plt.title('Classificação com variáves menos correlacionadas')



plt.show()
# Normalização



# Ruído
# Desempenho dos modelos gausianos, medido pela acurácia, com o mesmo conjunto de dados com diferentes pré-processamentos:



# Normalização (min=0 max=1)

X_norm = MinMaxScaler().fit_transform(X) 

                                         

# Seleção de modelos gausianos

gausian_models = [

    GaussianNB(),

    MultinomialNB(),

    BernoulliNB()

]



# Teste dos modelos

scores = []

for X_currrent,base in zip([X, X_norm],['Sem processamento','Normalizados']):

    X_current_train, X_current_test, y_current_train, y_current_test = train_test_split(X_currrent,y,random_state=42)

    for model in gausian_models:

        model.fit(X_current_train,y_current_train)

        score = model.score(X_current_test,y_current_test)

        scores.append({'Model':type(model).__name__, 'dataset':base,'score':score})

scores_df = pd.DataFrame(scores)



# Resultado

plt.figure(figsize=(10,6))

sns.barplot(x=scores_df.dataset, y=scores_df.score.values, hue=scores_df.Model);

plt.xlabel('')

plt.show()
# Desempenho dos modelo KNN, medido pela acurácia, em função do hiperparâmetro k, com diferentes distâncias:



plt.figure(figsize=(15,7))



# Ks a serem utilizados pela curva

K = range(1,11)



# Distâncias

metrics = ['euclidean','manhattan','chebyshev']



# Testes dos modelos

for m in metrics:

    scores = []

    for k in K:

        knn = KNeighborsClassifier(n_neighbors=k, metric=m)

        knn.fit(X_train,y_train)

        scores.append(knn.score(X_test,y_test))

    sns.lineplot(K, scores)

    

for p in [1.5,3,5]:

    scores = []

    for k in K:

        knn = KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=p)

        knn.fit(X_train,y_train)

        scores.append(knn.score(X_test,y_test))

    metrics.append('minkowski, p={}'.format(p))

    sns.lineplot(K, scores)



# Resultados

plt.legend(metrics)

plt.ylabel('Score')

plt.xlabel('K')

plt.show()
# Desempenho dos modelos testados, medido pela acurácia, em função do matanho do dataset de treinamento:



plt.figure(figsize=(8,5))



# Valores para test_size

P = [i/10 for i in range(1,10)]



# Teste dos modelos

for model in models:

    scores = []

    for p in P:

        X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X,y,random_state=42,test_size=p)

        model.fit(X_p_train,y_p_train)

        scores.append(model.score(X_p_test,y_p_test))

    sns.lineplot(P, scores)



# Resultados

plt.legend([type(m).__name__ for m in models])

plt.ylabel('Score')

plt.xlabel('Test Size')

plt.show()