import os
print(os.listdir("../input"))
# Bibliotecas
#import os
#print(os.listdir("../input"))
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# Datasets
train = pd.read_csv("../input/train_data.csv")
test = pd.read_csv("../input/test_data.csv")
train.describe()
train.shape
train.head()
plt.figure(figsize=(20,10))
train['age'].value_counts().sort_index().plot(kind="bar")
plt.title("Number vs Age")
plt.ylabel("Number")
plt.xlabel("Age")
plt.figure(figsize=(5,5))
train['workclass'].value_counts().sort_values(ascending = False).plot(kind="pie")
plt.title("Workclass")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['education'].value_counts().sort_values(ascending = False).plot(kind="pie")
plt.title("Education")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['education.num'].value_counts().sort_index().plot(kind="bar")
plt.title("Education Number")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['marital.status'].value_counts().sort_values(ascending = False).plot(kind="pie")
plt.title("Marital Status")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['occupation'].value_counts().sort_values(ascending = False).plot(kind="pie")
plt.title("Occupation")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['relationship'].value_counts().sort_values(ascending = False).plot(kind="pie")
plt.title("Relationship")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['race'].value_counts().sort_values(ascending = False).plot(kind="pie")
plt.title("Race")
plt.ylabel("")
plt.figure(figsize=(5,5))
train['sex'].value_counts().sort_values().plot(kind="bar")
plt.title("Sex")
plt.ylabel("Number")
plt.xlabel("Sex")
plt.figure(figsize=(5,5))
train['capital.gain'].value_counts().sort_index().plot()
plt.yscale('log')
plt.xscale('log')
plt.title("Number vs Capital Gain")
plt.ylabel("Number")
plt.xlabel("Capital Gain")
plt.figure(figsize=(5,5))
train['capital.loss'].value_counts().sort_index().plot()
plt.yscale('log')
plt.xscale('log')
plt.title("Number vs Capital Loss")
plt.ylabel("Number")
plt.xlabel("Capital Loss")
plt.figure(figsize=(5,5))
train['hours.per.week'].value_counts().sort_index().plot()
plt.title("Number vs Hours per Week")
plt.ylabel("Number")
plt.xlabel("Hours per Week")
plt.figure(figsize=(10,5))
train['native.country'].value_counts().sort_values().plot(kind="bar")
plt.title("Number vs Native Country")
plt.ylabel("Number")
plt.xlabel("Native Country")
train['income'].value_counts().plot(kind="bar")
plt.title("Number vs Income")
plt.ylabel("Number")
plt.xlabel("Income")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

simpleImputer = SimpleImputer(missing_values = '?',
                              strategy = 'most_frequent')
categoricalFeatures = ['workclass', 'education',
                       'education.num', 'marital.status', 'occupation',
                       'relationship', 'race', 'sex',
                       'native.country']

le_sex = LabelEncoder()
le_income = LabelEncoder()

encoder = ce.TargetEncoder(cols = categoricalFeatures)

normalizedFeatures = ['age',
    'workclass', 'education', 'education.num',
    'marital.status', 'occupation', 'relationship',
    'race', 'capital.gain', 'capital.loss',
    'hours.per.week', 'native.country']
normalizer = StandardScaler()

#Função que faz preprocessamento no dataset de treino
#fit_transform
def train_preprocessor(df):
    #Imputar features discretas
    imputed_df = pd.DataFrame(simpleImputer.fit_transform(df[categoricalFeatures]))
    imputed_df.columns = categoricalFeatures
    imputed_df.index = df.index
    for i in categoricalFeatures:
        if(i != 'occupation'):
            df[i] = imputed_df[i]
    
    #Binarização de sexo
    df["sex"] = le_sex.fit_transform(df["sex"])
    
    #Binarização de income
    df["income.binary"] = le_income.fit_transform(df["income"])
    
    #Target Encoding em features discretas
    df_encoder = df.copy()
    a = encoder.fit_transform(df_encoder, df_encoder["income.binary"])
    for i in a.columns:
        df[i] = a[i]
    
    #Normalização
    normalized_df = pd.DataFrame(normalizer.fit_transform(df[normalizedFeatures]))
    normalized_df.columns = normalizedFeatures
    normalized_df.index = df.index
    for i in normalizedFeatures:
        df[i] = normalized_df[i]

#Função que faz preprocessamento no dataset de teste
#transform
def test_preprocessor(df):
    #Imputar features discretas
    imputed_df = pd.DataFrame(simpleImputer.transform(df[categoricalFeatures]))
    imputed_df.columns = categoricalFeatures
    imputed_df.index = df.index
    for i in categoricalFeatures:
        if(i != 'occupation'):
            df[i] = imputed_df[i]
    
    #Binarização de sexo
    df["sex"] = le_sex.transform(df["sex"])
    
    #Criação de income e income.binary para compatibilidade
    df["income.binary"] = np.nan
    df["income"] = np.nan
    
    #Target Encoding em features discretas
    df_encoder = df.copy()
    a = encoder.transform(df_encoder)
    for i in a.columns:
        df[i] = a[i]
    
    #Normalização
    normalized_df = pd.DataFrame(normalizer.transform(df[normalizedFeatures]))
    normalized_df.columns = normalizedFeatures
    normalized_df.index = df.index
    for i in normalizedFeatures:
        df[i] = normalized_df[i]
#Aplicação das funções
train_preprocessor(train)
test_preprocessor(test)
#Separação dos datasets em dois pedaços
Xtrain = train[train.columns.difference(['income',
                                         'income.binary',
                                         'fnlwgt',
                                         'education.num',
                                         'Id'])]
Ytrain = train['income']

Xtest = test[test.columns.difference(['income',
                                      'income.binary',
                                      'fnlwgt',
                                      'education.num',
                                      'Id'])]
Ytest = test['income']
#Redução de dimensionalidade das features dos datasets
#usando PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 7)
Xtrain = pd.DataFrame(pca.fit_transform(Xtrain))
Xtest = pd.DataFrame(pca.transform(Xtest))
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import cross_val_score as cvs

list = []

#Entrar o melhor valor de k
for i in range(1, 60, 6):
    knn = knc(n_neighbors = i)
    knn.fit(Xtrain, Ytrain)
    scores = cvs(knn, Xtrain, Ytrain, cv=5)
    accuracy = sum(scores)/len(scores)
    list.append([i, accuracy])

resultados = pd.DataFrame(list, columns=["k", "Accuracy"])
resultados.plot(x="k",y="Accuracy",style="")
list = np.asarray(list)
max_accuracy = list[np.where(list[:,1] == np.amax(list[:,1]))[0]]
print("k = " + str(max_accuracy[0,0]) + " ; Accuracy = " + str(max_accuracy[0,1]))
knn = knc(n_neighbors = int(max_accuracy[0,0]))
knn.fit(Xtrain, Ytrain)
Ytest = knn.predict(Xtest)
Evaluation = pd.DataFrame(test.Id)
Evaluation["income"] = Ytest
Evaluation.to_csv("Evaluation_knn.csv", index=False)
train_sample_indexes = train.sample(5000).index
Xtrain_sample = Xtrain.iloc[train_sample_indexes]
Ytrain_sample = Ytrain.iloc[train_sample_indexes]
from sklearn.svm import SVC as svc
from sklearn.model_selection import cross_val_score as cvs

list = []

#Encotra melhor valor para parâmetro de soft margin
i = 0.001
while(i <= 1000):
    svm = svc(C=i, gamma = 'auto')
    svm.fit(Xtrain_sample, Ytrain_sample)
    scores = cvs(svm, Xtrain_sample, Ytrain_sample, cv=3)
    accuracy = sum(scores)/len(scores)
    list.append([i, accuracy])
    print(i)
    i *= 10

resultados = pd.DataFrame(list, columns=["C", "Accuracy"])
resultados.plot(x="C",y="Accuracy",style="")
list = np.asarray(list)
max_accuracy = list[np.where(list[:,1] == np.amax(list[:,1]))[0]]
print("C = " + str(max_accuracy[0,0]) + " ; Accuracy = " + str(max_accuracy[0,1]))
svm = svc(C=int(max_accuracy[0,0]), gamma = 'auto')
svm.fit(Xtrain_sample, Ytrain_sample)
Ytest = svm.predict(Xtest)
Evaluation = pd.DataFrame(test.Id)
Evaluation["income"] = Ytest
Evaluation.to_csv("Evaluation_svm.csv", index=False)
train_sample_indexes = train.sample(5000).index
Xtrain_sample = Xtrain.iloc[train_sample_indexes]
Ytrain_sample = Ytrain.iloc[train_sample_indexes]
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import cross_val_score as cvs

list = []

#Encontra o melhor valor para número de neurônios escondidos
#Na 1ª e 2ª camada
for i in range(1, 100, 10):
    for j in range(1, 10, 5):
        mlp = MLPC(hidden_layer_sizes = (i,j,),
               random_state = 0,
               activation = 'relu',
               alpha=0.0001,
               learning_rate='constant',
               learning_rate_init = 0.001,
               max_iter = 1000)
        mlp.fit(Xtrain_sample, Ytrain_sample)
        scores = cvs(mlp, Xtrain_sample, Ytrain_sample, cv=2)
        accuracy = sum(scores)/len(scores)
        list.append([i, j, accuracy])
        print(i, j)
resultados = pd.DataFrame(list, columns=["1st Hidden Layer n° Neurons", "2st Hidden Layer n° Neurons", "Accuracy"])
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

X = resultados.iloc[:,0]
Y = resultados.iloc[:,1]
Z = resultados.iloc[:,2]

surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('1st Hidden Layer n° Neurons')
ax.set_ylabel('2st Hidden Layer n° Neurons')
ax.set_zlabel('Accuracy')
list = np.asarray(list)
max_accuracy = list[np.where(list[:,2] == np.amax(list[:,2]))[0]]
print("1st Hidden Layer n° Neurons = " + str(max_accuracy[0,0]) + " ; 2st Hidden Layer n° Neurons = " + str(max_accuracy[0,1]) + " ; Accuracy = " + str(max_accuracy[0,2]))
mlp = MLPC(hidden_layer_sizes = (41, 6,),
               random_state = 0,
               activation = 'relu',
               alpha=0.0001,
               learning_rate='constant',
               learning_rate_init = 0.001,
               max_iter = 1000)
mlp.fit(Xtrain_sample, Ytrain_sample)
Ytest = mlp.predict(Xtest)
Evaluation = pd.DataFrame(test.Id)
Evaluation["income"] = Ytest
Evaluation.to_csv("Evaluation_mlp.csv", index=False)
