%matplotlib inline
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adultpmr/train_data.csv", na_values="?")
adult.shape # dimensões da tabela
len(adult) # tamanho da tabela
adult.head() 
adult["race"].value_counts() # mostra o pais de cada individuo
adult["native.country"].value_counts() # conta quantas vezes cada pais apareceu
adult["education.num"].value_counts().plot(kind="bar") # grafico da distribuicao decrescente em barras
adult.isnull().sum() # mostra quantos dados faltantes ha em cada coluna
adult.isnull().sum().sum() # mostra o numero de dados faltantes total
adult_semfaltantes = adult.dropna() 
adult_semfaltantes.shape
# comando que tira linhas com NA
# antes eram 32561 linhas
# ex: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
#Variaveis categoricas
import seaborn as sns
Education = sns.countplot(adult["income"])
sns.countplot(x="workclass",hue='income', data = adult_semfaltantes)
adult_semfaltantes[adult_semfaltantes['income']==">50K"].workclass.value_counts()
adult_semfaltantes[adult_semfaltantes['income']=="<=50K"].workclass.value_counts()
import numpy as np
print(adult_semfaltantes.corr())
# Treino
# variaveis explicativas
Xadult=adult_semfaltantes[["age","fnlwgt","education.num","capital.gain", "capital.loss", "hours.per.week"]]
# variavel a ser explicada
Yadult = adult_semfaltantes.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # escolha do k
from sklearn.model_selection import cross_val_score
grupos = cross_val_score(knn,Xadult,Yadult,cv=10) # validacao cruzada, geracao de vetor com grupos
grupos
grupos.mean()
# variacao de k para otimizar a acurácia
k = 30 # escolha do k
knn = KNeighborsClassifier(n_neighbors=k) 
from sklearn.model_selection import cross_val_score
grupos = cross_val_score(knn,Xadult,Yadult,cv=10) # validacao cruzada, geracao de vetor com grupos
grupos.mean()
sns.boxplot(data=adult_semfaltantes,x="education.num",orient="v")
# skewness
3*(adult_semfaltantes['education.num'].mean()-adult_semfaltantes['education.num'].median())/adult_semfaltantes['education.num'].std()
#atributo = 'education.num' # variar e vizualizar os resultados
#atributo = 'age'
#atributo = 'capital.gain'
#atributo = 'capital.loss'
atributo = 'hours.per.week'
#atributo = 'education.num'
Income = sns.distplot(adult_semfaltantes[atributo])
plt.title(atributo + " Distribution")
plt.show(Income)
adult1WithoutOutliers=adult_semfaltantes[np.abs(adult_semfaltantes[atributo]-adult_semfaltantes[atributo].mean())<=(3*adult_semfaltantes[atributo].std())]
len(adult1WithoutOutliers) # retirou 30161 - 29965 =  196 pontos (0,65%)
# age
atributo = 'age'
adult2WithoutOutliers=adult1WithoutOutliers[np.abs(adult1WithoutOutliers[atributo]-adult1WithoutOutliers[atributo].mean())<=(3*adult1WithoutOutliers[atributo].std())]

len(adult2WithoutOutliers)
# education.num
atributo = 'education.num'
adult3WithoutOutliers=adult2WithoutOutliers[np.abs(adult2WithoutOutliers[atributo]-adult2WithoutOutliers[atributo].mean())<=(3*adult2WithoutOutliers[atributo].std())]

len(adult3WithoutOutliers)
# capital.loss
atributo = 'capital.loss'
adult4WithoutOutliers=adult3WithoutOutliers[np.abs(adult3WithoutOutliers[atributo]-adult3WithoutOutliers[atributo].mean())<=(3*adult3WithoutOutliers[atributo].std())]

len(adult4WithoutOutliers)
# capital.gain 
atributo = 'capital.gain'
adult5WithoutOutliers=adult4WithoutOutliers[np.abs(adult4WithoutOutliers[atributo]-adult4WithoutOutliers[atributo].mean())<=(3*adult4WithoutOutliers[atributo].std())]

len(adult5WithoutOutliers)
# variaveis explicativas
Xadult=adult3WithoutOutliers[["age","fnlwgt","education.num","capital.gain", "capital.loss", "hours.per.week"]]
# variavel a ser explicada
Yadult = adult3WithoutOutliers.income
# variacao de k para otimizar a acurácia
k = 26 # k que consegue a mlehor acurácia 
knn = KNeighborsClassifier(n_neighbors=k) # escolha do k
from sklearn.model_selection import cross_val_score
grupos = cross_val_score(knn,Xadult,Yadult,cv=10) # validacao cruzada, geracao de vetor com grupos
grupos.mean()
# Sem retirar os outliers -> 0.830973965155016
# variaveis explicativas - variar para verificar influencia de cada uma
Xadult=adult3WithoutOutliers[["age",
                              #"fnlwgt",
                              "education.num",
                              "capital.gain", 
                              "capital.loss", 
                              "hours.per.week"
                             ]]
# variavel a ser explicada
Yadult = adult3WithoutOutliers.income
# variacao de k para otimizar a acurácia
k = 26 # k que consegue a mlehor acurácia 
knn = KNeighborsClassifier(n_neighbors=k) # escolha do k
from sklearn.model_selection import cross_val_score
grupos = cross_val_score(knn,Xadult,Yadult,cv=10) # validacao cruzada, geracao de vetor com grupos
grupos.mean()
# Sem retirar os outliers -> 0.830973965155016
# hours.per.week + age + education.num (k = 26) -> 0.8317495528300407
# Retirada de capital.gain e capital.loss nao melhorou a acuracia
from sklearn import preprocessing
num_adult3WithoutOutliers = adult3WithoutOutliers.apply(preprocessing.LabelEncoder().fit_transform)

# processo de tentativa e erro para verificar quais variaveis explicativas "atrapalham o modelo"
Xadult2 = num_adult3WithoutOutliers[["age",
                                     "workclass",
                                     #"fnlwgt",
                                     #"education",
                                     "education.num",
                                     "marital.status",
                                     #"occupation",
                                     #"relationship",
                                     "race",
                                     "sex",
                                     "capital.gain",
                                     "capital.loss",
                                     #"hours.per.week",
                                     #"native.country"
                                    ]]
Yadult2=adult3WithoutOutliers.income
from sklearn.neighbors import KNeighborsClassifier
k = 36
knn = KNeighborsClassifier(n_neighbors=k)
from sklearn.model_selection import cross_val_score
grupos = cross_val_score(knn,Xadult2,Yadult,cv=10)
grupos.mean()
# melhor ( k=36) -> 0.844004539990103
#Treino a base
knn.fit(Xadult2, Yadult2)
Testadult = pd.read_csv("../input/adultpmr/test_data.csv")
num_Testadult = Testadult.apply(preprocessing.LabelEncoder().fit_transform)
Testadult.shape
# Prection
XtestAdult = num_Testadult[["age",
                                     "workclass",
                                     #"fnlwgt",
                                     #"education",
                                     "education.num",
                                     "marital.status",
                                     #"occupation",
                                     #"relationship",
                                     "race",
                                     "sex",
                                     "capital.gain",
                                     "capital.loss",
                                     #"hours.per.week",
                                     #"native.country"
                                    ]]
YtestPred = knn.predict(XtestAdult)
YtestPred.shape
Id = num_Testadult.index.values
Id
import pandas as pd
Id = num_Testadult.index.values
d = {'Id' : Id, 'Income' : YtestPred}
my_df = pd.DataFrame(d)
#my_df.to_csv('C:/Users/palop/Documents/10semestre/Aprendizado de Maquinas/Priemeira entrega/PMR3508-f92c77d8dc-prediction.csv',
#            index=False, sep=',', line_terminator='\n', header = ["Id", "income"])