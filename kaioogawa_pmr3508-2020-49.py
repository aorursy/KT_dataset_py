import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (11,7)
train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
train.shape  #tamanho da tabela
train.head() # dados iniciais
pd.isna(train["workclass"]).value_counts()  #coluna workclass
pd.isna(train["occupation"]).value_counts()  #coluna occupation
pd.isna(train["native.country"]).value_counts() #coluna native.country
train["native.country"].value_counts()
# Por simplicidade, inicialmente, insere-se a moda na coluna "native.country"

train['native.country'] = train['native.country'].fillna('United-States')
# Com isso, resta apenas as 1843 linhas faltantes em que as colunas "workclass" ou "occupation",

#serão também preenchidas com a respectiva moda

train["workclass"].value_counts()
train["occupation"].value_counts()
train['workclass'] = train['workclass'].fillna('Private')

train['occupation'] = train['occupation'].fillna('Prof-specialty')

trainOk = train.dropna() #verificado se há dados faltantes
# Verificando o resultado final da tabela:

trainOk.shape  #tamanho da tabela
trainOk.head() # dados iniciais
pd.isna(trainOk["native.country"]).value_counts()
trainOk.describe()  #dados numéricos
def compare_histogram(df, obj_var, test_var, obj_labels = None, alpha = 0.7):

    

    if obj_labels is None:

        obj_labels = df[obj_var].unique()

    

    #obj_var = 'income'

    #obj_labels = ['>50K', '<=50K']

    #test_var = 'age' (for example)

    

    temp = []

    n_labels = len(obj_labels)

    for i in range(n_labels):

        temp.append(df[df[obj_var] == obj_labels[i]])

        temp[i] = np.array(temp[i][test_var]).reshape(-1,1)



    fig = plt.figure(figsize= (13,7))

    

    for i in range(n_labels):

        plt.hist(temp[i], alpha = alpha)

    plt.xlabel(test_var)

    plt.ylabel('quantity')

    plt.title('Histogram over \'' + test_var + '\' filtered by \'' + obj_var + '\'')

    plt.legend(obj_labels)
compare_histogram(trainOk, 'income','age')
compare_histogram(trainOk, 'income','workclass')
compare_histogram(trainOk, 'income','fnlwgt')
groups = trainOk.groupby(['education', 'income'], as_index=False).size()

groups.plot.barh()
groups = trainOk.groupby(['marital.status', 'income'], as_index=False).size()

groups.plot.barh()
groups = trainOk.groupby(['occupation', 'income'], as_index=False).size()

groups.plot.barh()
groups = trainOk.groupby(['relationship', 'income'], as_index=False).size()

groups.plot.barh()
compare_histogram(trainOk, 'income','race')
compare_histogram(trainOk, 'income','sex')
compare_histogram(trainOk, 'income','capital.gain')
compare_histogram(trainOk, 'income','capital.loss')
compare_histogram(trainOk, 'income','hours.per.week')
groups = trainOk.groupby(['native.country', 'income'], as_index=False).size()

groups

#groups.plot.barh()
trainOk = trainOk.drop(['workclass'], axis=1)

trainOk = trainOk.drop(['race'], axis=1)

trainOk = trainOk.drop(['capital.loss'], axis=1)

trainOk = trainOk.drop(['native.country'], axis=1)
trainOk.head()
cut_labels = [0,1,2,3,4]

cut_bins = [0,20,30,40,60,150]

trainOk['age'] = pd.cut(trainOk['age'], bins=cut_bins, labels=cut_labels)
trainOk['fnlwgt'] = pd.qcut(trainOk['fnlwgt'], 16, labels=False)
cut_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

cut_bins = [-1,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000]

trainOk['capital.gain'] = pd.cut(trainOk['capital.gain'], bins=cut_bins, labels=cut_labels)
cut_labels = [0,1,2,3,4,5,6,7,8,9]

cut_bins = [-1,10,20,30,40,50,60,70,80,90,100]

trainOk['hours.per.week'] = pd.cut(trainOk['hours.per.week'], bins=cut_bins, labels=cut_labels)
trainOk = trainOk.drop(['education'], axis=1)
trainOk = trainOk.drop(['Id'], axis=1)
trainOk['marital.status'].unique()
trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Divorced',value=0)

trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Married-civ-spouse',value=1)

trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Never-married',value=2)

trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Widowed',value=3)

trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Married-AF-spouse',value=4)

trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Married-spouse-absent',value=5)

trainOk['marital.status']=trainOk['marital.status'].replace(to_replace='Separated',value=6)

trainOk['occupation'].unique()
trainOk['occupation']=trainOk['occupation'].replace(to_replace='Exec-managerial',value=0)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Transport-moving',value=1)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Machine-op-inspct',value=2)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Adm-clerical',value=3)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Other-service',value=4)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Sales',value=5)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Handlers-cleaners',value=6)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Craft-repair',value=7)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Tech-support',value=8)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Prof-specialty',value=9)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Priv-house-serv',value=10)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Farming-fishing',value=11)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Protective-serv',value=12)

trainOk['occupation']=trainOk['occupation'].replace(to_replace='Armed-Forces',value=13)
trainOk['relationship'].unique()
trainOk['relationship']=trainOk['relationship'].replace('Own-child',0)

trainOk['relationship']=trainOk['relationship'].replace('Husband',value=1)

trainOk['relationship']=trainOk['relationship'].replace('Not-in-family',value=2)

trainOk['relationship']=trainOk['relationship'].replace('Wife',value=3)

trainOk['relationship']=trainOk['relationship'].replace('Unmarried',value=4)

trainOk['relationship']=trainOk['relationship'].replace('Other-relative',value=5)
trainOk['sex'].unique()
trainOk['sex']=trainOk['sex'].replace(to_replace='Male',value=0)

trainOk['sex']=trainOk['sex'].replace(to_replace='Female',value=1)
trainOk.head()
corrMatrix = trainOk.corr()

print (corrMatrix)
Xadult = trainOk[["age","fnlwgt","education.num", "marital.status", "occupation",

                  "relationship","sex","capital.gain","hours.per.week"]]

Yadult = trainOk.income
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
treino_K=[]

K=[]

for i in range(1,50):

    K.append(i)

    knn = KNeighborsClassifier(n_neighbors=i)

    score=cross_val_score(knn, Xadult, Yadult, cv=5).mean()

    treino_K.append(score)

    print("Para K=%d, temos score médio = %.5f"%(i,score))

    

plt.plot(K,treino_K)
bestK = K[np.argmax(treino_K)]

print(bestK)
knn.fit(Xadult,Yadult)
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult.shape  #tamanho da tabela
testAdult.head() # dados iniciais
pd.isna(testAdult["workclass"]).value_counts() 
pd.isna(testAdult["occupation"]).value_counts() 
pd.isna(testAdult["native.country"]).value_counts() 
testAdult["native.country"].value_counts()
# Por simplicidade, inicialmente, insere-se a moda na coluna "native.country"

testAdult['native.country'] = testAdult['native.country'].fillna('United-States')
# Com isso, resta apenas as 966 linhas em que as colunas "workclass" ou "occupation",inclusive,

# apresentam dados faltantes que serão preenchidos com suas respectivas modas

testAdult['workclass'] = testAdult['workclass'].fillna('Private')

testAdult['occupation'] = testAdult['occupation'].fillna('Prof-specialty')

testAdultOk = testAdult.dropna() #verificado se há dados faltantes
testAdultOk['income'] = np.nan
testAdultOk.head()
testAdultOk = testAdultOk.drop(['workclass'], axis=1)

testAdultOk = testAdultOk.drop(['race'], axis=1)

testAdultOk = testAdultOk.drop(['capital.loss'], axis=1)

testAdultOk = testAdultOk.drop(['native.country'], axis=1)
testAdultOk.head()
cut_labels = [0,1,2,3,4]

cut_bins = [0,20,30,40,60,150]

testAdultOk['age'] = pd.cut(testAdultOk['age'], bins=cut_bins, labels=cut_labels)
testAdultOk['fnlwgt'] = pd.qcut(testAdultOk['fnlwgt'], 16, labels=False)
cut_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

cut_bins = [-1,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000]

testAdultOk['capital.gain'] = pd.cut(testAdultOk['capital.gain'], bins=cut_bins, labels=cut_labels)
cut_labels = [0,1,2,3,4,5,6,7,8,9]

cut_bins = [-1,10,20,30,40,50,60,70,80,90,100]

testAdultOk['hours.per.week'] = pd.cut(testAdultOk['hours.per.week'], bins=cut_bins, labels=cut_labels)
colunaId = testAdultOk.drop(['education'], axis=1)

testAdultOk = testAdultOk.drop(['education'], axis=1)
testAdultOk = testAdultOk.drop(['Id'], axis=1)
testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Divorced',value=0)

testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Married-civ-spouse',value=1)

testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Never-married',value=2)

testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Widowed',value=3)

testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Married-AF-spouse',value=4)

testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Married-spouse-absent',value=5)

testAdultOk['marital.status']=testAdultOk['marital.status'].replace(to_replace='Separated',value=6)
testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Exec-managerial',value=0)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Transport-moving',value=1)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Machine-op-inspct',value=2)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Adm-clerical',value=3)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Other-service',value=4)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Sales',value=5)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Handlers-cleaners',value=6)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Craft-repair',value=7)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Tech-support',value=8)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Prof-specialty',value=9)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Priv-house-serv',value=10)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Farming-fishing',value=11)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Protective-serv',value=12)

testAdultOk['occupation']=testAdultOk['occupation'].replace(to_replace='Armed-Forces',value=13)
testAdultOk['relationship']=testAdultOk['relationship'].replace('Own-child',0)

testAdultOk['relationship']=testAdultOk['relationship'].replace('Husband',value=1)

testAdultOk['relationship']=testAdultOk['relationship'].replace('Not-in-family',value=2)

testAdultOk['relationship']=testAdultOk['relationship'].replace('Wife',value=3)

testAdultOk['relationship']=testAdultOk['relationship'].replace('Unmarried',value=4)

testAdultOk['relationship']=testAdultOk['relationship'].replace('Other-relative',value=5)
testAdultOk['sex']=testAdultOk['sex'].replace(to_replace='Male',value=0)

testAdultOk['sex']=testAdultOk['sex'].replace(to_replace='Female',value=1)
testAdultOk.head()
XtestAdult  = testAdultOk[["age","fnlwgt","education.num", "marital.status", "occupation",

                  "relationship","sex","capital.gain","hours.per.week"]]

YtestAdult  = testAdultOk.income
knn = KNeighborsClassifier(n_neighbors=bestK)
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)

import pandas as pd

resultado=pd.DataFrame()

resultado[0]=testAdult.index

resultado[1]=testAdult.index

resultado.columns = ['Id','income']

resultado.update(pd.Series(YtestPred,name='income'))



resultado.to_csv('entregaKaio.csv',index=False)
resultado.head()