import numpy as np # algebra linear

import pandas as pd # data frames, leitura CSV... 

import seaborn as sns # visualização, bibilioteca baseada na matplotlib

import statsmodels.api as sm  # estatística

import statsmodels.formula.api as smf # estatística

import pandas_profiling # análise de dataset 

from scipy import stats # estatística 

from tqdm import tqdm # barra de progresso

import matplotlib.pyplot as plt # plotar gráficos

from sklearn import metrics # importando a biblioteca de métricas

from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV # importando as bibliotecas de "ruídos"

from sklearn.decomposition import PCA # compressão de dados

from sklearn.model_selection import train_test_split # dividir dataset em treino e teste

from sklearn.preprocessing import StandardScaler # para deixar as variáveis na mesma escala



# importando as bibliotecas com os modelos classificadores

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression   

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB





###  KAGGLE ###

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Código para execução do Notebook no Kaggle



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# imprimindo uma barra de tempo para descrever o progresso do processamento

tqdm.pandas(desc="Operation Progress")

df_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

df_train.head()
df_test.head()

# EStá sem a variável target
df_train.info()
df_test.info()
# Explorando Dataset: estatísticas básicas do dataset de treino



df_train.describe().T
df_test.describe().T
# Explorando Dataset: verificando a distribuição da variável target 



df_train.target.value_counts()
# Plotando distribuição pelo histograma da variável target



df_train.target.plot.hist()
# Explorando Dataset: verificando a distribuição da variável target ( percentual)



val_0 = df_train.target.value_counts()[0]

val_1 = df_train.target.value_counts()[1]

total = val_0 + val_1

perc_1 = round(val_1/total,4)*100

# percentual de positivos ( target =1 )

perc_1

# Fazendo a correlação das variáveis

correlacoes = df_train.corr()
# Ordenando as correlações de acordo com a variável target

correlacoes.nlargest(200,'target')['target'].head(50)
# Ordenando as correlações de acordo com a variável target

correlacoes.nlargest(200,'target')['target'].tail(50)
# Ordenando as correlações de acordo com a variável target

correlacoes.nlargest(200,'target')['target'].head(100)
list(df_train)
# Teste: informações das variáveis, sem a target e ID

features = list(df_train)

df_train[features[2:22]].head()
features = list(df_train)

features

#inicio = 2

#fim = inicio +10

#sns.pairplot(df_train[features[inicio:fim]])
features = list(df_train)

inicio = 2

fim = inicio +10

sns.pairplot(df_train[features[inicio:fim]])

#%time sns.pairplot(df_train[features[inicio:fim]], hue='target')
# Analisando dataset com pandas_profiling

pandas_profiling.ProfileReport(df_train[features[inicio:fim]])



inicio = fim

fim = inicio +10

# Identificando Outliers



n = np.arange(0,df_train.shape[0])

plt.plot(n,df_train['var_185'],'o')

plt.show()
# Definindo a Máscara de retirada



mascara = (df_train['var_185']< 10) & (df_train['var_185']> -19)



n = np.arange(0,df_train[ mascara].shape[0]) 

plt.plot(n,df_train[mascara]['var_185'],'o')

plt.show()
mascara1 = (df_train['var_2']< 18) & (df_train['var_2']> 3.5)

mascara2 = (df_train['var_4']< 15.5) & (df_train['var_4']> 5.75)

mascara3 = (df_train['var_10']< 15.5) & (df_train['var_10']> -16)

mascara4 = (df_train['var_11']< 13) & (df_train['var_11']> -21)

mascara5 = (df_train['var_12']< 14.5) & (df_train['var_12']> 13.6)

mascara6 = (df_train['var_15']< 15.6) & (df_train['var_15']> 13.45)

mascara7 = (df_train['var_16']< 16.5) & (df_train['var_16']> 2.45)

mascara8 = (df_train['var_17']< 12) & (df_train['var_17']> -26)

mascara9 = (df_train['var_18']< 37) & (df_train['var_18']> -6)

mascara10 = (df_train['var_20']< 29) & (df_train['var_20']> -1.5)

mascara11 = (df_train['var_23']< 4.6) & (df_train['var_23']> 1.6)

mascara12= (df_train['var_24']< 21) & (df_train['var_24']> 1)

mascara13= (df_train['var_28']< 8.1) & (df_train['var_28']> 3.3)

mascara14= (df_train['var_31']< 17.7) & (df_train['var_31']> 3.6)

mascara15 = (df_train['var_32']< 6.2) & (df_train['var_32']> -7.5)

mascara16 = (df_train['var_33']< 26) & (df_train['var_33']> 2.7)

mascara17 = (df_train['var_38']< 24) & (df_train['var_38']> -2)

mascara18 = (df_train['var_40']< 15) & (df_train['var_40']> -31)

mascara19 = (df_train['var_44']< 25) & (df_train['var_44']> -9)

mascara20 = (df_train['var_45']< 50) & (df_train['var_45']> -75)

mascara21 = (df_train['var_47']< 13) & (df_train['var_47']> -41)

mascara22 = (df_train['var_52']< 11) & (df_train['var_52']> -18)

mascara23= (df_train['var_60']< 25) & (df_train['var_60']> 1)

mascara24= (df_train['var_65']< 12) & (df_train['var_65']> -11)

mascara25= (df_train['var_66']< 9) & (df_train['var_66']> 2.5)

mascara26= (df_train['var_70']< 58) & (df_train['var_70']> -10)

mascara27= (df_train['var_78']< 11) & (df_train['var_78']> -0.5)

mascara28= (df_train['var_80']< 26) & (df_train['var_80']> -16)

mascara29= (df_train['var_81']< 22) & (df_train['var_81']> 8.5)

mascara30= (df_train['var_87']< 28) & (df_train['var_87']> -5)

mascara31= (df_train['var_89']< 13) & (df_train['var_89']> -6)

mascara32= (df_train['var_107']< 41) & (df_train['var_107']> -3)

mascara33= (df_train['var_108']< 14.65) & (df_train['var_108']> 13.75)

mascara34= (df_train['var_110']< 17) & (df_train['var_110']> -6)

mascara35= (df_train['var_111']< 9.5) & (df_train['var_111']> 3.5)

mascara36= (df_train['var_115']< 10.5) & (df_train['var_115']> -6)

mascara37= (df_train['var_116']< 7) & (df_train['var_116']> -2.2)

mascara38= (df_train['var_120']< 61) & (df_train['var_120']> -12)

mascara39= (df_train['var_121']< 16.5) & (df_train['var_121']> 6.5)

mascara40= (df_train['var_124']< 12.5) & (df_train['var_124']> -5)

mascara41= (df_train['var_129']< 28) & (df_train['var_129']> 3)

mascara42= (df_train['var_131']< 1.8) & (df_train['var_131']> -0.8)

mascara43= (df_train['var_135']< 18) & (df_train['var_135']> -27)

mascara44 = (df_train['var_138']< 16) & (df_train['var_138']> -14)

mascara45 = (df_train['var_140']< 17) & (df_train['var_140']> -12)

mascara46 = (df_train['var_146']< 18) & (df_train['var_146']> 2)

mascara47 = (df_train['var_153']< 23) & (df_train['var_153']> 11.6)

mascara48 = (df_train['var_163']< 29) & (df_train['var_163']> -5)

mascara49 = (df_train['var_166']< 4.05) & (df_train['var_166']> 1.95)

mascara50 = (df_train['var_168']< 14) & (df_train['var_168']> -5)

mascara51 = (df_train['var_169']< 6.7) & (df_train['var_169']> 4.5)

mascara52 = (df_train['var_175']< 20) & (df_train['var_175']> 3)

mascara53 = (df_train['var_176']< 16) & (df_train['var_176']> -25)

mascara54 = (df_train['var_180']< 11) & (df_train['var_180']> -18)

mascara55 = (df_train['var_181']< 14) & (df_train['var_181']> 6)

mascara56 = (df_train['var_185']< 10) & (df_train['var_185']> -19)





df = df_train[mascara1 & mascara2 & mascara4 & mascara3 & mascara5 & mascara6 & mascara7 & mascara8 & mascara9 & mascara10 & 

              mascara11 & mascara12 & mascara13 & mascara14 & mascara15 & mascara16 & mascara17 & mascara18 & mascara19 & mascara20 &

              mascara21 & mascara22 & mascara23 & mascara24 & mascara25 & mascara26 & mascara27 & mascara28 & mascara29 & mascara30 &

              mascara31 & mascara32 & mascara33 & mascara34 & mascara35 & mascara36 & mascara37 & mascara38 & mascara39 & mascara40 &

              mascara41 & mascara42 & mascara43 & mascara44 & mascara45 & mascara46 & mascara47 & mascara48 & mascara49 & mascara50 &

              mascara51 & mascara52 & mascara53 & mascara54 & mascara55 & mascara56]

df.shape
# definindo variáveis para cada uma das classes

df1 = df[df.target==0]

df2 = df[df.target==1]
# verificando o desbalanceamento

len(df1),len(df2)
# fazendo um undersampling da classe com output zero (em maior número)

df1=df1.sample(n=18613)

len(df1)
# concatenando os dois DataSets com o mesmo tamanho

df = pd.concat([df1,df2])

df.target.value_counts()
X = df.iloc[:, 2:202].values 

y = df.iloc[:, 1].values 

# dividindo os dados  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 
# é preciso certificar que todas as features estão na mesma escala para evitar enviesamento.

scaler = StandardScaler()



# ajustando com os dados de treino função fit + transform juntas

X_train = scaler.fit_transform(X_train) 

# transformando os dados de teste

X_test = scaler.transform(X_test) 


pca = PCA().fit(X_train)

#Plotando a soma cumulativa para identificar o melhor número de componentes

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Numero de Componentes')

plt.ylabel('Variacao (%)') #para cada componente

plt.title('Explicacao por n.o de variaveis')

plt.show()
# definindo uma lista com todos os classificadores para identificar qual teria uma melhor performance

classifiers = [

    KNeighborsClassifier(3),

    GaussianNB(),

    LogisticRegression(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]



# definindo o tamanho da figura para o gráfico

plt.figure(figsize=(12,8))



# rotina para instanciar, predizer e medir os rasultados de todos os modelos

for clf in classifiers:

    # instanciando o modelo

    clf.fit(X_train, y_train)

    # armazenando o nome do modelo na variável name

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados do modelo

    print('****Resultados****')

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall:", metrics.recall_score(y_test, y_pred))

    

    

    # plotando a curva ROC

    y_pred_proba = clf.predict_proba(X_test)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label=name+", auc="+str(auc))

    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

    plt.legend(loc=4)

# GaussianNB 0.8 -  sem fazer PCA

# LogisticRegression 0.77 - c/ PCA de 10 componentes



#clf = LogisticRegression()

clf = GaussianNB()



clf.fit(X_train, y_train)

# armazenando o nome do modelo na variável name

name = clf.__class__.__name__

# imprimindo o nome do modelo

print("="*30)

print(name)

# imprimindo os resultados do modelo

print('****Resultados****')

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))



X = df_test.iloc[:, 1:201].values # variáveis , começa 1 antes, pois nao tem o target antes das variáveis

X = scaler.transform(X)           # só faz o transform, mas não o fit novamente, o mesmo com test

                                  # fit calcula os parametros, transform - transforma... não pode calcular novamente

y_pred = clf.predict_proba(X)     # predição retornando probabilidade é mais eficaz no kaggle do que 0 ou 1

y_pred

# primeiro valor no array é a probabilidade de ser 0

# segundo valor no array é a probabilidade de ser 1
df_test['target'] = y_pred[:,1] # apenas probabilidade de ser 1
df_test.head(5)
df_test[['ID_code','target']].head() # Apenas ID + target/predição
# Escrevendo arquivo para submissão ( máquina do usuário)

# df_test[['ID_code','target']].to_csv (r'C:\Users\Usuario\Desktop\submission5.csv', index = None, header=True)