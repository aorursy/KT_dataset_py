#import de bibliotecas

import  numpy as np

import pandas as pd



#graficos

import matplotlib.pyplot as plt

import seaborn as sns



#treino/teste

from sklearn.model_selection import train_test_split



#modelos

from sklearn.linear_model import LogisticRegression



#avaliar o modelo

from sklearn import metrics

#carregando os dados

dados = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
#visualizando as primeiras linhas

dados.head()
#verificando o tamanho

dados.shape
#verificando as colunas

dados.columns
#verificando o tipo de dados

dados.dtypes
#veficando os dados estatisticos da base

dados.describe(include='all').T.round(2)
#verificando se temos dados nulos

dados.isna().sum()
#outra forma de ver as informações

dados.info()
#verificando a correção em forma de tabela

dados.corr()
#verificando a correlação entre as variáveis de forma gráfica / visual

plt.figure(figsize=(15,10))

mascara=np.triu(np.ones(dados.corr().shape)).astype(np.bool)

sns.heatmap(dados.corr().round(2), annot=True, cmap="BuPu", mask = mascara)

plt.show()
#conhecendo a variabilidade das variaveis

dados.nunique()
#criando uma função para gerar os graficos

def f_grafico(df, coluna, target):

    #df

    dados=df

    

    #target

    #target='Outcome'

    target=target

    

    #coluna

    #coluna='Pregnancies'

    coluna=coluna



    #nome da analise

    analise=dados[coluna]

    #cria os graficos

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)}, figsize=(15,5))

    #titulo do grafico

    f.suptitle(pd.DataFrame(analise).columns[0])

    #faz o grafico

    sns.boxplot(analise, ax=ax_box)

    sns.distplot(analise, ax=ax_hist, bins=3, color="g" )

    #remove o nome do grafico no boxplot

    ax_box.set(xlabel='')

    #exibe o grafico

    plt.show()

    #exibe tabela com os dados

    #pd.DataFrame(analise).describe().round(2).T



    plt.figure(figsize=(15, 10))

    #plt.title('Pregnancies', fontsize=20)

    #plt.title(pd.DataFrame(analise).columns[0], fontsize=20)

    #sns.distplot(dados["Pregnancies"]) #,  bins=20

    sns.distplot(analise)

    plt.show()

    #dados[["Pregnancies"]].describe().round(2).T



    g = sns.FacetGrid(dados, col=target,height=(5))

    g = g.map(sns.distplot, coluna, kde=False, color='c')



    #exibe tabela com os dados

    print(pd.DataFrame(analise).describe().round(2).T)
f_grafico(dados,'Pregnancies','Outcome')
f_grafico(dados,'Glucose','Outcome')
f_grafico(dados,'BloodPressure','Outcome')
f_grafico(dados,'SkinThickness','Outcome')
f_grafico(dados,'Insulin','Outcome')
f_grafico(dados,'BMI','Outcome')
f_grafico(dados,'DiabetesPedigreeFunction','Outcome')
f_grafico(dados,'Age','Outcome')
dados['total']=1

dados.groupby('Outcome', as_index=False)['total'].count()
dados['total']=1

x=dados.groupby('Outcome', as_index=False)['total'].count()

x['Outcome'].tolist()
dados['total']=1

x=dados.groupby('Outcome', as_index=False)['total'].count()

#x=dados.groupby('Outcome', as_index=False).count()

labels = x['Outcome'].tolist()

sizes = x['total'].tolist()



# Plot

plt.figure(figsize=(15,5))

plt.title('Outcome')

plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=False, startangle=15)

plt.legend(labels, loc="best")

plt.axis('equal')

plt.show()
dados['total']=1

dados.groupby(by='Outcome')['total'].count().plot.bar()

plt.title('Outcome')

plt.show()

temp=dados.groupby('Outcome', as_index=False)['total'].count()

temp.rename(columns={'Outcome':'Quantidade'}, inplace=True)

temp['Percentual']=temp['total']/dados.shape[0]*100

temp
#Visualizando todos os graficos simultaneamente

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(20, 10))



ax.set_facecolor('#fafafa')

#ax.set(xlim=(-10, 200))

plt.ylabel('Variaveis')

plt.title("Visão Geral")

ax = sns.boxplot(data = dados, 

  orient = 'h', 

  palette = 'Set2')
#definindo o target

target=dados['Outcome']



target
explicativas=dados.drop('Outcome', axis=1)

explicativas
#separando em treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(explicativas, target, test_size=0.3, random_state=42)
#Criando o modelo de regressão logistica

modelo_log=LogisticRegression(random_state=42, max_iter=400) 



#o valore default para o max_iter é 100, mas para não gerar erro, alteramos para 400.
#treinando o modelo

modelo_log.fit(x_treino, y_treino)
#criando as predições

predict_log=modelo_log.predict(x_teste)
#criando a matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_teste, predict_log)

cm
#desenhando a matrix

plt.figure(figsize = (10, 6))

sns.heatmap(cm, annot = True, fmt='1.3g')

plt.xlabel("Valor Real")

plt.ylabel("Valor Predito")
#acuracia

acuracia = round(metrics.accuracy_score(y_teste, predict_log)*100,2)

acuracia
fpr_logreg, tpr_logreg, thresholds_logreg = metrics.roc_curve(y_teste, predict_log)



plt.plot(fpr_logreg, tpr_logreg, label = 'ACC = %0.2f' % acuracia, color = 'orange')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.rcParams['font.size'] = 12

plt.title('ROC curve - Receiver Operator Characteristic \n')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.legend(loc="lower right", fontsize=10)

plt.grid(True)
precisao = round(metrics.precision_score(y_teste, predict_log)*100,2)

precisao
recall = metrics.recall_score(y_teste, predict_log)

recall
average_precision = metrics.average_precision_score(y_teste, predict_log)

disp = metrics.plot_precision_recall_curve(modelo_log, x_teste, y_teste)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
f1 = metrics.f1_score(y_teste, predict_log)

f1
print(metrics.classification_report(y_teste, predict_log))