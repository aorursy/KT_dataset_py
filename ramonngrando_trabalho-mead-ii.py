import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import statsmodels.api as sm

import math



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats.mstats import winsorize

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# ver

# https://data-flair.training/blogs/python-statistics/

# https://www.kaggle.com/harshini564/ab-testing-life-expectancy-who

# https://www.kaggle.com/ramonngrando/trabalho-scd-ii

# fiq - https://pt.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/identifying-outliers-iqr-rule
# carregando arquivo

df = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv')
df.info()
# renomeando as colunas

df.rename(columns={'Country':'País', 

                   'Year':'Ano',

                   'Status':'Situação',

                   'Life expectancy ':'Expectativa de Vida',

                   'Adult Mortality':'Mortes Adultos',

                   'infant deaths':'Mortes Infantis',

                   'Alcohol':'Álcool',

                   'percentage expenditure':'Porcentagem de Despesas',

                   'Hepatitis B':'Hepatite B',                   

                   'Measles ':'Sarampo',

                   ' BMI ':'IMC',

                   'under-five deaths ':'Mortes - Menor de 5 anos',

                   'Polio':'Poliomielite',

                   'Total expenditure':'Despesa Total',                   

                   'Diphtheria ':'Difteria',

                   ' HIV/AIDS':'Mortes por HIV/AIDS',

                   'GDP':'PIB',

                   'Population':'População',

                   ' thinness  1-19 years':'Magreza 10-19 anos',

                   ' thinness 5-9 years':'Magreza 5-9 anos',

                   'Income composition of resources':'Composição de Renda',

                   'Schooling':'Escolaridade'

                  }, inplace=True)
df.describe()
# df.query('País.str.contains("boli")', engine='python') # checa se há venezuela e bolívia

# altera venezuela pois o valor está incorreta

df.loc[df['País'] == 'Venezuela (Bolivarian Republic of)', 'País'] = 'Venezuela'

df.loc[df['País'] == 'Bolivia (Plurinational State of)', 'País'] = 'Bolivia'
# apenas os países da américa do sul

ams = df[df.País.isin(('Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'))].copy()



ams.head()
ams.País.unique()
ams.count()
# Venezuela e Bolivia não possuem PIB, nem População

ams[ams['PIB'].isna()]
# Maiores números de casos com Sarampo

ams.nlargest(10, 'Sarampo')[{'País', 'Ano', 'Sarampo'}].style.hide_index()
# Trato com a média os valores nulos que não puderam ser tratados com a interpolação

ams['Álcool'] = ams['Álcool'].fillna(ams.groupby('País')['Álcool'].transform('mean'))

ams['Hepatite B'] = ams['Hepatite B'].fillna(ams.groupby('País')['Hepatite B'].transform('mean'))

ams['Despesa Total'] = ams['Despesa Total'].fillna(ams.groupby('País')['Despesa Total'].transform('mean'))



# Venezuela e Bolivia não possuem valores de PIB e População, portanto não utilizaremos estes dados

df.drop(['PIB', 'População'], axis=1)
ams.count()
# Dicionário das colunas

dicionario = {

    'Expectativa de Vida':1

    , 'Mortes Adultos':2

    , 'Mortes Infantis':3

    , 'Álcool':4

    , 'Porcentagem de Despesas':5

    , 'Hepatite B':6

    , 'Sarampo':7

    , 'IMC':8

    , 'Mortes - Menor de 5 anos':9

    , 'Poliomielite':10

    , 'Despesa Total':11

    , 'Difteria':12

    , 'Mortes por HIV/AIDS':13

    , 'PIB':14

    , 'População':15

    , 'Magreza 10-19 anos':16

    , 'Magreza 5-9 anos':17

    , 'Composição de Renda':18

    , 'Escolaridade':19

}



# Printo os boxplots para saber se tem outliers

plt.figure(figsize=(20,30))



for variavel,i in dicionario.items():

                     plt.subplot(5,4,i)

                     plt.boxplot(ams[variavel],whis=1.5)

                     plt.title(variavel)



plt.show()
# cópia do df da américa do sul - este será tratado

ams_tratado = ams.copy()



# dicionário para listar apenas as variáveis ajustadas, para comparação posterior

variaveis_ajustadas = {}



for variavel, y in dicionario.items():

    q75, q25 = np.percentile(ams[variavel], [75 ,25])

    

    # iqr é a diferença entre os quartis 3 e 1 (q3 - q1)

    iqr = q75 - q25

    

    # 1.5 é por conta da FIQ

    min_val = q25 - (iqr*1.5)

    max_val = q75 + (iqr*1.5)

    

    qtd_acima = 0

    qtd_abaixo = 0

    

    limite_acima = 0

    limite_abaixo = 0    

    

    # se tiver outlier

    if len((np.where((ams[variavel] > max_val) | (ams[variavel] < min_val))[0])) > 0:

        # se tiver outlier acima

        if len((np.where(ams[variavel] > max_val)[0])) > 0:             

            limite_acima = math.ceil((len((np.where((ams[variavel] > max_val))[0]))*100/ams[variavel].count()))/100

            qtd_acima = len((np.where(ams[variavel] > max_val)[0]))

        # se tiver outlier abaixo

        if len((np.where(ams[variavel] < min_val)[0])) > 0:             

            limite_abaixo = math.ceil((len((np.where((ams[variavel] < min_val))[0]))*100/ams[variavel].count()))/100

            qtd_abaixo = len((np.where(ams[variavel] < min_val)[0]))

                

        ams_tratado[variavel] = winsorize(ams_tratado[variavel],(limite_abaixo, limite_acima))

        print('{}: Acima: {} - {} / Abaixo: {} - {} '.format(variavel, qtd_acima, limite_acima, qtd_abaixo, limite_abaixo))

        #print('Ajuste efetuado: ams_tratado[{}] = winsorize(ams_tratado[{}],({}, {}))'.format(variavel, variavel, limite_abaixo, limite_acima))

        

        variaveis_ajustadas[variavel] = y
# Printo os boxplots para saber como ficou o ajuste

plt.figure(figsize=(20,30))



y = 1



for variavel,i in variaveis_ajustadas.items():

    plt.subplot(10,4,y)

    plt.boxplot(ams[variavel],whis=1.5)

    plt.title(variavel)

    

    plt.subplot(10,4,y+1)

    plt.boxplot(ams_tratado[variavel],whis=1.5)

    plt.title(variavel + ' - Tratada')

    

    y = y + 2



plt.show()
ams_tratado.head()
# Maiores números de Mortes Adultos

ams_tratado.nlargest(10, 'Mortes Adultos')[{'País', 'Ano', 'Mortes Adultos'}].style.hide_index()
# Maiores números de Expectativa de Vida

ams_tratado.nlargest(10, 'Expectativa de Vida')[{'País', 'Ano', 'Expectativa de Vida'}].style.hide_index()
# Países Com os Maiores Índices de Mortes Adultos - América do Sul

f, ax = plt.subplots(figsize=(10,7))



plot = sns.scatterplot(data=ams_tratado[ams_tratado['País'].isin(('Bolivia','Guyana'))], y='Mortes Adultos', x='Ano', hue='País', ax=ax, s=50)



plot.set_title("Países Com os Maiores Índices de Mortes Adultos - América do Sul",fontsize=13)

plot.legend(loc='center left', ncol=1)



f.tight_layout()
sns.pairplot(ams_tratado[{"Mortes Adultos", "Expectativa de Vida"}], kind="reg")
# Maiores números de Mortes Infantis

ams_tratado.nlargest(10, 'Mortes Infantis')[{'País', 'Ano', 'Mortes Infantis'}].style.hide_index()
# Maiores Índices de Consumo de Álcool

ams_tratado.nlargest(10, 'Álcool')[['País', 'Ano', 'Álcool']].style.hide_index()
# Países Com os Maiores Índices de Consumo de Álcool - América do Sul

f, ax = plt.subplots(figsize=(10,7))



plot = sns.scatterplot(data=ams_tratado[ams_tratado['País'].isin(('Venezuela','Argentina'))], y='Álcool', x='Ano', hue='País', ax=ax, s=50)



#plot.set_title("Países Com os Maiores Índices de Consumo de Álcool",fontsize=13)

plot.legend(loc='upper left', ncol=1)



f.tight_layout()
# Consumo de Álcool por ano

f, ax = plt.subplots(figsize=(15,6))

sns.barplot(y='Álcool', x='Ano',  data=ams_tratado)
# Mortes por HIV/AIDS por Ano

f, ax = plt.subplots(figsize=(15,6))

plot1 = sns.barplot(y='Mortes por HIV/AIDS', x='Ano', data=ams_tratado, palette='husl')

plot1.set_title("Mortes por HIV/AIDS por Ano",fontsize=13)



plt.show()
# Cobertura de Diftería por Ano - América do Sul

f, ax = plt.subplots(figsize=(15,6))

plot1 = sns.barplot(y='Difteria', x='Ano', data=ams_tratado, palette='husl')

plot1.set_title("Cobertura de Diftería por Ano - América do Sul",fontsize=13)



plt.show()
# Expectativa de Vida por Ano - América do Sul

f, ax = plt.subplots(figsize=(10,7))



plot = sns.scatterplot(data=ams_tratado

                , y='Expectativa de Vida'

                , x='Ano'

                , hue='País'

                , palette='colorblind'

                , ax=ax

                , s=50

               )



#plot.set_title("Expectativa de Vida por Ano - América do Sul",fontsize=13)

plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)



f.tight_layout()
ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Expectativa de Vida'}].style.hide_index()
# Maiores Índices de Consumo de Álcool

#ams_tratado.nlargest(10, 'Hepatite B')[['País', 'Ano', 'Hepatite B']].style.hide_index()

ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Hepatite B'}].style.hide_index()
# Cobertura Hepatite B por Ano e País - América do Sul

f, ax = plt.subplots(figsize=(10,7))



plot = sns.scatterplot(data=ams_tratado

                , y='Hepatite B'

                , x='Ano'

                , hue='País'

                , palette='colorblind'

                , ax=ax

                , s=50

               )



plot.set_title("Cobertura Hepatite B por Ano e País - América do Sul",fontsize=13)

plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)



f.tight_layout()
ams_tratado.info()
f, ax = plt.subplots(figsize=(12,7))



plot = ams_tratado[{'Ano', 'Magreza 10-19 anos', 'Magreza 5-9 anos'}].groupby(['Ano']).mean()[{'Magreza 10-19 anos', 'Magreza 5-9 anos'}].plot(ax=ax)

#plot.set_title("% Média de Prevalência de Magreza - América do Sul",fontsize=13)

plot.legend(loc='center right', ncol=1)



plt.show()



f.tight_layout()
ams_tratado.nlargest(10, 'Magreza 5-9 anos')[{'País', 'Ano', 'Magreza 5-9 anos'}].style.hide_index()
ams_tratado.nlargest(10, 'Magreza 5-9 anos')[{'País', 'Ano', 'Magreza 10-19 anos'}].style.hide_index()
f, ax = plt.subplots(figsize=(12,7))



plot = ams_tratado[{'Ano', 'Mortes Adultos', 'Mortes por HIV/AIDS', 'Mortes Infantis', 'Mortes - Menor de 5 anos'}].groupby(['Ano']).mean().round()[{'Mortes Adultos', 'Mortes por HIV/AIDS', 'Mortes Infantis', 'Mortes - Menor de 5 anos'}].plot(ax=ax)

plot.set_title("Quantidade Média de Mortes a Cada 1000 - América do Sul",fontsize=13)

plot.legend(loc='center right', ncol=1)



plt.show()



f.tight_layout()
ams_tratado[{'Ano', 'Mortes Adultos'}].groupby(['Ano']).mean().round().astype(int).transpose()
ams_tratado[{'Ano', 'Hepatite B', 'Difteria', 'Poliomielite'}].groupby(['Ano']).mean().round(2)
f, ax = plt.subplots(figsize=(12,7))



plot = ams_tratado[{'Ano', 'Hepatite B', 'Difteria', 'Poliomielite'}].groupby(['Ano']).mean().round(2)[{'Difteria', 'Hepatite B', 'Poliomielite'}].plot(ax=ax)

plot.set_title("% Média de Cobertura da Imunização - América do Sul",fontsize=13)

plot.legend(loc='upper left', ncol=1)



plt.show()



f.tight_layout()
ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Hepatite B'}].style.hide_index()
sns.barplot(x='Hepatite B', y='País', data=ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Hepatite B'}], palette='husl')
ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Poliomielite'}].style.hide_index()
sns.barplot(x='Poliomielite', y='País', data=ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Poliomielite'}], palette='husl')
ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Difteria'}].style.hide_index()
sns.barplot(x='Difteria', y='País', data=ams_tratado[ams_tratado['Ano'] == 2015][{'Ano', 'País', 'Difteria'}], palette='husl')
# Correlação dos dados



f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(ams_tratado.corr(), annot=True, fmt='.2f', linecolor='black' , ax=ax, lw=.7)
ams_tratado.head()
# Determino o p-valor das variáveis

df_rl = ams_tratado.drop({"País", "Ano", "Situação", "PIB", "População", "Sarampo"},1).copy()

df_rl['intercept']=1



lm=sm.OLS(df_rl['Expectativa de Vida'],df_rl.drop('Expectativa de Vida', 1))

slr_results = lm.fit()

slr_results.summary()
# alineriquetti@gmail.com



df_corr = df_rl.corr()

# Correlação com a variável target

cor_target = abs(df_corr["Expectativa de Vida"])



# Seleciono as correlações maiores que 0.5

variaveis_relevantes = cor_target[cor_target>0.5]

variaveis_relevantes

# Determino o p-valor das variáveis

#df_rl = ams_tratado.drop({"País", "Ano", "Situação", "PIB", "População", "Sarampo"},1).copy()



#lm=sm.OLS(oi['Expectativa de Vida'],df_rl.drop('Expectativa de Vida', 1))

#slr_results = lm.fit()

#slr_results.summary()
# Dicionário das colunas com p-valor < 0.05

cols_regressao = {

      'Mortes Adultos':1

    , 'Mortes Infantis':2

    , 'Mortes - Menor de 5 anos':3

    , 'Mortes por HIV/AIDS':4

    , 'Magreza 5-9 anos':5

    , 'Magreza 10-19 anos':6

    , 'Hepatite B':7

    , 'Poliomielite':8    

    , 'Difteria':9

    , 'IMC':10

    , 'Álcool':11

    , 'Despesa Total':12

    , 'Porcentagem de Despesas':13

    , 'Composição de Renda':14

    , 'Escolaridade':15

}
# regressão linear simples (apenas para análise)

plt.figure(figsize=(20,30))



x = np.array(ams_tratado[{'Expectativa de Vida'}].iloc[:,0].values.reshape(-1, 1))



cont = 1



valores_regressao = []



for variavel,i in cols_regressao.items():

    y = np.array(ams_tratado[{variavel}].iloc[:,0].values.reshape(-1, 1))

    

    model = LinearRegression().fit(x, y)

    

    if slr_results.pvalues.loc[variavel] <= 0.05:

        plt.subplot(4,3,cont)

        plt.scatter(x, y)

        plt.plot(x, model.predict(x), color='red')

        plt.title('Expectativa de Vida x ' + variavel)

        

        cont = cont + 1

        

        valores_regressao.append(variavel + ': CD: ' + ((model.score(x, y) * 100).round(2)).astype(str) + ' | b0: ' + (model.intercept_[0].round(2)).astype(str) + ' | b1: ' + (model.coef_[0][0].round(2)).astype(str))

    

plt.show()    
valores_regressao
rl_n = ams_tratado.drop({"País", "Ano", "Situação", "PIB", "População", "Sarampo"},1).copy()

rl_n.to_csv('rl_n.csv',index=False)
X = rl_n.drop('Expectativa de Vida', 1)

y = rl_n['Expectativa de Vida']
model = LinearRegression()

#Initializing RFE model

rfe = RFE(model, 7)

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)
#no of features

nof_list=np.arange(1,13)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    model = LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))
# https://medium.com/@mayankshah_85820/machine-learning-feature-selection-with-backward-elimination-955894654026

ams_tratado.info()
# separo em dois arrays

rl_n = ams_tratado.drop({"País", "Ano", "Situação", "PIB", "População", "Sarampo"},1).copy()



X = rl_n.iloc[:, :-1].values

y = rl_n.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
scaler = StandardScaler()

X_train[:,3:] = scaler.fit_transform(X_train[:,3:])

X_test[:,3:] = scaler.transform(X_test[:,3:])
#from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))
#import numpy as np

X_train = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
#import statsmodels.api as sm

X_opt = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

regressor = sm.OLS(Y_train, X_train[:,X_opt]).fit()

print(regressor.summary())
X = df_rl.drop('Expectativa de Vida', 1)

y = df_rl['Expectativa de Vida']





def stepwise_selection(X, y, 

                       initial_list=[], 

                       threshold_in=0.01, 

                       threshold_out = 0.05, 

                       verbose=True):

    """ Perform a forward-backward feature selection 

    based on p-value from statsmodels.api.OLS

    Arguments:

        X - pandas.DataFrame with candidate features

        y - list-like with the target

        initial_list - list of features to start with (column names of X)

        threshold_in - include a feature if its p-value < threshold_in

        threshold_out - exclude a feature if its p-value > threshold_out

        verbose - whether to print the sequence of inclusions and exclusions

    Returns: list of selected features 

    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details

    """

    included = list(initial_list)

    while True:

        changed=False

        # forward step

        excluded = list(set(X.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included



result = stepwise_selection(X, y)



print('resulting features:')

print(result)