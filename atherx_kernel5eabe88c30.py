# Importação das bibliotecas necessárias e leitura da base
import numpy as np # realização de calculos computacionais
import pandas as pd # manipulação dos dados
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df2 = df.copy()
# Seleciona uma amostra com 10 observações da base e as mostra na tela
display(df.sample(10).T)

# Informa a quantidade de linhas e colunas respectivamente
display(df.shape)
# Informações básicas da base como quantidade de colunas e seus respectivos nomes, quantidade des observações não nulas e seus respectivos tipos
df.info()
def missing_values(data_frame):
    """
    Função responsável por mostrar quantos 
    valores missing há na base em cada 
    coluna
    """
    display(data_frame.isnull().sum().rename_axis('Colunas').reset_index(name='Missing Values'))


#chamada da função
missing_values(df)
'''
seleciona as variáveis númericas da base para uma primeira análise, 
deixando de fora apenas a varíavel target "BAD", pois pelo fato da mesma 
ser binária, vizulizaremos-a melhor posteriormente em outro gráfico
'''
numeric_feats = [c for c in df.columns if df[c].dtype != 'object' and c not in ['BAD']]
df_numeric_feats = df[numeric_feats]

# Cria um gráfico de paridade relacionado cada uma das variáveis entre si
sns.pairplot(df_numeric_feats)
# Cria histogramas das variáveis selecionadas anteriormente
df_numeric_feats.hist(figsize=(20,8), bins=30)
plt.tight_layout() 
'''
Cria uma sequência de gráficos relacionando 
as variáveis númericas anteriormente 
selecionadas com a variável target "BAD"
'''
plt.figure(figsize=(18,18))
c = 1
for i in df_numeric_feats.columns:
    if c < len(df_numeric_feats.columns):
        plt.subplot(3,3,c)
        sns.boxplot(x='BAD' , y= i, data=df)
        c+=1
    else:
        sns.boxplot(x='BAD' , y= i, data=df)
plt.tight_layout() 
# Cria um sequência de gráficos relacionando as varáveis númericas com a variável "JOB".
plt.figure(figsize=(18,18))
c = 1
for i in df_numeric_feats.columns:
    if c < len(df_numeric_feats.columns):
        plt.subplot(3,3,c)
        sns.boxplot(x='JOB' , y= i, data=df)
        c+=1
    else:
        sns.boxplot(x='JOB' , y= i, data=df)
plt.tight_layout() 
#Dropa os campos com valores faltantes na coluna "JOB", sem alterar o datafram original, e pega os possíveis valores nesta coluna.
jobs = df['JOB'].dropna().unique()

#Cria uma série de histogramas da váriavel "VALUE" segmentando pela variável "JOB"
plt.figure(figsize=(14,15))
c=1
for i in jobs:
    plt.subplot(7,1,c)
    plt.title(i)
    df[df['JOB'] == i]['VALUE'].hist(bins=20)
    c+=1
plt.tight_layout() 

# Motra um histograma dos valores presentes na variável "VALUE" onde na coluna JOB está faltando dado
print(df[df['JOB'].isnull()]['VALUE'].hist(bins=20))
#Cria uma série de boxplot relacionando as variáveis númericas com a variável categórica "REASON".
plt.figure(figsize=(18,18))
c = 1
for i in df_numeric_feats.columns:
    if c < len(df_numeric_feats.columns):
        plt.subplot(3,3,c)
        sns.boxplot(x='REASON' , y= i, data=df)
        c+=1
    else:
        sns.boxplot(x='REASON' , y= i, data=df)

# Importando as bibliotecas necessárias
import scipy.stats as stats
from scipy.stats import ttest_ind

# Seleciona os valores da variável "VALUE" onde o valor da varoável "REASON" e igual a "HomeImp"
df_reason_homeimp = df[df['REASON']=='HomeImp']['VALUE']
# Seleciona os valores da variável "VALUE" onde o valor da varoável "REASON" e igual a "DebtCon"
df_reason_debtcon = df[df['REASON']=='DebtCon']['VALUE']

# teste Shapiro-Wilk (Normalidade) para o subconjunto da variável "VALUE" onde o valor da variável "REASON" e igual a "HomeImp"
shapiro_stat_reason_homeimp, shapiro_p_valor_reason_homeimp = stats.shapiro(df_reason_homeimp)
# teste Shapiro-Wilk (Normalidade) para o subconjunto da variável "VALUE" onde o valor da variável "REASON" e igual a "DebtCon"
shapiro_stat_reason_debtcon, shapiro_p_valor_reason_debtcon = stats.shapiro(df_reason_debtcon)

# Mostra o P valor do teste de normalidade
print('teste de normalidade')
print('reason homeimp: {}'.format(shapiro_p_valor_reason_homeimp))
print('reason_debtcon: {}'.format(shapiro_p_valor_reason_debtcon))
# Seleciona os valores da variável "VALUE" onde o valor da varoável "BAD" e igual a "1"
df_bad1 = df[df['BAD']== 1]['VALUE']
# Seleciona os valores da variável "VALUE" onde o valor da varoável "BAD" e igual a "0"
df_bad0 = df[df['BAD']== 0]['VALUE']

# teste Shapiro-Wilk (Normalidade) da biblioteca scipy para o subconjunto da variável "VALUE" onde o valor da variável "bAD" e igual a "1"
shapiro_stat_bad1,  shapiro_p_valor_bad1 = stats.shapiro(df_bad1)
# teste Shapiro-Wilk (Normalidade) da biblioteca scipy para o subconjunto da variável "VALUE" onde o valor da variável "bAD" e igual a "0"
shapiro_stat_bad0, shapiro_p_valor_bad0 = stats.shapiro(df_bad0)

# Mostra o P valor do teste de normalidade
print('teste de normalidade')
print('Value com Bad igual a 1: {}'.format(shapiro_p_valor_bad1))
print('Value com Bad igual a 0: {}'.format(shapiro_p_valor_bad0))
# Seleciona os valores da variável "DEBTINC" onde o valor da varoável "BAD" e igual a "1"
df_debtinc_bad1 = df[df['BAD']== 1]['DEBTINC']
# Seleciona os valores da variável "DEBTINC" onde o valor da varoável "BAD" e igual a "0"
df_debtinc_bad0 = df[df['BAD']== 0]['DEBTINC']
shapiro_stat_bad1,  shapiro_p_valor_debtinc_bad1 = stats.shapiro(df_debtinc_bad1)
shapiro_stat_bad0, shapiro_p_valor_debtinc_bad0 = stats.shapiro(df_debtinc_bad0)

# Mostra o P valor do teste de normalidade
print('teste de normalidade')
print('Debtinc com Bad igual a 1: {}'.format(shapiro_p_valor_debtinc_bad1))
print('Debtinc com Bad igual a 0: {}'.format(shapiro_p_valor_debtinc_bad0))
# remove os valores campos com valores faltantes, sem alterar o dataframe original e realiza o t-test atraves da função ttest_ind da biblioteca scipy
_, ttest_p_value = ttest_ind(df_reason_homeimp.dropna(), df_reason_debtcon.dropna())

# Mostra o P valor  do t-teste
print('T-teste: {:.4f}'.format(ttest_p_value))
# remove os valores campos com valores faltantes, sem alterar o dataframe original e realiza o t-test atraves da função ttest_ind da biblioteca scipy
_, ttest_p_value_bad = ttest_ind(df_bad1.dropna(), df_bad0.dropna())

# Mostr o P valor do teste
print('T-teste: {:.4f}'.format(ttest_p_value_bad))
# remove os valores campos com valores faltantes, sem alterar o dataframe original e realiza o t-test atraves da função ttest_ind da biblioteca scipy
_, ttest_p_value_debtinc_bad = ttest_ind(df_debtinc_bad1.dropna(), df_debtinc_bad0.dropna())

# Mostr o P valor do teste
print('T-teste: {:.4f}'.format(ttest_p_value_debtinc_bad))
# pegando os valores da variável "VALUE" e agrupando por ocupação "JOB"
anova_value_by_job = {job:df['VALUE'][df['JOB'] == job] for job in jobs}

# realizando o teste de análise de variácia
_, anova_value_job_p = stats.f_oneway(anova_value_by_job['Other'].dropna(),
                                          anova_value_by_job['Office'].dropna(),
                                          anova_value_by_job['Sales'].dropna(),
                                          anova_value_by_job['Mgr'].dropna(),
                                          anova_value_by_job['ProfExe'].dropna(), 
                                          anova_value_by_job['Self'].dropna())
# Mostra o P value do teste
print('One Way Anova: {:.4f}'.format(anova_value_job_p))
# Calculando o primeiro quartil da variável "VALUE"
q1 = df['VALUE'].quantile(0.25)
# Calculando o terceiro quartil da variável "VALUE"
q3 = df['VALUE'].quantile(0.75)

# Calculando o IQR
iqr = q3 - q1 

# Guardando domente os valores que não são considerados outliers 
df_value_and_job_no_outlier = df[~((df['VALUE'] < (q1 - 1.5  * iqr)) | (df['VALUE']  > (q3 + 1.5 * iqr)))][['VALUE', 'JOB', 'BAD']]
# Verificando se as medias continuao diferentes
anova_value_by_job = {job:df_value_and_job_no_outlier['VALUE'][df_value_and_job_no_outlier['JOB'] == job] for job in jobs}
anova_job_f, anova_job_p = stats.f_oneway(anova_value_by_job['Other'].dropna(),
                                          anova_value_by_job['Office'].dropna(),
                                          anova_value_by_job['Sales'].dropna(),
                                          anova_value_by_job['Mgr'].dropna(),
                                          anova_value_by_job['ProfExe'].dropna(), 
                                          anova_value_by_job['Self'].dropna())
#Mostra o P valor do teste
print('One Way Anova: {:.4f}'.format(anova_job_p))
sns.boxplot(x='JOB', y='VALUE', data=df_value_and_job_no_outlier)

anova_debtinc_by_job = {job:df['DEBTINC'][df['JOB'] == job] for job in jobs}
anova_debtinc_f, anova_debtinc_p = stats.f_oneway(anova_debtinc_by_job['Other'].dropna(),
                                          anova_debtinc_by_job['Office'].dropna(),
                                          anova_debtinc_by_job['Sales'].dropna(),
                                          anova_debtinc_by_job['Mgr'].dropna(),
                                          anova_debtinc_by_job['ProfExe'].dropna(), 
                                          anova_debtinc_by_job['Self'].dropna())
#Ao menos um dos Jobs tem valores diferentes entre si, estatisticamente falando
print('One Way Anova: {:.4f}'.format(anova_debtinc_p))
# Selecionando o primeiro Quartil da variável "YOJ"
q1 = df['YOJ'].quantile(0.25)
# Selecionando o segundo Quartil da variável "YOJ"
q3 = df['YOJ'].quantile(0.75)

#Ralizando o calculo do iqr
iqr = q3 - q1

#descartando os outliers e 
df_yoj_and_job_no_outlier = df[~((df['YOJ'] < (q1 - 1.5  * iqr)) | (df['YOJ']  > (q3 + 1.5 * iqr)))][['YOJ', 'JOB']]

anova_yoj_by_job = {job:df_yoj_and_job_no_outlier['YOJ'][df_yoj_and_job_no_outlier['JOB'] == job] for job in jobs}
anova_yoj_f, anova_yoj_p = stats.f_oneway(anova_yoj_by_job['Other'].dropna(),
                                          anova_yoj_by_job['Office'].dropna(),
                                          anova_yoj_by_job['Sales'].dropna(),
                                          anova_yoj_by_job['Mgr'].dropna(),
                                          anova_yoj_by_job['ProfExe'].dropna(), 
                                          anova_yoj_by_job['Self'].dropna())
#Ao menos um dos Jobs tem valores diferentes entre si, estatisticamente falando
print('One Way Anova: {:.4f}'.format(anova_yoj_p))
sns.boxplot(x='JOB', y= 'YOJ', data=df_yoj_and_job_no_outlier)
# Salvando as médias da variável VALUE por ocupação
value_mean_by_job = df_value_and_job_no_outlier.groupby(['JOB', 'BAD'])['VALUE'].mean()

# instancia um objeto pandas series sem conteúdo.
imp_value = pd.Series([]) 

'''
reseta o idice do data frame para garantir 
que cada iteração verifique um ídice do 
dataframe evitando com que observações 
fiquem sem ser verificadas.
'''
df.reset_index()
'''
itera sobre o dataframe e, caso valor do 
campo "VALUE" esteja nulo, verifica a 
ocupação do indivíduo e coloca a média de 
"VALUE" para aquele "JOB" naquela posição 
dentro de um objeto Series, caso "VALUE"
não seja nulo, atribui ao objeto o próprio
valor de "VALUE"
'''
for i in range(len(df)):
    if df['VALUE'][i] != df['VALUE'][i]:
        if df['JOB'][i] == 'Mgr':
            if df['BAD'][i] == 0:
                imp_debtinc[i] = value_mean_by_job['Mgr'][0]
            else:
                imp_value[i] = value_mean_by_job['Mgr'][1]
        if df['JOB'][i] == 'Office':
            if df['BAD'][i] == 0:
                imp_value[i] = value_mean_by_job['Office'][0]
            else:
                imp_value[i] = value_mean_by_job['Office'][1]
        if df['JOB'][i] == 'Other':
            if df['BAD'][i] == 0:
                imp_value[i] = value_mean_by_job['Other'][0]
            else:
                imp_value[i] = value_mean_by_job['Other'][1]
        if df['JOB'][i] == 'ProfExe':
            if df['BAD'][i] == 0:
                imp_value[i] = value_mean_by_job['ProfExe'][0]
            else:
                imp_value[i] = value_mean_by_job['ProfExe'][1]
        if df['JOB'][i] == 'Sales':
            if df['BAD'][i] == 0:
                imp_value[i] = value_mean_by_job['Sales'][0]
            else:
                imp_value[i] = value_mean_by_job['Sales'][1]
        if df['JOB'][i] == 'Self':
            if df['BAD'][i] == 0:
                imp_value[i] = value_mean_by_job['Self'][0]
            else:
                imp_value[i] = value_mean_by_job['Self'][1]
            
    else: 
        imp_value[i] = df['VALUE'][i]
'''
casi já exista alguma coluna com o nome IMP_VALUE
realiza a exclusão o mesmo
'''
if "IMP_VALUE" in np.array(df.columns):
    df.drop("IMP_VALUE", axis=1, inplace=True)
    
# Inserie o objeto no dataframe como uma coluna
df.insert(13, "IMP_VALUE", imp_value) 
df.head().T
# Seleciona as observações do dataframe onde a coluna "IMP_VALUE" apresenta valores faltantes
df[df['IMP_VALUE'].isnull()]
# Descarta todas as observações que possuam mais que 10 campos com valores faltantes.
df.dropna(thresh=10, inplace=True)
# Mostra a estrutura do dataframe
df.shape
missing_values(df)
# Seleciona as observações do dataframe onde a coluna "IMP_VALUE" apresenta valores faltantes
df[df['IMP_VALUE'].isnull()]
df.dropna(axis=0,subset=['IMP_VALUE'], inplace=True)
df.shape
df.drop('VALUE', axis=1, inplace=True)
df.head()
missing_values(df)
df.dropna(axis=0, subset=['JOB'], inplace=True)
df.dropna(axis=0, subset=['REASON'], inplace=True)
df.shape
missing_values(df)
df[['IMP_VALUE', 'MORTDUE']].corr()
plt.scatter(df['IMP_VALUE'], df['MORTDUE'])
plt.ylabel('IMP_VALUE')
plt.xlabel('MORTDUE')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
missing_mortdue = df[df['MORTDUE'].isnull()][['IMP_VALUE', 'MORTDUE']]
not_missing_mortdue = df[df['MORTDUE'].notnull()][['IMP_VALUE', 'MORTDUE']]
X = not_missing_mortdue['IMP_VALUE'].values.reshape(-1, 1)
y = not_missing_mortdue['MORTDUE'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
mortdue_pred = lr.predict(X_test)
real_vs_pred = pd.DataFrame({'Real': y_test.flatten(), 'Predito': mortdue_pred.flatten()})
real_vs_pred.sample(20)
plt.figure(figsize=(10,10))
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, mortdue_pred, color='red', linewidth=2)
plt.show()
print('Raiz quadrada do Erro medio ao quadrado: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, mortdue_pred))))
print('R quadrado: {}'.format(metrics.r2_score(y_test, mortdue_pred)))
# trantando outliers para tentar diminuir o erro.
# calculando o iqr
q1 = not_missing_mortdue.quantile(0.25)
q3 = not_missing_mortdue.quantile(0.75)

iqr = q3-q1

print(iqr)
not_missing_and_outliers_mortdue = not_missing_mortdue[~((not_missing_mortdue < (q1 - 1.5  * iqr)) | (not_missing_mortdue > (q3 + 1.5 * iqr))).any(axis=1)]
not_missing_mortdue.shape
not_missing_and_outliers_mortdue.shape
X_no_outliers = not_missing_and_outliers_mortdue['IMP_VALUE'].values.reshape(-1, 1)
y_no_outliers = not_missing_and_outliers_mortdue['MORTDUE'].values.reshape(-1, 1)
X_no_outliers_train, X_no_outliers_test, y_no_outliers_train, y_no_outliers_test = train_test_split(X_no_outliers, y_no_outliers, test_size=0.20, random_state=42)
lr_no_outliers = LinearRegression()
lr_no_outliers.fit(X_no_outliers_train, y_no_outliers_train)
mortdue_pred = lr.predict(X_test)
real_vs_pred = pd.DataFrame({'Real': y_test.flatten(), 'Predito': mortdue_pred.flatten()})
real_vs_pred.sample(20)
print('Raiz quadrada do Erro medio ao quadrado: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, mortdue_pred))))
print('R quadrado: {}'.format(metrics.r2_score(y_test, mortdue_pred)))
plt.figure(figsize=(10,10))
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, mortdue_pred, color='red', linewidth=2)
plt.show()
imp_mortdue = pd.Series([])
imp_mortdue = lr.predict(df['IMP_VALUE'].values.reshape(-1,1))
imp_mortdue
df.insert(13, 'IMP_MORTDUE', np.round(imp_mortdue, 2))
df.drop('MORTDUE', axis=1, inplace=True)

df.head()
missing_values(df)
sns.boxplot(x='JOB', y='DEBTINC', data=df)
sns.boxplot(x='BAD', y='DEBTINC', data=df)
q1 = df['DEBTINC'].quantile(0.25)
q3 = df['DEBTINC'].quantile(0.75)

iqr = q3 - q1

df_debtinc_and_job_no_outlier = df[~((df['DEBTINC'] < (q1 - 1.5  * iqr)) | (df['DEBTINC']  > (q3 + 1.5 * iqr)))][['DEBTINC', 'JOB', 'BAD']]
debtinc_mean_by_job = df_debtinc_and_job_no_outlier.groupby(['JOB', 'BAD'])['DEBTINC'].mean()
debtinc_mean_by_job

debtinc_mean_by_job['Mgr'][1]
imp_debtinc = pd.Series([]) 

df.reset_index(inplace=True)
for i in range(len(df)):
    if df['DEBTINC'][i] != df['DEBTINC'][i]:
        if df['JOB'][i] == 'Mgr':
            if df['BAD'][i] == 0:
                imp_debtinc[i] =  debtinc_mean_by_job['Mgr'][0]
            else:
                imp_debtinc[i] =  debtinc_mean_by_job['Mgr'][1]
        if df['JOB'][i] == 'Office':
            if df['BAD'][i] == 0:
                imp_debtinc[i] =  debtinc_mean_by_job['Office'][0]
            else:
                imp_debtinc[i] =  debtinc_mean_by_job['Office'][1]
        if df['JOB'][i] == 'Other':
            if df['BAD'][i] == 0:
                imp_debtinc[i] =  debtinc_mean_by_job['Other'][0]
            else:
                imp_debtinc[i] =  debtinc_mean_by_job['Other'][1]
        if df['JOB'][i] == 'ProfExe':
            if df['BAD'][i] == 0:
                imp_debtinc[i] =  debtinc_mean_by_job['ProfExe'][0]
            else:
                imp_debtinc[i] =  debtinc_mean_by_job['ProfExe'][1]
        if df['JOB'][i] == 'Sales':
            if df['BAD'][i] == 0:
                imp_debtinc[i] =  debtinc_mean_by_job['Sales'][0]
            else:
                imp_debtinc[i] =  debtinc_mean_by_job['Sales'][1]
        if df['JOB'][i] == 'Self':
            if df['BAD'][i] == 0:
                imp_debtinc[i] =  debtinc_mean_by_job['Self'][0]
            else:
                imp_debtinc[i] =  debtinc_mean_by_job['Self'][1]
    else: 
        imp_debtinc[i] = df['DEBTINC'][i]

if "IMP_DEBTINC" in np.array(df.columns):
    df.drop("IMP_DEBTINC", axis=1, inplace=True)
    
df.insert(13, "IMP_DEBTINC", imp_debtinc) 
df.head().T
df.shape
missing_values(df)
df.drop('DEBTINC', axis=1, inplace=True)
sns.boxplot(x='JOB', y='YOJ', data=df)
yoj_mean_by_job = df_yoj_and_job_no_outlier.groupby(['JOB'])['YOJ'].mean()
imp_yoj = pd.Series([]) 

df.reset_index(inplace=True)
for i in range(len(df)):
    if df['YOJ'][i] != df['YOJ'][i]:
        if df['JOB'][i] == 'Mgr':
            imp_yoj[i] =  yoj_mean_by_job['Mgr']
        if df['JOB'][i] == 'Office':
            imp_yoj[i] = yoj_mean_by_job['Office']
        if df['JOB'][i] == 'Other':
            imp_yoj[i] = yoj_mean_by_job['Other']
        if df['JOB'][i] == 'ProfExe':
            imp_yoj[i] = yoj_mean_by_job['ProfExe']
        if df['JOB'][i] == 'Sales':
            imp_yoj[i] = yoj_mean_by_job['Sales']
        if df['JOB'][i] == 'Self':
            imp_yoj[i] = yoj_mean_by_job['Self']
    else: 
        imp_yoj[i] = df['YOJ'][i]
        
if "IMP_YOJ" in np.array(df.columns):
    df.drop("IMP_YOJ", axis=1, inplace=True)
    
df.insert(13, "IMP_YOJ", imp_yoj) 
df.head().T
df.drop('YOJ', axis=1, inplace=True)
missing_values(df)
# Em virtude a falta de tempo para as outra variaveis (DEROG,DELINQ, CLAGE e NINQ)  fora imputadas pela media 
df.fillna(df.mean(), inplace=True)
missing_values(df)
# antes de ir para o modelo propriamente dito vou realizar um one hot encoding nas variaveis categoricas.
df = pd.get_dummies(data=df, columns=['JOB', 'REASON'])
df.head()
# Iniciando o modelo de predicao
#pip install -U scikit-learn == 0.22.1

#pip install -U imbalanced-learn
# Classes desbalanceadas, vamos rodar um modelo com as classes do jeito que estao.
df['BAD'].value_counts().plot(kind='bar')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report

feats = [c for c in df.columns if c not in ['BAD']]
X = df[feats]
y = df['BAD']
param_grid_decision_tree = {
    'criterion': ('gini', 'entropy'),
    'splitter': ('best', 'random'),
    'max_features': ('auto', 'sqrt', 'log2')
}
grid_decision_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_decision_tree)
grid_decision_tree.fit(X,y)
print('\nOs melhores parametros foram: \n' + str(grid_decision_tree.best_params_))
cv_decision_tree = cross_val_score(grid_decision_tree, X, y, cv=10)
print(cv_decision_tree)
print(cv_decision_tree.mean())
param_grid_random_forest = {
    'criterion': ('gini', 'entropy'),
    'max_features': ('log2', 'sqrt')
}

grid_random_forest_classifier = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, bootstrap = True, oob_score = True), param_grid_random_forest)
grid_random_forest_classifier.fit(X,y)
print('\nOs melhores parametros foram: \n' + str(grid_random_forest_classifier.best_params_))
cv_random_forest = cross_val_score(grid_random_forest_classifier, X, y, cv=10, n_jobs=-1)
print(cv_random_forest)
print(cv_random_forest.mean())
param_grid_gbost_classifier = {
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'max_features': ('log2', 'sqrt')
} 
grid_gradiente_boost_machine = GridSearchCV(GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42, n_jobs=-1), param_grid_gbost_classifier)
grid_gradiente_boost_machine.fit(X, y)
print('\nOs melhores parametros foram: \n' + str(grid_gradiente_boost_machine.best_params_))
cv_gradiente_boost_machine = cross_val_score(grid_gradiente_boost_machine, X, y, cv=10)
print(cv_gradiente_boost_machine)
print(cv_gradiente_boost_machine.mean())
# realizando um over sapling da classe minoritaria
smt = SMOTE(sampling_strategy=0.80)
X, y = smt.fit_sample(X,y)
y.value_counts().plot(kind='bar')
grid_decision_tree.fit(X, y)
cv_decision_tree = cross_val_score(grid_decision_tree, X, y, cv=10, n_jobs=-1)
print(cv_decision_tree)
print(cv_decision_tree.mean())
grid_random_forest_classifier.fit(X, y)
cv_random_forest = cross_val_score(grid_random_forest_classifier, X, y, cv=10, n_jobs=-1)
print(cv_random_forest)
print(cv_random_forest.mean())
grid_gradiente_boost_machine.fit(X, y)
cv_gradiente_boost_machine = cross_val_score(grid_gradiente_boost_machine, X, y, cv=10, n_jobs=-1)
print(cv_gradiente_boost_machine)
print(cv_gradiente_boost_machine.mean())