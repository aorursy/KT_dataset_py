# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Bibliotecas necesárias para análise

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from sklearn import linear_model





import scipy as sp

from scipy.stats import norm



from IPython.core.pylabtools import figsize





from sklearn import preprocessing

from sklearn.model_selection import train_test_split 

from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error, accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder





import warnings

warnings.filterwarnings('ignore')



import plotly

import plotly as py

import plotly.graph_objs as go

import plotly.offline as py

plotly.offline.init_notebook_mode(connected=True)

from plotly.offline import plot, iplot

dados = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
dados.head(5)
dados.tail(5)
list(dados.columns.values)
dados.shape
print('No. de atributos categóricos: ', dados.select_dtypes(exclude = ['int64','float64']).columns.size)
print('No. de atributos numéricos: ', dados.select_dtypes(exclude = ['object']).columns.size)
plt.figure(figsize=(20,6))

sns.heatmap(dados.select_dtypes(include=['object']).isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

plt.title('Valores Nulos presentes nas variáveis Object',fontsize=18)

plt.show()
dados.select_dtypes(include=['object'])
dados['SARS-Cov-2 exam result'].unique()
dados.groupby(['SARS-Cov-2 exam result']).count()
# 'SARS-Cov-2 exam result' #negativo 0 e positivo 1



def transforma(s):

        if s == 'negative':

            return 0

        elif s == 'positive':

            return 1

        

dados['SARS-Cov-2 exam result'] = dados['SARS-Cov-2 exam result'].apply(transforma)
dados.dtypes
dados = dados.fillna(0)  #Preencher os valores nulos com 0 
dados.isnull().sum().sort_values(ascending=False)
#Exclusão de dados sem representatividade



dados.drop('Mycoplasma pneumoniae', axis = 1, inplace=True)

dados.drop('Fio2 (venous blood gas analysis)', axis = 1, inplace=True)

dados.drop('Promyelocytes', axis = 1, inplace=True)

dados.drop('Myeloblasts', axis = 1, inplace=True)

dados.drop('Partial thromboplastin time\xa0(PTT)\xa0', axis = 1, inplace=True)

dados.drop('Prothrombin time (PT), Activity', axis = 1, inplace=True)

dados.drop('D-Dimer', axis = 1, inplace=True)

dados.drop('Urine - Sugar', axis = 1, inplace=True)

def transformacao(s):

        if s == 'not_detected':

            return 0

        if s == 'None':

            return 0

        elif s == 'detected':

            return 1

        elif s == 'negative':

            return 0

        elif s == 'positive':

            return 1

        

#DICIONÁRIO DE DADOS: # 0 not-detected, 1 detected

dados['Respiratory Syncytial Virus'] = dados['Respiratory Syncytial Virus'].apply(transformacao)

dados['Influenza A'] = dados['Influenza A'].apply(transformacao)

dados['Influenza B'] = dados['Influenza B'].apply(transformacao)

dados['Parainfluenza 1'] = dados['Parainfluenza 1'].apply(transformacao)

dados['CoronavirusNL63'] = dados['CoronavirusNL63'].apply(transformacao)

dados['Rhinovirus/Enterovirus'] = dados['Rhinovirus/Enterovirus'].apply(transformacao)

dados['Coronavirus HKU1'] = dados['Coronavirus HKU1'].apply(transformacao)

dados['Parainfluenza 3'] = dados['Parainfluenza 3'].apply(transformacao)

dados['Chlamydophila pneumoniae'] = dados['Chlamydophila pneumoniae'].apply(transformacao)

dados['Adenovirus'] = dados['Adenovirus'].apply(transformacao)

dados['Parainfluenza 4'] = dados['Parainfluenza 4'].apply(transformacao)

dados['Coronavirus229E'] = dados['Coronavirus229E'].apply(transformacao)

dados['CoronavirusOC43'] = dados['CoronavirusOC43'].apply(transformacao)

dados['Inf A H1N1 2009'] = dados['Inf A H1N1 2009'].apply(transformacao)

dados['Bordetella pertussis'] = dados['Bordetella pertussis'].apply(transformacao)

dados['Metapneumovirus'] = dados['Metapneumovirus'].apply(transformacao)

dados['Parainfluenza 2'] = dados['Parainfluenza 2'].apply(transformacao)

    

    

#DICIONÁRIO DE DADOS: # 0 negative, 1 positive    

dados['Influenza B, rapid test'] = dados['Influenza B, rapid test'].apply(transformacao)

dados['Influenza A, rapid test'] = dados['Influenza A, rapid test'].apply(transformacao)

def transformacao_2(s):

        if s == 'not_done':

            return 0

        if s == 'None':

            return 0

        elif s == 'absent':

            return 1

        elif s == 'present':

            return 2

        elif s == 'negative':

            return 0

        elif s == 'positive':

            return 1

        elif s == 'Não Realizado':

            return 0

        

#DICIONÁRIO DE DADOS: # 0 = not-done, 1 = absent, 2 = present

dados['Urine - Hemoglobin'] = dados['Urine - Hemoglobin'].apply(transformacao_2)

dados['Urine - Esterase'] = dados['Urine - Esterase'].apply(transformacao_2)

dados['Urine - pH'] = dados['Urine - pH'].apply(transformacao_2)

#dados['Urine - Hemoglobin'] = dados['Urine - Hemoglobin'].apply(transformacao_2)

dados['Urine - Bile pigments'] = dados['Urine - Bile pigments'].apply(transformacao_2)

dados['Urine - Ketone Bodies'] = dados['Urine - Ketone Bodies'].apply(transformacao_2)

dados['Urine - Nitrite'] = dados['Urine - Nitrite'].apply(transformacao_2)

dados['Urine - Protein'] = dados['Urine - Protein'].apply(transformacao_2)

dados['Urine - Hyaline cylinders'] = dados['Urine - Hyaline cylinders'].apply(transformacao_2)

dados['Urine - Granular cylinders'] = dados['Urine - Granular cylinders'].apply(transformacao_2)

dados['Urine - Yeasts'] = dados['Urine - Yeasts'].apply(transformacao_2)





#DICIONÁRIO DE DADOS: # 0 negative, 1 positive

dados['Strepto A'] = dados['Strepto A'].apply(transformacao_2)  
def transformacao_3(s):

        if s == 'clear':

            return 1

        elif s == 'cloudy':

            return 2

        elif s == 'altered_coloring':

            return 3

        elif s == 'lightly_cloudy':

            return 4

        elif s == 'light_yellow':

            return 1

        elif s == 'yellow':

            return 2

        elif s == 'orange':

            return 3

        elif s == 'citrus_yellow':

            return 4

        elif s == 'normal':

            return 1

        elif s == 'not_done':

            return 0

        elif s == 'Ausentes':

            return 1

        elif s == 'Urato Amorfo --+':

            return 2

        elif s == 'Oxalato de Cálcio +++':

            return 3

        elif s == 'Oxalato de Cálcio -++':

            return 4

        elif s == 'Urato Amorfo +++':

            return 5

        

        

        

#DICIONÁRIO DE DADOS: # 1'clear', 2'cloudy', 3'altered_coloring', 4'lightly_cloudy'

dados['Urine - Aspect'] = dados['Urine - Aspect'].apply(transformacao_3)      

        

#DICIONÁRIO DE DADOS: # 1'normal', 0'not_done'

dados['Urine - Urobilinogen'] = dados['Urine - Urobilinogen'].apply(transformacao_3)  



#DICIONÁRIO DE DADOS: # 1'Ausentes', 2'Urato Amorfo --+', 3'Oxalato de Cálcio +++',4'Oxalato de Cálcio -++', 5'Urato Amorfo +++'

dados['Urine - Crystals'] = dados['Urine - Crystals'].apply(transformacao_3)  



#DICIONÁRIO DE DADOS: # 1'light_yellow', 2'yellow', 3'orange', 4'citrus_yellow'

dados['Urine - Color'] = dados['Urine - Color'].apply(transformacao_3)
dados = dados.fillna(0)  #Preencher os valores nulos com 0 
# Foi analisado cada resultado únicos apresentados nas variáveis.



dados['Respiratory Syncytial Virus'].unique()
dados['Urine - Hemoglobin'].unique()

dados['Urine - Leukocytes'].unique()

#Transformando Urine - Leukocytes de Object para Numérico:



dados['Urine - Leukocytes'] = pd.to_numeric(dados['Urine - Leukocytes'], errors='coerce')
dados.info()
dados.select_dtypes(include=['object'])
dados['SARS-Cov-2 exam result'].value_counts()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Hematocrit', 'Platelets', 'Mean platelet volume ', 'Red blood Cells', 'Lymphocytes', 'Hemoglobin']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Mean corpuscular hemoglobin concentration\xa0(MCHC)', 'Leukocytes', 'Basophils', 'Mean corpuscular hemoglobin (MCH)', 'Eosinophils']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Mean corpuscular volume (MCV)', 'Monocytes', 'Red blood cell distribution width (RDW)', 'Serum Glucose', 'Neutrophils']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Urea', 'Proteina C reativa mg/dL', 'Creatinine', 'Potassium', 'Sodium']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Alanine transaminase', 'Aspartate transaminase', 'Gamma-glutamyltransferase\xa0', 'Total Bilirubin', 'Direct Bilirubin']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Indirect Bilirubin', 'Alkaline phosphatase', 'Ionized calcium\xa0', 'Strepto A', 'Magnesium']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','pCO2 (venous blood gas analysis)', 'Hb saturation (venous blood gas analysis)', 'Base excess (venous blood gas analysis)', 'pO2 (venous blood gas analysis)', 'Total CO2 (venous blood gas analysis)']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','pH (venous blood gas analysis)', 'HCO3 (venous blood gas analysis)', 'Rods #', 'Segmented', 'Metamyelocytes', 'Myelocytes']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Urine - Nitrite', 'Urine - Density', 'Urine - Urobilinogen', 'Urine - Protein', 'Urine - Leukocytes', 'Urine - Red blood cells']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Relationship (Patient/Normal)', 'International normalized ratio (INR)', 'Lactic Dehydrogenase', 'Creatine phosphokinase\xa0(CPK)\xa0', 'Vitamin B12']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','Ferritin', 'Arterial Lactic Acid', 'Lipase dosage', 'Albumin', 'Hb saturation (arterial blood gases)']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','pCO2 (arterial blood gas analysis)', 'Base excess (arterial blood gas analysis)', 'pH (arterial blood gas analysis)', 'Total CO2 (arterial blood gas analysis)', 'HCO3 (arterial blood gas analysis)']

sns.pairplot(dados[cols])

plt.show()
sns.set_style('whitegrid')

cols = ['SARS-Cov-2 exam result','pO2 (arterial blood gas analysis)', 'Arteiral Fio2', 'Phosphor', 'ctO2 (arterial blood gas analysis)']

sns.pairplot(dados[cols])

plt.show()
def bar_chart(feature):

    positivo = dados[dados['SARS-Cov-2 exam result']==1][feature].value_counts()

    negativo = dados[dados['SARS-Cov-2 exam result']==0][feature].value_counts()

    df = pd.DataFrame([positivo,negativo])

    df.index = ['positivo','negativo']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Patient age quantile')
dados.groupby('SARS-Cov-2 exam result')[u'Patient age quantile'].value_counts()
add_regular = dados[dados['Patient addmited to regular ward (1=yes, 0=no)']==1]['SARS-Cov-2 exam result'].value_counts()

add_semi = dados[dados['Patient addmited to semi-intensive unit (1=yes, 0=no)']==1]['SARS-Cov-2 exam result'].value_counts()

add_intensive = dados[dados['Patient addmited to intensive care unit (1=yes, 0=no)']==1]['SARS-Cov-2 exam result'].value_counts()

df = pd.DataFrame([add_regular, add_semi, add_intensive])

df.index = ['Regular','Semi', 'Intensive']

df.plot(kind='bar',stacked=True, figsize=(10,5))
#bar_chart('Influenza A')



influA = dados[dados['Influenza A']==1]['SARS-Cov-2 exam result'].value_counts()

influB = dados[dados['Influenza B']==1]['SARS-Cov-2 exam result'].value_counts()

df = pd.DataFrame([influA, influB])

df.index = ['Influenza A','Influenza B']

df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Parainfluenza 1')
bar_chart('Parainfluenza 2')
bar_chart('Parainfluenza 3')
bar_chart('Parainfluenza 4')
bar_chart('Respiratory Syncytial Virus')
bar_chart('CoronavirusNL63')
bar_chart('Coronavirus HKU1')
bar_chart('Rhinovirus/Enterovirus')
bar_chart('Chlamydophila pneumoniae')
bar_chart('Adenovirus')
bar_chart('Coronavirus229E')
bar_chart('CoronavirusOC43')
bar_chart('Inf A H1N1 2009')
bar_chart('Bordetella pertussis')
bar_chart('Metapneumovirus')
bar_chart('Influenza B, rapid test')
bar_chart('Influenza A, rapid test')
bar_chart('Strepto A')
bar_chart('Urine - Hemoglobin')
bar_chart('Urine - Esterase')
bar_chart('Urine - pH')
bar_chart('Urine - Bile pigments')
bar_chart('Urine - Ketone Bodies')
bar_chart('Urine - Nitrite')
bar_chart('Urine - Protein')
bar_chart('Urine - Hyaline cylinders')
bar_chart('Urine - Granular cylinders')
bar_chart('Urine - Yeasts')
bar_chart('Urine - Aspect')
bar_chart('Urine - Urobilinogen')
bar_chart('Urine - Crystals')
bar_chart('Urine - Color')
corr = dados.corr()



dados_corr = corr[corr>=.8]

plt.figure(figsize=(12,8))

sns.heatmap(dados_corr, cmap="Greens")
dados_matrix = dados.corr().abs()

dados_corr_var = np.where(dados_matrix>0.8)

dados_corr_var=[(dados_matrix.columns[x],dados_matrix.columns[y]) for x,y in zip(*dados_corr_var) if x!=y and x<y]
list(dados_corr_var)
dados['SARS-Cov-2 exam result'].value_counts()
dados_exames_positivos = dados.loc[dados['SARS-Cov-2 exam result']==1]

dados_exames_negativos = dados.loc[dados['SARS-Cov-2 exam result']==0]

dados_exames_positivos.drop('Patient ID', axis = 1, inplace=True)

dados_exames_negativos.drop('Patient ID', axis = 1, inplace=True)
dados_exames_positivos.shape
dados_exames_negativos.shape
dados_exames_negativos
dados_exames_negativos['Hemoglobin'].value_counts()
dados_exames_negativos = dados_exames_negativos.loc[dados_exames_negativos['Hemoglobin'] !=0]
dados_exames_negativos['Hemoglobin'].value_counts()
dados_exames_negativos.shape
base = dados_exames_negativos.append(dados_exames_positivos, ignore_index=True, sort=False)
base.shape
base
dados_modelo = base[['Influenza B','Respiratory Syncytial Virus',

                      'CoronavirusNL63','Coronavirus HKU1','Rhinovirus/Enterovirus','Chlamydophila pneumoniae','Adenovirus',

                      'Coronavirus229E','CoronavirusOC43','Inf A H1N1 2009','Metapneumovirus','Influenza B, rapid test',

                      'Influenza A, rapid test','Strepto A','Hemoglobin','Red blood Cells','Hematocrit','Platelets',

                      'Patient age quantile','Basophils','Lymphocytes','Leukocytes','Hematocrit','SARS-Cov-2 exam result']]



dados_modelo_outros = base[['Mean corpuscular hemoglobin (MCH)','Mean corpuscular volume (MCV)','Alanine transaminase',

                           'Aspartate transaminase','Total Bilirubin','Direct Bilirubin','Indirect Bilirubin',

                           'Hb saturation (venous blood gas analysis)','pO2 (venous blood gas analysis)','Base excess (venous blood gas analysis)',

                           'Total CO2 (venous blood gas analysis)','HCO3 (venous blood gas analysis)','Urine - Esterase',

                           'Urine - Hemoglobin','Urine - Ketone Bodies','Urine - Urobilinogen','Urine - Protein',

                           'Urine - Hyaline cylinders','Urine - Granular cylinders','Urine - Yeasts','Urine - Color',

                           'Urine - Aspect','Urine - Bile pigments','Urine - Crystals','Urine - Ketone Bodies','Urine - Protein',

                           'pH (arterial blood gas analysis)','Total CO2 (arterial blood gas analysis)','HCO3 (arterial blood gas analysis)', 'SARS-Cov-2 exam result']]



#Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)',

#'Patient addmited to intensive care unit (1=yes, 0=no)',
dados_modelo
X = dados_modelo.iloc[:, 1:-1].values    

y = dados_modelo.iloc[:, -1].values
X.shape
y.shape
y
# Dividindo o dataset em treino e teste. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
tree = DecisionTreeClassifier(max_leaf_nodes=38)

model_tree = tree.fit(X_train, y_train)
resultado_tree = model_tree.predict(X_test)
mean_squared_error(y_test, resultado_tree)
model_tree.score(X_test, y_test)
print("Acurácia para o Modelo Decision Tree: ",accuracy_score(resultado_tree,y_test), "%")
#Iniciando o modelo de regressão logistica

model = LogisticRegression()

model.fit(X_train, y_train)
y_train_pred_lg = model.predict(X_train)

y_test_pred_lg = model.predict(X_test)
#Apresentando a acurácia do modelo, o quanto o modelo conseguiu aprender com os dados. 

print("Acurácia para o Modelo Regressão Logistica: ",accuracy_score(y_train_pred_lg, y_train), "%")
#Observando o valor de F1-Score, é recomendado que o modelo seja avaliado por este número pois ele é o 

#balanceamento entre a precisão e recall apresentados abaixo:  

print(classification_report(y_train, y_train_pred_lg))
#Observa-se a Matriz de Confusão, valores previsto corretamente (Verdadeiro positivo), valores previsto incorretamente (Falso positivo), 

#valores que não estavamos buscando prever e foi prevista corretamente(Falso verdadeiro) e valores que não estavamos buscando prever foi prevista incorretamente (Falso negativo). 

cm = confusion_matrix(y_test_pred_lg, y_test)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Matriz de Confusão para a Classificação')

fig.colorbar(cax)

ax.set_xticklabels([''])

ax.set_yticklabels([''])

plt.xlabel('Previsto')

plt.ylabel('Atual')

plt.show()
#Importanto biblioteca do segundo modelo, a Gaussiana é usada para classificaçào e assume-se que segue uma disribuição normal

from sklearn.naive_bayes import GaussianNB
#Executando modelo de predição

NB_sk = GaussianNB()

NB_sk.fit(X_train, y_train)

sk_pred = NB_sk.predict(X_test)
#Definindo uma função para calculo da acurácia, com objetivo de apresentar visualmente em gráficos a dimensão dos dados de 0 a 100000, 

#assim como a faixa de dados testes x previsto.

def accuracy(y_tes, y_pred):

    correct = 0

    for i in range(len(y_pred)):

        if(y_tes[i] == y_pred[i]):

            correct += 1

            

    #Quantidade de Acertos

    return (correct/len(y_tes))*100
print("Acurácia para o Modelo Naive Bayes: ",accuracy_score(sk_pred, y_test), "%")
def sigmoid(z):

  '''

  Retorna o sigmóide, (conhecido como log das probabilidades), o sigmoide inverte o log e retorna o valor original das probabilidades.

  '''

  return 1 / (1 + np.exp(-z))





def predict(features, weights):

  '''

  Retorna uma matriz 1D de probabilidades de que o rótulo da classe seja True / Yes / 1

  O produto escalar dos recursos (Verdadeiro ou Falso) e theta / beta / pesos (B_k) fornecerá o logit (p), que é B0 + B1 Verdadeiro + B2 Falso

  O sigmóide inverte o logit para nos dar p, a probabilidade.

  '''

  return sigmoid( np.dot(features, weights) )
def cost_function_for_all_training_samples(features, labels, weights):

  m = features.shape[0]  # m = número de samples

  predictions = predict(features, weights)

  return -(1/m) * np.sum( labels*np.log(predictions) + (1-labels)*np.log(1-predictions) )
def decision_boundary(probability, threshold=0.5):

  return 1 if probability >= threshold else 0
def calculate_gradient(features, labels, weights):

    predictions = predict(features, weights)

    matrixOfAggregateSlopeOfCostFunction = np.dot(features.T, predictions - labels)

    return matrixOfAggregateSlopeOfCostFunction

  



def update_weights(features, labels, weights, lr):



    matrixOfAggregateSlopeOfCostFunction = calculate_gradient(features, labels, weights)



    m = len(features)

    averageCostDerivativeForEachFeature = matrixOfAggregateSlopeOfCostFunction / m



    gradient = averageCostDerivativeForEachFeature * lr



    return weights - gradient





def fit(features, labels, weights, lr, iterations):



    for i in range(iterations):

        weights = update_weights(features, labels, weights, lr)



        # Processo Log

        if i % 100 == 0:

          cost = cost_function_for_all_training_samples(features, labels, weights)

          print("iteration:", str(i), "cost:", str(cost))



    return weights

  

  

def classify(predictions):

  '''

  input - matriz de elementos N de previsões entre 0 e 1

  output - matriz do elemento N de 0s (False) e 1s (True)

  '''

  decide = np.vectorize(decision_boundary)

  return decide(predictions).flatten()
initial_weights = [0] * X_train.shape[1]

lr = 0.2

iterations = 3001



weights = fit(X_train, y_train, initial_weights, lr, iterations)
#Resultado Final do Modelo - Gradient - Sigmoid

y_test_probabilities = predict(X_test, weights).flatten()

y_test_pred = classify(y_test_probabilities)



#accuracy_score(y_test_pred, y_test)



print("Acurácia para o Modelo Gradient Descent: ",accuracy_score(y_test_pred, y_test), "%")
#Regression scored 0.82 and the scikit learn one scored 0.84.

#For 0, this was slightly more precise, but with worse recall and f1-score. For 1, this was less precise but had a better recall and f1-score.

# Scikit was this:

#                 precision    recall  f1-score   support

#           0       0.88      0.93      0.90      4945

#           1       0.72      0.60      0.65      1568



print(classification_report(y_test, y_test_pred))
cm = confusion_matrix(y_test_pred, y_test)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Matriz de Confusão para a Classificação')

fig.colorbar(cax)

ax.set_xticklabels([''])

ax.set_yticklabels([''])

plt.xlabel('Predição')

plt.ylabel('Atual')

plt.show()
comparacao = pd.DataFrame(y_test, columns=["Ocorrido"])

comparacao.insert(loc=1, column="Previsao", value=resultado_tree)

comparacao.head(25)
comparacao.tail(25)
plt.figure(figsize =(15,4))

plt.plot(np.arange(len(y_train)), y_train, label = 'Treino')

plt.plot(np.arange(len(y_train), (len(y_test)+len(y_train)), 1), y_test, label = 'Teste')

plt.plot(np.arange(len(y_train), (len(resultado_tree)+len(y_train)), 1), resultado_tree, label = 'Previsto')

plt.legend(loc = 'best')

plt.title('Previsao: ' + str(accuracy(y_test, resultado_tree)) + '%')
plt.figure(figsize =(15,4))

plt.plot(np.arange(len(y_train), (len(y_test)+len(y_train)), 1), y_test, label = 'Teste')

plt.plot(np.arange(len(y_train), (len(resultado_tree)+len(y_train)), 1), resultado_tree, label = 'Previsto')

plt.legend(loc = 'best')

plt.title('Previsao: ' + str(accuracy(y_test, resultado_tree)) + '%')