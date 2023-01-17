import pandas as pd #Biblioteca utilizada para manipulação do dataframe
import seaborn as sns # Biblioteca utilizada para explorar as características (Features) das variáveis
import matplotlib.pyplot as plt #Biblioteca utilizada para criação/visualização de gráficos
import tensorflow as tf #Biblioteca utilizada para implementação da Rede Neural Densa
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.shape
df.head()
#Usando a variável 'SARS-Cov-2 exam result', para criar a variavel 'Target', onde negative =0 e positive=1
target = []
for i in range(df.shape[0]):
    if df['SARS-Cov-2 exam result'][i]=='negative':
        target.append(0)
    else:
        target.append(1)
        
dfTarget  =pd.DataFrame({'Target': target})

#juntar o df com o dfTarget
df = pd.concat([df, dfTarget], sort=False, axis=1)
df
#Nome das variáveis
variaveis =list(df.columns)
variaveis

# criaremos uma nova lista (varExcluir) a partir da lista variaveis, 
# apenas usaremos o '#' para selecionar as variáveis que pretendemos 
# usar. Essa "flexibilidade" nos permite testar outras variáveis, (com os devidos ajustes)

varExcluir = ['Patient ID', 
 #'Patient age quantile',
 #'SARS-Cov-2 exam result',
 'Patient addmited to regular ward (1=yes, 0=no)',
 'Patient addmited to semi-intensive unit (1=yes, 0=no)',
 'Patient addmited to intensive care unit (1=yes, 0=no)',
 #'Hematocrit',
 #'Hemoglobin',
 #'Platelets',
 #'Mean platelet volume ',
 #'Red blood Cells',
 #'Lymphocytes',
 #'Mean corpuscular hemoglobin concentration\xa0(MCHC)',
 #'Leukocytes',
 #'Basophils',
 #'Mean corpuscular hemoglobin (MCH)',
 #'Eosinophils',
 #'Mean corpuscular volume (MCV)',
 #'Monocytes',
 #'Red blood cell distribution width (RDW)',
 #'Serum Glucose',
 'Respiratory Syncytial Virus',
 'Influenza A',
 'Influenza B',
 'Parainfluenza 1',
 'CoronavirusNL63',
 'Rhinovirus/Enterovirus',
 'Mycoplasma pneumoniae',
 'Coronavirus HKU1',
 'Parainfluenza 3',
 'Chlamydophila pneumoniae',
 'Adenovirus',
 'Parainfluenza 4',
 'Coronavirus229E',
 'CoronavirusOC43',
 'Inf A H1N1 2009',
 'Bordetella pertussis',
 'Metapneumovirus',
 'Parainfluenza 2',
 #'Neutrophils',
 #'Urea',
 'Proteina C reativa mg/dL',
 #'Creatinine',
 #'Potassium',
 #'Sodium',
 'Influenza B, rapid test',
 'Influenza A, rapid test',
 #'Alanine transaminase',
 #'Aspartate transaminase',
 #'Gamma-glutamyltransferase\xa0',
 #'Total Bilirubin',
 #'Direct Bilirubin',
 #'Indirect Bilirubin',
 #'Alkaline phosphatase',
 'Ionized calcium\xa0',
 'Strepto A',
 'Magnesium',
 'pCO2 (venous blood gas analysis)',
 'Hb saturation (venous blood gas analysis)',
 'Base excess (venous blood gas analysis)',
 'pO2 (venous blood gas analysis)',
 'Fio2 (venous blood gas analysis)',
 'Total CO2 (venous blood gas analysis)',
 'pH (venous blood gas analysis)',
 'HCO3 (venous blood gas analysis)',
 'Rods #',
 'Segmented',
 'Promyelocytes',
 'Metamyelocytes',
 'Myelocytes',
 'Myeloblasts',
 'Urine - Esterase',
 'Urine - Aspect',
 'Urine - pH',
 'Urine - Hemoglobin',
 'Urine - Bile pigments',
 'Urine - Ketone Bodies',
 'Urine - Nitrite',
 'Urine - Density',
 'Urine - Urobilinogen',
 'Urine - Protein',
 'Urine - Sugar',
 'Urine - Leukocytes',
 'Urine - Crystals',
 'Urine - Red blood cells',
 'Urine - Hyaline cylinders',
 'Urine - Granular cylinders',
 'Urine - Yeasts',
 'Urine - Color',
 'Partial thromboplastin time\xa0(PTT)\xa0',
 'Relationship (Patient/Normal)',
 'International normalized ratio (INR)',
 'Lactic Dehydrogenase',
 'Prothrombin time (PT), Activity',
 'Vitamin B12',
 'Creatine phosphokinase\xa0(CPK)\xa0',
 'Ferritin',
 'Arterial Lactic Acid',
 'Lipase dosage',
 'D-Dimer',
 'Albumin',
 'Hb saturation (arterial blood gases)',
 'pCO2 (arterial blood gas analysis)',
 'Base excess (arterial blood gas analysis)',
 'pH (arterial blood gas analysis)',
 'Total CO2 (arterial blood gas analysis)',
 'HCO3 (arterial blood gas analysis)',
 'pO2 (arterial blood gas analysis)',
 'Arteiral Fio2',
 'Phosphor',
 'ctO2 (arterial blood gas analysis)']

#vizualização das variáveis que serão exluidas
varExcluir

#excluindo as variaveis
df.drop(varExcluir, axis=1, inplace=True)
#checando se existem miss values
df.isnull().sum()
# infelizmente, esse dataset possui poucas informações completas...não seria conveniente preencher os valores vazios com "médias"
# manteremos apenas as observações que contem todos os valores. Assim, excluiremos as linhas que tiverem miss values

df=df.dropna()
     
#checando se ainda ficaram algum miss values
df.isnull().sum()
#Checando as quantidades entre as Classes
plt.figure(figsize=(12,6))

cont1 = sns.countplot(x=df['SARS-Cov-2 exam result'])
plt.title("Contagem de pacientes que testaram para COVID -19 ")
plt.xlabel('Contagem de testes Negativos (0) e Positivos (1) ')
plt.ylabel('Contagem ')
#trecho para exibir a contagem na parte superior das barras
cont1.set_xticklabels(cont1.get_xticklabels(),rotation=0)
i=0
for p in cont1.patches:
    height = p.get_height()
    cont1.text(p.get_x()+p.get_width()/2., height + 0.1,
        df['SARS-Cov-2 exam result'].value_counts()[i],ha="center")
    i += 1
#mapa de calor, da correlação entre as variávies
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
#separando as variáveis independentes da variável dependente 
X = df.drop(['SARS-Cov-2 exam result', 'Target'], axis=1)
Y = df['Target']
#separando os dados entre dados de treino e de teste
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30)
#instanciando a rede neural
dnn = tf.keras.models.Sequential()
#exibindo o numedo de reunônios na camada de entrada
X.shape[1] 
#camada de entrada 

neuroniosEnt =X.shape[1] 

#camada oculta 1 
#28-17-
dnn.add(tf.keras.layers.Dense(units=17,kernel_initializer='uniform',activation='relu', input_dim=neuroniosEnt))

#camada oculta 2 
#28-17-17
dnn.add(tf.keras.layers.Dense(units=17,kernel_initializer='uniform',activation='relu'))

#camada oculta 3 
#28-17-17-15
dnn.add(tf.keras.layers.Dense(units=15,kernel_initializer='uniform',activation='relu'))

#camada oculta 4 
#28-17-17-15-15
dnn.add(tf.keras.layers.Dense(units=15,kernel_initializer='uniform',activation='relu'))

#camada oculta 5 
#28-17-17-15-15-8
#dnn.add(tf.keras.layers.Dense(units=8,kernel_initializer='uniform',activation='relu'))

#camada oculta 6 
#28-17-17-15-15-8-8
#dnn.add(tf.keras.layers.Dense(units=8,kernel_initializer='uniform',activation='relu'))


#camada de saida
#28-17-17-15-15-8-8-1
dnn.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#desenho da rede
dnn.summary()
#compilando a rede
dnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#treinando o modelo
batch=32
epocas = 400

dnn.fit(X_train,Y_train, batch_size=batch, epochs=epocas)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
    
# Testando a performance
y_pred_train = dnn.predict(X_train)
y_pred_train= (y_pred_train>0.5)
y_true_train = Y_train
print("Acurácia treino: {0}".format(accuracy_score(y_true_train, y_pred_train).round(2)))
y_pred_test = dnn.predict(X_test)
y_pred_test = (y_pred_test>0.5)
y_true_test = Y_test
print("Acurácia teste : {0}".format(accuracy_score(y_true_test, y_pred_test).round(2)))

# Mas nos bastidores...
print('\n')
print('Relatório de Classificação')
print('\n')
print(classification_report(Y_test, y_pred_test))

#Matriz de confusão 
print('\n')
print('Matriz de Confusão ')

print( confusion_matrix(Y_test, y_pred_test))
#selecionando as variaveis que mais se correlacionam com o Target
corr = df.corr()['Target']
corr = abs(corr)
corr.reset_index()
corr = pd.DataFrame( corr.sort_values(ascending=True))
corr = corr[corr['Target']<0.1]
#lista com os valores das variáveis com valor absoluto menor que 0.1
corr
listaCorr = corr.index
listaCorr
#criando um dataframe, excluindo as variáveis menos relevantes (<0.1)
nX = X.drop(listaCorr, axis=1)
# exibindo o dataframe
nX 
#variáveis
nX.columns
nX.shape
#separando os dados entre dados de treino e de teste
X_train, X_test, Y_train, Y_test = train_test_split(nX,Y, test_size=0.30, random_state=42)
#instanciando a rede neural
dnn = tf.keras.models.Sequential()
#exibindo o numedo de reunônios na camada de entrada
nX.shape[1] 
#camada de entrada 
neuroniosEnt =nX.shape[1] 
#camada oculta 1 
#17-17-
dnn.add(tf.keras.layers.Dense(units=17,kernel_initializer='uniform',activation='relu', input_dim=neuroniosEnt))

#camada oculta 2 
#17-17-17
dnn.add(tf.keras.layers.Dense(units=17,kernel_initializer='uniform',activation='relu'))

#camada oculta 3 
#17-17-17-15
dnn.add(tf.keras.layers.Dense(units=15,kernel_initializer='uniform',activation='relu'))

#camada oculta 4 
#17-17-17-15-15
dnn.add(tf.keras.layers.Dense(units=15,kernel_initializer='uniform',activation='relu'))

#camada de saida
#17-17-17-15-15-1
dnn.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#desenho da rede
dnn.summary()

#compilando a rede
dnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#treinando o modelo
batch=32
epocas = 330

dnn.fit(X_train,Y_train, batch_size=batch, epochs=epocas)
# Testando a performance
y_pred_train = dnn.predict(X_train)
y_pred_train= (y_pred_train>0.5)
y_true_train = Y_train
print("Acurácia treino: {0}".format(accuracy_score(y_true_train, y_pred_train).round(2)))
y_pred_test = dnn.predict(X_test)
y_pred_test = (y_pred_test>0.5)
y_true_test = Y_test
print("Acurácia teste : {0}".format(accuracy_score(y_true_test, y_pred_test).round(2)))

# Nos bastidores...
print('\n')
print('Relatório de Classificação')
print('\n')
print(classification_report(Y_test, y_pred_test))

#Matriz de confusão 
print('\n')
print('Matriz de Confusão ')
print( confusion_matrix(Y_test, y_pred_test))
# Testando todo dataset

y_pred_test = dnn.predict(nX)
y_pred_test = (y_pred_test>0.5)
y_true_test = Y
print("Acurácia no dataset : {0}".format(accuracy_score(y_true_test, y_pred_test).round(2)))

# Nos bastidores..
print('\n')
print('Relatório de Classificação')
print('\n')
print(classification_report(Y, y_pred_test))

# Matriz de confusão
print('\n')
print('Matriz de Confusão ')

print( confusion_matrix(Y, y_pred_test))
