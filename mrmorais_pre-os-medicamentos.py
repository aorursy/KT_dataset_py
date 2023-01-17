import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

from keras.optimizers import SGD

warnings.filterwarnings('ignore')

%matplotlib inline
medicamentos_df = pd.read_csv('../input/TA_PRECO_MEDICAMENTO.csv', delimiter=";")
medicamentos_df.columns
# Remover colunas irrelevantes

list_to_drop = ['SUBSTÂNCIA', 'CNPJ', 'LABORATÓRIO', 'CÓDIGO GGREM', 'REGISTRO', 'EAN 1', 'EAN 2', 'EAN 3', 'PRODUTO', 'APRESENTAÇÃO', 'REGIME DE PREÇO', 'PF 0%', 'PF 12%', 'PF 17%', 'PF 17% ALC', 'PF 17,5%', 'PF 17,5% ALC', 'PF 18%', 'PF 18% ALC', 'PF 20%', 'PMC 0%', 'PMC 12%', 'PMC 17%', 'PMC 17% ALC', 'PMC 17,5%', 'PMC 17,5% ALC', 'PMC 18%', 'PMC 18% ALC', 'PMC 20%', 'CAP', 'CONFAZ 87', 'ANÁLISE RECURSAL', 'LISTA DE CONCESSÃO DE CRÉDITO TRIBUTÁRIO (PIS/COFINS)', 'COMERCIALIZAÇÃO 2018']

medicamentos_df = medicamentos_df.drop(labels=list_to_drop, axis=1)

medicamentos_df.columns
# Renomeando as colunas

medicamentos_df.columns = ['classe_terapeutica', 'tipo_produto', 'preco', 'restricao_hospitalar', 'icms_0', 'tarja']
# Selecionando as 80 classes com maior quantidade de medicamentos

classes_mais_relevantes = medicamentos_df.groupby('classe_terapeutica')['classe_terapeutica'].count().sort_values(ascending=False).head(80)
print("Total de medicamentos relevantes:", classes_mais_relevantes.sum())

classes_mais_relevantes
f, ax = plt.subplots(figsize=(12, 9))

plt.tick_params(

    axis='x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom=False,      # ticks along the bottom edge are off

    top=False,         # ticks along the top edge are off

    labelbottom=False) # labels along the bottom edge are off

sns.barplot(classes_mais_relevantes.index, classes_mais_relevantes.values)
# Filtrando o número de classes irrelevantes

classes = classes_mais_relevantes.index.tolist()

print("Nova quantidade de classes terapeuticas: ", len(classes))

medicamentos_df = medicamentos_df[medicamentos_df['classe_terapeutica'].isin(classes)]
medicamentos_df['tipo_produto'].unique()
# Renomeando tipos de produtos

medicamentos_df.loc[medicamentos_df['tipo_produto'] == 'Novo (Referência)', 'tipo_produto'] = 'Novo'

medicamentos_df.loc[medicamentos_df['tipo_produto'] == 'Genérico (Referência)', 'tipo_produto'] = 'Genérico'

medicamentos_df.loc[medicamentos_df['tipo_produto'] == 'Similar (Referência)', 'tipo_produto'] = 'Similar'

medicamentos_df.loc[medicamentos_df['tipo_produto'] == 'Biológico Novo', 'tipo_produto'] = 'Biológicos'

medicamentos_df['tipo_produto'].unique()
medicamentos_df.groupby('tipo_produto')['tipo_produto'].count().sort_values(ascending=False)
# Convertendo preço para numérico

medicamentos_df['preco'] = medicamentos_df['preco'].str.replace(",", ".")

medicamentos_df['preco'] = pd.to_numeric(medicamentos_df['preco'])
sns.distplot(medicamentos_df['preco'])
medicamentos_df = medicamentos_df[(medicamentos_df.preco >= 0.5) & (medicamentos_df.preco <= 150)]
sns.distplot(medicamentos_df['preco'])
print("Skewness: %f" % medicamentos_df.preco.skew())

print("Kurtosis: %f" % medicamentos_df.preco.kurt())
# Relação do preço com tarja

data = pd.concat([medicamentos_df.preco, medicamentos_df.tarja], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x='preco', y='tarja', data=data);
# Relação do preço com tipo de produto

data = pd.concat([medicamentos_df.preco, medicamentos_df.tipo_produto], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x='preco', y='tipo_produto', data=data);
data = pd.concat([medicamentos_df.preco, medicamentos_df.classe_terapeutica], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='classe_terapeutica', y='preco', data=data)

fig.axis(ymin=0, ymax=130);

plt.xticks(rotation=90);
sns.distplot(medicamentos_df.preco, fit=norm);

fig = plt.figure()

res = stats.probplot(medicamentos_df.preco, plot=plt)
medicamentos_df['preco'] = np.log(medicamentos_df['preco'])
sns.distplot(medicamentos_df.preco, fit=norm);

fig = plt.figure()

res = stats.probplot(medicamentos_df.preco, plot=plt)
medicamentos_df.preco.describe()
print(medicamentos_df[(medicamentos_df.preco >= -0.415515) & (medicamentos_df.preco <= 2.659210)].shape)

print(medicamentos_df[(medicamentos_df.preco > 2.659210) & (medicamentos_df.preco <= 3.355502)].shape)

print(medicamentos_df[(medicamentos_df.preco > 3.355502) & (medicamentos_df.preco <= 4.119037)].shape)

print(medicamentos_df[(medicamentos_df.preco > 4.119037) & (medicamentos_df.preco <= 5.010235)].shape)
# Bucketing preço

medicamentos_df.loc[(medicamentos_df.preco >= -0.5) & (medicamentos_df.preco <= 2.659210), 'preco'] = 0

medicamentos_df.loc[(medicamentos_df.preco > 2.659210) & (medicamentos_df.preco <= 3.355502), 'preco'] = 1

medicamentos_df.loc[(medicamentos_df.preco > 3.355502) & (medicamentos_df.preco <= 4.119037), 'preco'] = 2

medicamentos_df.loc[(medicamentos_df.preco > 4.119037) & (medicamentos_df.preco <= 6), 'preco'] = 3
# Bucketing tipo_produto

tipos_produto = medicamentos_df.tipo_produto.unique().tolist()

for i in range(0, len(tipos_produto)):

    medicamentos_df.loc[medicamentos_df.tipo_produto == tipos_produto[i], 'tipo_produto'] = i
# Bucketing classe_terapeutica

classes = medicamentos_df.classe_terapeutica.unique().tolist()

for i in range(0, len(classes)):

    medicamentos_df.loc[medicamentos_df.classe_terapeutica == classes[i], 'classe_terapeutica'] = i
# Bucketing restricao_hospitalar

restricao = medicamentos_df.restricao_hospitalar.unique().tolist()

for i in range(0, len(restricao)):

    medicamentos_df.loc[medicamentos_df.restricao_hospitalar == restricao[i], 'restricao_hospitalar'] = i
# Bucketing icms_0

icms_0_classes = medicamentos_df.icms_0.unique().tolist()

for i in range(0, len(icms_0_classes)):

    medicamentos_df.loc[medicamentos_df.icms_0 == icms_0_classes[i], 'icms_0'] = i
# Bucketing tarja

tarja_classes = medicamentos_df.tarja.unique().tolist()

for i in range(0, len(tarja_classes)):

    medicamentos_df.loc[medicamentos_df.tarja == tarja_classes[i], 'tarja'] = i
sns.pairplot(medicamentos_df, diag_kind="kde")
corrmat = medicamentos_df.corr()

sns.heatmap(corrmat, square=True)
from sklearn.preprocessing import label_binarize



train_labels_ohe = label_binarize(medicamentos_df.preco, classes=[0, 1, 2,3])
medicamentos_df.pop('preco')
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(medicamentos_df, train_labels_ohe, test_size=0.25, random_state=21)
print('X_train size: {}'.format(x_train.shape))

print('y_train size: {}'.format(y_train.shape))

print('X_test size: {}'.format(x_test.shape))

print('y_test size: {}'.format(y_test.shape))
model = Sequential()

model.add(Dense(128, activation='relu', input_dim=5))

model.add(Dense(128, activation='relu'))

model.add(Dense(4, activation='softmax'))



model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
epochs = 200



estimator = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1)
print("Training accuracy: %.2f%% " % 

      (100*estimator.history['acc'][-1]))
sns.reset_orig()   # Reset seaborn settings to get rid of black background

plt.plot(estimator.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()



# Plot model loss over epochs

plt.plot(estimator.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
loss, accuracy = model.evaluate(x=x_test, y=y_test)

print("Accuracy", accuracy)