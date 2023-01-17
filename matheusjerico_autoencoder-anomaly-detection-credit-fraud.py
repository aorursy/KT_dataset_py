import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split 

from sklearn.utils import shuffle

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve



from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

dataset = pd.read_csv("../input/creditcardfraud/creditcard.csv")
dataset.info()
dataset.describe()
print("Distribuição das classificações dos dados:")

print(f"Transações Não Fraudulentas: {round(dataset['Class'].value_counts()[0]/len(dataset) * 100,2)}%.")

print(f"Transações Fraudulentas: {round(dataset['Class'].value_counts()[1]/len(dataset) * 100,2)}%.")
labels = ['Não Fraude', 'Fraude']

sns.countplot('Class', data = dataset, palette = ['red','blue'])

plt.title('Distribuição das Classes', fontsize=14)

plt.xticks(range(2), labels)

plt.xlabel("Classe")

plt.ylabel("Quantidade");
fraude = dataset[dataset['Class'] == 1]['Amount']

n_fraude= dataset[dataset['Class'] == 0]['Amount']
print(f"Transações Não Fraudulentas:\n{n_fraude.describe()}")

print()

print(f"Transações Fraudulentas:\n{fraude.describe()}")
fig, ax = plt.subplots(2, 2, figsize = (18,10))



# Dados de todas as transações

valores = dataset['Amount'].values

tempo = dataset['Time'].values



sns.distplot(valores, ax = ax[0][0], color = 'red')

ax[0][0].set_title('Distribuição dos Valores (Todas as transações)', fontsize = 12)

ax[0][0].set_xlim([min(valores), max(valores)])

ax[0][0].set_xlabel('Valor da Transação')



sns.distplot(tempo, ax = ax[0][1], color = 'green')

ax[0][1].set_title('Distribuição das Transações no Tempo (Todas as transações)', fontsize = 12)

ax[0][1].set_xlim([min(tempo), max(tempo)])

ax[0][1].set_xlabel('Segundos desde a primeira transação')



# Dados apenas das transações fraudulentas

valores_fraude = dataset[dataset['Class'] == 1]['Amount'].values

tempo_fraude = dataset[dataset['Class'] == 1]['Time'].values



sns.distplot(valores_fraude, ax = ax[1][0], color = 'red')

ax[1][0].set_title('Distribuição dos Valores (Transações Fraudulentas)', fontsize = 12)

ax[1][0].set_xlim([min(valores), max(valores)])

ax[1][0].set_xlabel('Valor da Transação')



sns.distplot(tempo_fraude, ax = ax[1][1], color = 'green')

ax[1][1].set_title('Distribuição das Transações no Tempo (Transações Fraudulentas)', fontsize = 12)

ax[1][1].set_xlim([min(tempo), max(tempo)])

ax[1][1].set_xlabel('Segundos desde a primeira transação')



plt.show()
# Remoção da coluna relacionada ao tempo

df = dataset.drop(['Time'], axis=1)



# Normalização da coluna relacionada ao valor da transação

sc = StandardScaler()

df['Amount'] = sc.fit_transform(df['Amount'].values.reshape(-1, 1))
# Separação das classes de dados

fraud_data = df[df['Class'] == 1]

n_fraud_data = df[df['Class'] == 0]
# Separação dos dados de treino e de teste

X_train, X_test = train_test_split(n_fraud_data, test_size = 0.2)



X_train = X_train.drop(columns = ['Class']).copy()



X_test_full = pd.concat([X_test, fraud_data], ignore_index=True, sort=False)

X_test_full = shuffle(X_test_full)



y_test = X_test_full['Class'].copy()

X_test = X_test_full.drop(columns = ['Class'])



print('Dimensionalidade dos dados:')

print(f'Treinamento: {X_train.shape}')

print(f'Teste (Fraudulentos): {fraud_data.shape}, Teste (Não Fraudulentos): {X_test.shape}')
# Dimensões dos dados de entrada

input_dim = X_train.shape[1]



input_layer = Input(shape=(input_dim, ))



# Camadas de encoding

encoder = Dense(18, activation="relu")(input_layer)

encoder = Dense(14, activation="relu", activity_regularizer=regularizers.l2(10e-5))(encoder) 

encoder = Dense(10, activation="relu")(encoder)



# Camadas de decoding

decoder = Dense(14, activation='relu')(encoder)

decoder = Dense(18, activation='relu')(decoder)

decoder = Dense(input_dim, activation='relu')(decoder)



autoencoder = Model(inputs=input_layer, outputs=decoder)



autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])



checkpointer = ModelCheckpoint(filepath="./model/model.h5", verbose=0, save_best_only=True)



tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
autoencoder.summary()
# Treinamento da rede utilizando 15 Épocas e um tamanho de batch de 32



EPOCHS = 50

BS = 64



history = autoencoder.fit(X_train, X_train,

                          epochs = EPOCHS, 

                          batch_size = BS,      

                          shuffle=True,

                          validation_data=(X_test, X_test),

                          verbose=1).history
# Representação gráfica dos valores de perda para os dados de Treino e de Validação ao longo do treinamento

fig, ax = plt.subplots(figsize=[14,8])

ax.plot(history['loss'], label='Treino')

ax.plot(history['val_loss'], label='Validação')

ax.set_title('Valores de Perda', fontdict={'fontsize': 20})

ax.set_ylabel('Perda', fontdict={'fontsize': 15})

ax.set_xlabel('Época', fontdict={'fontsize': 15})

ax.legend(fontsize=12, loc='upper right');
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_test})
error_df.describe()
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]

fraud_error_df = error_df[error_df['true_class'] == 1]



fig, ax = plt.subplots(1, 2, figsize = (25,8))

bins=20



ax[0].hist(normal_error_df.reconstruction_error.values, bins=bins)

ax[0].set_title('Erros de Reconstrução (Transações Não Fraudulentas)', fontsize = 17)

ax[0].set_xlabel('Erro de Reconstrução', fontsize = 15)



ax[1].hist(fraud_error_df.reconstruction_error.values, bins=bins)

ax[1].set_title('Erros de Reconstrução (Transações Fraudulentas)', fontsize = 17)

_ = ax[1].set_xlabel('Erro de Reconstrução', fontsize = 15)
# Cálculo das curvas de precisão e revocação

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)



# Cálculo da curva do F1 Score

F1_score=[]

F1_score = 2 * (precision * recall) / (precision + recall)

where_are_NaNs = np.isnan(F1_score)

F1_score[where_are_NaNs] = 0

threshold_max_f1 = th[np.argmax(F1_score)]
fig, ax = plt.subplots(figsize = (16,9))



ax.plot(recall, precision, 'b', label='Precision-Recall curve')

ax.set_title('Precisão vs Revocação', fontsize=20)

ax.set_xlabel('Revocação', fontsize=15)

_ = ax.set_ylabel('Precisão', fontsize=15)
fig, ax = plt.subplots(figsize = (16,9))



ax.plot(th, precision[1:], 'r', label='Precisão')

ax.plot(th, recall[1:], 'b', label='Revocação')

ax.plot(th, F1_score[1:], 'g', label='F1 Score')



ax.set_title('Precisão, Revocação e F1 Score Vs Threshold', fontsize=20)

ax.set_xlabel('Erro de Reconstrução', fontsize=15)

ax.set_ylabel('Precisão, Revocação e F1', fontsize=15)

_ = ax.legend(fontsize=12)
print('Valor máximo de  F1 Score: ', str(max(F1_score)))
def plot_reconstruction_error(error_df, threshold):

    groups = error_df.groupby('true_class')

    fig, ax = plt.subplots(figsize=(16, 9))



    for name, group in groups:

        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',

                label= "Fraude" if name == 1 else "Normal")

        

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Limite')

    ax.set_title("Erros de reconstrução para as diferentes transações", fontsize=20)

    ax.set_ylabel("Erro de Reconstrução", fontsize=15)

    ax.set_xlabel("Índice dos dados", fontsize=15)

    ax.legend(fontsize=12)
def plot_confusion_matrix(error_df, threshold):

    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

    conf_matrix = confusion_matrix(error_df.true_class, y_pred)



    fig, ax = plt.subplots(figsize=(16, 9))



    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");

    ax.set_title("Matriz de Confusão", fontsize=20)

    ax.set_ylabel('Classe Verdadeira', fontsize=15)

    ax.set_xlabel('Classe Predita', fontsize=15)



    return y_pred
plot_reconstruction_error(error_df, threshold_max_f1)
y_pred = plot_confusion_matrix(error_df, threshold_max_f1)
print(classification_report(error_df.true_class, y_pred))
threshold = 3.2
plot_reconstruction_error(error_df, threshold)
y_pred = plot_confusion_matrix(error_df, threshold)
print(classification_report(error_df.true_class, y_pred))
## FIM