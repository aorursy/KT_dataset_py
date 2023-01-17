# !pip install scikit-plot



import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression



import scikitplot as skplt

import matplotlib

import matplotlib.pyplot as plt



from imblearn.over_sampling import SMOTE

from itertools import groupby



import pickle
df_TRN = pd.read_csv('../input/projeto-rn/TRN.csv', sep='\t', index_col=0)

df_TRN.drop_duplicates(inplace=True)
df_TRN.head()
df_TRN.describe()
# IND_BOM_1_1 e IND_BOM_1_2 indicam quem é bom pagador e quem não, como uma é o inverso da outra, basta utilizar somente uma delas



classe1 = df_TRN[df_TRN['IND_BOM_1_1'] == 0]

classe2 = df_TRN[df_TRN['IND_BOM_1_1'] == 1]

# # classe1.to_csv('classe1.csv')

# classe2.to_csv('classe2.csv')



# # from google.colab import files

# # files.download('classe1.csv')

# # files.download('classe2.csv')



# uploaded = drive.CreateFile({'title': 'classe2.csv'})

# uploaded.SetContentFile('classe2.csv')

# uploaded.Upload()

# print('Uploaded file with ID {}'.format(uploaded.get('id')))
# link = 'https://drive.google.com/open?id=1xQCTHhAbjW6YXhjgwSx0yrHnRtwy-Kjg'

# fluff, id = link.split('=')

# downloaded = drive.CreateFile({'id':id}) 

# downloaded.GetContentFile('classe1.csv')  

df_classe1 = pd.read_csv('../input/projeto-rn2/classe1.csv', sep=',', index_col=0)
df_classe1.head()
# link = 'https://drive.google.com/open?id=13phTJn2_mNv7n_LHE5BsHVEnBcKod3IQ'

# fluff, id = link.split('=')

# downloaded = drive.CreateFile({'id':id}) 

# downloaded.GetContentFile('classe2.csv')  

df_classe2 = pd.read_csv('../input/projeto-rn2/classe2.csv', sep=',', index_col=0)
df_classe2.head()
# X = df_TRN.drop(['INDEX', 'IND_BOM_1_1', 'IND_BOM_1_2'], axis = 1)

# y = df_TRN[['IND_BOM_1_1']]



X1 = df_classe1.iloc[:, 0:-2].values

y1 = df_classe1.iloc[:, -2].values



X2 = df_classe2.iloc[:, 0:-2].values

y2 = df_classe2.iloc[:, -2].values
## Treino: 50%, Validação: 25%, Teste: 25%

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=1/4, 

                                                    random_state=42, stratify=y1)



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=1/4,

                                                  random_state=42, stratify=y2)

# X_test = np.concatenate((X1_test,X2_test))

# y_test = np.concatenate((y1_test,y2_test))
# # with open('X_test.pickle', 'wb') as fp:

# #   pickle.dump(X_test, fp)



# with open('y_test.pickle', 'wb') as fp:

#   pickle.dump(y_test, fp)



# uploaded = drive.CreateFile({'title': 'y_test.pickle'})

# uploaded.SetContentFile('y_test.pickle')

# uploaded.Upload()
# link = 'https://drive.google.com/open?id=1NNLtB_i51PzRcXqHnifqrebLsyF1megE'

# fluff, id = link.split('=')

# downloaded = drive.CreateFile({'id':id}) 

# downloaded.GetContentFile('X_test.pickle')



pickle_off = open('../input/projeto-rn2/X_test.pickle',"rb")

X_test = pickle.load(pickle_off)
# link = 'https://drive.google.com/open?id=1-j1DgCDKlR00TGtRGtVFNfDzTyI1se5_'

# fluff, id = link.split('=')

# downloaded = drive.CreateFile({'id':id}) 

# downloaded.GetContentFile('y_test.pickle')



pickle_off = open('../input/projeto-rn2/y_test.pickle', 'rb')

y_test = pickle.load(pickle_off)

y_test
# for i in range(5):

#   X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=1/3, 

#                                                   random_state=42, stratify=y1_train)

  

#   X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=1/3,

#                                                   random_state=42, stratify=y2_train)

  

#   X_train = np.concatenate((X1_train,X2_train))

#   y_train = np.concatenate((y1_train,y2_train))

  

#   X_val = np.concatenate((X1_val,X2_val))

#   y_val = np.concatenate((y1_val,y2_val))

  

#   path_Xtrain = 'X_train' + str(i) + '.pickle'

#   path_ytrain = 'y_train' + str(i) + '.pickle'

#   path_Xval = 'X_val' + str(i) + '.pickle'

#   path_yval = 'y_val' + str(i) + '.pickle'

  

  

#   with open(path_Xtrain, 'wb') as fp:

#     pickle.dump(X_train, fp)

  

#   with open(path_ytrain, 'wb') as fp:

#     pickle.dump(y_train, fp)

  

#   with open(path_Xval, 'wb') as fp:

#     pickle.dump(X_val, fp)

  

#   with open(path_yval, 'wb') as fp:

#     pickle.dump(y_val, fp)



#   uploaded = drive.CreateFile({'title': path_Xtrain})

#   uploaded.SetContentFile(path_Xtrain)

#   uploaded.Upload()

  

#   uploaded = drive.CreateFile({'title': path_ytrain})

#   uploaded.SetContentFile(path_ytrain)

#   uploaded.Upload()

  

#   uploaded = drive.CreateFile({'title': path_Xval})

#   uploaded.SetContentFile(path_Xval)

#   uploaded.Upload()

  

#   uploaded = drive.CreateFile({'title': path_yval})

#   uploaded.SetContentFile(path_yval)

#   uploaded.Upload()
import os



print(os.listdir('../input'))
# #### BASE 0

# # link = 'https://drive.google.com/open?id=1M4OHrK83fYPTSV4AU8XCrloqIcrrDFep'

# # fluff, id = link.split('=')

# # downloaded = drive.CreateFile({'id':id}) 

# # downloaded.GetContentFile('X_train0.pickle')



# pickle_off = open('../input/projetorn/X_train0.pickle', 'rb')

# X_train0 = pickle.load(pickle_off)



# # link = 'https://drive.google.com/open?id=19D4XVzLLptwHZWF63bfPxegX4fz00EC8'

# # fluff, id = link.split('=')

# # downloaded = drive.CreateFile({'id':id}) 

# # downloaded.GetContentFile('y_train0.pickle')



# pickle_off = open('../input/projetorn/y_train0.pickle', 'rb')

# y_train0 = pickle.load(pickle_off)



# # link = 'https://drive.google.com/open?id=1tWOG9vqoJAu5qp4m-XBhjqlp_OxLurF1'

# # fluff, id = link.split('=')

# # downloaded = drive.CreateFile({'id':id}) 

# # downloaded.GetContentFile('X_val0.pickle')



# pickle_off = open('../input/projetorn/X_val0.pickle', 'rb')

# X_val0 = pickle.load(pickle_off)



# # link = 'https://drive.google.com/open?id=1g-HiwVCWFf1SxRQAJoWA94wfVesWWoNF'

# # fluff, id = link.split('=')

# # downloaded = drive.CreateFile({'id':id}) 

# # downloaded.GetContentFile('y_val0.pickle')



# pickle_off = open('../input/projetorn/y_val0.pickle', 'rb')

# y_val0 = pickle.load(pickle_off)



# # BASE 1



# pickle_off = open('../input/projetorn/X_train1.pickle', 'rb')

# X_train1 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_train1.pickle', 'rb')

# y_train1 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/X_val1.pickle', 'rb')

# X_val1 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_val1.pickle', 'rb')

# y_val1 = pickle.load(pickle_off)



# # BASE 2



# pickle_off = open('../input/projetorn/X_train2.pickle', 'rb')

# X_train2 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_train2.pickle', 'rb')

# y_train2 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/X_val2.pickle', 'rb')

# X_val2 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_val2.pickle', 'rb')

# y_val2 = pickle.load(pickle_off)



# # BASE 3



# pickle_off = open('../input/projetorn/X_train3.pickle', 'rb')

# X_train3 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_train3.pickle', 'rb')

# y_train3 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/X_val3.pickle', 'rb')

# X_val3 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_val3.pickle', 'rb')

# y_val3 = pickle.load(pickle_off)



# # BASE 4



# pickle_off = open('../input/projetorn/X_train4.pickle', 'rb')

# X_train4 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_train4.pickle', 'rb')

# y_train4 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/X_val4.pickle', 'rb')

# X_val4 = pickle.load(pickle_off)



# pickle_off = open('../input/projetorn/y_val4.pickle', 'rb')

# y_val4 = pickle.load(pickle_off)

    

# print('## y_train0 ##')

# print(pd.value_counts(y_train0, normalize=True))

# print('## y_val0 ##')

# print(pd.value_counts(y_val0, normalize=True))

# print('## y_train1 ##')

# print(pd.value_counts(y_train1, normalize=True))

# print('## y_val1 ##')

# print(pd.value_counts(y_val1, normalize=True))

# print('## y_train2 ##')

# print(pd.value_counts(y_train2, normalize=True))

# print('## y_val2 ##')

# print(pd.value_counts(y_val2, normalize=True))

# print('## y_train3 ##')

# print(pd.value_counts(y_train3, normalize=True))

# print('## y_val3 ##')

# print(pd.value_counts(y_val3, normalize=True))

# print('## y_train4 ##')

# print(pd.value_counts(y_train4, normalize=True))

# print('## y_val4 ##')

# print(pd.value_counts(y_val4, normalize=True))
# scaler = StandardScaler()

# X_train0 = scaler.fit_transform(X_train0)

# X_val0 = scaler.fit_transform(X_val0)

# X_test = scaler.transform(X_test)

# X_train1 = scaler.fit_transform(X_train0)

# X_val1 = scaler.fit_transform(X_val0)

# X_train2 = scaler.fit_transform(X_train0)

# X_val2 = scaler.fit_transform(X_val0)

# X_train3 = scaler.fit_transform(X_train0)

# X_val3 = scaler.fit_transform(X_val0)

# X_train4 = scaler.fit_transform(X_train0)

# X_val4 = scaler.fit_transform(X_val0)
# smt = SMOTE(random_state=2, ratio = 1)

# X_train0, y_train0 = smt.fit_sample(X_train0, y_train0)

# X_val0, y_val0 = smt.fit_sample(X_val0, y_val0)
# X_train0s = pickle.dump(X_train0, open('X_train0s.pickle', 'wb'))

# y_train0s = pickle.dump(y_train0, open('y_train0s.pickle', 'wb'))

# X_val0s = pickle.dump(X_val0, open('X_val0s.pickle', 'wb'))

# y_val0s = pickle.dump(y_val0, open('y_val0s.pickle', 'wb'))
# from IPython.display import HTML



# def create_download_link(title = "Download CSV file", filename = "data.pickle"):  

#     html = '<a href={filename}>{title}</a>'

#     html = html.format(title=title,filename=filename)

#     return HTML(html)



# # create a link to download the dataframe which was saved with .to_csv method

# create_download_link(filename='X_train0s.pickle')

# create_download_link(filename='y_train0s.pickle')

# create_download_link(filename='X_val0s.pickle')

# create_download_link(filename='y_val0s.pickle')
pickle_off = open('../input/smotebasezero/X_train0s.pickle', 'rb')

X_train0 = pickle.load(pickle_off)

pickle_off = open('../input/smotebasezero/y_train0s.pickle', 'rb')

y_train0 = pickle.load(pickle_off)

pickle_off = open('../input/smotebasezero/X_val0s.pickle', 'rb')

X_val0 = pickle.load(pickle_off)

pickle_off = open('../input/smotebasezero/y_val0s.pickle', 'rb')

y_val0 = pickle.load(pickle_off)
# X_train1, y_train1 = smt.fit_sample(X_train1, y_train1)

# X_val1, y_val1 = smt.fit_sample(X_val1, y_val1)
# X_train2, y_train2 = smt.fit_sample(X_train2, y_train2)

# X_val2, y_val2 = smt.fit_sample(X_val2, y_val2)
# X_train3, y_train3 = smt.fit_sample(X_train3, y_train3)

# X_val3, y_val3 = smt.fit_sample(X_val3, y_val3)
# X_train4, y_train4 = smt.fit_sample(X_train4, y_train4)

# X_val4, y_val4 = smt.fit_sample(X_val4, y_val4)
print('## y_train0 ##')

print(pd.value_counts(y_train0, normalize=True))

print('## y_val0 ##')

print(pd.value_counts(y_val0, normalize=True))

# print('## y_train1 ##')

# print(pd.value_counts(y_train1, normalize=True))

# print('## y_val1 ##')

# print(pd.value_counts(y_val1, normalize=True))

# print('## y_train2 ##')

# print(pd.value_counts(y_train2, normalize=True))

# print('## y_val2 ##')

# print(pd.value_counts(y_val2, normalize=True))

# print('## y_train3 ##')

# print(pd.value_counts(y_train3, normalize=True))

# print('## y_val3 ##')

# print(pd.value_counts(y_val3, normalize=True))

# print('## y_train4 ##')

# print(pd.value_counts(y_train4, normalize=True))

# print('## y_val4 ##')

# print(pd.value_counts(y_val4, normalize=True))
# Número de features do nosso data set.

input_dim = X_train0.shape[1]



# Aqui criamos o esboço da rede.

classifier = Sequential()



# Agora adicionamos a primeira camada escondida contendo 16 neurônios e função de ativação

# tangente hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar

# a dimensão de entrada (número de features do data set).

classifier.add(Dense(16, activation='tanh', input_dim=input_dim))



# Em seguida adicionamos a camada de saída. Como nosso problema é binário só precisamos de

# 1 neurônio com função de ativação sigmoidal. A partir da segunda camada adicionada keras já

# consegue inferir o número de neurônios de entrada (16) e nós não precisamos mais especificar.

classifier.add(Dense(1, activation='sigmoid'))



# Por fim compilamos o modelo especificando um otimizador, a função de custo, e opcionalmente

# métricas para serem observadas durante treinamento.

classifier.compile(optimizer='adam', loss='mean_squared_error')
# Para treinar a rede passamos o conjunto de treinamento e especificamos o tamanho do mini-batch,

# o número máximo de épocas, e opcionalmente callbacks. No seguinte exemplo utilizamos early

# stopping para interromper o treinamento caso a performance não melhore em um conjunto de validação.

history = classifier.fit(X_train0, y_train0, batch_size=64, epochs=1000, 

                         callbacks=[EarlyStopping(patience=3)], validation_data=(X_val0, y_val0))
def extract_final_losses(history):

    """Função para extrair o melhor loss de treino e validação.

    

    Argumento(s):

    history -- Objeto retornado pela função fit do keras.

    

    Retorno:

    Dicionário contendo o melhor loss de treino e de validação baseado 

    no menor loss de validação.

    """

    train_loss = history.history['loss']

    val_loss = history.history['val_loss']

    idx_min_val_loss = np.argmin(val_loss)

    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}



def plot_training_error_curves(history):

    """Função para plotar as curvas de erro do treinamento da rede neural.

    

    Argumento(s):

    history -- Objeto retornado pela função fit do keras.

    

    Retorno:

    A função gera o gráfico do treino da rede e retorna None.

    """

    train_loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    fig, ax = plt.subplots()

    ax.plot(train_loss, label='Train')

    ax.plot(val_loss, label='Validation')

    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')

    ax.legend()

    plt.show()



def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):

    accuracy = accuracy_score(y, y_pred_class)

    recall = recall_score(y, y_pred_class)

    precision = precision_score(y, y_pred_class)

    f1 = f1_score(y, y_pred_class)

    performance_metrics = (accuracy, recall, precision, f1)

    if y_pred_scores is not None:

        skplt.metrics.plot_ks_statistic(y, y_pred_scores)

        plt.show()

        y_pred_scores = y_pred_scores[:, 1]

        auroc = roc_auc_score(y, y_pred_scores)

        aupr = average_precision_score(y, y_pred_scores)

        performance_metrics = performance_metrics + (auroc, aupr)

    return performance_metrics



def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):

    print()

    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))

    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))

    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))

    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))

    if auroc is not None:

        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))

    if aupr is not None:

        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
plot_training_error_curves(history)
# Fazer predições no conjunto de teste

y_pred_scores = classifier.predict(X_test)

y_pred_class = classifier.predict_classes(X_test, verbose=0)

y_pred_scores_0 = 1 - y_pred_scores

y_pred_scores = np.concatenate([y_pred_scores_0, y_pred_scores], axis=1)



## Matriz de confusão

print('Matriz de confusão no conjunto de teste:')

print(confusion_matrix(y_test, y_pred_class))



## Resumo dos resultados

losses = extract_final_losses(history)

print()

print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))

print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))

print('\nPerformance no conjunto de teste:')

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_pred_class, y_pred_scores)

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
def create_sklearn_compatible_model():

    model = Sequential()

    model.add(Dense(20, activation='tanh', input_dim=input_dim))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
mlp_clf = KerasClassifier(build_fn=create_sklearn_compatible_model, 

                          batch_size=64, epochs=100,

                          verbose=0)

mlp_clf.fit(X_train0, y_train0)

mlp_pred_class = mlp_clf.predict(X_val0)

mlp_pred_scores = mlp_clf.predict_proba(X_val0)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val0, mlp_pred_class, mlp_pred_scores)

print('Performance no conjunto de validação:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
# svc_clf = SVC(probability=True)  # Modifique aqui os hyperparâmetros

# svc_clf.fit(X_train0, y_train0)

# svc_pred_class = svc_clf.predict(X_val0)

# svc_pred_scores = svc_clf.predict_proba(X_val0)

# accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val0, svc_pred_class, svc_pred_scores)

# print('Performance no conjunto de validação:')

# print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
rf_clf = RandomForestClassifier()  # Modifique aqui os hyperparâmetros

rf_clf.fit(X_train0, y_train0)

rf_pred_class = rf_clf.predict(X_val0)

rf_pred_scores = rf_clf.predict_proba(X_val0)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val0, rf_pred_class, rf_pred_scores)

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
gb_clf = GradientBoostingClassifier()  # Modifique aqui os hyperparâmetros

gb_clf.fit(X_train0, y_train0)

gb_pred_class = gb_clf.predict(X_val0)

gb_pred_scores = gb_clf.predict_proba(X_val0)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val0, gb_pred_class, gb_pred_scores)

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

mlp_ens_clf = KerasClassifier(build_fn=create_sklearn_compatible_model,

                              batch_size=64, epochs=50, verbose=0)

# svc_ens_clf = SVC(probability=True)

gb_ens_clf = GradientBoostingClassifier()

rf_ens_clf = RandomForestClassifier()

ens_clf = VotingClassifier([('mlp', mlp_ens_clf), ('gb', gb_ens_clf), ('rf', rf_ens_clf)], 

                           voting='soft')



ens_clf.fit(X_train0, y_train0)

ens_pred_class = ens_clf.predict(X_val0)

ens_pred_scores = ens_clf.predict_proba(X_val0)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val0, ens_pred_class, ens_pred_scores)

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
mlp_pred_class = mlp_clf.predict(X_test0)

mlp_pred_scores = mlp_clf.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, mlp_pred_class, mlp_pred_scores)

print('MLP')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



# svc_pred_class = svc_clf.predict(X_test)

# svc_pred_scores = svc_clf.predict_proba(X_test)

# accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, svc_pred_class, svc_pred_scores)

# print('\n\nSupport Vector Machine')

# print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



gb_pred_class = gb_clf.predict(X_test)

gb_pred_scores = gb_clf.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, gb_pred_class, gb_pred_scores)

print('\n\nGradient Boosting')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



rf_pred_class = rf_clf.predict(X_test)

rf_pred_scores = rf_clf.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, rf_pred_class, rf_pred_scores)

print('\n\nRandom Forest')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



ens_pred_class = ens_clf.predict(X_test)

ens_pred_scores = ens_clf.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, ens_pred_class, ens_pred_scores)

print('\n\nEnsemble')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)