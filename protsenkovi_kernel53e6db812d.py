import pandas as pd
creditdata = pd.read_csv("/kaggle/input/creditcard.csv")
X = creditdata.drop(['Class', 'Time'], axis=1)

y = creditdata['Class']



X['Amount'] = X['Amount']/X['Amount'].max()
train_test_ratio = 0.8



frauds_idx = X[y==1].index



X_train = X.sample(int(X.shape[0]*train_test_ratio))

y_train = y.loc[X_train.index]



X_train_frauds = X_train.drop(frauds_idx, axis=0, inplace=True, errors='ignore')

X_test = X.loc[(set(X.index) - set(X_train.index)).union(set(frauds_idx))]

y_test = y.loc[X_test.index]



X_train.shape, X_test.shape
def reconstruction_error(X_test, model):

    error = np.sqrt(np.sum(((X_test-model.predict(X_test))**2).values, axis=-1))

    return pd.Series(error, index=X_test.index)



def is_anomaly(X_test, model, threshold):

    rec_error = reconstruction_error(X_test, model)

    y_pred = [1 if err > threshold else 0 for err in rec_error]    

    return pd.Series(y_pred, index=X_test.index)
import tensorflow as tf

import tensorflow.keras as k

import tensorflow.keras.layers as kl

import numpy as np

tf.__version__
features_num = X.shape[1]

encoding_num = features_num*2

learning_rate = 1e-7



input = tf.keras.Input((features_num,))

e1 = kl.Dense(features_num, activation='relu')(input)

e2 = kl.LeakyReLU(alpha=0.2)(kl.Dense(encoding_num, activity_regularizer=k.regularizers.l1(learning_rate))(e1))

d1 = kl.Dense(encoding_num, activation='relu', activity_regularizer=k.regularizers.l1(learning_rate))(e2)

d2 = kl.LeakyReLU(alpha=0.2)(kl.Dense(features_num)(d1))

model = k.Model(inputs=input, outputs=d2)

model_encoder = k.Model(inputs=input, outputs=e2)



model.compile(optimizer='rmsprop', loss='mse')

print(model.summary())
model.fit(X_train, X_train, epochs=50, batch_size=2**7, verbose=0)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

import matplotlib.pyplot as plt

rec_error = reconstruction_error(X_test, model)



tpr, fpr, threshold = roc_curve(y_test, rec_error)

precision, recall, threshold = precision_recall_curve(y_test, rec_error)



fig, ax = plt.subplots(1,1, figsize=(3,3))



ax.plot(tpr, fpr)

ax.plot(recall, precision)



ax.set_title("ROCAUC: {:.3f}, AURPC: {:.3f}".format(auc(tpr, fpr), auc(recall, precision)))
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,3))



ax1.plot(tpr, fpr, label="AUC = {:.2f}".format(auc(tpr, fpr)))

ax1.set_xlabel('Доля ложно-положительных\n срабатываний')

ax1.set_ylabel('Доля\n истинно-положительных\n срабатываний')

ax1.set_title('ROC-кривая')



ax1.legend(handlelength=0, handletextpad=0, fancybox=True)



ax2.plot(recall, precision, label="PR-AUC = {:.2f}".format(auc(recall, precision)))

ax2.set_xlabel('Recall')

ax2.set_ylabel('Precision')

ax2.set_title('PR-кривая')



ax2.legend(handlelength=0, handletextpad=0, fancybox=True)



import seaborn as sn



threshold_fixed = 1

y_pred = is_anomaly(X_test, model, threshold_fixed)

data = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))

df_cm.index.name = 'Истинный класс'

df_cm.columns.name = 'Предсказанный класс'

plt.figure(figsize = (3,3))

sn.set(font_scale=1.2)

sn.heatmap(df_cm, cmap="Blues", annot=True, fmt='g', annot_kws={"size": 12})# font size
encoded_train = pd.DataFrame(model_encoder.predict(X_train), index=X_train.index)

encoded_test = pd.DataFrame(model_encoder.predict(X_test), index=X_test.index)



encoded_train['Class'] = y_train.loc[X_train.index]

encoded_test['Class'] = y_test.loc[X_test.index]



encoded_train.to_csv('/kaggle/working/train_embeddings.csv.zip')

encoded_test.to_csv('/kaggle/working/test_embeddings.csv.zip')
from sklearn.manifold import TSNE
frauds = X[y==1]

frauds['false_negative'] = False

frauds.loc[(y_pred!=y_test), 'false_negative'] = True



frauds.drop(['Amount'], inplace=True, axis=1) 



[p1x, p2x], [p1y, p2y] = [-15, -5], [-10, 15]

[p3x, p4x], [p3y, p4y] = [5, 15], [-13, 10]



fraud_tsne = pd.DataFrame(TSNE(n_components=2, verbose=0, n_iter=300, perplexity=50, random_state = 6).fit_transform(frauds.drop(columns='false_negative')), index=frauds.index)



frauds_0 = np.array([idx for idx, (px, py) in fraud_tsne.reset_index(drop=True).iterrows() if py>((p2y-p1y)/(p2x-p1x))*(px-p2x)+p2y])

frauds_1 = np.array([idx for idx, (px, py) in fraud_tsne.reset_index(drop=True).iterrows() if (py<((p2y-p1y)/(p2x-p1x))*(px-p2x)+p2y) and (py>((p4y-p3y)/(p4x-p3x))*(px-p4x)+p4y)])

frauds_2 = np.array([idx for idx, (px, py) in fraud_tsne.reset_index(drop=True).iterrows() if py<((p4y-p3y)/(p4x-p3x))*(px-p4x)+p4y])

fraud_clusters = pd.concat([pd.Series(np.full(len(frauds_0), 0), index=frauds_0), pd.Series(np.full(len(frauds_1), 1), index=frauds_1), pd.Series(np.full(len(frauds_2), 2), index=frauds_2)])

fraud_clusters = fraud_clusters.sort_index()

frauds['cluster'] = fraud_clusters.values

missed_frauds = frauds[frauds.false_negative]



fig, ax = plt.subplots(1, 1, figsize=(8,5))



ax.scatter(fraud_tsne.iloc[:,0], fraud_tsne.iloc[:,1], label=None)

ax.scatter(fraud_tsne[frauds.false_negative][0], fraud_tsne[frauds.false_negative][1], color='red', label='Пропущенные мошеннические транзакции')



plt.plot([p1x, p2x], [p1y, p2y], color='r')

plt.plot([p3x, p4x], [p3y, p4y], color='r')



plt.legend()

plt.title("T-SNE проекция {} мошеннических транзакций из\n предобработанного пространства признаков".format(frauds.shape[0]))
encoded_x_test = encoded_test.drop('Class', axis=1)

sampled_encoded_x_test = pd.concat([encoded_x_test[y_test==1], encoded_x_test[y_test==0].sample(492, random_state=3)])

embedded_test_tsne = pd.DataFrame(TSNE(n_components=2, random_state=6).fit_transform(sampled_encoded_x_test), index=sampled_encoded_x_test.index)



[p1x, p2x], [p1y, p2y] = [-30, 50], [10, 10]



idx_missed_frauds = frauds[frauds.false_negative].index

idx_detected_frauds = frauds[~frauds.false_negative].index



idx_fraud_A = np.array([idx for idx, (px, py) in embedded_test_tsne[y_test==1].iterrows() if py>((p2y-p1y)/(p2x-p1x))*(px-p2x)+p2y])

idx_fraud_B = np.array([idx for idx, (px, py) in embedded_test_tsne[y_test==1].iterrows() if py<((p2y-p1y)/(p2x-p1x))*(px-p2x)+p2y])

print(len(idx_fraud_A), len(idx_fraud_B))



fig, ax = plt.subplots(1, 1, figsize=(8,5))



ax.scatter(embedded_test_tsne.loc[y_test==0,0], embedded_test_tsne.loc[y_test==0,1], label='Обычные')



ax.scatter(embedded_test_tsne.loc[idx_detected_frauds,0], embedded_test_tsne.loc[idx_detected_frauds,1], color='green', marker = '+', label='Найденные мошеннические')

ax.scatter(embedded_test_tsne.loc[idx_missed_frauds,0], embedded_test_tsne.loc[idx_missed_frauds,1], color='red', label='Пропущенные мошеннические')





ax.set_ylim([-48, 80])

plt.plot([p1x, p2x], [p1y, p2y], color='r')



plt.legend()

plt.title("T-SNE проекция {} мошеннических транзакций и\n 492 случайно выбранных обычных транзакций\n из пространства отклика кодировщика".format(frauds.shape[0]))