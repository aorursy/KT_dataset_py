import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="white")
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
# Load csv as pandas frame and drop useless columns
data = pd.read_csv('../input/churn-for-bank-customers/churn.csv', index_col='RowNumber')\
    .drop(['Surname', 'CustomerId'], axis=1)
data
data.info()
data['Geography'].value_counts()
data['Gender'].value_counts()
# converts categorical features to integers
def label_encoder(data_: pd.DataFrame(), columns_name_: list):
    le = LabelEncoder()
    for i in columns_name_:
        le.fit(data_[i])
        data_[i] = le.transform(data_[i])
    return data_
data = label_encoder(data, ['Geography', 'Gender'])
data
print(data.CreditScore.describe())

plt.title('CreditScore')
plt.hist(data.CreditScore)
plt.show()
print(data.Balance.describe())

plt.title('Balance')
plt.hist(data.Balance)
plt.show()
print(data.EstimatedSalary.describe())

plt.title('EstimatedSalary')
plt.hist(data.EstimatedSalary)
plt.show()
plt.title('Exited')

sns.barplot(x=data['Exited'].value_counts().keys(),
            y=data['Exited'].value_counts().values)
plt.title('Tenure')

sns.barplot(x=data['Tenure'].value_counts().keys(),
            y=data['Tenure'].value_counts().values)
plt.title('Gender')

sns.barplot(x=['male', 'female'],
            y=data['Gender'].value_counts().values)
plt.title('NumOfProducts')

sns.barplot(x=data['NumOfProducts'].value_counts().keys(),
            y=data['NumOfProducts'].value_counts().values)
plt.title('HasCrCard')

sns.barplot(x=data['HasCrCard'].value_counts().keys(),
            y=data['HasCrCard'].value_counts().values)
plt.title('IsActiveMember')

sns.barplot(x=data['IsActiveMember'].value_counts().keys(),
            y=data['IsActiveMember'].value_counts().values)
print('Min:', data['Age'].min(),
      '\nMax:', data['Age'].max())
val_count = data['Age'].value_counts()
plt.title('Age')
plt.plot([i for i in range(data['Age'].min(), data['Age'].max() + 1)],
         [val_count[i] if i in val_count else 0 for i in range(data['Age'].min(), data['Age'].max() + 1)])
sns.barplot(x=['France', 'Germany', 'Spain'],
            y=[*data['Geography'].value_counts().values])
# correlation table

corr = data.corr()
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=None, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(data, hue="Exited", palette="husl")
# split data on train, val and test
x_tr, x_te, y_tr, y_te = train_test_split(
    data.iloc[:, :-1], data['Exited'], random_state=42, test_size=0.2, shuffle=True)
x_tr, x_val, y_tr, y_val = train_test_split(
    x_tr, y_tr, random_state=42, test_size=0.2, shuffle=True)
print(x_tr.shape)
print(x_val.shape)
print(x_te.shape)
final_score = {}
# calculate auc (if possible), accuracy, f-metric and recall; return pandas frame
def calc_score(y_true, y_pred, y_pred_proba=None):
    return pd.DataFrame(data={'metrics': ['auc', 'acc', 'f1', 'recall'],
                              'single model': [roc_auc_score(y_true, y_pred_proba).round(3)\
                                               if y_pred_proba is not None else '-',
                                               accuracy_score(y_true, y_pred).round(3),
                                               f1_score(y_true, y_pred).round(3),
                                               recall_score(y_true, y_pred).round(3)]})
import copy

# coss-validation
def kfold(model, split: int, X, y, x_test):
    """
    :param model: sklearn model
    :param split: number of folds
    :param X: train data
    :param y: target
    :param x_test: test data, which need to predict
    :return: np.array with predictions on x_test
    """
    pred_cross_val = []
    # init KFold
    kf = KFold(n_splits=split, shuffle=False)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # init new model with same parameters
        model_ = copy.copy(model)
        model_.fit(X_train, y_train)
        pred_cross_val.append(model_.predict(x_test))
    # mean prediction
    pred_cross_val = np.array(pred_cross_val).mean(axis=0)
    pred_cross_val = np.around(pred_cross_val)
    return pred_cross_val
# LogReg model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_tr, y_tr)
# calculate score for single model on validation sample
score = calc_score(y_val, lr.predict(x_val))

# predict and calculate score for cross-validation model on validation sample
pred_cross_val = kfold(LogisticRegression(), 2, x_tr.values, y_tr.values, x_val)
# add column 'cross. val. model' in score
score['cross. val. model'] = calc_score(y_val, pred_cross_val)['single model']

# predict and calculate score for cross-validation model on test sample
pred_cross_val = kfold(LogisticRegression(), 2, x_tr.values, y_tr.values, x_te)
final_score['LogisticRegression'] = calc_score(y_te, pred_cross_val)['single model'][1]
score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(10)
knn.fit(x_tr, y_tr)
# calculate score for single model on validation sample
score = calc_score(y_val, knn.predict(x_val))

# predict and calculate score for cross-validation model on validation sample
pred_cross_val = kfold(KNeighborsClassifier(10), 2, x_tr.values, y_tr.values, x_val)
score['cross. val. model'] = calc_score(y_val, pred_cross_val)['single model']

# predict and calculate score for cross-validation model on test sample
pred_cross_val = kfold(KNeighborsClassifier(10), 2, x_tr.values, y_tr.values, x_te)
final_score['KNeighborsClassifier'] = calc_score(y_te, pred_cross_val)['single model'][1]
score
from catboost import CatBoostClassifier

catb = CatBoostClassifier(learning_rate=0.1, boosting_type='Ordered', verbose=0)
catb.fit(x_tr, y_tr, eval_set=(x_val, y_val), use_best_model=True)

# predict and calculate score for single model on validation sample
score = calc_score(y_val, catb.predict(x_val), catb.predict_proba(x_val)[:, 1])

# predict and calculate score for cross-validation model on test sample
final_score['CatBoostClassifier'] = calc_score(y_te, catb.predict(x_te),
                                        catb.predict_proba(x_te)[:, 1])['single model'][1]

score
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate=0.1, n_estimators=100)
lgbm.fit(x_tr, y_tr)

# predict and calculate score for single model on validation sample
score = calc_score(y_val, lgbm.predict(x_val), lgbm.predict_proba(x_val)[:, 1])

# predict and calculate score for cross-validation model on test sample
final_score['LGBMClassifier'] = calc_score(y_te, lgbm.predict(x_te),
                                              lgbm.predict_proba(x_te)[:, 1])['single model'][1]
score
pred = []
pred_proba = []
pred.append(catb.predict(x_te))
pred_proba.append(catb.predict_proba(x_te)[:, 1])
pred.append(lgbm.predict(x_te))
pred_proba.append(lgbm.predict_proba(x_te)[:, 1])
# mean prediction
pred = np.array(pred).mean(axis=0).round()
pred_proba = np.array(pred_proba).mean(axis=0)

# calculate ensemble score
calc_score(y_te, pred, pred_proba)
import keras
from keras.layers import Dense, Dropout, LeakyReLU
from keras import Sequential
from keras.metrics import Accuracy, AUC
from keras.optimizers import Adam
# plot loss and auc on each epoch
def ploting(history):
    # print(history.history.keys())
    ac = []
    for i in history.history.keys():
        ac.append(i)
    loss = history.history[ac[0]]
    val_loss = history.history[ac[2]]
    acc = history.history[ac[1]]
    val_acc = history.history[ac[3]]
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation loss', color='r')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, acc, 'bo', label='Training acc')
    ax2.plot(epochs, val_acc, 'b', label='Validation acc', color='r')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    for ax in fig.axes:
        ax.grid(True)
    plt.show()
# normalize and split data on train, val and test
scaler = StandardScaler()
scaler.fit(data.iloc[:, :-1])

x_tr, x_te, y_tr, y_te = train_test_split(
    data.iloc[:, :-1], data['Exited'], random_state=42, test_size=0.2, shuffle=True)
x_tr, x_val, y_tr, y_val = train_test_split(
    x_tr, y_tr, random_state=42, test_size=0.2, shuffle=True)
print(x_tr.shape)
print(x_val.shape)
print(x_te.shape)

x_tr = scaler.transform(x_tr)
x_val = scaler.transform(x_val)
x_te = scaler.transform(x_te)

y_tr = y_tr.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)
y_te = y_te.values.reshape(-1, 1)
activation = LeakyReLU(alpha=0.2)

# create model
model = Sequential()
model.add(Dense(64, input_dim=x_tr.shape[-1], activation=activation))
model.add(Dropout(0.2))
model.add(Dense(32, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(16, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(8, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compile model
model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=[AUC()])

# fit model
hist = model.fit(x_tr, y_tr,
          batch_size=64, epochs=150,
          validation_data=(x_val, y_val), verbose=2)
ploting(hist)
score = calc_score(y_val,
                      model.predict(x_val).reshape(-1).round(),
                      model.predict(x_val).reshape(-1))
final_score['NeuroClassifier'] = calc_score(y_te,
                model.predict(x_te).reshape(-1).round(),
                model.predict(x_te).reshape(-1))['single model'][1]
score
import keras
from keras.layers import (Dropout, LeakyReLU, Conv1D,
                          MaxPooling1D, GlobalAveragePooling1D, BatchNormalization)
from keras import Sequential
from keras.metrics import Accuracy, AUC
from keras.optimizers import Adam
activation = 'sigmoid'

model = Sequential()
model.add(Conv1D(64, 3, input_shape=(10, 1), padding='same', activation=activation))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv1D(64, 3, padding='same', activation=activation))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=[AUC()])

print(x_tr.reshape(-1, x_tr.shape[1], 1).shape)
print(x_val.reshape(-1, x_tr.shape[1], 1).shape)
hist = model.fit(x_tr.reshape(-1, x_tr.shape[1], 1), y_tr,
          batch_size=256, epochs=200,
          validation_data=(x_val.reshape(-1, x_val.shape[1], 1),
                           y_val))
ploting(hist)
score = calc_score(y_val,
                      model.predict(x_val.reshape(-1, x_tr.shape[1], 1)).reshape(-1).round(),
                      model.predict(x_val.reshape(-1, x_tr.shape[1], 1)).reshape(-1))
final_score['CNNClassifier'] = calc_score(y_te,
                model.predict(x_te.reshape(-1, x_tr.shape[1], 1)).reshape(-1).round(),
                model.predict(x_te.reshape(-1, x_tr.shape[1], 1)).reshape(-1))['single model'][1]
score
final_score
# sort models' score
import operator
sort_dict = sorted(final_score.items(), key=operator.itemgetter(1), reverse=True)
sort_dict
# rating
def visualize(column=0):
    y = [x[1] for x in sort_dict]
    labels = [x[0] for x in sort_dict]
    shift = 0.78
    plt.figure(figsize=(15, 10))
    graph = sns.barplot(x=(np.asarray(y) - shift), y=labels,
                        palette=sns.color_palette("RdYlGn_r", len(y)),
                        edgecolor=".2", linewidth=2)
    plt.xticks([i / 100 for i in range(0, 11)], ["%.2f" % (i / 100 + shift) for i in range(0, 11)])
    for i, v in enumerate(y):
        graph.text(v - shift - 0.009, i + 0.05, "%.4f" % v, color='darkslategray', fontweight='bold', size=14)
    plt.title('Rating accuracy')
    plt.show()

visualize(0)