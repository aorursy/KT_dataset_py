import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/X_train_selecionado.csv')

train = train.drop(train.columns[0], axis=1)

train.sample(5)
train.target.value_counts()
X = train.drop(['target'], axis=1)

y = train['target']
#under-sampling

# from imblearn.under_sampling import RandomUnderSampler

# np.random.seed(1000)

# rus = RandomUnderSampler(return_indices=True)

# X_rus, y_rus, id_rus = rus.fit_sample(X, y)



# print('Quantidade de índices removidos:', len(X)-len(X_rus))

# print(X_rus.shape)

# print(y_rus.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

unique, counts = np.unique(y_train, return_counts=True)

print('y_train possui: ', dict(zip(unique, counts)))

unique, counts = np.unique(y_test, return_counts=True)

print('y_test possui: ', dict(zip(unique, counts)))
print(y_train.head())



from sklearn.utils import class_weight

from sklearn.utils.class_weight import compute_sample_weight

y_weigth = compute_sample_weight(class_weight='balanced', y = y_train)

    

print(y_weigth)
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(X_train, y_train, sample_weight=y_weigth)

lr_predictions = lr.predict(X_test); lr_predictions

unique, counts = np.unique(lr_predictions, return_counts=True); unique, counts

lr_score = lr.score(X_test, y_test)

print(lr_score)

lr_predict = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_predict); lr_acc

columns = train.columns[1:]
print("Accuracy:",metrics.accuracy_score(y_test, lr_predict))

print("Precisão:",metrics.average_precision_score(y_test, lr_predict, average='macro', pos_label=1, sample_weight=None))

print("Sensibilidade/Especificidade/F1:")

print(metrics.classification_report(y_test, lr_predict, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False))
lr_cm = metrics.confusion_matrix(y_test, lr_predictions)

fig, ax = plt.subplots(figsize=(4,4))

sns.heatmap(lr_cm, annot=True, fmt='.2f', linewidths=0.5, square=True, cmap='Blues_r', cbar=False, ax=ax)

plt.ylabel('Real')

plt.xlabel('Previsto')
lr_probs = lr.predict_proba(X_test)

lr_preds = lr_probs[:,1]

lr_fpr, lr_tpr, lr_threshold = metrics.roc_curve(y_test, lr_preds)

lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)

print('A AUC da Curva ROC é: ', lr_roc_auc)
plt.figure(figsize=(10,10))

plt.title('Curva ROC')

plt.plot(lr_fpr, lr_tpr, label = 'AUC LR = %0.2f' % lr_roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('Taxa Verdadeiro Positivo')

plt.xlabel('Taxa Falso Positivo');