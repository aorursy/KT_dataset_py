# Imports
import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
# Importando Spam Base - Treino e Teste:
spamTreino = pd.read_csv("../input/traindata/train_data.csv",
        engine='python')

spamTeste = pd.read_csv("../input/testdata/test_features.csv",
        engine='python')
# Tamanho da base (número de linhas e colunas)
spamTreino.shape
# Visualizando amostra de dados da base:
spamTreino.sample(n=10)
# Lista de colunas da base de treino:
features = list(spamTreino.iloc[:,0:57])
features
# Número de observações de spam e ham na base de treino
spamTreino["ham"].value_counts().plot(kind="bar")
spamTreino["ham"].value_counts()
# Extraino observações classificadas como spam:
spam = spamTreino[(spamTreino['ham'] == 0)]

# Extraindo observações classificadas como ham:
ham = spamTreino[(spamTreino['ham'] == 1)]
spam.iloc[:,0:54].mean().plot(kind="bar")
spam.iloc[:,0:54].mean(axis=0)
spam.iloc[:,54:57].mean().plot(kind="bar")
spam.iloc[:,54:57].mean(axis=0)
ham.iloc[:,0:54].mean().plot(kind="bar")
ham.iloc[:,0:54].mean(axis=0)
ham.iloc[:,54:57].mean().plot(kind="bar")
ham.iloc[:,54:57].mean(axis=0)
feat = np.asarray(features)
arrHam = np.asarray(ham.iloc[:,0:57].mean().tolist())
arrSpam = np.asarray(spam.iloc[:,0:57].mean().tolist())
comparacao = pd.DataFrame({'FEATURES':feat[:],'HAM - Média':arrHam[:], 'SPAM - Média':arrSpam[:]})
comparacao

# Separação da base em Treino (80% da base original) e Teste (20% da base original):
Y = spamTreino[["ham"]]
X = spamTreino.iloc[:,0:57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=30)

# Scaler para ajustar a escala das features:
scaler = MinMaxScaler()
# KNN with Select K Best - avaliando número ideal de features para maximizar a acurácia:
warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
kmax = 0
clf = KNeighborsClassifier(n_neighbors=3)
for num in range(1, 58):
  select = SelectKBest(score_func=f_classif, k=num)
  Xselect_1 = select.fit_transform(X_train, Y_train)
  Xselect = scaler.fit_transform(Xselect_1)
  scores = cross_val_score(clf, Xselect, Y_train, cv=10)
  score_list.append(scores.mean())
  if scores.mean() >= maxscore:
    maxscore = scores.mean()
    kmax = num
plt.plot(np.arange(1, 58), score_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.show()
print(maxscore, kmax)
# KNN with Select K Best - avaliando número ideal de neighbors para maximizar a acurácia:
score_list = []
maxscore = 0
kmax = 0
select = SelectKBest(score_func=f_classif, k=57)
Xselect_1 = select.fit_transform(X_train, Y_train)
Xselect = scaler.fit_transform(Xselect_1)
for k in range(1, 100):
  clf = KNeighborsClassifier(n_neighbors=k)  
  scores = cross_val_score(clf, Xselect, Y_train, cv=10)
  score_list.append(scores.mean())
  if scores.mean() >= maxscore:
    maxscore = scores.mean()
    kmax = k
plt.plot(np.arange(1, 100), score_list)
plt.ylabel('Accuracy')
plt.xlabel('K Neighbors')
plt.show()
print(maxscore, kmax)
# KNN with Select K Best - avaliando número ideal de features para maximizar a f3 score:
score_list = []
maxscore = 0
kmax = 0
clf = KNeighborsClassifier(n_neighbors=63)
for num in range(1, 58):
  select = SelectKBest(score_func=f_classif, k=num)
  Xselect_1 = select.fit_transform(X_train, Y_train)
  Xselect = scaler.fit_transform(Xselect_1)
  scores = cross_validate(clf, Xselect, Y_train, cv=10, scoring=['precision', 'recall'])
  precision = scores['test_precision'].mean()
  recall = scores['test_recall'].mean()
  f3 = 10*((precision*recall)/(9*precision + recall))  
  score_list.append(f3)
  if f3 >= maxscore:
    maxscore = f3
    kmax = num
plt.plot(np.arange(1, 58), score_list)
plt.ylabel('F3 Score')
plt.xlabel('Number of Features')
plt.show()
print(maxscore, kmax)
# KNN with Select K Best - avaliando número ideal de neighbors para maximizar a f3 score:
score_list = []
maxscore = 0
kmax = 0
select = SelectKBest(score_func=f_classif, k=19)
Xselect_1 = select.fit_transform(X_train, Y_train)
Xselect = scaler.fit_transform(Xselect_1)
for k in range(1, 100):
  clf = KNeighborsClassifier(n_neighbors=k)
  scores = cross_validate(clf, Xselect, Y_train, cv=10, scoring=['precision', 'recall'])
  precision = scores['test_precision'].mean()
  recall = scores['test_recall'].mean()
  f3 = 10*((precision*recall)/(9*precision + recall))  
  score_list.append(f3)
  if f3 >= maxscore:
    maxscore = f3
    kmax = k
plt.plot(np.arange(1, 100), score_list)
plt.ylabel('F3 Score')
plt.xlabel('K Neighbors')
plt.show()
print(maxscore, kmax)
# Teste de predição
clf = KNeighborsClassifier(n_neighbors=63)
select = SelectKBest(score_func=f_classif, k=19)
Xselect_1 = select.fit_transform(X_train, Y_train)
Xselect = scaler.fit_transform(Xselect_1)
clf.fit(Xselect, Y_train)
Xselect_2 = select.transform(X_test)
Xselect_test = scaler.transform(Xselect_2)
prediction = clf.predict(Xselect_test)
fbeta_score(Y_test, prediction, beta=3)
# Naive Bayes with Select K Best - avaliando número ideal de features para maximizar a f3 score:
score_list = []
maxscore = 0
kmax = 0
clf = MultinomialNB()
for num in range(1, 58):
  select = SelectKBest(score_func=f_classif, k=num)
  Xselect_1 = select.fit_transform(X_train, Y_train)
  Xselect = scaler.fit_transform(Xselect_1)
  scores = cross_validate(clf, Xselect, Y_train, cv=10, scoring=['precision', 'recall'])
  precision = scores['test_precision'].mean()
  recall = scores['test_recall'].mean()
  f3 = 10*((precision*recall)/(9*precision + recall))  
  score_list.append(f3)
  if f3 >= maxscore:
    maxscore = f3
    kmax = num
plt.plot(np.arange(1, 58), score_list)
plt.ylabel('F3 Score')
plt.xlabel('Number of Features')
plt.show()
print(maxscore, kmax)
# Teste de predição
clf = MultinomialNB()
select = SelectKBest(score_func=f_classif, k=28)
Xselect_1 = select.fit_transform(X_train, Y_train)
Xselect = scaler.fit_transform(Xselect_1)
clf.fit(Xselect, Y_train)
Xselect_2 = select.transform(X_test)
Xselect_test = scaler.transform(Xselect_2)
NB_prediction = clf.predict(Xselect_test)
fbeta_score(Y_test, NB_prediction, beta=3)
X_spamTeste = spamTeste.iloc[:,0:57]
Ids = spamTeste['Id']
clf = MultinomialNB()
select = SelectKBest(score_func=f_classif, k=28)
X1 = select.fit_transform(X,Y)
X2 = scaler.fit_transform(X1)
clf.fit(X2,Y)
X_test_select = select.transform(X_spamTeste)
X_test_final = scaler.transform(X_test_select)
pred = clf.predict(X_test_final)
submission = pd.DataFrame({'Id':Ids,'ham':pred})
submission.to_csv('submission.csv',index = False)
cv = StratifiedKFold(n_splits=6)
classifier = MultinomialNB()
select = SelectKBest(score_func=f_classif, k=28)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, Y):
    Xselect_1 = select.fit_transform(X.iloc[train,:], Y.iloc[train,:])
    Xselect = scaler.fit_transform(Xselect_1)
    Xtest_1 = select.transform(X.iloc[test,:])
    Xtest = scaler.transform(Xtest_1)
    probas_ = classifier.fit(Xselect, Y.iloc[train,:]).predict_proba(Xtest)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y.iloc[test,:], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()