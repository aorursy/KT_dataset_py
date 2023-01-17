from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
arquivo = r'1_9_d.xlsx'
pd = pd.read_excel(arquivo, head='None', sep=';', decimal = ',')
pd.describe
print(pd.isnull().sum())
pd.head()
pd.tail()
x_train=pd.iloc[0:80,0:2]
x_train.tail()
x_test=pd.iloc[80:123,0:2]
x_test.head()
y_train=pd.iloc[0:80,2]
y_train.head()
y_test=pd.iloc[80:123,2]
y_test.tail()
import seaborn as sns
plt.figure(figsize=(15,6))
c = pd.corr()
sns.heatmap(c,cmap='BrBG',annot=True)
c
fsh = LinearDiscriminantAnalysis()
fsh.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
fsh_pred = fsh.predict(x_test)
fsh_cm = confusion_matrix(y_test,fsh_pred)

acuracia_fsh = accuracy_score(y_pred=fsh_pred,y_true=y_test)
precisao_fsh = precision_score(y_pred=fsh_pred,y_true=y_test)
recall_fsh= recall_score(y_pred=fsh_pred,y_true=y_test)
print("acuracia_fsh=%.5f, precisão_fsh=%.5f, recall_fsh=%.4f" % (acuracia_fsh,precisao_fsh,recall_fsh))
print("Matriz_Confusao =")
print (fsh_cm)
x_train_1=pd.iloc[0:80,0:1]
x_train_1.head()
x_test_1=pd.iloc[80:123,0:1]
y_train_1=pd.iloc[0:80,2]
y_test_1=pd.iloc[80:123,2]
y_test_1.tail()
fsh.fit(x_train_1,y_train_1)
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
fsh1_pred = fsh.predict(x_test_1)
fsh1_cm = confusion_matrix(y_test_1,fsh1_pred)

acuracia1_fsh = accuracy_score(y_pred=fsh1_pred,y_true=y_test_1)
precisao1_fsh = precision_score(y_pred=fsh1_pred,y_true=y_test_1)
recall1_fsh= recall_score(y_pred=fsh1_pred,y_true=y_test_1)
print("acuracia_fsh=%.5f, precisão_fsh=%.5f, recall_fsh=%.4f" % (acuracia_fsh,precisao_fsh,recall_fsh))
print("Matriz_Confusao =")
print (fsh_cm)
# Sensibilidade = VP/(VP+FN) = 0,809
# Especificidade = VN/(VN+FP) = 1.00
# Melhor preditor = com duas variáveis.
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
arquivo = r'1_9_d.xlsx'
arq = pd
x_train_1=arq.iloc[0:80,0:1]
x_test_1=arq.iloc[80:123,0:1]
y_train_1=arq.iloc[0:80,2]
y_test_1=arq.iloc[80:123,2]
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train_1, y_train_1)
y_pred = knn.predict(x_test_1)
print('accuracy: ', knn.score(x_test_1, y_test_1))
scores = cross_val_score(knn, x_test_1, y_test_1, cv=5, scoring='accuracy')
print(scores)
loocv = LeaveOneOut()
knn_loocv = knn
results_loocv = cross_val_score(knn_loocv, x_test_1, y_test_1, cv=loocv)
print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
k_range = range(1, 6)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_test_1, y_test_1, cv=loocv, scoring='accuracy')
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(x_train_1,y_train_1)
neigh_pred = neigh.predict(x_test_1)
neigh_cm=confusion_matrix(y_test_1,neigh_pred)

acuracia_neigh = accuracy_score(y_pred=neigh_pred ,y_true=y_test_1)
precisao_neigh = precision_score(y_pred=neigh_pred ,y_true=y_test_1)
recall_neigh = recall_score(y_pred=neigh_pred ,y_true=y_test_1)
neigh_cm=confusion_matrix(y_test_1,neigh_pred)
print(acuracia_neigh,precisao_neigh,recall_neigh)
print(neigh_cm)
# VP | FP
# FN | VN
# Sensibilidade = VP/(VP+FN) = 17/(17+3) = 0.85
# Especificidade = VN/(VN+FP) = 4/(4+0) = 1.00
# Melhor preditor = [knn=3]
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(x_train_1,y_train_1)
neigh_pred = neigh.predict(x_train_1)
neigh_cm=confusion_matrix(y_train_1,neigh_pred)

acuracia_neigh = accuracy_score(y_pred=neigh_pred ,y_true=y_train_1)
precisao_neigh = precision_score(y_pred=neigh_pred ,y_true=y_train_1)
recall_neigh = recall_score(y_pred=neigh_pred ,y_true=y_train_1)
neigh_cm=confusion_matrix(y_train_1,neigh_pred)
print(acuracia_neigh,precisao_neigh,recall_neigh)
print(neigh_cm)
# VP | FP
# FN | VN
# Os scores estão próximos entre treino e teste. Pode ocorrer overfitting.
# fpr = false positive rate ... tpr = true positive rate
# k = 4
from sklearn import metrics
fpr_knn, tpr_knn, threshold = metrics.roc_curve(y_test_1, neigh_pred)
roc_auc_knn = metrics.auc(fpr_knn,tpr_knn)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_knn, tpr_knn, 'b', label = 'AUC = %0.2f' % roc_auc_knn)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
fpr_knn
tpr_knn
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train_1,y_train_1)
neigh_pred = neigh.predict(x_test_1)

acuracia_neigh = accuracy_score(y_pred=neigh_pred ,y_true=y_test_1)
precisao_neigh = precision_score(y_pred=neigh_pred ,y_true=y_test_1)
recall_neigh = recall_score(y_pred=neigh_pred ,y_true=y_test_1)
neigh_cm=confusion_matrix(y_test_1,neigh_pred)
print(acuracia_neigh,precisao_neigh,recall_neigh)
print(neigh_cm)
# VP | FP
# FN | VN
# fpr = false positive rate ... tpr = true positive rate
# k = 1
from sklearn import metrics
fpr_knn, tpr_knn, threshold = metrics.roc_curve(y_test_1, neigh_pred)
roc_auc_knn = metrics.auc(fpr_knn,tpr_knn)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_knn, tpr_knn, 'b', label = 'AUC = %0.2f' % roc_auc_knn)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Considere uma distribuição Chi-quadrado
# pvalue (5%) = 3,841. Se o valor do teste for maior, rejeitar a Ho: o Disco Aberto faz diferença
# McNemar = (b-c)^2/(b+c) | b=3 c=0 | 9/3 = 3, que é menor que pvalue. 
# Assim, aceitamos que o Disco aberto não faz diferença.
# http://www.leg.ufpr.br/lib/exe/fetch.php/disciplinas:ce001:teste_de_mcnemar_pronto.pdf
from statsmodels.stats.contingency_tables import mcnemar
resultado = mcnemar(neigh_cm, exact=True)
print('statistic=%.3f, p-value=%.3f' % (resultado.statistic, resultado.pvalue))
alpha = 0.05
if resultado.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')

# Sensibilidade = 0.85 - Total dos VP em relação a todos os V (VP + FN)
# Especificidade = 1.0 -- Total do VN em relação a todos os F (VN + FP)
