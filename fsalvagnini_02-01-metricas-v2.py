from sklearn.datasets import load_breast_cancer
import numpy as np
bc = load_breast_cancer()

data = bc.data
target = bc.target
print("Formato dos dados (linhas x colunas) => ", data.shape)
print("Labels do problema => ", np.unique(target))
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
probas = knn.predict_proba(X_test)
print(probas)
probas[:,1]
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, probas[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (10, 5))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Aleatório')
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, 'ro', label='Thresholds')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Falsos Positivos', fontsize = 15)
plt.ylabel('Positivos Verdadeiros', fontsize = 15)
plt.title('ROC do KNN', fontsize = 20)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()

for tp, fp, t in zip(tpr, fpr, thresholds):
    print('tp = %.2f, fp = %.2f, t=%.2f' % (tp, fp, t))
from statsmodels.stats.contingency_tables import mcnemar
def build_contingence_table(Y, Y_pred_1, Y_pred_2):
    y1_and_y2 = 0
    y1_and_not_y2 = 0
    y2_and_not_y1 = 0
    not_y1_and_not_y2 = 0
    for y, y1, y2 in zip(Y, Y_pred_1, Y_pred_2):
        if y == y1 == y2:
            y1_and_y2 += 1
        elif y != y1 and y != y2:
            not_y1_and_not_y2 += 1
        elif y == y1 and y != y2:
            y1_and_not_y2 += 1
        elif y != y1 and y == y2:
            y2_and_not_y1 += 1
            
    contingency_table = [[y1_and_y2, y1_and_not_y2], 
                         [y2_and_not_y1, not_y1_and_not_y2]]
    
    return contingency_table
knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(X_train, y_train)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)

y_pred2 = knn2.predict(X_test)
y_pred3 = knn3.predict(X_test)
contingence_table = build_contingence_table(y_test, y_pred2, y_pred3)

import pprint

pprint.pprint(contingence_table)
result = mcnemar(contingence_table)
    
    
if result.pvalue >= 0.001:
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
else:
    print('statistic=%.3f, p-value=%.3e' % (result.statistic, result.pvalue))

# interpretando o p-value
alpha = 0.05
if result.pvalue > alpha:
    print('Mesma proporção de erros (falhou em rejeitar H0)')
else:
    print('Proporções de erros diferentes (rejeitou H0)')

