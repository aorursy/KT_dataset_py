# Importando o que utilizaremos no programa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,fbeta_score
from sklearn.neighbors import KNeighborsClassifier
trainDb = pd.read_csv("../input/myspamdb/train_data.csv")
testDb = pd.read_csv("../input/myspamdb/test_features.csv")
trainDb.head()
trainDb.info()
correl = trainDb.corr()
plt.matshow(correl)
plt.colorbar()
correlList = list(correl["ham"].abs().sort_values()[abs(correl.ham)>0.15].index.drop("ham"))
correlList
filteredTrain = trainDb[correlList]
filteredTest = testDb[correlList]
hamTrain = trainDb.ham
gcScores, bcScores, mcScores = [], [], []
gcMethod = naive_bayes.GaussianNB()
bcMethod = naive_bayes.BernoulliNB()
mcMethod = naive_bayes.MultinomialNB()
cvList = [3, 5, 8, 10, 14, 18, 22, 25, 30]

for i in cvList:
    scores = cross_val_score(gcMethod, filteredTrain, hamTrain, cv=i)
    gcScores.append(scores.mean())
    
    scores = cross_val_score(bcMethod, filteredTrain, hamTrain, cv=i)
    bcScores.append(scores.mean())
    
    scores = cross_val_score(mcMethod, filteredTrain, hamTrain, cv=i)
    mcScores.append(scores.mean())

plt.title("Relação de pontuação entre os métodos")
plt.plot(cvList, gcScores, color = 'blue', label = 'Gaussian')
plt.plot(cvList,bcScores, color = 'red', label = 'Bernoulli')
plt.plot(cvList, mcScores, color = 'yellow', label = 'Multinomial')
plt.xlabel("Cross-validation (cv)")
plt.ylabel("Pontuação obtida")
plt.legend()
gcMethod.fit(filteredTrain, hamTrain)
NaivesANS = gcMethod.predict(filteredTrain)
print('Pontuação de acurácia: ' + str(accuracy_score(hamTrain, NaivesANS)))
print(classification_report(hamTrain, NaivesANS))
mConfusao = confusion_matrix(hamTrain, NaivesANS)
pd.DataFrame(mConfusao, columns = ['Considerado Ham', 'Considerado Spam'], index = ['De fato Ham', 'De fato Spam'])
beta = 3
p = mConfusao[0][0]/(mConfusao[0][0]+mConfusao[1][0])
r = mConfusao[0][0]/(mConfusao[0][0]+mConfusao[0][1])
F3 = 10 * p * r / (9*p + r)
print ('Pontuação F-3: ' + str(F3))
kNNScores = []
gcMethod = naive_bayes.GaussianNB()
bcMethod = naive_bayes.BernoulliNB()
mcMethod = naive_bayes.MultinomialNB()
kList = [3, 5, 8, 10, 14, 18, 22, 25, 30]

for i in kList:
    kNNMethod = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(kNNMethod, filteredTrain, hamTrain, cv=3)
    kNNScores.append(scores.mean())

plt.title("Escolha do número de vizinhos (k)")
plt.plot(kList, kNNScores, color = 'blue', label = 'kNN')
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Pontuação obtida")
kNNMethod.fit(filteredTrain, hamTrain)
kNNANS = kNNMethod.predict(filteredTrain)
print('Pontuação de acurácia: ' + str(accuracy_score(hamTrain, kNNANS)))
print(classification_report(hamTrain, kNNANS))
gcFinal = naive_bayes.GaussianNB()
gcFinal.fit(filteredTrain, hamTrain)
predictionTest = gcFinal.predict(filteredTest)
predictionTest = pd.DataFrame({'Id':testDb.Id,'ham':predictionTest})
predictionTest.to_csv("ans.csv", index = False)
predictionTest