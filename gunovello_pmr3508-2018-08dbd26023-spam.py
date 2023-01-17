import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
treino =  pd.read_csv("../input/basespam/train_data.csv")
treino.shape
treino.head()
Xtreino = treino.drop('ham', axis=1)
Ytreino = treino.ham
Cross = []
for i in range (1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    Cross.append(cross_val_score(knn,Xtreino,Ytreino,cv=10).mean())
for i in range (0,25):
    print(2*i+1,":  ",Cross[2*i],"    ", 2*i+2,":  ",Cross[2*i+1])
NaiveBayes = MultinomialNB()
NaiveBayes.fit(Xtreino,Ytreino)
teste =  pd.read_csv("../input/basespam/test_features.csv")
Resultado = NaiveBayes.predict(teste)
Pred=pd.DataFrame(columns = ['Id','ham'])
Pred['ham'] = Resultado
Pred['Id'] = teste.Id
Pred.to_csv("Spam.csv",index=False)
#Comparando o resultado da validação cruzada (método KNN), com o resultado obtido na competição (Naive Bayes),
#observamos que o classficador Naive Bayes apresentou acurácia 10% maior que o classificador KNN na cross validation.
#O classificador plug-in Naive Bayes demonstrou ser muito eficiente nessa aplicação, pois obteve um resultado satisfatório
#para uma base de dados que não recebeu nenhum tratamento especial.