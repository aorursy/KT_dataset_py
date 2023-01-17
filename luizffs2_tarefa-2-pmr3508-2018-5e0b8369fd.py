import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.metrics import accuracy_score as acs
from sklearn.model_selection import cross_val_score as cvs
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.metrics import fbeta_score as fb, make_scorer as ms
base = pd.read_csv("../input/spambase/train_data.csv")
base.shape
base.head()
base['ham'].value_counts().plot(kind='bar')
base['ham'].mean()
dados = base.drop(['ham','Id'],axis=1)
dados.head()
Id = base['Id']
Id.head()
ham = base['ham']
ham.head()
Mult = MNB()
Mult.fit(dados,ham)
Mult_pred = Mult.predict(dados)
print(acs(ham,Mult_pred))
print(cm(ham,Mult_pred))
print(cr(ham,Mult_pred))
Bern = BNB()
Bern.fit(dados,ham)
Bern_pred = Bern.predict(dados)
print(acs(ham,Bern_pred))
print(cm(ham,Bern_pred))
print(cr(ham,Bern_pred))
def melhor_knn(base,p,u):
    '''Recebe a base de testes, o primeiro e o último número de nearest neighbors a ser 
       verificado e retorna a melhor acurácia obtida bem como o número de nearest neighbors
       utilizado
    '''
    score = 0
    for i in range(p,u):
        knn = KNN(n_neighbors=i)
        new_scores = cvs(knn,base,ham,cv=10)
        if new_scores.mean() > score:
            score = new_scores.mean()
            nearn = i
            
    return score, nearn
melhor_knn(dados,10,50)
knn = KNN(n_neighbors=11)
knn.fit(dados,ham)
knn_pred = knn.predict(dados)
print(acs(ham,knn_pred))
print(cm(ham,knn_pred))
print(cr(ham,knn_pred))
base.corr(min_periods=30)['ham']
dados2 = dados.drop(['word_freq_address','word_freq_3d','word_freq_will','word_freq_report',
                     'word_freq_font','word_freq_parts','word_freq_direct','word_freq_cs',
                     'word_freq_project','word_freq_table','word_freq_conference','char_freq_;',
                     'char_freq_(','char_freq_[','char_freq_#'],axis=1)
Bern2 = Bern.fit(dados2,ham)
Bern2_pred = Bern2.predict(dados2)
print(acs(ham,Bern2_pred))
print(cm(ham,Bern2_pred))
print(cr(ham,Bern2_pred))
base2 = pd.read_csv("../input/spambase/test_features.csv")
Id = base2['Id']
Predict = Bern2.predict(base2.drop(['Id','word_freq_address','word_freq_3d','word_freq_will','word_freq_report',
                     'word_freq_font','word_freq_parts','word_freq_direct','word_freq_cs',
                     'word_freq_project','word_freq_table','word_freq_conference','char_freq_;',
                     'char_freq_(','char_freq_[','char_freq_#'],axis=1))
Submit = pd.DataFrame()
Submit.insert(0,'Id',Id)
Submit.insert(1,'ham',Predict)
file = open('sample_submission.csv','w')
file.write(pd.DataFrame.to_csv(Submit, index=False))
file.close()