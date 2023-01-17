import pandas as pd
bow = pd.read_csv("../input/bag-of-words-train/train_data.csv")
bow.head()
bow_false = bow[bow['ham'] == False] 
bow_true = bow[bow['ham'] == True] 

bow_mean = pd.DataFrame(bow_false.mean(axis=0),columns=['False'])
bow_mean['True'] = bow_true.mean(axis=0)
bow_mean['Compare'] = bow_mean['False']/bow_mean['True']
bow_mean


diff = 0.5
bow_mean.loc[(bow_mean['Compare'] > (1-diff)) & (bow_mean['Compare'] < (1+diff))]
bow_input = bow.drop(['ham','Id'],axis=1)
bow_output = bow.ham

from sklearn.model_selection import train_test_split
bow_input_train, bow_input_test, bow_output_train, bow_output_test = train_test_split(bow_input, bow_output, train_size=0.75,random_state=1)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(bow_input_train,bow_output_train)

bow_predict_knn = knn.predict(bow_input_test)

from sklearn.metrics import fbeta_score
fbeta_score(bow_output_test,bow_predict_knn,beta=3)
K = []
F3 = []
for k in range (1,100):
    K.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(bow_input_train,bow_output_train)
    bow_predict_knn = knn.predict(bow_input_test)
    F3.append(fbeta_score(bow_output_test,bow_predict_knn,beta=3))
    
import matplotlib.pyplot as plt
plt.plot(K,F3)
    
    
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(bow_input_train,bow_output_train)

bow_predict_NB = NB.predict(bow_input_test)
fbeta_score(bow_output_test,bow_predict_NB,beta=3)


new_bow_input_train = bow_input_train.drop(["word_freq_address","word_freq_will","char_freq_("],axis = 1)
new_bow_input_test = bow_input_test.drop(["word_freq_address","word_freq_will","char_freq_("],axis = 1)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(new_bow_input_train,bow_output_train)

new_bow_predict_knn = knn.predict(new_bow_input_test)

from sklearn.metrics import fbeta_score
fbeta_score(bow_output_test,new_bow_predict_knn,beta=3)

NB = MultinomialNB()
NB.fit(new_bow_input_train,bow_output_train)

new_bow_predict_NB = NB.predict(new_bow_input_test)
fbeta_score(bow_output_test,new_bow_predict_NB,beta=3)