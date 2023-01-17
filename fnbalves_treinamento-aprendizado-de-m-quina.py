#Operacoes basicas de python

print(2 + 2)

print(7/2)



a = 123.4

b = 5.0

c = a*b

print(c)



import math as m

print(m.sqrt(b))

print(m.pow(b, 2))

print(b**2)

#Lacos, controle de fluxo e manipulacao de listas

a = 35

if a > 12:

    print('Maior que 12')

elif a > 5:

    print('Menor que 12 e maior que 5')

else:

    print('Caso contrario')



for i in range(3):

    print('Laco de repeticao')



control_var = 0

while control_var < 3:

    control_var += 1

    print('Laco de repeticao while')



my_list = [1,2,3,4,5,6]

for L in my_list:

    print('elemento da lista', L)



squared_list = [L*L for L in my_list]

str_list = ['STRING 1', 'STRING 2', 'STRING 3']

lower_list = [S.lower() for S in str_list]

print('SQUARED LIST', squared_list)

print('LOWER LIST', lower_list)



multidimensional_list = [[1,2,3], [4,5,6]]

print(multidimensional_list)

print(multidimensional_list[1][2])
#Conjuntos e dicionarios

my_list = [1,2,1,3, 3, 3, 4, 5, 5, 5, 5]

my_set = set(my_list)

print('List set', my_set)

my_set = list(my_set)

print('Posicao do elemento "3"', my_set.index(3))



my_dict = {}

my_dict['valor 1'] = 1

my_dict['valor 2'] = 35



print('valor 1' in my_dict)

print('valor 3' in my_dict)



print('Valor para a chave "valor_1"', my_dict['valor 1'])
#Manipulando matrizes

import numpy as np



mat_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

mat_2 = np.array([[3, 7, 3], [5, 12, 35], [3, 9, 2]], dtype=np.float32)



mat_mul = np.dot(mat_1, mat_2)

mat_1_t = np.transpose(mat_mul)

inversed = np.linalg.inv(mat_2)



print('Mat 1\n', mat_1)

print('Mat 2\n', mat_2)

print('Mat mult\n', mat_mul)

print('Mat 1 transpose\n', mat_1_t)

print('Mat 2 inverse\n', inversed)
#Criando conjuntos de treinamento, teste e validacao

from sklearn import datasets

from sklearn.model_selection import train_test_split

import numpy as np



iris = datasets.load_iris()

X = iris.data

Y = iris.target



print('SHAPE X', np.shape(X))

print('SHAPE Y', np.shape(Y))



tag_set = set(Y)

print('TAG SET', tag_set)



X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(X, Y, 

                                                                    test_size=0.30, 

                                                                    stratify=Y,

                                                                    shuffle=True)



print('SHAPE X_train_test', np.shape(X_train_initial))

print('SHAPE X_test', np.shape(X_test))

print('SET Y_train', set(Y_train_initial))

print('SET Y_test', set(Y_test))



X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, 

                                                                test_size=0.2, 

                                                                stratify=Y_train_initial,

                                                                shuffle=True)



print('SHAPE train', np.shape(X_train))

print('SHAPE validation', np.shape(X_validation))

print('SHAPE test', np.shape(X_test))



#Testando classificadores no conjunto de validação e no conjunto de teste

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB



model_1 = GaussianNB()

model_1.fit(X_train, Y_train)



model_2 = BernoulliNB()

model_2.fit(X_train, Y_train)



accuracy_validation_1 = model_1.score(X_validation, Y_validation)

accuracy_validation_2 = model_2.score(X_validation, Y_validation)



print('Validation accuracy GaussianNB', accuracy_validation_1)

print('Validation accuracy BernoulliNB', accuracy_validation_2)



#For testing

#accuracy_validation_2 = 1.0



if accuracy_validation_1 > accuracy_validation_2:

    accuracy_test = model_1.score(X_test, Y_test)

    print('Test accuracy GaussianNB', accuracy_test)

else:

    accuracy_test = model_2.score(X_test, Y_test)

    print('Test accuracy BernoulliNB', accuracy_test)

#Calculando metricas

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix



Y_pred_1 = model_1.predict(X_test)

Y_pred_2 = model_2.predict(X_test)



precision_1 = precision_score(Y_test, Y_pred_1, average=None)

precision_1_average = precision_score(Y_test, Y_pred_1, average='weighted')

print('Precision GaussianNB per class', precision_1)

print('Precision GaussianNB average', precision_1_average)



precision_2 = precision_score(Y_test, Y_pred_2, average=None)

precision_2_average = precision_score(Y_test, Y_pred_2, average='weighted')

print('Precision BernoulliNB per class', precision_2)

print('Precision BernoulliNB average', precision_2_average)



recall_1 = recall_score(Y_test, Y_pred_1, average=None)

recall_1_average = recall_score(Y_test, Y_pred_1, average='weighted')

print('Recall GaussianNB per class', recall_1)

print('Recall GaussianNB average', recall_1_average)



recall_2 = recall_score(Y_test, Y_pred_2, average=None)

recall_2_average = recall_score(Y_test, Y_pred_2, average='weighted')

print('Recall BernoulliNB per class', recall_2)

print('Recall BernoulliNB average', recall_2_average)



cm_1 = confusion_matrix(Y_test, Y_pred_1)

print('Confusion matrix GaussianNB\n', cm_1)



cm_2 = confusion_matrix(Y_test, Y_pred_2)

print('Confusion matrix BernoulliNB\n', cm_2)
#Carregando o digits dataset

from sklearn import datasets

from sklearn.model_selection import train_test_split

import numpy as np



iris = datasets.load_digits()

X = iris.data

Y = iris.target



print('SHAPE X', np.shape(X))

print('SHAPE Y', np.shape(Y))



tag_set = set(Y)

print('TAG SET', tag_set)



X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(X, Y, 

                                                                    test_size=0.30, 

                                                                    stratify=Y,

                                                                    shuffle=True)



print('SHAPE X_train_test', np.shape(X_train_initial))

print('SHAPE X_test', np.shape(X_test))

print('SET Y_train', set(Y_train_initial))

print('SET Y_test', set(Y_test))



X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, 

                                                                test_size=0.2, 

                                                                stratify=Y_train_initial,

                                                                shuffle=True)



print('SHAPE train', np.shape(X_train))

print('SHAPE validation', np.shape(X_validation))

print('SHAPE test', np.shape(X_test))



#Comparando um classificador bayesiano com um classificador de regressao logistica

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression



model_1 = GaussianNB()

model_1.fit(X_train, Y_train)



model_2 = LogisticRegression()

model_2.fit(X_train, Y_train)



accuracy_validation_1 = model_1.score(X_validation, Y_validation)

accuracy_validation_2 = model_2.score(X_validation, Y_validation)



print('Validation accuracy GaussianNB', accuracy_validation_1)

print('Validation accuracy LogisticRegression', accuracy_validation_2)



print('Logistic regression coefficients', model_2.coef_)
#Comparando um classificador bayesiano, um classificador de regressao logistica e

#um classificador mlp

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



model_1 = GaussianNB()

model_1.fit(X_train, Y_train)



model_2 = LogisticRegression()

model_2.fit(X_train, Y_train)



model_3 = MLPClassifier((10,), activation='tanh')

model_3.fit(X_train, Y_train)



accuracy_validation_1 = model_1.score(X_validation, Y_validation)

accuracy_validation_2 = model_2.score(X_validation, Y_validation)

accuracy_validation_3 = model_3.score(X_validation, Y_validation)



print('Validation accuracy GaussianNB', accuracy_validation_1)

print('Validation accuracy LogisticRegression', accuracy_validation_2)

print('Validation accuracy MLPClassifier', accuracy_validation_3)
#Comparando um classificador bayesiano, um classificador de regressao logistica e

#um classificador svc

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



model_1 = GaussianNB()

model_1.fit(X_train, Y_train)



model_2 = LogisticRegression()

model_2.fit(X_train, Y_train)



model_3 = SVC(kernel='linear')

model_3.fit(X_train, Y_train)



accuracy_validation_1 = model_1.score(X_validation, Y_validation)

accuracy_validation_2 = model_2.score(X_validation, Y_validation)

accuracy_validation_3 = model_3.score(X_validation, Y_validation)



print('Validation accuracy GaussianNB', accuracy_validation_1)

print('Validation accuracy LogisticRegression', accuracy_validation_2)

print('Validation accuracy SVC', accuracy_validation_3)
#Treinamento em batches

import math

from sklearn.linear_model import SGDClassifier



def create_batches(X_set, Y_set, size_batch):

    len_X = np.shape(X_set)[0]

    num_batches = int(math.floor(len_X/size_batch))

    start = 0

    for i in range(num_batches):

        x_batch = X_set[start:start + size_batch]

        y_batch = Y_set[start:start + size_batch]

        start += size_batch

        yield x_batch, y_batch



model = SGDClassifier(loss='hinge')

num_epochs = 3

my_classes = (0,1,2,3,4,5,6,7,8,9)

for epoch in range(num_epochs):

    print('Epoch', epoch, 'of', num_epochs)

    batch_generator = create_batches(X_train, Y_train, 30)

    for batch_x, batch_y in batch_generator:

        model.partial_fit(batch_x, batch_y, classes=my_classes)



accuracy_validation = model.score(X_validation, Y_validation)

print('Accuracy', accuracy_validation)
#Classificacao de texto

import json



raw_file = open('../input/reviews.json', 'r').read()

as_json = json.loads(raw_file)

num_texts = len(as_json['paper'])

entries = [j for j in as_json['paper']]

entries[0]
import numpy as np



texts = [' '.join([x['text'] for x in j['review']]) for j in entries]

classifications = [j['preliminary_decision'] for j in entries]



class_set = list(set(classifications))

print('SET classifications', class_set)

numeric_classifications = [class_set.index(c) for c in classifications]

Y = np.array(numeric_classifications).ravel()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB



vectorizer_1 = CountVectorizer()

vectorizer_2 = TfidfVectorizer()



X_1 = vectorizer_1.fit_transform(texts)

X_2 = vectorizer_2.fit_transform(texts)



X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_1, Y, 

                                                    test_size=0.30, 

                                                    stratify=Y,

                                                    shuffle=True,

                                                    random_state=42)



X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X_2, Y, 

                                                    test_size=0.30, 

                                                    stratify=Y,

                                                    shuffle=True,

                                                    random_state=42)



model1_1 = GaussianNB()

model1_2 = GaussianNB()

model2_1 = LogisticRegression()

model2_2 = LogisticRegression()

model3_1 = MLPClassifier((10,), activation='logistic')

model3_2 = MLPClassifier((10,), activation='logistic')



model1_1.fit(X_train_1.todense(), Y_train_1)

model1_2.fit(X_train_2.todense(), Y_train_2)

model2_1.fit(X_train_1, Y_train_1)

model2_2.fit(X_train_2, Y_train_2)

model3_1.fit(X_train_1, Y_train_1)

model3_2.fit(X_train_2, Y_train_2)



accuracy1_1 = model1_1.score(X_test_1.todense(), Y_test_1)

accuracy1_2 = model1_2.score(X_test_2.todense(), Y_test_2)

accuracy2_1 = model2_1.score(X_test_1, Y_test_1)

accuracy2_2 = model2_2.score(X_test_2, Y_test_2)

accuracy3_1 = model3_1.score(X_test_1, Y_test_1)

accuracy3_2 = model3_2.score(X_test_2, Y_test_2)



print('Accuracy GaussianNB + CountVectorizer', accuracy1_1)

print('Accuracy GaussianNB + Tfidf', accuracy1_2)

print('Accuracy LogisticRegression + CountVectorizer', accuracy2_1)

print('Accuracy LogisticRegression + Tfidf', accuracy2_2)

print('Accuracy MLPClassifier + CountVectorizer', accuracy3_1)

print('Accuracy MLPClassifier + Tfidf', accuracy3_2)
