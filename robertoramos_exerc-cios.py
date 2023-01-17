#Crie classificadores para o MNIST dataset
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data, target = load_digits(return_X_y=True)

label_set = list(set(target))
#print('Num examples', len(data))
#print('Shape', np.shape(data[0]))
#print('Label set', label_set)

X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(data, target, 
                                                                    test_size=0.30, 
                                                                    stratify=target,
                                                                    shuffle=True)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, 
                                                                test_size=0.2, 
                                                                stratify=Y_train_initial,
                                                                shuffle=True)

#print('SHAPE train', np.shape(X_train))
#print('SHAPE validation', np.shape(X_validation))
#print('SHAPE test', np.shape(X_test))

#Criação modelos
model_1 = GaussianNB()
model_1.fit(X_train, Y_train)

model_2 = BernoulliNB()
model_2.fit(X_train, Y_train)

model_3 = LogisticRegression()
model_3.fit(X_train, Y_train)

#Verificação accuracy
accuracy_validation_1 = model_1.score(X_validation, Y_validation)
accuracy_validation_2 = model_2.score(X_validation, Y_validation)
accuracy_validation_3 = model_3.score(X_validation, Y_validation)

print('Validation accuracy GaussianNB', accuracy_validation_1)
print('Validation accuracy BernoulliNB', accuracy_validation_2)
print('Validation accuracy LogisticRegression', accuracy_validation_3)

accuracy_test1 = model_1.score(X_test, Y_test)
accuracy_test2 = model_2.score(X_test, Y_test)
accuracy_test3 = model_3.score(X_test, Y_test)

print('Test accuracy GaussianNB', accuracy_test1)
print('Test accuracy BernoulliNB', accuracy_test2)
print('Test accuracy LogisticRegression', accuracy_test3)

Y_pred_1 = model_1.predict(X_test)
Y_pred_2 = model_2.predict(X_test)
Y_pred_3 = model_3.predict(X_test)

precision_1 = precision_score(Y_test, Y_pred_1, average=None)
precision_1_average = precision_score(Y_test, Y_pred_1, average='weighted')
#print('Precision GaussianNB per class', precision_1)
print('Precision GaussianNB average', precision_1_average)

precision_2 = precision_score(Y_test, Y_pred_2, average=None)
precision_2_average = precision_score(Y_test, Y_pred_2, average='weighted')
#print('Precision BernoulliNB per class', precision_2)
print('Precision BernoulliNB average', precision_2_average)

precision_3 = precision_score(Y_test, Y_pred_3, average=None)
precision_3_average = precision_score(Y_test, Y_pred_3, average='weighted')
#print('Precision BernoulliNB per class', precision_2)
print('Precision LogisticRegression average', precision_3_average)


recall_1 = recall_score(Y_test, Y_pred_1, average=None)
recall_1_average = recall_score(Y_test, Y_pred_1, average='weighted')
#print('Recall GaussianNB per class', recall_1)
print('Recall GaussianNB average', recall_1_average)

recall_2 = recall_score(Y_test, Y_pred_2, average=None)
recall_2_average = recall_score(Y_test, Y_pred_2, average='weighted')
#print('Recall BernoulliNB per class', recall_2)
print('Recall BernoulliNB average', recall_2_average)

recall_3 = recall_score(Y_test, Y_pred_3, average=None)
recall_3_average = recall_score(Y_test, Y_pred_3, average='weighted')
#print('Recall BernoulliNB per class', recall_2)
print('Recall LogisticRegression average', recall_3_average)


#cm_1 = confusion_matrix(Y_test, Y_pred_1)
#print('Confusion matrix GaussianNB\n', cm_1)

#cm_2 = confusion_matrix(Y_test, Y_pred_2)
#print('Confusion matrix BernoulliNB\n', cm_2)

#cm_3 = confusion_matrix(Y_test, Y_pred_3)
#print('Confusion matrix BernoulliNB\n', cm_3)

#Desenvolva seus estudos a partir daqui
#Crie um classificador para diferenciar pessoas fisicas ou juridicas
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

corpus = pickle.load(open('../input/name_company_corpus.pickle', 'rb'))

names = [x[0] for x in corpus if x[1] == 'NAME']
companies = [x[0] for x in corpus if x[1] == 'COMPANY']

all_texts = names + companies
labels = [0]*(len(names)) + [1]*(len(companies))
#print('NUM PEOPLE', len(names))
#print('NUM COMPANIES', len(companies))

vectorizer_1 = CountVectorizer()
vectorizer_2 = TfidfVectorizer()

X_1 = vectorizer_1.fit_transform(all_texts)
X_2 = vectorizer_2.fit_transform(all_texts)

X_train_1_initial, X_test_1, Y_train_1_initial, Y_test_1 = train_test_split(X_1, labels, 
                                                    test_size=0.30, 
                                                    stratify=labels,
                                                    shuffle=True,
                                                    random_state=10)

X_train_1, X_validation_1, Y_train_1, Y_validation_1 = train_test_split(X_train_1_initial, Y_train_1_initial, 
                                                    test_size=0.20, 
                                                    stratify=Y_train_1_initial,
                                                    shuffle=True,
                                                    random_state=10)


X_train_2_initial, X_test_2, Y_train_2_initial, Y_test_2 = train_test_split(X_2, labels, 
                                                    test_size=0.30, 
                                                    stratify=labels,
                                                    shuffle=True,
                                                    random_state=10)

X_train_2, X_validation_2, Y_train_2, Y_validation_2 = train_test_split(X_train_2_initial, Y_train_2_initial, 
                                                    test_size=0.30, 
                                                    stratify=Y_train_2_initial,
                                                    shuffle=True,
                                                    random_state=10)

#model1_1 = GaussianNB()
#model1_2 = GaussianNB()
model2_1 = LogisticRegression()
"""model2_2 = LogisticRegression()
model3_1 = MLPClassifier((10,), activation='logistic')
model3_2 = MLPClassifier((10,), activation='logistic')"""

#model1_1.fit(X_train_1.todense(), Y_train_1)
#model1_2.fit(X_train_2.todense(), Y_train_2)
model2_1.fit(X_train_1, Y_train_1)
"""
model2_2.fit(X_train_2, Y_train_2)
model3_1.fit(X_train_1, Y_train_1)
model3_2.fit(X_train_2, Y_train_2)
"""

#accuracy1_1 = model1_1.score(X_validation_1.todense(), Y_validation_1)
#accuracy1_2 = model1_2.score(X_validation_2.todense(), Y_validation_2)
#accuracy1_3 = model1_1.score(X_test_1.todense(), Y_test_1)
#accuracy1_4 = model1_2.score(X_test_2.todense(), Y_test_2)

accuracy2_1 = model2_1.score(X_validation_1, Y_validation_1)
"""
accuracy2_2 = model2_2.score(X_validation_2, Y_validation_2)
accuracy2_3 = model2_1.score(X_test_1, Y_test_1)
accuracy2_4 = model2_2.score(X_test_2, Y_test_2)

accuracy3_1 = model3_1.score(X_validation_1, Y_validation_1)
accuracy3_2 = model3_2.score(X_validation_2, Y_validation_2)
accuracy3_3 = model3_1.score(X_test_1, Y_test_1)
accuracy3_4 = model3_2.score(X_test_2, Y_test_2)"""

#print('Validation Accuracy GaussianNB + CountVectorizer', accuracy1_1)
#print('Validation Accuracy GaussianNB + Tfidf', accuracy1_2)
#print('Test Accuracy GaussianNB + CountVectorizer', accuracy1_3)
#print('Test Accuracy GaussianNB + Tfidf', accuracy1_4)
print('Validation Accuracy LogisticRegression + CountVectorizer', accuracy2_1)
"""
print('Validation Accuracy LogisticRegression + Tfidf', accuracy2_2)
print('Test Accuracy LogisticRegression + CountVectorizer', accuracy2_3)
print('Test Accuracy LogisticRegression + Tfidf', accuracy2_4)
print('Validation Accuracy MLPClassifier + CountVectorizer', accuracy3_1)
print('Validation Accuracy MLPClassifier + Tfidf', accuracy3_2)
print('Test Accuracy MLPClassifier + CountVectorizer', accuracy3_3)
print('Test Accuracy MLPClassifier + Tfidf', accuracy3_4)"""
#Desenvolva seus estudos a partir daqui


