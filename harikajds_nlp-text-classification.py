#Importing Libraries

import numpy as np

import pandas as pd



#For visualization

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow



#Importing sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer





#reading data

df=pd.read_csv("../input/spam.csv", encoding='latin-1')
#making ham=0,spam=1 used as target matrix in label_num column

df['v1'] = df.v1.map({'ham':0,'spam':1})

features = df["v2"]

labels=df["v1"]





#plotting data

sns.countplot(df["v1"])

print(plt.show())
#spliting training and testing data

f_train,f_test,l_train,l_test= train_test_split(features,labels,test_size=.1)

f_train=np.array(f_train)

f_test=np.array(f_test)

l_train=np.array(l_train)

l_test=np.array(l_test)





#Inorder to train model we need to convert text to appropriate numerical values using vetorization

vect = CountVectorizer()

f_train_count = vect.fit_transform(f_train)

f_test_count= vect.transform(f_test)

trans = TfidfTransformer()
#DecisionTree Algorithm

from sklearn.tree import DecisionTreeClassifier



classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state =0)

classifier1.fit(f_train_count,l_train)

print("\n\n\033[1;32;40m [+]train score: ",classifier1.score(f_train_count,l_train))

print("\n\n[+]test score:  ",classifier1.score(f_test_count,l_test))

y_pred1 = classifier1.predict(f_train_count)

scDT = classifier1.score(f_test_count,l_test)

print("\n\n[+]accuracy score of DecisionTree Algorithm= " ,scDT)
#KNeighbors Algorithm

from sklearn.neighbors import KNeighborsClassifier



classifier2 = KNeighborsClassifier(n_neighbors =2, metric ='minkowski', p=2)

classifier2.fit(f_train_count,l_train)

print("\n\n\033[1;32;40m [+]train score: ",classifier2.score(f_train_count,l_train))

print("\n\n[+]test score:  ",classifier2.score(f_test_count,l_test))

y_pred2 = classifier2.predict(f_train_count)

scKNN = classifier2.score(f_test_count,l_test)

print("\n\n[+]accuracy score of KNeighbors Algorithm = " ,scKNN)
#RandomForest Algorithm

from sklearn.ensemble import RandomForestClassifier



classifier3 = RandomForestClassifier(n_estimators = 55,criterion = 'entropy', random_state =0)

classifier3.fit(f_train_count,l_train)

print("\n\n\033[1;32;40m [+]train score: ",classifier3.score(f_train_count,l_train))

print("\n\n[+]test score:  ",classifier3.score(f_test_count,l_test))

y_pred3 = classifier3.predict(f_train_count)

scRF = classifier3.score(f_test_count,l_test)

print("\n\n[+]accuracy score of RandomForest Algorithm = " ,scRF)
#Naive_Bayes Algorithm

from sklearn.naive_bayes import MultinomialNB



classifier4 = MultinomialNB()

classifier4.fit(f_train_count,l_train)

print("\n\n\033[1;32;40m [+]train score: ",classifier4.score(f_train_count,l_train))

print("\n\n[+]test score:  ",classifier4.score(f_test_count,l_test))

y_pred4 = classifier4.predict(f_test_count)

scNB = accuracy_score(l_test,y_pred4)

#precision_score = precision_score(l_test,y_pred4)

#recall_score = recall_score(l_test,y_pred4)

#f1_score = f1_score(l_test,y_pred4)

print("\n\n[+]accuracy score of Naive_Bayes Algorithm = " ,scNB)

report = classification_report(y_pred4,l_test)

print("\n",report)
#Plotting seaborn comapritive Accuracies Graph 

scores = [scDT,scKNN,scRF,scNB]

algorithms = ["Decision Tree","KNeighbors","RandomForest","Naive Bayes"] 

sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy Score")



sns.barplot(algorithms,scores)
#Testing HAM and SPAM Predection SMS

print("\n\n[+] lets test whether SMS is HAM or SPAM: ") 

t=['hello']

t=np.array(t)

t=vect.transform(t)

prediction = classifier4.predict(t)

if prediction == 0:

    print("HAM!")

else:

    print("\033[1;31;40m [+]SPAM!")