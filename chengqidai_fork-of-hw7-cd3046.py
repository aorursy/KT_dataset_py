#Data setup
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt 

xin,yin = fetch_openml(name='tic-tac-toe', return_X_y=True)

# labelencoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(yin)
y = le.transform(yin)

#data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xin, y)

#naive bayes
from sklearn.naive_bayes import MultinomialNB
#Train naive bayes using default parameter
nb_a = MultinomialNB()
nb_a.fit(X_train, y_train)
#calculate accuracy on test data and train data
pred_test = nb_a.predict(X_test)
score_test = nb_a.score(X_test,y_test)
pred_train = nb_a.predict(X_train)
score_train = nb_a.score(X_train,y_train)
#print result (test first,train second)
print(format(score_test,'.4f'),format(score_train,'.4f'))
del score_test,score_train,pred_test,pred_train

#Set fit_prior to False and train naive bayes
nb_b = MultinomialNB(fit_prior=False)
nb_b.fit(X_train, y_train)
#calculate accuracy on test data and train data
pred_test = nb_b.predict(X_test)
score_test = nb_b.score(X_test,y_test)
pred_train = nb_b.predict(X_train)
score_train = nb_b.score(X_train,y_train)
#print result (test first,train second)
print(format(score_test,'.4f'),format(score_train,'.4f'))
del score_test,score_train,pred_test,pred_train
#initialize the list
test_list = []
train_list = []
index_list = np.arange(1.0e-10,5.1,0.25).tolist()

#change the value of alpha
for a in index_list:
    #train the Naive Bayes with different alpha
    nb_c = MultinomialNB(alpha=a)
    nb_c.fit(X_train, y_train)
    #calculate accuracy on test data and train data
    pred_test = nb_c.predict(X_test)
    score_test = nb_c.score(X_test,y_test)
    pred_train = nb_c.predict(X_train)
    score_train = nb_c.score(X_train,y_train)
    #print result (test first,train second)
    print(format(a,'.2f'),format(score_test,'.4f'),format(score_train,'.4f'))
    test_list.append(score_test)
    train_list.append(score_train)
    del score_test,score_train,pred_test,pred_train
#produce plot graph
plt.plot(index_list,test_list,color='red',label= "test")
plt.plot(index_list,train_list,color='blue',label="train")
plt.xlabel('alpha') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs alpha')
plt.legend() 
plt.show()
del index_list,test_list,train_list
#5.2(d) set train_size to 0.01
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(xin, y,train_size = 0.01)

#5.2(d) Train naive bayes using default parameter
nb_d1 = MultinomialNB()
nb_d1.fit(X_new_train, y_new_train)
#calculate accuracy on test data and train data
pred_test = nb_d1.predict(X_new_test)
score_test = nb_d1.score(X_new_test,y_new_test)
pred_train = nb_d1.predict(X_new_train)
score_train = nb_d1.score(X_new_train,y_new_train)
#print result (test first,train second)
print(format(score_test,'.4f'),format(score_train,'.4f'))
del score_test,score_train,pred_test,pred_train
print("\n")

#5.2(d) Set fit_prior as False
nb_d2 = MultinomialNB(fit_prior=False)
nb_d2.fit(X_new_train, y_new_train)
#calculate accuracy on test data and train data
pred_test = nb_d2.predict(X_new_test)
score_test = nb_d2.score(X_new_test,y_new_test)
pred_train = nb_d2.predict(X_new_train)
score_train = nb_d2.score(X_new_train,y_new_train)
#print result (test first,train second)
print(format(score_test,'.4f'),format(score_train,'.4f'))
del score_test,score_train,pred_test,pred_train
print("\n")

#5.2(d) Change alpha value
#initialize the list
test_list = []
train_list = []
index_list = np.arange(1.0e-10,5.1,0.25).tolist()
#change the value of alpha
for a in index_list:
    #train the Naive Bayes with different alpha
    nb_d3 = MultinomialNB(alpha=a)
    nb_d3.fit(X_new_train, y_new_train)
    #calculate accuracy on test data and train data
    pred_test = nb_d3.predict(X_new_test)
    score_test = nb_d3.score(X_new_test,y_new_test)
    pred_train = nb_d3.predict(X_new_train)
    score_train = nb_d3.score(X_new_train,y_new_train)
    #print result (test first,train second)
    print(format(a,'.2f'),format(score_test,'.4f'),format(score_train,'.4f'))
    test_list.append(score_test)
    train_list.append(score_train)
    del score_test,score_train,pred_test,pred_train
#produce plot graph
plt.plot(index_list,test_list,color='red',label= "test")
plt.plot(index_list,train_list,color='blue',label="train")
plt.xlabel('alpha') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs alpha under train_size 0.01')
plt.legend() 
plt.show()
del index_list,test_list,train_list


#Decision Tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)

#report depth and number of leaves
depth = dt.get_depth()
print(depth,dt.get_n_leaves())

#calculate accuracy on test data and train data
pred_test = dt.predict(X_test)
score_test = dt.score(X_test,y_test)
pred_train = dt.predict(X_train)
score_train = dt.score(X_train,y_train)
#print result (test first,train second)
print(format(score_test,'.4f'),format(score_train,'.4f'))
del score_test,score_train,pred_test,pred_train
print(dt.feature_importances_)

#initialize the list
test_list = []
train_list = []

#change the value of max_depth
for d in range(1,depth+1):
    #train the decision tree with max_depth
    dt_c = tree.DecisionTreeClassifier(max_depth=d)
    dt_c.fit(X_train, y_train)
    #calculate accuracy on test data and train data
    pred_test = dt_c.predict(X_test)
    score_test = dt_c.score(X_test,y_test)
    pred_train = dt_c.predict(X_train)
    score_train = dt_c.score(X_train,y_train)
    #print result (test first,train second)
    print(d,format(score_test,'.4f'),format(score_train,'.4f'))
    test_list.append(score_test)
    train_list.append(score_train)
    del score_test,score_train,pred_test,pred_train
#produce plot graph
plt.plot(range(1,depth+1),test_list,color='red',label= "test")
plt.plot(range(1,depth+1),train_list,color='blue',label="train")
plt.xlabel('Depth') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs depth')
plt.legend() 
plt.show()
del test_list,train_list
#perceptron
from sklearn.linear_model import Perceptron
pc_a = Perceptron()
pc_a.fit(X_train, y_train)
#calculate accuracy on test data and train data
pred_test = pc_a.predict(X_test)
score_test = pc_a.score(X_test,y_test)
pred_train = pc_a.predict(X_train)
score_train = pc_a.score(X_train,y_train)
#print result (test first,train second)
print(format(score_test,'.4f'),format(score_train,'.4f'))
del score_test,score_train,pred_test,pred_train
from sklearn.preprocessing import PolynomialFeatures
test_list = []
train_list = []
for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    xx_train = poly.fit_transform(X_train,y_train)
    xx_test = poly.fit_transform(X_test,y_test)
    pc_b = Perceptron()
    pc_b.fit(xx_train, y_train)
    #calculate accuracy on test data and train data
    pred_test = pc_b.predict(xx_test)
    score_test = pc_b.score(xx_test,y_test)
    pred_train = pc_b.predict(xx_train)
    score_train = pc_b.score(xx_train,y_train)
    #print result (test first,train second)
    print(i,format(score_test,'.4f'),format(score_train,'.4f'))
    test_list.append(score_test)
    train_list.append(score_train)
    del score_test,score_train,pred_test,pred_train,xx_train,xx_test
#produce plot graph
plt.plot(range(1,11),test_list,color='red',label= "test")
plt.plot(range(1,11),train_list,color='blue',label="train")
plt.xlabel('Degree') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs degree')
plt.legend() 
plt.show()
del test_list,train_list

#kNN
from sklearn.neighbors import KNeighborsClassifier

test_list = []
train_list = []
for k in range(80):
    #Set k between 1 nd 80
    neigh = KNeighborsClassifier(n_neighbors=k+1)
    neigh.fit(X_train, y_train)
    pred_test = neigh.predict(X_test)
    score_test = neigh.score(X_test,y_test)
    pred_train = neigh.predict(X_train)
    score_train = neigh.score(X_train,y_train)
    #print result (test first,train second)
    test_list.append(score_test)
    train_list.append(score_train)
    print(k+1,format(score_test,'.4f'),format(score_train,'.4f'))
    del score_test,score_train,pred_test,pred_train
print("highest test score, k = ",np.argmax(test_list)+1,"; highest train score, k = ",np.argmax(train_list)+1)
plt.plot(range(1,81),test_list,color='red',label='test')
plt.plot(range(1,81),train_list,color='blue',label='train')
plt.xlabel('K') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs K-value')
plt.legend() 
plt.show()
del test_list,train_list