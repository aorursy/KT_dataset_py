import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import datasets
class LogisticRegression:

    def __init__(self,learning_rate=0.0001,n_iters=10000):

        self.lr = learning_rate

        self.n_iters = n_iters

        self.weights = None

        self.bias = None

    def sigmoid(self,X):

        return 1/(1+np.exp(-X))

    def fit(self,X,y):

        n_row,n_col = X.shape #no of samples and no of features

        

        self.weights = np.zeros(n_col)

        self.bias = 0

        

        for i in range(self.n_iters):  #gradient descent

            Z = np.dot(X,self.weights)+self.bias #Z = C + mX

            predictions = self.sigmoid(Z)

            

            dw = (1/n_row)*np.dot(X.T,(predictions-y))   #calculating gradients

            db = (1/n_row)*np.sum(predictions - y)

            

            self.weights -= self.lr*dw #updating the weights and bias

            self.bias -= self.lr*db

        return self.weights,self.bias

    def predict(self,X):

        Z = np.dot(X,self.weights)+self.bias

        y_predict = self.sigmoid(Z)   #calculating probabilities

        clas = [1 if j>0.5 else 0 for j in y_predict]  #converting probabilities to classes

        return np.array(clas)

            

        

        
data = datasets.load_breast_cancer() 

X,y = data.data,data.target

print(X.shape)

print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
clf = LogisticRegression(learning_rate=0.0001,n_iters=10000)

final_weights,final_bias = clf.fit(X_train,y_train)

predict = clf.predict(X_test)

print(f''' 

 final weights: {final_weights}

 final bias : {final_bias}

 ''')

print(predict.shape)
def accuracy(y_true,y_pred):

    acc = np.sum(y_true == y_pred)/len(y_true)

    return acc
print('Accuracy:',accuracy(y_test,predict))