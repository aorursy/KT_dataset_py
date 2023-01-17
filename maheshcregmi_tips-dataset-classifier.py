import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = sns.load_dataset("tips")
df.info()
df.describe()
df.head()
df['tip'] = df['tip'].map(lambda x: 1 if x>2.99 else 0)
X = df.drop('tip', axis=1)
Y = df['tip']
X = pd.concat([X,pd.get_dummies(X['sex'], prefix='sex', drop_first=True)], axis=1)
X.drop("sex", axis=1, inplace=True)
X = pd.concat([X,pd.get_dummies(X['smoker'], prefix='smoker', drop_first=True)], axis=1)
X.drop("smoker", axis=1, inplace=True)
X.rename(columns={"smoker_No": "non_smoker", "sex_Female": "Female"}, inplace=True)
X.head()
X = pd.concat([X,pd.get_dummies(X['day'], prefix='day')], axis=1)

X.drop("day", axis=1, inplace=True)

X = pd.concat([X,pd.get_dummies(X['time'], prefix='time')], axis=1)
X.drop("time", axis=1, inplace=True)


X.rename(columns={"Female": "sex", "non_smoker": "smoker"}, inplace=True)
inv = {0 : 1, 1 : 0}
X["sex"] = X["sex"].map(inv)
X["smoker"] = X["smoker"].map(inv)
X.info()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1, stratify=Y)
plt.plot(X_train.T, ".")
plt.show()
class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
    
    def model(self, x):
        return 1 if (np.dot(x,self.w) >= self.b) else 0
    
    def predict(self,X):
        Y = []
        for x in X:
            y_pred = self.model(x)
            Y.append(y_pred)
        return np.array(Y)
    
    def fit(self, X , Y, lr=1, epochs=10):
        
        self.w = np.ones(X.shape[1])
        self.b = 0
        
        max_acc = 0
        
        
        
        accuracy = {}
        
        for i in range(epochs):
            
            for (x,y) in zip(X,Y):
                y_pred = self.model(x)
                
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b + lr * 1
                
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b - lr * 1
                    
            accuracy[i] = accuracy_score(self.predict(X), Y)
                
            if accuracy[i] > max_acc:
                max_acc = accuracy[i]
                max_wt = self.w
                max_b = self.b
                
        
        self.w = max_wt
        self.b = max_b
        
        plt.plot(*zip(*sorted(accuracy.items())))
        plt.show()
                
        print(max_acc)        
        
perceptron = Perceptron()
perceptron.fit(X_train.values, Y_train.values, epochs=100)
Y_pred = perceptron.predict(X_test.values)

acc = accuracy_score(Y_pred, Y_test.values)
acc
