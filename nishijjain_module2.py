!pip install jovian --upgrade --quiet
import sklearn.datasets
import numpy as np

breast_cancer = sklearn.datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
print(X)
print(Y)
print(X.shape,Y.shape)
import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target
data.head()
data.describe()
print(data['class'].value_counts())
print(breast_cancer.target_names)
data.groupby('class').mean()
from sklearn.model_selection import train_test_split
X = data.drop('class',axis=1)
Y = data['class']
type(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1)
print(Y.mean(), Y_train.mean(), Y_test.mean())
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1, stratify=Y)
#as in the previous cell, the ratio of malignant cases in train and test set is not close
#so stratify is used for more accuracy
print(Y.mean(), Y_train.mean(), Y_test.mean())
print(X.mean(), X_train.mean(),X_test.mean())
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1, stratify=Y, random_state=1)
import matplotlib.pyplot as plt
plt.plot(X_train.T,'*')
plt.xticks(rotation='vertical')
plt.show()
plt.plot(X_test.T,'*')
plt.xticks(rotation='vertical')
plt.show()
X_binarised_3_train = X_train['mean area'].map(lambda x:0 if x<1000 else 1)
plt.plot(X_binarised_3_train,'*')
plt.show()
X_binarised_train = X_train.apply(pd.cut,bins=2,labels=[0,1])
plt.plot(X_binarised_train.T,'*')
plt.xticks(rotation='vertical')
plt.show()
X_binarised_test = X_test.apply(pd.cut,bins=2,labels=[0,1])
X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values
type(X_binarised_test)
b = 3
i = 100
if(np.sum(X_binarised_train[100,:])>=b):
    print('Malignant')
else:
    print('Benign')
    
if(Y_train[i]==1):
    print('Ground truth is malignant')
else:
    print('Ground truth is benign')
    
from random import randint
b = 3
i = randint(0,X_binarised_train.shape[0])
print('For row',i)
if(np.sum(X_binarised_train[100,:])>=b):
    print('Malignant')
else:
    print('Benign')
    
if(Y_train[i]==1):
    print('Ground truth is malignant')
else:
    print('Ground truth is benign')
b=3

Y_pred_train = []
accurate_rows = 0

for x,y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x)>=b)
    Y_pred_train.append(y_pred)
    accurate_rows += (y==y_pred)
    
print(accurate_rows,accurate_rows/X_binarised_train.shape[0])
print(X_binarised_train.shape[1])
for b in range(X_binarised_train.shape[1]+1):
    Y_pred_train = []
    accurate_rows = 0

    for x,y in zip(X_binarised_train, Y_train):
        y_pred = (np.sum(x)>=b)
        Y_pred_train.append(y_pred)
        accurate_rows += (y==y_pred)
    
    print(b,accurate_rows/X_binarised_train.shape[0])
X_binarised_train = X_train.apply(pd.cut,bins=2,labels=[1,0])
X_binarised_test = X_test.apply(pd.cut,bins=2,labels=[1,0])
X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values
for b in range(X_binarised_train.shape[1]+1):
    Y_pred_train = []
    accurate_rows = 0

    for x,y in zip(X_binarised_train, Y_train):
        y_pred = (np.sum(x)>=b)
        Y_pred_train.append(y_pred)
        accurate_rows += (y==y_pred)
    
    print(b,accurate_rows/X_binarised_train.shape[0])
from sklearn.metrics import accuracy_score
b = 28
Y_pred_test = []

for x in X_binarised_test:
    y_pred = (np.sum(x)>=b)
    Y_pred_test.append(y_pred)
accuracy = accuracy_score(Y_pred_test,Y_test)
print(b,accuracy)
class MPNeuron:
    
    def __init__(self):
        self.b = None
        
    def model(self,x):
        return(sum(x)>=self.b)
    
    def predict(self,X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    def fit(self,X,Y):
        accuracy = {}
        
        for b in range(X.shape[1]+1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y_pred,Y)
            
        best_b = max(accuracy,key=accuracy.get)
        self.b = best_b
        
        print('Optimal value of b',best_b)
        print('Highest Accuracy is',accuracy[best_b])
mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train,Y_train)
Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred,Y_test)
print(accuracy_test)
class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        
    def model(self,x):
        return 1 if(np.dot(self.w,x)>=self.b) else 0
    
    def predict(self,X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    def fit(self,X,Y):
        self.w = np.ones(X.shape[1])
        self.b = 0
        
        for x,y in zip(X,Y):
            y_pred = self.model(x)
            if y == 1 and y_pred == 0:
                self.w = self.w + x
                self.b = self.b + 1
            elif y == 0 and y_pred == 1:
                self.w = self.w - x
                self.b = self.b - 1
        
perceptron = Perceptron()
X_train = X_train.values
X_test = X_test.values
perceptron.fit(X_train,Y_train)
plt.plot(perceptron.w)

Y_pred_train = perceptron.predict(X_train)
print(accuracy_score(Y_pred_train,Y_train))
Y_pred_test = perceptron.predict(X_test)
print(accuracy_score(Y_pred_test,Y_test))
class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        
    def model(self,x):
        return 1 if(np.dot(self.w,x)>=self.b) else 0
    
    def predict(self,X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    def fit(self,X,Y,epochs=1,lr=1):
        self.w = np.ones(X.shape[1])
        self.b = 0
        
        accuracy = {}
        max_accuracy = 0
        
        wt_matrix = []
        
        for i in range(epochs):
            for x,y in zip(X,Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b + lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b - lr * 1
            
            wt_matrix.append(self.w)    
                    
            accuracy[i] = accuracy_score(self.predict(X),Y)
            if (accuracy[i]>max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b
        
        self.w = chkptw
        self.b = chkptb
        
        print(max_accuracy)
        plt.plot(list(accuracy.values()))
        return np.array(wt_matrix)
perceptron = Perceptron()
wt_matrix = perceptron.fit(X_train,Y_train,100)
Y_pred_train = perceptron.predict(X_train)
print(accuracy_score(Y_pred_train,Y_train))
Y_pred_test = perceptron.predict(X_test)
print(accuracy_score(Y_pred_test,Y_test))
plt.plot(wt_matrix[-1,:])
plt.show()
%matplotlib inline
from matplotlib import animation, rc
from IPython.display import HTML
# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

ax.set_xlim(( 0, wt_matrix.shape[1]))
ax.set_ylim((-10000, 25000))

line, = ax.plot([], [], lw=2)
# animation function. This is called sequentially
def animate(i):
    x = list(range(wt_matrix.shape[1]))
    y = wt_matrix[i,:]
    line.set_data(x, y)
    return (line,)
# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=100, interval=200, blit=True)
HTML(anim.to_html5_video())
