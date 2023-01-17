import sklearn.datasets

import numpy as np
breast_cancer=sklearn.datasets.load_breast_cancer()
X=breast_cancer.data

Y=breast_cancer.target
print(X)

print(Y)
print(X.shape,Y.shape)
import pandas as pd
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['class']=breast_cancer.target
data.head()
data.describe()
print(data['class'].value_counts())
print(breast_cancer.target_names)
data.groupby('class').mean()
from sklearn.model_selection import train_test_split
X=data.drop(['class'],axis=1)

Y=data['class']
type(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
print(Y.mean(),Y_train.mean(),Y_test.mean())
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
print(X_train.mean(),X_test.mean(),Y_train.mean(),Y_test.mean())
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=1)
print(Y.mean(),Y_train.mean(),Y_test.mean())
print(X_train.mean(),X_test.mean(),Y_train.mean(),Y_test.mean())
import matplotlib.pyplot as plt
plt.plot(X_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_binarised_3_train=X_train['mean area'].map(lambda x : 0 if x<1000 else 1)
plt.plot(X_binarised_3_train,'*')

plt.show()
X_binarised_train=X_train.apply(pd.cut,bins=2,labels=[0,1])

plt.plot(X_binarised_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_binarised_test=X_test.apply(pd.cut,bins=2,labels=[0,1])
type(X_binarised_test)
#Convert into array

X_binarised_train=X_binarised_train.values

X_binarised_test=X_binarised_test.values

Y_train=Y_train.values

Y_test=Y_test.values
type(X_binarised_train)
from random import randint
#Initialize the parameter b of y=ax+b

b=3

#Get the inference fro a particular row

#i=100

i=randint(0,X_binarised_train.shape[0])

print('For row=',i)

if(np.sum(X_binarised_train[i:])>=b):

  print('MP Neuron inference is malignant')

else:

  print('MP Neuron inference is benign')

if(Y_train[i]==1):

  print('Ground truth is malignant')

else:

  print('Ground truth is benign')
b=3

Y_pred_train=[]

accurate_rows=0

for x,y in zip(X_binarised_train,Y_train):

  y_pred=(np.sum(x)>=b)

  Y_pred_train.append(y_pred)

  accurate_rows+=(y==y_pred)

print(accurate_rows,accurate_rows/X_binarised_train.shape[0])
np.sum([True,True])
for i in range(0,X_binarised_train.shape[1]+1):

  b=i

  Y_pred_train=[]

  accurate_rows=0

  for x,y in zip(X_binarised_train,Y_train):

    y_pred=(np.sum(x)>=b)

    Y_pred_train.append(y_pred)

    accurate_rows+=(y==y_pred)

  print(b,accurate_rows/X_binarised_train.shape[0])
X_binarised_train=X_train.apply(pd.cut,bins=2,labels=[1,0])

X_binarised_test=X_test.apply(pd.cut,bins=2,labels=[1,0])

#Convert into array

X_binarised_train=X_binarised_train.values

X_binarised_test=X_binarised_test.values
train_accuracy={}

for i in range(0,X_binarised_train.shape[1]+1):

  b=i

  Y_pred_train=[]

  accurate_rows=0

  for x,y in zip(X_binarised_train,Y_train):

    y_pred=(np.sum(x)>=b)

    Y_pred_train.append(y_pred)

    accurate_rows+=(y==y_pred)

  print(b,accurate_rows/X_binarised_train.shape[0])

  train_accuracy[i]=accurate_rows/X_binarised_train.shape[0]

#b=28

test_acc={}

y_pred_test=[]

for i in range(X_binarised_test.shape[1]):

  b=i

  accurate_rows=0

  for x,y in zip(X_binarised_test,Y_test):

    y_pred=(np.sum(x)>=b)

    y_pred_test.append(y_pred)

    accurate_rows+=(y_pred==y)



  print(b,accurate_rows/X_binarised_test.shape[0])

  test_acc[i]=accurate_rows/X_binarised_test.shape[0]

plt.plot(test_acc.values())

  

plt.plot(train_accuracy.values(),label='train')

#plt.show()

 

plt.show()
from sklearn.metrics import accuracy_score
b=28

y_pred_test=[]

for x in X_binarised_test:

  y_pred_test.append((np.sum(x)>=b))

accuracy=accuracy_score(y_pred_test,Y_test)

print(b,accuracy)
class MPNeuron:

  def __init__(self):

    self.b=None

  def model(self,x):

    return (sum(x)>=self.b)

  

  def predict(self,X):

    Y=[]

    for x in X:

      result=self.model(x)

      Y.append(result)

    return np.array(Y)

  def fit(self,X,Y):

    accuracy={}

    for b in range(0,X.shape[1]+1):

      self.b=b

      y_pred=self.predict(X)

      accuracy[b]=accuracy_score(y_pred,Y)

    best_b=max(accuracy,key=accuracy.get)

    self.b=best_b

    

    print('Optimal value of b =',best_b)

    print('Highest accuracy is =',accuracy[best_b])

    
mp_neuron=MPNeuron()

mp_neuron.fit(X_binarised_train,Y_train)
Y_test_pred=mp_neuron.predict(X_binarised_test)

accuracy_test=  accuracy_score(y_pred_test,Y_test)

print('Test accuracy=',accuracy_test)
class Perceptron:

  

  

  def __init__(self):

    self.w=None #Array

    self.b=None #Scaler

  

  

  def model(self,x):

    return 0 if (np.dot(x,self.w)<self.b) else 1

    

  def predict(self,X):

    y_pred=[]

    for x in X:

      y_pred.append(self.model(x))

      

    return np.array(y_pred)

  

  

  def fit(self,X,Y):

    

    #Initialize the value of weights w and constant b

    self.w=np.ones(X.shape[1])

    self.b=0

    for x,y in zip(X,Y):

      y_pred=self.model(x)

      if y==1 and y_pred==0:

        self.w=self.w+x

        self.b=self.b+1

      elif y==0 and y_pred==1:

        self.w=self.w-x

        self.b=self.b-1

        

  

  
perceptron=Perceptron()
#Here we don't have constraint to binarize the data. We can directly use the input data



X_train=X_train.values

X_test=X_test.values
perceptron.fit(X_train,Y_train)
plt.plot(perceptron.w)

plt.show()
Y_pred_train=perceptron.predict(X_train)

print(accuracy_score(Y_pred_train,Y_train))
Y_pred_test=perceptron.predict(X_test)

print(accuracy_score(Y_pred_test,Y_test))
class Perceptron:

  

  

  def __init__(self):

    self.w=None #Array

    self.b=None #Scaler

  

  

  def model(self,x):

    return 0 if (np.dot(x,self.w)<self.b) else 1

    

  def predict(self,X):

    y_pred=[]

    for x in X:

      y_pred.append(self.model(x))

      

    return np.array(y_pred)

  

  

  def fit(self,X,Y,epochs):

    #Initialize the value of weights w and constant b

    self.w=np.ones(X.shape[1])

    self.b=0

    accuracy={}

    max_accuracy=0

    for i in range(epochs):

      for x,y in zip(X,Y):

        y_pred=self.model(x)

        if y==1 and y_pred==0:

          self.w=self.w+x

          self.b=self.b+1

        elif y==0 and y_pred==1:

          self.w=self.w-x

          self.b=self.b-1

      accuracy[i]=accuracy_score(self.predict(X),Y)

      if accuracy[i]>max_accuracy:

        max_accuracy=accuracy[i]

        #Collect the parameter values corresponding to  max value seen so far for accuracy

        chkptw=self.w

        chkptb=self.b

        

    self.w=chkptw

    self.b=chkptb

    print('Max Accuracy=',max_accuracy)

    plt.plot(accuracy.values())

    #print(accuracy.values())

    plt.show()

        

      

      

  

  
perceptron=Perceptron()

perceptron.fit(X_train,Y_train,1000)



Y_pred_train=perceptron.predict(X_train)

print('Train accuracy',accuracy_score(Y_pred_train,Y_train))



Y_pred_test=perceptron.predict(X_test)

print('Test accuracy',accuracy_score(Y_pred_test,Y_test))



class Perceptron:

  

  

  def __init__(self):

    self.w=None #Array

    self.b=None #Scaler

  

  

  def model(self,x):

    return 0 if (np.dot(x,self.w)<self.b) else 1

    

  def predict(self,X):

    y_pred=[]

    for x in X:

      y_pred.append(self.model(x))

      

    return np.array(y_pred)

  

  

  def fit(self,X,Y,epochs=1,lr=1):

    #Initialize the value of weights w and constant b

    self.w=np.ones(X.shape[1])

    self.b=0

    accuracy={}

    max_accuracy=0

    wt_matrix=[]

    for i in range(epochs):

      for x,y in zip(X,Y):

        y_pred=self.model(x)

        if y==1 and y_pred==0:

          self.w=self.w+lr*x

          self.b=self.b-lr*1

        elif y==0 and y_pred==1:

          self.w=self.w-lr*x

          self.b=self.b+lr*1

          

          

      wt_matrix.append(self.w)

      accuracy[i]=accuracy_score(self.predict(X),Y)

      if accuracy[i]>max_accuracy:

        max_accuracy=accuracy[i]

        #Collect the parameter values corresponding to  max value seen so far for accuracy

        chkptw=self.w

        chkptb=self.b

        

    self.w=chkptw

    self.b=chkptb

    print('Max Accuracy=',max_accuracy)

    plt.plot(accuracy.values())

    #Limit the y axis as value of accuracy will be between 0 and 1. If dnt limit then you will see more oscillatory graph

    plt.ylim([0,1])

    plt.show()

    

    return np.array(wt_matrix)

        

      

      

  

  
perceptron=Perceptron()

wt_matrix=perceptron.fit(X_train,Y_train,100,0.5)



Y_pred_train=perceptron.predict(X_train)

print('Train accuracy',accuracy_score(Y_pred_train,Y_train))



Y_pred_test=perceptron.predict(X_test)

print('Test accuracy',accuracy_score(Y_pred_test,Y_test))



wt_matrix=perceptron.fit(X_train,Y_train,5,10)



Y_pred_train=perceptron.predict(X_train)

print('Train accuracy',accuracy_score(Y_pred_train,Y_train))



Y_pred_test=perceptron.predict(X_test)

print('Test accuracy',accuracy_score(Y_pred_test,Y_test))



plt.plot(perceptron.w)

plt.show()



#Different features are having different weights.Features for the weights are neart to zero may not play the significant role comare to features having large positive or negative values
%matplotlib inline

from matplotlib import animation, rc

from IPython.display import HTML



# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots()



ax.set_xlim(( 0, wt_matrix.shape[1]))

ax.set_ylim((-15000, 25000))



line, = ax.plot([], [], lw=2)



# animation function. This is called sequentially

def animate(i):

    x = list(range(wt_matrix.shape[1]))

    y = wt_matrix[i, :]

    line.set_data(x, y)

    return (line,)



# call the animator. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=100, interval=200, blit=True)



HTML(anim.to_html5_video())
#Initialize w and b randomly

from random import randint,uniform

#print()



class Perceptron:

  

  

  def __init__(self):

    self.w=None #Array

    self.b=None #Scaler

  

  

  def model(self,x):

    return 0 if (np.dot(x,self.w)<self.b) else 1

    

  def predict(self,X):

    y_pred=[]

    for x in X:

      y_pred.append(self.model(x))

      

    return np.array(y_pred)

  

  

  def fit(self,X,Y,epochs=1,lr=1):

    #Initialize the value of weights w and constant b

    self.w=[uniform(0,10) for i in range(0,X.shape[1])]

    #np.ones(X.shape[1])

    self.b=uniform(0,100)

    accuracy={}

    max_accuracy=0

    wt_matrix=[]

    for i in range(epochs):

      for x,y in zip(X,Y):

        y_pred=self.model(x)

        if y==1 and y_pred==0:

          self.w=self.w+lr*x

          self.b=self.b-lr*1

        elif y==0 and y_pred==1:

          self.w=self.w-lr*x

          self.b=self.b+lr*1

          

          

      wt_matrix.append(self.w)

      accuracy[i]=accuracy_score(self.predict(X),Y)

      if accuracy[i]>max_accuracy:

        max_accuracy=accuracy[i]

        #Collect the parameter values corresponding to  max value seen so far for accuracy

        chkptw=self.w

        chkptb=self.b

        

    self.w=chkptw

    self.b=chkptb

    print('Max Accuracy=',max_accuracy)

    plt.plot(accuracy.values())

    #Limit the y axis as value of accuracy will be between 0 and 1. If dnt limit then you will see more oscillatory graph

    plt.ylim([0,1])

    plt.show()

    

    return np.array(wt_matrix)

        

      

      

  

  
perceptron=Perceptron()

wt_matrix=perceptron.fit(X_train,Y_train,100000,0.0001)



Y_pred_train=perceptron.predict(X_train)

print('Train accuracy',accuracy_score(Y_pred_train,Y_train))



Y_pred_test=perceptron.predict(X_test)

print('Test accuracy',accuracy_score(Y_pred_test,Y_test))



%matplotlib inline

from matplotlib import animation, rc

from IPython.display import HTML



# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots()



ax.set_xlim(( 0, wt_matrix.shape[1]))

ax.set_ylim((-15000, 25000))



line, = ax.plot([], [], lw=2)



# animation function. This is called sequentially

def animate(i):

    x = list(range(wt_matrix.shape[1]))

    y = wt_matrix[i, :]

    line.set_data(x, y)

    return (line,)



# call the animator. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=100, interval=200, blit=True)



HTML(anim.to_html5_video())