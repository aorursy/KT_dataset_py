#import libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
%matplotlib inline

# Reading Data 
df = pd.read_csv('../input/Iris.csv')
#df.describe()
df.info()
df['Class']=df['Species']
df['Class'] = df['Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df["Class"].unique()
df.head()
# Define a function for Euclidien distance
def eucdist(p1, p2): 
    dist = 0
    for i in range(len(p1)):
        dist = dist + np.square(p1[i]-p2[i])
    dist = np.sqrt(dist)
    return dist;
# Define kNN function
def kNN(X,y,k,test):
    # Calculating distances for our test point
    newdist = np.zeros(len(y))

    for j in range(len(y)):
        newdist[j] = eucdist(test, X[j,:])

    # Merging actual labels with calculated distances
    newdist = np.array([newdist, y])

    ## Finding the closest k neighbors
    # Sorting index
    idx = np.argsort(newdist[0,:])

    # Sorting the all newdist
    newdist = newdist[:,idx]
    #print(newdist)

    # We should count neighbor labels and take the label which has max count
    # Define a dictionary for the counts
    c = {'0':0,'1':0,'2':0 }

    for i in range(k):
        c[str(int(newdist[1,i]))] = c[str(int(newdist[1,i]))] + 1

    key_max = max(c.keys(), key=(lambda k: c[k]))
    return int(key_max)
# Define sigmoid function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Define cost function
# I do not use cost function in the code but 
# use it to check theta and choice of alpha
# in gradientdescent function
def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
# To minimize our cost function we use Gradient Descent briefly:
# Repeat {
#  1. Calculate gradient average
#  2. Multiply by learning rate alpha
#  3. Subtract from weights }
def gradientdescent(X, y, alpha, num_iter):

    # select initial values zero
    theta = np.zeros(X.shape[1])

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta = theta - alpha * gradient

        # This part is used for the check  
        #calcloss = cost(h, y)
        #if( i % 1000 == 0): 
        #    print('Number of Iterations: ', i, 'Cost : ', calcloss, 'Theta: ', theta)
    return theta
# Define predict function to calculate probabilities
# for each class
def predict(test, theta):
    z = np.dot(np.transpose(test), theta)
    return sigmoid(z)
def logistic(X, y,  alpha, num_iter, test):
    # We have 3 classifiers because of this, I need to select a method
    # and I decided to apply one vs all (OvA)

    # Model for class 0
    y0 = np.copy(y)
    y0[y==2] = 1
    y0 = y0 - 1
    y0 = y0 * -1 
    theta0 = gradientdescent(X, y0, alpha, num_iter)

    # Model for class 1 
    y1 = np.copy(y)
    y1[y==2] = 0
    theta1 = gradientdescent(X, y1, alpha, num_iter)

    # Model for class 2
    y2 = np.copy(y)
    y2[y==1] = 0
    y2[y==2] = 1
    theta2 = gradientdescent(X, y2, alpha, num_iter)

    # Calculate probabilties
    preds = []
    preds.append(predict(test,theta0))
    preds.append(predict(test,theta1))
    preds.append(predict(test,theta2))
    
    # Select max probability
    m=max(preds)
    c=0
    if(m==preds[1]):
        c=1
    if(m==preds[2]):
        c=2
        
    return c
# Define X (features) and Y (target)
adf = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Class']].values
X = adf[:,:4]
y = adf[:,4]

# I chose data points close to the real data points
test0 = [5.77,4.44,1.55,0.44] # close to X[15]
test1 = [5.66,3.01,4.55,1.55] # close to X[66]
test2 = [7.44, 2.88, 6.11, 1.99] # close to X[130]
plt.figure(figsize=(10,10))
t=np.unique(y)

ax1=plt.subplot(2, 2, 1)
ax1.set(xlabel='Sepal Length (cm)', ylabel='Sepal Width (cm)')
plt.plot(X[y==t[0],0], X[y==t[0],1], 'o', color='y')
plt.plot(X[y==t[1],0], X[y==t[1],1], 'o', color='r')
plt.plot(X[y==t[2],0], X[y==t[2],1], 'o', color='b')
# test datapoints
plt.plot(test0[0],test0[1],'*',color="k")
plt.plot(test1[0],test1[1],'*',color="k")
plt.plot(test2[0],test2[1],'*',color="k")

ax2=plt.subplot(2, 2, 2)
ax2.set(xlabel='Petal Length (cm)', ylabel='Petal Width (cm)')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
plt.plot(X[y==t[0],2], X[y==t[0],3], 'o', color='y')
plt.plot(X[y==t[1],2], X[y==t[1],3], 'o', color='r')
plt.plot(X[y==t[2],2], X[y==t[2],3], 'o', color='b')
# test datapoints
plt.plot(test0[2],test0[3],'*',color="k")
plt.plot(test1[2],test1[3],'*',color="k")
plt.plot(test2[2],test2[3],'*',color="k")

ax3=plt.subplot(2, 2, 3)
ax3.set(xlabel='Sepal Length (cm)', ylabel='Petal Length (cm)')
plt.plot(X[y==t[0],0], X[y==t[0],2], 'o', color='y')
plt.plot(X[y==t[1],0], X[y==t[1],2], 'o', color='r')
plt.plot(X[y==t[2],0], X[y==t[2],2], 'o', color='b')
# test datapoints
plt.plot(test0[0],test0[2],'*',color="k")
plt.plot(test1[0],test1[2],'*',color="k")
plt.plot(test2[0],test2[2],'*',color="k")

ax4=plt.subplot(2, 2, 4)
ax4.set(xlabel='Sepal Width (cm)', ylabel='Petal Width (cm)')
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
plt.plot(X[y==t[0],1], X[y==t[0],3], 'o', color='y')
plt.plot(X[y==t[1],1], X[y==t[1],3], 'o', color='r')
plt.plot(X[y==t[2],1], X[y==t[2],3], 'o', color='b')
# test datapoints
plt.plot(test0[1],test0[3],'*',color="k")
plt.plot(test1[1],test1[3],'*',color="k")
plt.plot(test2[1],test2[3],'*',color="k")
# Predicting the classes of the test data by kNN 

# Decide k value
k = 5

print("kNN\n")
c = kNN(X,y,k,test0)
print("Test point "+str(test0)+" has label "+str(c)+" according to "
      +str(k)+"-nearest neighbors.\n")

c=kNN(X,y,k,test1)
print("Test point "+str(test1)+" has label "+str(c)+" according to "
      +str(k)+"-nearest neighbors.\n")

c=kNN(X,y,k,test2)
print("Test point "+str(test2)+" has label "+str(c)+" according to "
      +str(k)+"-nearest neighbors.\n")
# Predicting the classes of the test data by Logistic Regression

# Adjustments for the logistic regression
alpha = 0.1
num_iter = 30000

# Add intercept
intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis=1)
test0 = np.concatenate(([1],test0))
test1 = np.concatenate(([1],test1))
test2 = np.concatenate(([1],test2))

# Run logistic function and determine results
print("Logistic Regression\n")
c=logistic(X,y, alpha, num_iter, test0)
print("Test point "+str(test0)+" has label "+str(c)+" according to logistic regression.\n")

c=logistic(X,y, alpha, num_iter, test1)
print("Test point "+str(test1)+" has label "+str(c)+" according to logistic regression.\n")

c=logistic(X,y, alpha, num_iter, test2)
print("Test point "+str(test2)+" has label "+str(c)+" according to logistic regression.\n")