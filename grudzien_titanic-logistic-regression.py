import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
data_train_file = "../input/titanic-train/train.csv"
data_test_file = "../input/titanic-test/test.csv"
dane = pd.read_csv(data_train_file)
m = len(dane)

                # L E A R N I N G

#numerizing embark
Q = dane.Embarked == "Q"
C = 2*(dane.Embarked == "C")
S = 3*(dane.Embarked == "S")
x9 = Q+C+S
#numerizing sex
G = dane.Sex=="male"
int = np.vectorize(int)
G=int(G)
dane["Gender"] = G #Gender = 1 iff Sex = "male"

#creating x0
x0 =[] #vector of ones
i=1
while i<=m:
    x0.append(1)
    i+=1

#creating bilety feature
bilety=[]
for a in dane.Ticket:
    bilety.append(len(a))
dane["Bilet"] = bilety #length of a ticket name


#getting data
x0 = np.array(x0)
x1 = np.array(dane["SibSp"])
x2 = np.array(dane["Fare"])
x2 = (x2 - np.mean(x2)) / (np.max(x2) -np.min(x2))  # feature normalization
x3 = np.array(dane["Pclass"])
x4 = np.array(dane["Gender"])
x5 = np.array(dane["Parch"])
x6 = np.array(np.log( dane["Bilet"] +100) ) # avoiding log(0)
x7 = np.array(np.log( dane["Fare"]+100 ) )
x8 =np.array( dane.Fare * dane.Bilet )
x8 = x8/10000
#x9 declared previosuly
n = 9
X = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9])
Y = np.mat(dane.Survived)
theta = np.random.rand(1, n+1)



#defining sigmoids function
def g(z):
    return 1/(1+math.exp(-z))
g = np.vectorize(g)

#defining hypothesis
def h(theta, x):
    return g( np.dot(theta, x) )

#defining cost function
def cost(h, y):
    return -np.dot( np.log(h), y.transpose() )/m -np.dot( np.log(1-h), (1-y).transpose() )/m + 10000/m

#defining derivatives
def der(theta, x, y, m):
    return (1/m)* np.dot( ( h(theta,x) - y) , x.transpose() )

#debuggers
#print("X ",X)
#print("theta ",theta)
#print("h ",h(theta, X))
#print("cost ", cost( h(theta,X), Y) )
#print("der ",der( theta, X, Y, m))


#learning
it = 1
I=[]
C=[]
alpha = 0.1

while( it <= 4000):
    I.append(it)
    c = cost( h(theta,X), Y )
    c = float(c)
    C.append(c)
    theta = theta - alpha*der(theta, X, Y, m)
    it += 1

#print(h(theta,X))
pred = h(theta,X) > 0.65
#print(pred)
acc = int( pred==Y )
print("Accuracy is " , np.sum(acc)/m)


#learning curve
plt.plot(I,C)
plt.title("Learning Curve")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()




                    # P R E D I C T I O N


dane = pd.read_csv( data_test_file )
#print(dane)
m = len(dane)

#numerizing embark
Q = dane.Embarked == "Q"
C = 2*(dane.Embarked == "C")
S = 3*(dane.Embarked == "S")
x9 = Q+C+S
#numerizing sex
G = dane.Sex=="male"
int = np.vectorize(int)
G = int(G)
dane["Gender"] = G

#creating x0
x0 =[]
i=1
while i<=m:
    x0.append(1)
    i+=1

#the bilety feature
bilety=[]
for a in dane.Ticket:
    bilety.append(len(a))
dane["Bilet"] = bilety


#getting data
x0 = np.array(x0)
x1 = np.array(dane["SibSp"])
x2 = np.array(dane["Fare"])
x2[152] = 36 #replacing null with the average
x2 = (x2 - np.mean(x2)) / (np.max(x2) -np.min(x2))
x3 = np.array(dane["Pclass"])
x4 = np.array(dane["Gender"])
x5 = np.array(dane["Parch"])
x6 = np.array( np.log( dane["Bilet"] + 100) )
x7 = np.array( np.log( x2 + 100) ) #x2 is Fare
x8 = np.array( x2 * dane.Bilet)
x8 = x8/10000
n = 9
X = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9])

pred = int( h(theta,X) > 0.65 )

prediction_file = open( "first_submission.csv", "wt" ,newline = '\n')
prediction_file_object = csv.writer(prediction_file)

k=892 #smallest indicator of the test set
prediction_file_object.writerow(["PassengerId", "Survived"])
i=0
while i<m:
    prediction_file_object.writerow([i+k, pred[0,i]])
    i+=1

prediction_file.close()