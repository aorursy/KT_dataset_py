#Importing Numpy and Pandas
import numpy as np 
import pandas as pd 
#Using the Social_Network_Ads Dataset
data = pd.read_csv('../input/Social_Network_Ads.csv')
data.drop(columns=['User ID','Gender',],axis=1,inplace=True)
data.head()
#Outcome
y = data.iloc[:,-1].values
#Features
X = data.iloc[:,:-1].values
#Splitting the  Dataset into Training and Testing Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Performing Feature Scaling on Feature variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#y_pred, to store results of test set
#len_x, to store number of features
#w and b will be coefficients and intercept
y_pred = []
len_x = len(X_train[0])
w = []
b = 0.2
print(len_x)
#entries, to store number of rows in training set
entries = len(X_train[:,0])
entries
#Initially all coefficients w1,w2 etc set to 0
for weights in range(len_x):
    w.append(0)
w
#Sigmoid Function to return probability between 0 and 1
def sigmoid(z):
    return (1/(1+np.exp(-z)))
#Predict the probability of y being 1 given features of X
def predict(inputs):
    z = np.dot(w,inputs)+b
    a = sigmoid(z)
    return a
#Loss function
def loss_func(y,a):
    J = -(y*np.log(a) + (1-y)*np.log(1-a))
    return J         
#dw will store d(Loss(a,y))/d(w(i)) where d denotes differentiation
#db will store d(Loss(a,y))/db where d denotes differentiation
#J, total loss
#alpha, learning rate
dw = []
db = 0
J = 0
alpha = 0.1
for x in range(len_x):
    dw.append(0)
#Repeating the gradient descent process 1000 times
for iterations in range(1000):
    for i in range(entries):
        localx = X_train[i]                      
        a = predict(localx)                     
        dz = a - y_train[i]                     
        J += loss_func(y_train[i],a)            
        for j in range(len_x):                    
            dw[j] = dw[j]+(localx[j]*dz)
        db += dz
    J = J/entries
    db = db/entries
    for x in range(len_x):
        dw[x]=dw[x]/entries
    for x in range(len_x):              #Updating the coefficients and intercept
        w[x] = w[x]-(alpha*dw[x])       #w(x) = w(x) - learning_rate * dw(x)
    b = b-(alpha*db)         
    J=0
#localx will be the i(th) row from training set
#a will be the predicted value when features are localx
#dz is the differentiation of d(Loss(a,y)) w.r.t dz where z = (w1 * x1) + (w2 * x2)+....+b 
#J is the total cost, only used to check if model is converging
#Calculating Individual dw(dw1,dw2 etc...) where dw(j) = dz * x(j)
#Printing the coefficients
print(w)
#Printing the intercept
print(b)
#Predicting on test data and appending results to y_pred
for x in range(len(y_test)):
    y_pred.append(predict(X_test[x]))
for x in range(len(y_pred)):
    #Displaying Actual vs Predicted Values
    print('Actual ',y_test[x],' Predicted ',y_pred[x])
    #Rounding off values of y_pred, round(y_pred[x]) can also be used
    if y_pred[x]>=0.5:
        y_pred[x]=1
    else:
        y_pred[x]=0
#Checking the number of Correct Results
count = 0
for x in range(len(y_pred)):
    if(y_pred[x]==y_test[x]):
        count=count+1
#Displaying the accuracy
print('Accuracy:',(count/(len(y_pred)))*100)
#Analysing the test results
correct1 = 0  #True results correctly classified(Will buy) 
correct0 = 0  #False results correctly classified(Will not buy)
false_pos = 0 #False Positive
false_neg = 0 #False Negative
for x in range(len(y_pred)):
    if(y_pred[x]==1 and y_test[x]==1):
        correct1 += 1
    elif (y_pred[x]==0 and y_test[x]==0):
        correct0 += 1
    elif(y_pred[x]==0 and y_test[x]==1):
        false_pos += 1
    else: 
        false_neg += 1
print('Test Cases correctly classified :',correct1+correct0)
print('No of false positives :',false_pos)
print('No of false negatives :',false_neg)

#Note : Confusion Matrix Could also be used     