#import libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from random import randrange

from random import seed

from statistics import mean 

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Reading Data 

df = pd.read_csv('../input/Iris.csv')

#df.describe()

#df.info()

df['Class']=df['Species']

df['Class'] = df['Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

df["Class"].unique()

df.head()
# Distances

def euclidian(p1, p2): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + np.square(p1[i]-p2[i])

    dist = np.sqrt(dist)

    return dist;



def manhattan(p1, p2): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + abs(p1[i]-p2[i])

    return dist;



def minkowski(p1, p2, q): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + abs(p1[i]-p2[i])**q

    dist = np.sqrt(dist)**(1/q)

    return dist;
# kNN Function

def kNN(X_train,y_train, X_test, k, dist='euclidian',q=2):

    pred = []

    # Adjusting the data type

    if isinstance(X_test, np.ndarray):

        X_test=pd.DataFrame(X_test)

    if isinstance(X_train, np.ndarray):

        X_train=pd.DataFrame(X_train)

        

    for i in range(len(X_test)):    

        # Calculating distances for our test point

        newdist = np.zeros(len(y_train))



        if dist=='euclidian':

            for j in range(len(y_train)):

                newdist[j] = euclidian(X_train.iloc[j,:], X_test.iloc[i,:])

    

        if dist=='manhattan':

            for j in range(len(y_train)):

                newdist[j] = manhattan(X_train.iloc[j,:], X_test.iloc[i,:])

    

        if dist=='minkowski':

            for j in range(len(y_train)):

                newdist[j] = minkowski(X_train.iloc[j,:], X_test.iloc[i,:],q)



        # Merging actual labels with calculated distances

        newdist = np.array([newdist, y_train])



        ## Finding the closest k neighbors

        # Sorting index

        idx = np.argsort(newdist[0,:])



        # Sorting the all newdist

        newdist = newdist[:,idx]

        #print(newdist)



        # We should count neighbor labels and take the label which has max count

        # Define a dictionary for the counts

        c = {'0':0,'1':0,'2':0 }

        # Update counts in the dictionary 

        for j in range(k):

            c[str(int(newdist[1,j]))] = c[str(int(newdist[1,j]))] + 1



        key_max = max(c.keys(), key=(lambda k: c[k]))

        pred.append(int(key_max))

        

    return pred
# Sigmoid Function 

def sigmoid(z):

    return 1 / (1 + np.exp(-z))
# Cost Function

def J(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
# Gradient Descent Function

def gradientdescent(X, y, lmd, alpha, num_iter, print_cost):



    # select initial values zero

    theta = np.zeros(X.shape[1])

    

    costs = []  

    

    for i in range(num_iter):

        z = np.dot(X, theta)

        h = sigmoid(z)

        

        # adding regularization 

        reg = lmd / y.size * theta

        # first theta is intercept

        # it is not regularized

        reg[0] = 0

        cost = J(h, y)

        

        gradient = np.dot(X.T, (h - y)) / y.size + reg

        theta = theta - alpha * gradient

    

        if print_cost and i % 100 == 0: 

            print('Number of Iterations: ', i, 'Cost : ', cost, 'Theta: ', theta)

        if i % 100 == 0:

            costs.append(cost)

      

    return theta, costs
# Predict Function 

def predict(X_test, theta):

    z = np.dot(X_test, theta)

    return sigmoid(z)
# Main Logistic Function

def logistic(X_train, y_train, X_test, lmd=0, alpha=0.1, num_iter=30000, print_cost = False):

    # Adding intercept

    intercept = np.ones((X_train.shape[0], 1))

    X_train = np.concatenate((intercept, X_train), axis=1)

    

    intercept = np.ones((X_test.shape[0], 1))

    X_test = np.concatenate((intercept, X_test), axis=1)



    # one vs rest

    u=set(y_train)

    t=[]

    allCosts=[]   

    for c in u:

        # set the labels to 0 and 1

        ynew = np.array(y_train == c, dtype = int)

        theta_onevsrest, costs_onevsrest = gradientdescent(X_train, ynew, lmd, alpha, num_iter, print_cost)

        t.append(theta_onevsrest)

        

        # Save costs

        allCosts.append(costs_onevsrest)

        

    # Calculate probabilties

    pred_test = np.zeros((len(u),len(X_test)))

    for i in range(len(u)):

        pred_test[i,:] = predict(X_test,t[i])

    

    # Select max probability

    prediction_test = np.argmax(pred_test, axis=0)

    

    # Calculate probabilties

    pred_train = np.zeros((len(u),len(X_train)))

    for i in range(len(u)):

        pred_train[i,:] = predict(X_train,t[i])

    

    # Select max probability

    prediction_train = np.argmax(pred_train, axis=0)

    

    d = {"costs": allCosts,

         "Y_prediction_test": prediction_test, 

         "Y_prediction_train" : prediction_train, 

         "learning_rate" : alpha,

         "num_iterations": num_iter,

         "lambda": lmd}

        

    return d
# Sigmoid Function

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



# Select initial values zero

def initialize_with_zeros(dim):

    return np.zeros((dim,1)), 0
def propagate(w, b, X, Y):

    m = X.shape[1]

    

    # FORWARD PROPAGATION (FROM X TO COST)

    A = sigmoid(np.dot(w.T,X)+b) # compute activation

    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) # compute cost

    

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = 1/m*np.dot(X,(A-Y).T)

    db = 1/m*np.sum(A-Y)

    

    # keep grads in a dictionary 

    grads = {"dw": dw,

             "db": db}

    

    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):    

    costs = []

    

    for i in range(num_iterations):

        # Cost and gradient calculation

        grads, cost = propagate(w, b, X, Y)

        

        # Retrieve derivatives from grads

        dw = grads["dw"]

        db = grads["db"]

        

        # update rule

        w = w-learning_rate*dw

        b = b-learning_rate*db 

        

        # Record the costs

        if i % 100 == 0:

            costs.append(cost)

            

        # Print the cost every 100 training iterations

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    # Save pameters and gradients

    params = {"w": w,

              "b": b}

    

    grads = {"dw": dw,

             "db": db}

    

    return params, grads, costs
def predict_nn(w, b, X):    

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    # Compute vector "A" predicting the probabilities

    A = sigmoid(np.dot(w.T,X)+b)

        

    return A
def model(X_train, Y_train, X_test, Y_test, num_iterations = 30000, learning_rate = 0.1, print_cost = False): 

    # pandas to numpy

    X_train = X_train.values

    Y_train = Y_train.values.reshape((1,Y_train.shape[0]))

    X_test = X_test.values

    Y_test = Y_test.values.reshape((1,Y_test.shape[0]))

    

    # take transpose of X

    X_train = X_train.T

    X_test = X_test.T

    

    # initialize parameters with zeros 

    w, b = initialize_with_zeros(X_train.shape[0])

    

    # one vs all

    u = set(y_train)

    param_w = []

    param_b = []

    allCosts = []

    for c in u:

        # set the labels to 0 and 1

        ynew = np.array(y_train == c, dtype = int)

        # Gradient descent 

        parameters, grads, costs = optimize(w, b, X_train, ynew, num_iterations, learning_rate, print_cost = print_cost)

        

        # Save costs

        allCosts.append(costs)

        

        # Retrieve parameters w and b from dictionary "parameters"

        param_w.append(parameters["w"])

        param_b.append(parameters["b"])

    

    # Calculate probabilties

    pred_test = np.zeros((len(u),X_test.shape[1]))

    for i in range(len(u)):

        pred_test[i,:] = predict_nn(param_w[i], param_b[i], X_test)

    

    # Select max probability

    Y_prediction_test = np.argmax(pred_test, axis=0)

    

    # Calculate probabilties

    pred_train = np.zeros((len(u),X_train.shape[1]))

    for i in range(len(u)):

        pred_train[i,:] = predict_nn(param_w[i], param_b[i], X_train)

    

    # Select max probability

    Y_prediction_train = np.argmax(pred_train, axis=0)

        

    d = {"costs": allCosts,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
# I chose data points close to the real data points X[15], X[66] and X[130]

test = np.array([[5.77,4.44,1.55,0.44],[5.66,3.01,4.55,1.55],[7.44, 2.88, 6.11, 1.99]])

print("TEST POINTS\n", test)



all_X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

all_y = df['Class']



# split data as training and test

df=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Class']]

train_data,test_data = train_test_split(df,train_size = 0.8,random_state=2)

X_train = train_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_train = train_data['Class']

X_test = test_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_test = test_data['Class']



def transform(i):

    if i == 0:

        return 'Iris-setosa'

    if i == 1:

        return 'Iris-versicolor'

    if i == 2:

        return 'Iris-virginica'
plt.figure(figsize=(10,10))

t=np.unique(all_y)



ax1=plt.subplot(2, 2, 1)

ax1.set(xlabel='Sepal Length (cm)', ylabel='Sepal Width (cm)')

plt.plot(df[df['Class']==t[0]].iloc[:,0], df[df['Class']==t[0]].iloc[:,1], 'o', color='y')

plt.plot(df[df['Class']==t[1]].iloc[:,0], df[df['Class']==t[1]].iloc[:,1], 'o', color='r')

plt.plot(df[df['Class']==t[2]].iloc[:,0], df[df['Class']==t[2]].iloc[:,1], 'o', color='b')

# test datapoints

plt.plot(test[0,0],test[0,1],'*',color="k")

plt.plot(test[1,0],test[1,1],'*',color="k")

plt.plot(test[2,0],test[2,1],'*',color="k")



ax2=plt.subplot(2, 2, 2)

ax2.set(xlabel='Petal Length (cm)', ylabel='Petal Width (cm)')

ax2.yaxis.set_label_position("right")

ax2.yaxis.tick_right()

plt.plot(df[df['Class']==t[0]].iloc[:,2], df[df['Class']==t[0]].iloc[:,3], 'o', color='y')

plt.plot(df[df['Class']==t[1]].iloc[:,2], df[df['Class']==t[1]].iloc[:,3], 'o', color='r')

plt.plot(df[df['Class']==t[2]].iloc[:,2], df[df['Class']==t[2]].iloc[:,3], 'o', color='b')

# test datapoints

plt.plot(test[0,2],test[0,3],'*',color="k")

plt.plot(test[1,2],test[1,3],'*',color="k")

plt.plot(test[2,2],test[2,3],'*',color="k")



ax3=plt.subplot(2, 2, 3)

ax3.set(xlabel='Sepal Length (cm)', ylabel='Petal Length (cm)')

plt.plot(df[df['Class']==t[0]].iloc[:,0], df[df['Class']==t[0]].iloc[:,2], 'o', color='y')

plt.plot(df[df['Class']==t[1]].iloc[:,0], df[df['Class']==t[1]].iloc[:,2], 'o', color='r')

plt.plot(df[df['Class']==t[2]].iloc[:,0], df[df['Class']==t[2]].iloc[:,2], 'o', color='b')

# test datapoints

plt.plot(test[0,0],test[0,2],'*',color="k")

plt.plot(test[1,0],test[1,2],'*',color="k")

plt.plot(test[2,0],test[2,2],'*',color="k")



ax4=plt.subplot(2, 2, 4)

ax4.set(xlabel='Sepal Width (cm)', ylabel='Petal Width (cm)')

ax4.yaxis.set_label_position("right")

ax4.yaxis.tick_right()

plt.plot(df[df['Class']==t[0]].iloc[:,1], df[df['Class']==t[0]].iloc[:,3], 'o', color='y')

plt.plot(df[df['Class']==t[1]].iloc[:,1], df[df['Class']==t[1]].iloc[:,3], 'o', color='r')

plt.plot(df[df['Class']==t[2]].iloc[:,1], df[df['Class']==t[2]].iloc[:,3], 'o', color='b')

# test datapoints

plt.plot(test[0,1],test[0,3],'*',color="k")

plt.plot(test[1,1],test[1,3],'*',color="k")

plt.plot(test[2,1],test[2,3],'*',color="k");

# Predicting the classes of the test data by kNN 

# Decide k value

k = 5

# print results

print("k-NN ("+str(k)+"-nearest neighbors)\n")

c = kNN(all_X,all_y,test,k)

for i in range(len(c)):

    ct=set(map(transform,[c[i]]))

    print("Test point: "+str(test[i,:])+"  Label: "+str(c[i])+" "+str(ct))
# k-NN from scratch

c=kNN(X_train,y_train,X_test,k)

cm=confusion_matrix(y_test, c)



# logistic regression - scikit learn

sck = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

sck_cm=confusion_matrix(y_test, sck.predict(X_test))



plt.figure(figsize=(15,6))

plt.suptitle("Confusion Matrixes",fontsize=24)



plt.subplot(1,2,1)

plt.title("k-NN from Scratch")

sns.heatmap(cm, annot = True, cmap="Greens",cbar=False);



plt.subplot(1,2,2)

plt.title("k-NN - scikit learn")

sns.heatmap(sck_cm, annot = True, cmap="Greens",cbar=False);
# Predicting the classes of the test data by Logistic Regression

print("Logistic Regression\n")

c=logistic(X_train,y_train,test)

# print results

for i in range(len(c['Y_prediction_test'])):

    ct=set(map(transform,[c['Y_prediction_test'][i]]))

    print("Test point: "+str(test[i,:])+"  Label: "+str(c['Y_prediction_test'][i])+" "+str(ct))
# logistic regression from scratch

start=dt.datetime.now()

c=logistic(X_train,y_train,X_test)

# Print train/test Errors

print('Elapsed time of logistic regression from scratch: ',str(dt.datetime.now()-start))

print("train accuracy: {} %".format(100 - np.mean(np.abs(c["Y_prediction_train"] - y_train)) * 100))

print("test accuracy: {} %".format(100 - np.mean(np.abs(c["Y_prediction_test"] - y_test)) * 100))





# Logistic Regression from Neural Network Perspective

start=dt.datetime.now()

d = model(X_train, y_train, X_test, y_test)

print('\nElapsed time of Logistic Regression from Neural Network Perspective: ',str(dt.datetime.now()-start))

print("train accuracy: {} %".format(100 - np.mean(np.abs(d["Y_prediction_train"] - y_train)) * 100))

print("test accuracy: {} %".format(100 - np.mean(np.abs(d["Y_prediction_test"] - y_test)) * 100))





cm=confusion_matrix(y_test, c['Y_prediction_test'])



plt.figure(figsize=(15,6))

plt.suptitle("Confusion Matrixes",fontsize=24)



plt.subplot(1,2,1)

plt.title("Logistic Regression from Scratch")

sns.heatmap(cm, annot = True, cmap="Greens",cbar=False);



cm=confusion_matrix(y_test, d['Y_prediction_test'].reshape(30,))



plt.subplot(1,2,2)

plt.title("Logistic Regression from Neural Network Perspective")

sns.heatmap(cm, annot = True, cmap="Greens",cbar=False);
# Learning rates

lr = [0.1, 0.01, 0.001]



for i in range(len(lr)):

    # Run the model for different learning rates

    c = logistic(X_train,y_train,X_test, alpha = lr[i])

    

    # Adjust results to plot

    dfcost = pd.DataFrame(list(c['costs'])).transpose()

    dfcost.columns = ['0 (Iris-setosa) vs rest','1 (Iris-versicolor) vs rest','2 (Iris-virginica) vs rest']

    

    # Plot the costs

    if i==0 : f, axes = plt.subplots(1, 3,figsize=(24,4))

    sns.lineplot(data = dfcost.iloc[:, :3], ax=axes[i])

    sns.despine(right=True, offset=True)

    axes[i].set(xlabel='Iterations (hundreds)', ylabel='Cost ' +'(Learning Rate: ' + str(lr[i]) + ')')

    

plt.suptitle("Logistic Regression from Scratch\n",fontsize=24);  



for i in range(len(lr)):

    # Run the model for different learning rates

    d = model(X_train, y_train, X_test, y_test, learning_rate = lr[i])

    

    # Adjust results to plot

    dfcost = pd.DataFrame(list(d['costs'])).transpose()

    dfcost.columns = ['0 (Iris-setosa) vs rest','1 (Iris-versicolor) vs rest','2 (Iris-virginica) vs rest']

    

    # Plot the costs

    if i==0 : f, axes = plt.subplots(1, 3,figsize=(30,5))

    sns.lineplot(data = dfcost.iloc[:, :3], ax=axes[i])

    sns.despine(right=True, offset=True)

    axes[i].set(xlabel='Iterations (hundreds)', ylabel='Cost ' +'(Learning Rate: ' + str(lr[i]) + ')')

    

plt.suptitle("Logistic Regression from Neural Network Perspective\n",fontsize=24);    
# logistic regression from scratch

c=logistic(X_train,y_train,X_test)

cm=confusion_matrix(y_test, c['Y_prediction_test'])



# logistic regression - scikit learn

sck = LogisticRegression().fit(X_train, y_train)

sck_cm=confusion_matrix(y_test, sck.predict(X_test))



# logistic regression from scratch

c_r=logistic(X_train,y_train,X_test,lmd=0.01)

cm_r=confusion_matrix(y_test, c_r['Y_prediction_test'])



# logistic regression - scikit learn

sck_r = LogisticRegression(C=100).fit(X_train, y_train)

sck_cm_r=confusion_matrix(y_test, sck_r.predict(X_test))



plt.figure(figsize=(15,12))

plt.suptitle("Confusion Matrixes",fontsize=24)



plt.subplot(2,2,1)

plt.title("Logistic Regression from Scratch")

sns.heatmap(cm, annot = True, cmap="Greens",cbar=False);



plt.subplot(2,2,2)

plt.title("Logistic Regression - scikit learn")

sns.heatmap(sck_cm, annot = True, cmap="Greens",cbar=False);



plt.subplot(2,2,3)

plt.title("Logistic Regression from Scratch ( $\lambda$ = 0.01 )")

sns.heatmap(cm_r, annot = True, cmap="Greens",cbar=False);



plt.subplot(2,2,4)

plt.title("Logistic Regression ( $\lambda$ = 0.01 / C = 100 ) - scikit learn")

sns.heatmap(sck_cm_r, annot = True, cmap="Greens",cbar=False);
def cross_validation_split(dataset, folds):

        dataset_split = []

        df_copy = dataset

        fold_size = int(df_copy.shape[0] / folds)

        

        # for loop to save each fold

        for i in range(folds):

            fold = []

            # while loop to add elements to the folds

            while len(fold) < fold_size:

                # select a random element

                r = randrange(df_copy.shape[0])

                # determine the index of this element 

                index = df_copy.index[r]

                # save the randomly selected line 

                fold.append(df_copy.loc[index].values.tolist())

                # delete the randomly selected line from

                # dataframe not to select again

                df_copy = df_copy.drop(index)

            # save the fold     

            dataset_split.append(np.asarray(fold))

            

        return dataset_split 
def kfoldCV(dataset, f=5, k=5, model="logistic"):

    data=cross_validation_split(dataset,f)

    result=[]

    # determine training and test sets 

    for i in range(f):

        r = list(range(f))

        r.pop(i)

        for j in r :

            if j == r[0]:

                cv = data[j]

            else:    

                cv=np.concatenate((cv,data[j]), axis=0)

        

        # apply the selected model

        # default is logistic regression

        if model == "logistic":

            # default: alpha=0.1, num_iter=30000

            # if you change alpha or num_iter, adjust the below line         

            c = logistic(cv[:,0:4],cv[:,4],data[i][:,0:4])

            test = c['Y_prediction_test']

        elif model == "knn":

            test = kNN(cv[:,0:4],cv[:,4],data[i][:,0:4],k)

            

        # calculate accuracy    

        acc=(test == data[i][:,4]).sum()

        result.append(acc/len(test))

        

    return result
print("3-Fold Cross Validation for Logistic Regression from Scratch")

print("Fold Size:",int(df.shape[0] / 3))

seed(1)

acc=kfoldCV(df,3)

print("Accuricies:", acc)

print("Average of the Accuracy:", round(mean(acc),2))



print("\n3-Fold Cross Validation for k-NN from Scratch")

print("Fold Size:",int(df.shape[0] / 3))

seed(1)

acc=kfoldCV(df,3,model="knn")

print("Accuricies:", acc)

print("Average of the Accuracy:", round(mean(acc), 2))
seed(1)

bva_lr=[]

bva_knn=[]

for f in range(2,11):

    # k-fold cv from scratch for logistic regression

    bva_lr.append(mean(kfoldCV(df,f)))

    # k-fold cv from scratch for k-NN

    bva_knn.append(mean(kfoldCV(df,f,model="knn")))



# plot the change in the average accuracy according to k 

plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

plt.title("Logistic Regression")

plt.xlabel("Number of Folds (k)")

plt.ylabel("Average Accuracy")

plt.plot(range(2,11),bva_lr);



plt.subplot(1,2,2)

plt.title("k-NN")

plt.xlabel("Number of Folds (k)")

plt.ylabel("Average Accuracy")

plt.plot(range(2,11),bva_knn);
seed(1)

lr_scratch=kfoldCV(df,3)

knn_scratch=kfoldCV(df,3,model="knn")

lr_sck=cross_val_score(LogisticRegression(), all_X, all_y, cv=3)

knn_sck=cross_val_score(KNeighborsClassifier(n_neighbors = k), all_X, all_y, cv=3)



print("RESULTS")

print("Logistic Regression & k-Fold Cross Validation from Scratch: ",lr_scratch,"\nMean: ",round(mean(lr_scratch),2))

print("\nLogistic Regression & k-Fold Cross Validation (scikit-learn): ",lr_sck,"\nMean: ",round(mean(lr_sck),2))

print("\nk-NN & k-Fold Cross Validation from Scratch: ",knn_scratch,"\nMean: ",round(mean(knn_scratch),2))

print("\nk-NN & k-Fold Cross Validation (scikit-learn): ",knn_sck,"\nMean: ",round(mean(knn_sck),2))