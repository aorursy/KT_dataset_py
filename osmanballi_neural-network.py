

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.patches as patches

from PIL import Image

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/face-mask-detection-dataset/train.csv")

print(train.shape[0])

train.head()

train.sort_values("name")
ax = sns.catplot(x='classname',kind='count',data=train,orient="h",height=10,aspect=1)

ax.fig.suptitle('Count of Classnames',fontsize=16,color="r")

ax.fig.autofmt_xdate()
def img_reg(id):

    train_name=train["name"][id]

    classname=train[train["name"]==train_name]["classname"]

    reg=train[train["name"]==train_name][["x1","x2","y1","y2"]]

    x1=reg["x1"][id]

    x2=reg["x2"][id]

    y1=reg["y1"][id]

    y2=reg["y2"][id]

    classname=classname[id]

    images = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

    img=plt.imread(os.path.join(images,train_name))

    return x1,x2,y1,y2,classname,img





# Create suubplot

fig,[ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15,15))



#first image



x1,x2,y1,y2,classname,img=img_reg(14835)

ax1.imshow(img)

rect = patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2,edgecolor='r',facecolor='none')

ax1.set_title(classname)

#second image



x1,x2,y1,y2,classname,img=img_reg(14836)

ax2.imshow(img)

rect2 = patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2,edgecolor='r',facecolor='none')

ax2.set_title(classname)



# Add the patch

ax1.add_patch(rect)

ax2.add_patch(rect2)



plt.show()
submission = pd.read_csv("/kaggle/input/face-mask-detection-dataset/submission.csv")

print(submission.shape[0])

submission.head()

submission.sort_values("name")
len(os.listdir("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"))
face_with_mask=train[train["classname"]=="face_with_mask"]

face_no_mask=train[train["classname"]=="face_no_mask"]

print("count of face with mask: "+str(len(face_with_mask))+"\ncount of face no mask: "+str(len(face_no_mask)))
import cv2



name=face_with_mask.iloc[0]["name"]

images = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

img=plt.imread(os.path.join(images,name))

resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)

face_with_flatten=resized.flatten().reshape(-1,1)



name=face_with_mask["name"]

for i in name:

    images = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

    img=plt.imread(os.path.join(images,i))

    resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)

    z=resized.flatten().reshape(-1,1)

    face_with_flatten=np.append(face_with_flatten,z,axis=1)



print(face_with_flatten.shape)


name=face_no_mask.iloc[0]["name"]

images = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

img=plt.imread(os.path.join(images,name))

resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)

face_no_flatten=resized.flatten().reshape(-1,1)



name=face_no_mask["name"]

for i in name:

    images = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

    img=plt.imread(os.path.join(images,i))

    resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)

    z=resized.flatten().reshape(-1,1)

    face_no_flatten=np.append(face_no_flatten,z,axis=1)



print(face_no_flatten.shape)
face_with_flatten=face_with_flatten/255.

face_no_flatten=face_no_flatten/255.



Train_x=np.append(face_with_flatten[:,1:1401],face_no_flatten[:,1:1401],axis=1)

Train_y=np.append(np.ones(1400),np.zeros(1400),axis=0).reshape(-1,1).T

Test_x=np.append(face_with_flatten[:,1401:1570],face_no_flatten[:,1401:1570],axis=1)

Test_y=np.append(np.ones(169),np.zeros(169),axis=0).reshape(-1,1).T



print(Train_x.shape)

print(Train_y.shape)

print(Test_x.shape)

print(Test_y.shape)
#sigmoid function

def sigmoid(z):

    

    s = 1/(1+np.exp(-z))  

    return s
def initialize_with_zeros(dim):

    

    w = np.zeros((dim,1))

    b = 0



    assert(w.shape == (dim, 1))

    assert(isinstance(b, float) or isinstance(b, int))

    

    return w, b
def propagate(w, b, X, Y):

    """

    Arguments:

    w -- weights, a numpy array of size (num_px * num_px * 3, 1)

    b -- bias, a scalar

    X -- data of size (num_px * num_px * 3, number of examples)

    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)



    Return:

    cost -- negative log-likelihood cost for logistic regression

    dw -- gradient of the loss with respect to w, thus same shape as w

    db -- gradient of the loss with respect to b, thus same shape as b

    """   

    

    m = X.shape[1]

    

    # FORWARD PROPAGATION



    A = sigmoid(np.dot(w.T, X)+ b)                                

    cost = -1/m*np.sum(Y*np.log(A)+ (1-Y)*np.log(1-A))                             



    

    # BACKWARD PROPAGATION



    dw = 1/m*(np.dot(X, (A-Y).T))

    db = 1/m*np.sum(A-Y)





    assert(dw.shape == w.shape)

    assert(db.dtype == float)

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    

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



        w = w - learning_rate*dw

        b = b - learning_rate*db



        

        # Record the costs

        if i % 100 == 0:

            costs.append(cost)

        

        # Print the cost every 100 training iterations

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    params = {"w": w,

              "b": b}

    

    grads = {"dw": dw,

             "db": db}

    

    return params, grads, costs
def predict(w, b, X):



    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    # Compute vector "A" predicting the probabilities of a cat being present in the picture



    A = sigmoid(np.dot(w.T, X)+ b) 



    

    for i in range(A.shape[1]):

        

        if A[0,i]< 0.5:

            Y_prediction[0,i] = 0

        else: 

            Y_prediction[0,i] = 1

    

    assert(Y_prediction.shape == (1, m))

    

    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):





    w, b = initialize_with_zeros(X_train.shape[0])



    # Gradient descent 

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)

    

    # Retrieve parameters w and b from dictionary "parameters"

    w = parameters["w"]

    b = parameters["b"]

    

    # Predict test/train set examples 

    Y_prediction_test = predict(w, b, X_test)

    Y_prediction_train = predict(w, b, X_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))



    

    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
d = model(Train_x, Train_y, Test_x, Test_y, num_iterations = 2000, learning_rate = 0.01, print_cost = True)
a=Test_x[:,13]*255

plt.imshow(a.reshape((64, 64, 3)))

print(d["Y_prediction_test"][0,13])
a=Test_x[:,95]*255

plt.imshow(a.reshape((64, 64, 3)))

print(d["Y_prediction_test"][0,95])
a=Test_x[:,300]

plt.imshow(a.reshape((64, 64, 3)))

print(d["Y_prediction_test"][0,300])
# Plot learning curve (with costs)

costs = np.squeeze(d['costs'])

plt.figure(figsize=(15,10))

plt.plot(costs)

plt.grid(color='r', linestyle='dotted', linewidth=0.5)

plt.ylabel('cost')

plt.xlabel('iterations (per hundreds)')

plt.title("Learning rate =" + str(d["learning_rate"]))

plt.show()