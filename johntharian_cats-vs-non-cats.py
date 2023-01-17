import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import scipy
import cv2
train_dataset = h5py.File('../input/catvsnoncat/train_catvnoncat.h5', "r")
train_x_orig = np.array(train_dataset["train_set_x"][:]) 
train_y = np.array(train_dataset["train_set_y"][:]) 

test_dataset=h5py.File('../input/catvsnoncat/test_catvnoncat.h5', "r")
test_x_orig=np.array(test_dataset['test_set_x'][:])
test_y=np.array(test_dataset['test_set_y'][:])

classes = np.array(test_dataset["list_classes"][:])
# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print(f"This image is {classes[train_y[index]].decode('utf-8')}")
m_train=train_x_orig.shape[0]
num_px=train_x_orig.shape[1]
m_test=test_x_orig.shape[0]

print("Number of training smaples ",m_train)
print("Number of testing smaples ",m_test)

print("Each immage is of size ",num_px)

print("Shape of train samples ",train_x_orig.shape)
print("Shape of test samples",test_x_orig.shape)

print("Shape of Y of train", train_y.shape)
print("Shape of Y of test", test_y.shape)
test_y=test_y.reshape(1,50)
train_y=train_y.reshape(1,209)
# Flatten
train_x_flatten=train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten=test_x_orig.reshape(test_x_orig.shape[0],-1).T

train_x=train_x_flatten/255
test_x=test_x_flatten/255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
#helper functions
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
# initialize parameters
def initialize_with_zeros(dim):
    
    w=np.zeros((dim,1))
    b=0
    
    return w,b
# Propagation
def propagate(w,b,X,Y):
    
    m=X.shape[1]
    
#     Forward Propagation
    A=sigmoid(np.dot(w.T,X)+b)
    
    cost=-1/m * (np.sum(Y*np.log(A)+(1-Y)*np.log(1-A),axis=1,keepdims=True))
    
#     Backward propagation

    dw=1/m * (np.dot(X,(A-Y).T))
    db=1/m * (np.sum(A-Y))
    
    grads={
        "dw":dw,
        "db":db
    }
    
    return grads,cost
# OPtimization
def optimize(w,b,X,Y,num_iter,learning_rate,print_cost=False):
    
    costs=[]
    
    for i in range(num_iter):
        
        grads,cost=propagate(w,b,X,Y)
        
        dw=grads["dw"]
        db=grads["db"]
        
        w=w-learning_rate*dw
        b=b-learning_rate*db
  
        if i%100==0:
            costs.append(cost)
            
        if print_cost and i%100==0:
            print(f"Cost after iteration {i}:{cost}")
            
        params={
            "w":w,
            "b":b
        }
        
    return params,grads,costs
# predict
def predict(w,b,X):
    
    m=X.shape[1]
    
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    
    A=sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        Y_prediction[0,i]=np.where(A[0,i]>0.5,1,0)
        
    return Y_prediction
# model
def model(X_train,Y_train,X_test,Y_test,num_iter=2000,learning_rate=0.005,print_cost=False):
    
    w,b=initialize_with_zeros(X_train.shape[0])
    
    params,grads,costs=optimize(w,b,X_train,Y_train,num_iter,learning_rate,print_cost)
    
    w=params["w"]
    b=params["b"]
    
    Y_predicted_test=predict(w,b,X_test)
    Y_predicted_train=predict(w,b,X_train)
    
    print(f"train accurcay: {100-np.mean(np.abs(Y_predicted_train-Y_train))*100}")
    print(f"test accurcay: {100-np.mean(np.abs(Y_predicted_test-Y_test))*100}")
    
    d={
        "costs":costs,
        "Y_prediction_test": Y_predicted_test, 
        "Y_prediction_train" : Y_predicted_train, 
        "w" : w, 
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iter
    }
        
    return d
d=model(train_x,train_y,test_x,test_y,num_iter=2000,learning_rate=0.005,print_cost=True)
y_pred=d['Y_prediction_test']
lr=d['learning_rate']
# example of wrongly classified piccture
index=6
plt.imshow(test_x[:,index].reshape((num_px,num_px,3)))

y_pred=y_pred.reshape(50,1)

print(f"This picture is {classes[int(y_pred[index])].decode('utf-8')}")
# plot of cost function
costs=np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundred)')
plt.title(f"Learning Rate= {lr}")
plt.show()
# experimenting with the model
learning_rates = [0.01, 0.001, 0.0001, 0.5, 0.005]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_x, train_y, test_x, test_y, num_iter = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

image = Image.open('../input/catvsnoncat/cat1.jpg')
image=np.array(image.resize((num_px,num_px)))
my_image = image.reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")