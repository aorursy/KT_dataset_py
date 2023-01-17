import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train_data.head()
train_labels = train_data.pop("label")

train_data
# train_X=COMPLETE_X[:5000]

# train_Y=(COMPLETE_Y[:5000])
import matplotlib.pyplot as plt

%matplotlib inline
def show_example(i):

    image = train_data.values

    temp=image[i].reshape((28,28))

    plt.imshow(temp,cmap='gray')

    plt.show()

    

    print("Label : " + str(train_labels[i]))

    

show_example(2)

show_example(23)
def cleanXY(X,Y=None):

    X=np.matrix(X.values,dtype=np.float128)

    if Y is not None:

        Y=pd.get_dummies(Y)

        Y=np.matrix(Y.values,dtype=np.float128)

    return X,Y
def normalize(x):

    mean_X=np.sum(x,axis=0).reshape(1,-1)/x.shape[0]

    sd_X=np.sum(np.square(x),axis=0).reshape(1,-1)/x.shape[0]

    norm_x=(x-mean_X.reshape(1,-1)  )/(np.sqrt(sd_X)+(10**-5)).reshape(1,-1)



    return norm_x
print(train_data.shape)

print(train_labels.shape)
def init_params(*args):

    np.random.seed(3)

    params={}

    for i in range(len(args)-1):

        params[f"W{str(i+1)}"]= np.random.randn(args[i],args[i+1])*0.01#np.sqrt(1/n_0) # Xaviers init

        params[f"W{str(i+1)}"]=np.array(params[f"W{str(i+1)}"],dtype=np.float128)

    

        params[f"b{str(i+1)}"]=np.zeros((args[i+1],1))

        params[f"b{str(i+1)}"]=np.array(params[f"b{str(i+1)}"],dtype=np.float128)

       

    return params
#test init_params

params=init_params(10,5)

display([k+":"+str(x.shape) for k,x in params.items()])
def softmax(z):

    z_exp = np.exp(z)

    sum_z_exp = np.sum(z_exp,axis=1).reshape(-1,1)

    lst=z_exp/sum_z_exp

    return np.array(lst,dtype=np.float128)
'''Cost Function for 2 layer neural network'''

def cost(Y_pred,Y):

    m=Y.shape[0]

    logprobs = np.multiply(Y,np.log(Y_pred),dtype=np.float128)

    cost = - (1/m)*((logprobs))     

    return np.sum(cost)
#Test cost

A2=softmax(np.matrix([[0.02,0.23,0.94,0.001],[0.02,0.23,0.94,0.001],[0.02,0.23,0.94,0.001]]))



Y= np.matrix([[0, 1, 0, 0],[0, 1, 0, 0],[0, 1, 0, 0]])

print(A2.shape)

print(Y.shape)

print(cost(A2,Y))
def forward_propogation(X,weights,num_of_layers):



    m=X.shape[0]



    cache_params={"A0":X}

    for i in range(num_of_layers-1):

        cache_params["Z"+str(i+1)]=np.dot(cache_params["A"+str(i)],weights["W"+str(i+1)])+weights["b"+str(i+1)].T        

        cache_params["A"+str(i+1)]=np.tanh(cache_params["Z"+str(i+1)])



        

    cache_params["Z"+str(num_of_layers)]=np.dot(cache_params["A"+str(num_of_layers-1)],weights["W"+str(num_of_layers)])+weights["b"+str(num_of_layers)].T        

    cache_params["A"+str(num_of_layers)]=softmax(cache_params["Z"+str(num_of_layers)])

    

    return cache_params["A"+str(num_of_layers)],cache_params
def back_propogation(Y_pre,Y,X,cache_params,params,num_of_layers,learning_rate=0.1):    

    m = Y.shape[0]

    updated_params={}

    

    updated_params["dZ"+str(num_of_layers)]= Y_pre-Y

    updated_params["dW"+str(num_of_layers)]=(1/m)*(np.dot(updated_params["dZ"+str(num_of_layers)].T,cache_params["A"+str(num_of_layers-1)])).T

    updated_params["db"+str(num_of_layers)]=(1/m)*np.sum(updated_params["dZ"+str(num_of_layers)].T,axis=1)

    

    for x in range(num_of_layers-1,0,-1):

    

        derivative=np.dot(params["W"+str(x+1)],updated_params["dZ"+str(x+1)].T).T

        temp2=cache_params["Z"+str(x)]

        val1= 1/np.square(np.cosh(temp2))

        updated_params["dZ"+str(x)]=np.multiply(derivative,val1) 

    

        updated_params["dW"+str(x)]=(1/m)*(np.dot(updated_params["dZ"+str(x)].T,cache_params["A"+str(x-1)])).T

        updated_params["db"+str(x)]=(1/m)*np.sum(updated_params["dZ"+str(x)].T,axis=1)

    

    for i in range(1,num_of_layers):

        params["W"+str(i)]=params["W"+str(i)]-learning_rate* updated_params["dW"+str(i)]

        params["b"+str(i)]=params["b"+str(i)]-learning_rate* updated_params["db"+str(i)]

    return params
def show_accuracy(Y_expanded,predictions):

    flags=np.argmax(Y_expanded,axis=1).reshape(1,-1)

    preds=np.argmax(predictions,axis=1)

# print(preds)





    success=(flags==preds).sum()/len(preds)



    print("Accuracy = " +str(success*100)+"%")
def train(X,Y,mini_batch_size=512,number_of_iter=100,hidden_layers=[]):

    layer_dims=[X.shape[1]]+hidden_layers

    layer_dims.append(Y.shape[1])

    params=init_params(*layer_dims)

    num_of_layers=len(layer_dims)-1

    num_mini_batches = np.math.ceil(X.shape[0]/mini_batch_size)



    cost_tracker = []

    for mini_batch in range(num_mini_batches):

        mini_X=normalize(X[mini_batch*mini_batch_size :(mini_batch+1)*mini_batch_size ])

        mini_Y=Y[mini_batch*mini_batch_size :(mini_batch+1)*mini_batch_size]

        mini_cost_tracker=[0]



        decay_rate = 0.001 #((decay_rate)/2)

        if mini_batch==num_mini_batches*0.95:

            decay_rate=0.00001

        print(f"BATCH : {mini_batch+1}",)

        for i in range(number_of_iter):

            

            

            A3,cache_params=forward_propogation(mini_X,params,num_of_layers)

            if i%10==0:

                print(f"Training : {i+1}",end=" ")

                my_cost=cost(A3,mini_Y)

                print("COST : " + str(my_cost) +" Change : "+ str(my_cost-mini_cost_tracker[-1]),end=" ")

                show_accuracy(mini_Y,A3)    

            

            mini_cost_tracker.append(my_cost)



            params = back_propogation(A3,mini_Y,mini_X,cache_params,params,num_of_layers,decay_rate)

            

            if i==number_of_iter*0.8:

                decay_rate=0.0008

                

            if i==number_of_iter*0.8:

                decay_rate=0.0001

        

        cost_tracker.append((mini_batch,np.average(mini_cost_tracker)))  

    predictions,cache_params=forward_propogation(normalize(X),params,num_of_layers)

    my_cost=cost(predictions,Y)

    print("Final COST : " + str(my_cost),end=" " )

    show_accuracy(Y,predictions)

    return params,cost_tracker
np.seterr(all='raise')

my_x,my_y= cleanXY(train_data,train_labels)

my_x_norm= normalize(my_x)



%time params,cost_tracker=train(my_x,my_y,32,40,[128,64])



print("DONE")
plt.plot([x for x,y in cost_tracker],[y for x,y in cost_tracker])
i=28



predictions,cache_params=forward_propogation(my_x_norm[i][0].reshape(1,-1),params,3)

print(f"Number is {np.argmax(my_y[i])} ")

# print(predictions)



print("Prediction : " +str(np.argmax(predictions)) )

# print(np.argmax(predictions,axis=1))
my_test_x,_=cleanXY(test_data)

my_norm_test_x=normalize(my_test_x)
predictions,cache_params=forward_propogation(my_norm_test_x,params,3)

temp=(pd.DataFrame(np.argmax(predictions,axis=1)))

soln=pd.DataFrame(temp)

soln.index+=1

display(soln)
x=soln.to_csv("/kaggle/working/submission.csv",header=["Label"],index=True,index_label=["ImageId"])
weights.x=np.save("/kaggle/working/weights.npy",params)