import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
train_data = np.genfromtxt("../input/digit-recognizer/train.csv",delimiter=',')

test_data = np.genfromtxt("../input/digit-recognizer/test.csv",delimiter=',')

sample_data = np.genfromtxt("../input/digit-recognizer/sample_submission.csv",delimiter=',')
print("train_data= " , train_data.shape)

print("test_data= " , test_data.shape)

print("sample_data= " , sample_data.shape)
print("train_data= \n",train_data,"\n")

print("test_data= \n",test_data,"\n")

print("sample_data= \n",sample_data,"\n")
train_x = train_data[1::,1::]

train_y = train_data[1::,0]
new_y=[]

for i in train_y:

    a=np.zeros(10)

    a[int(i)]=1

    new_y.append(a)

train_y=np.array(new_y)
print("train_x= ",train_x.shape)

print("train_y= ",train_y.shape)
evalu_x=train_x[:4200]

train_x=train_x[4200:]



evalu_y=train_y[:4200]

train_y=train_y[4200:]



print("train_x= ",train_x.shape)

print("train_y= ",train_y.shape)

print("eval_x= ",evalu_x.shape)

print("eval_y= ",evalu_y.shape)
test_data=test_data[1::]

sample_data=sample_data[1::]
image_to_show=12345 #We have 42000 images, so choose this value between 42000 and -42000 



print("Digit= ",train_y[image_to_show])

plt.imshow(train_x[image_to_show].reshape(28,28))

plt.show()
def mean_squared_error(y, y_hat):

    return np.mean(np.power(y-y_hat,2))



def mean_squared_error_der(y,y_hat):

    return (y-y_hat)
def sigmoid(x):

    return 1/(1+np.exp(-x))



def sigmoid_der(x):

    return sigmoid(x)*(1-sigmoid(x))

    
class dense_layer:

    update_count=0

    delta=0

    

    def __init__(self,input_shape,output_shape,is_last):

        self.input_shape=input_shape

        self.output_shape=output_shape

        self.is_last=is_last

        self.weights=np.random.random((self.input_shape,self.output_shape))-0.5

        self.vw=np.zeros((self.input_shape,self.output_shape))

        self.sw=np.zeros((self.input_shape,self.output_shape))

        

    def feed_forward(self,x):

        self.input_values=x

        self.output = sigmoid(np.dot(self.input_values,self.weights))

        return self.output

    

    def backprop(self,expected=0,next_layer_gamma=0,next_layer_weights=0):

        if self.is_last:

            self.error=mean_squared_error_der(expected,self.output)

        else:

            self.error=np.dot(next_layer_gamma,next_layer_weights.T)

        

        self.gamma=self.error*sigmoid_der(self.output)

          

        self.delta+=np.dot(self.input_values.T,self.gamma)

    

    def update_weights(self):

        self.update_count+=1

        

        self.sw=self.sw*beta1+self.delta*(1-beta1)

        self.vw=self.vw*beta2+self.delta**2*(1-beta2)

        

        swc=self.sw/(1-beta1**self.update_count)

        vwc=self.vw/(1-beta2**self.update_count)

        

        self.weights+=swc/(np.power(vwc,1/2)+epsilon)*lr

        

        self.delta=0
l1=dense_layer(784,392,False)

l2=dense_layer(392,98,False)

l3=dense_layer(98,49,False)

l4=dense_layer(49,10,True)
beta1=0.99

beta2=0.999

epsilon=000000000000.1

lr=0.001



batch_size=512

epochs=30
def fit(x,y):

    error_list=[]

    for ep in range(epochs):

        seen_points=0

        error=0

        

        for i in range(x.shape[0]):

            

            o1=l1.feed_forward(x[i].reshape(1,-1))

            o2=l2.feed_forward(o1)

            o3=l3.feed_forward(o2)

            o4=l4.feed_forward(o3)

            

            l4.backprop(y[i])

            l3.backprop(next_layer_gamma=l4.gamma,next_layer_weights=l4.weights)

            l2.backprop(next_layer_gamma=l3.gamma,next_layer_weights=l3.weights)

            l1.backprop(next_layer_gamma=l2.gamma,next_layer_weights=l2.weights)

            

            error+=np.mean(l4.error**2)

            

            if seen_points%batch_size==0:

                l4.update_weights()

                l3.update_weights()

                l2.update_weights()

                l1.update_weights()

                

                #print("Epochs: ",ep+1,"/",epochs," - Batches: ", i+1,"/",x.shape[0])

            

            seen_points+=1

            

        error=error/x.shape[0]/batch_size

        

        error_list.append(error)

        

        print("Epochs: ",ep+1,"/",epochs," - Error: ", error)

        

    return error_list
history=fit(train_x/train_x.max(),train_y)
print("Error")

plt.plot(history)

plt.show()
def prediction(x):

    o1=l1.feed_forward(x)

    o2=l2.feed_forward(o1)

    o3=l3.feed_forward(o2)

    o4=l4.feed_forward(o3)

    return o4



def evaluate(x,y):

    true=0

    for i in range(x.shape[0]):

        if np.argmax(prediction(x[i]))==np.argmax(y[i]):

            true+=1

    return true,x.shape[0]
true,total=evaluate(evalu_x/evalu_x.max(),evalu_y)

print(true/total)
show_id=300



plt.imshow(train_x[show_id].reshape(28,28))

print("Prediction= ",np.argmax(prediction(train_x[show_id].reshape(1,-1)/train_x.max())))

print("Real= ",np.argmax(train_y[show_id]))
submission=pd.DataFrame()



preds=[]

for i in range(test_data.shape[0]):

    preds.append(np.argmax(prediction(test_data[i]/test_data.max())))



submission["ImageId"]=np.arange(len(preds))+1

submission["Label"]=preds



submission
submission.to_csv("submission.csv",index=False)