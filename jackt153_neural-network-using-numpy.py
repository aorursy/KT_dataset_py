# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sn



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



def Graph_Confusion_Matrix(CM, labels):

    #np.fill_diagonal(CM,0)



    plt.figure(figsize = (8,8))

    sn.set(font_scale=1.4)#for label size

    sn.heatmap(CM, annot=True,annot_kws={"size": 16}, fmt='g'

               ,xticklabels = labels

               ,yticklabels = labels)# font size

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

#This randomise the trainning data (frac = 1 is sample of 100%)

df_train = df_train.sample(frac=1).reset_index(drop=True)



df_test = pd.read_csv("../input/test.csv")



y_ = df_train["label"].copy()

X_ = df_train.copy()

X_.drop(columns=["label"], inplace=True)



y_ = pd.get_dummies(y_)



#Pandas Dataframe to Numpy array

X_ = X_.values

y_ = y_.values



#Normalisation

X_norm = X_/255.0

    

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_, test_size=0.5, random_state=42)

fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize=(7,7))



count = 0

for i in range (0,9):

    

    ax[i//3, count].imshow(X_[i].reshape(28,28), cmap="gray")

    ax[i//3, count].set_yticks([])

    ax[i//3, count].set_xticks([])

    

    count = count +1

    if count == 3: count = 0
def Tanh(X):

    return(np.tanh(X))

    

def Diff_Tanh(X):

    return(1 - np.power(np.tanh(X),2)) 



def Softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)
def Forward(A_0, W_dict):

    

    cache = {}

    

    Z_1 = np.dot(W_dict.get("W_1"), A_0) + W_dict.get("B_1")

    A_1 = Tanh(Z_1)

    

    Z_2 = np.dot(W_dict.get("W_2"), A_1) + W_dict.get("B_2")

    A_2 = Softmax(Z_2)

    

    cache.update({"A_0":A_0})

    cache.update({"Z_1":Z_1})

    cache.update({"Z_2":Z_2})

    cache.update({"A_1":A_1})

    cache.update({"A_2":A_2})

    

    return(cache)
def Cost(Y_actual, Y_pred, mini_batch):

    

    Cost_list = []

    

    for i in range(0,mini_batch):

    

        Temp_Cost = np.sum(-1 * np.multiply(Y_actual[:,i].reshape(-1,1), np.log(Y_pred[:,i]).reshape(-1,1)))

    

        Cost_list.append(float(Temp_Cost))

    

    return(np.mean(Cost_list))
def Back_Prop(cache, Y_actual, m):



    dZ_2 = cache.get("A_2") - Y_actual

    dW_2 = np.multiply( (1.0/m) , np.dot(dZ_2, cache.get("A_1").T))

    dB_2 = np.multiply( (1.0/m) , np.sum(dZ_2, axis=1, keepdims=True))

    

    dZ_1 = np.dot(W_dict.get("W_2").T, dZ_2) * Diff_Tanh(cache.get("Z_1"))

    dW_1 = np.multiply( (1.0/m) , np.dot(dZ_1, cache.get("A_0").T))

    dB_1 = np.multiply( (1.0/m) , np.sum(dZ_1, axis=1, keepdims=True))



    cache.update({"dW_2":dW_2})

    cache.update({"dB_2":dB_2})

    cache.update({"dW_1":dW_1})

    cache.update({"dB_1":dB_1}) 

    

    return(cache)
def Grad_Descent(cache, W_dict, alpha):

    

    for i in range(1,3):

        W_dict.update({"W_"+str(i): W_dict.get("W_"+str(i)) - np.multiply(alpha, cache.get("dW_"+str(i)))}) 

    

    for i in range(1,3):

        W_dict.update({"B_"+str(i): W_dict.get("B_"+str(i)) - np.multiply(alpha, cache.get("dB_"+str(i)))})



    return(W_dict)

X_input = X_train.shape[1]

hidden_1 = 250

output = 10

alpha = 0.1



#Randomly initialising the weights

W_1 = np.random.randn(hidden_1, X_input)*0.01

B_1 = np.zeros([hidden_1,1])



W_2 = np.random.randn(output, hidden_1)*0.01

B_2 = np.zeros([output,1])



W_dict = {"W_1": W_1,

          "W_2": W_2,

          "B_1": B_1,

          "B_2": B_2}


batch_size = 128

n_epochs = 35

batchnumber = 0



Cost_Overall = []



for epoch in range(n_epochs):

        Cost_Tr_List = []

        Cost_Val_List = []

        Cost_Val_Acc = []

        #epoch_start_time = time.time()

        for i in range(len(y_train)// batch_size):

            batchnumber = batchnumber+1

            

            batch_start_idx = (i * batch_size) % (X_train.shape[0] - batch_size)

            batch_end_idx = batch_start_idx + batch_size

            batch_X = X_train[batch_start_idx:batch_end_idx]

            batch_Y = y_train[batch_start_idx:batch_end_idx]

            

            cache = Forward(batch_X.T, W_dict)

            

            Cost_Tr_List.append(Cost(batch_Y.T, cache.get("A_2"), batch_size))

            

            cache = Back_Prop(cache, batch_Y.T, batch_size)

            

            W_dict = Grad_Descent(cache, W_dict, alpha)

            

        print("Cost at epoch" +str(epoch)+ " Cost:"+str(np.mean(Cost_Tr_List)))

        Cost_Overall.append(np.mean(Cost_Tr_List))
plt.figure(figsize=(7,5))            

plt.plot(Cost_Overall)

plt.xlabel("Epoch")

plt.ylabel("Cost")

plt.title("Training Set Cost")

plt.show()
cache = Forward(X_test.T, W_dict)



Y_pred = np.argmax(cache.get("A_2"),axis=0)

Y_actual = np.argmax(y_test,axis=1)



CM = confusion_matrix(Y_actual, Y_pred)

Graph_Confusion_Matrix(CM, list(np.linspace(0,9,10)))

print("Accuary:", accuracy_score(Y_actual, Y_pred)*100)


for i in range(0,5):



    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5))



    r_int = np.random.randint(0,X_test.shape[0],1)



    ax[0].imshow(X_test[r_int].reshape(28,28), cmap="gray")

    ax[0].set_yticks([])

    ax[0].set_xticks([])

    ax[0].set_title(int(Y_pred[r_int]),size=20)



    ax[1].barh(np.linspace(0,9,10).astype(str), cache.get("A_2")[:,r_int].reshape(-1))

    ax[1].set_xlim(0,1)

    ax[1].set_title("Distribution of the Predicted Labels")

    ax[1].set_xlabel("Probability")