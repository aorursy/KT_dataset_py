import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn. feature_selection import SelectKBest,mutual_info_classif
from sklearn.neural_network import MLPClassifier
df = pd.read_csv("../input/health-care-data-set-on-heart-attack-possibility/heart.csv")
df.head(20)
df = df.loc[(df['trestbps'] >=100) & (df['trestbps'] <=180)]
df2 = df['age'].replace(np.arange(0,38),value = 0)
df2 = df2.replace(np.arange(38,53),value = 1)
df2 = df2.replace(np.arange(53,67),value = 2)
df2 = df2.replace(np.arange(67,100),value = 3)
df['age'] = df2
df2 = df['trestbps'].replace(np.arange(100,118),0)
df2 = df2.replace(np.arange(118,143),1)
df2 = df2.replace(np.arange(143,159),2)
df2 = df2.replace(np.arange(159,180),3)
df['trestbps'] = df2
temp = df['trestbps'].map(str) + '_' +df['exang'].map(str) 
encoder = LabelEncoder()
temp = encoder.fit_transform(temp)
df['exang_trestbps'] = temp
df.describe()
corr = df.corr("pearson")
corrs = []
for i in df.columns:
    temp = []
    for j in df.columns:
        temp.append(corr[i][j])
    corrs.append(np.array(temp))
corrs = np.array(corrs)
fig,ax = plt.subplots(1,1,figsize = (12,10))
sns.heatmap(corrs,annot = True,xticklabels = corr.keys(),yticklabels = corr.keys(),ax = ax)
class Logistic_Regression:
    def __init__(self,epochs,learning_rate,beta,regularization,err_thr,func):
        self.__epochs = epochs
        self.__alpha = learning_rate
        self.__err_thr = err_thr
        self.__act = func
        self.__rparam = regularization 
        self.__parameters = {}
        self.__beta = beta
        self.__losses = []
        self.__val_losses = []
    
    def __initialize_parameters(self,x_h):
        np.random.seed(1)
        self.__parameters['W'] = np.random.randn(1,x_h) * 0.01
        self.__parameters['b'] = 0
        self.__parameters['Vdw'] = np.zeros((1,x_h))
        self.__parameters['Vdb'] = 0
    
    def __compute_cost(self,A,Y):
        m = Y.shape[1]
        temp = (self.__rparam/2) * np.matmul(self.__parameters['W'],self.__parameters['W'].T) / m
        val = np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))) * -1/m
        val += temp[0,0]
        
        
        return val
        
    def __activation(self,X):
        val = 0
        if(self.__act == "sigmoid"):
            t = 1 + np.exp(-1 * X)
            val = 1/t
        elif(self.__act == "relu"):
            val = np.maximum(0,X)
        else:
            val = np.tanh(X)
        return val

    def __activation_derivative(self,X):
        val = 0
        if(self.__act == "sigmoid"):
            val = np.multiply(X,1-X)
        elif self.__act == "relu":
            val = X>0
        else:
            val = 1 - X**2
        return val
    
    def __linear_forward(self,X):
        Z = np.matmul(self.__parameters['W'],X) + self.__parameters['b']
        A = self.__activation(Z)
        
        return A
    def __linear_backward(self,A,X,Y):
        m = X.shape[0]
        
        dZ = A-Y
        
        dW = (self.__rparam * self.__parameters['W']) + (np.matmul(dZ,X.T) * 1/m)
        db = np.sum(dZ,axis = 1) * 1/m
        Vdw = (self.__beta * self.__parameters['Vdw']) + (1 - self.__beta)*dW
        Vdb = (self.__beta * self.__parameters['Vdb']) + (1 - self.__beta)*db
        
        grads = {'Vdw':Vdw,'Vdb':Vdb}
        return grads
    
    def __update_parameters(self,grads):
        self.__parameters['W'] -= self.__alpha * grads['Vdw']
        self.__parameters['b'] -= self.__alpha * grads['Vdb']
        self.__parameters['Vdw'] = grads['Vdw']
        self.__parameters['Vdb'] = grads['Vdb']
    
    def fit(self,X,Y,X_val,Y_val,print_cost = True):
        self.__initialize_parameters(X.shape[0])
        #print(self.__parameters)
        
        for i in range(self.__epochs):
            A = self.__linear_forward(X)
            
            cost = self.__compute_cost(A,Y)
            self.__losses.append(cost)
            Aval = self.__linear_forward(X_val)
            val_cost = self.__compute_cost(Aval,Y_val)
            self.__val_losses.append(val_cost)
            if(cost < self.__err_thr):
                print("error threshold reached")
                break
            grads = self.__linear_backward(A,X,Y)
            
            self.__update_parameters(grads)
            
            if(print_cost):
                print("epoch {0}: loss-> {1},validation loss-> {2}".format(i,cost,val_cost))
    
    def training_losses(self):
        return self.__losses
    
    def validation_losses(self):
        return self.__val_losses
    
    def predict(self,X,chance = False):
        A = self.__linear_forward(X)
        
        pred = [int(i>0.5) for i in A[0]]
        if(not chance):
            return np.array(pred)
        else:
            return np.array(pred),A[0]
    
    def confusion_matrix(self,X,Y,classes,show_table = True):
        pred = self.predict(X)    
        cmat = np.zeros((classes,classes))
        print(cmat.shape)
        for j in range(len(pred)):
            cmat[pred[j]][Y[j]]+=1
    
        if(show_table):
            fig,ax = plt.subplots(1,1,figsize = (10,8))
            sns.heatmap(cmat,annot = True,xticklabels = range(classes),yticklabels = range(classes),ax = ax);
        else:
            return cmat
class MLP:
    def __init__(self,layer_dims,func,loss_type,epochs,
                 regularization = 0.001,learning_rate = 0.1,beta = 0.0):
        self.__layer_dims = layer_dims
        self.__layers = len(layer_dims) - 1
        self.__func = func
        self.__epochs = epochs
        self.__alpha = learning_rate
        self.__beta = beta
        self.__rparam = regularization
        self.__loss_type = loss_type
        self.__parameters = {}
        self.__losses = []
        self._val_losses = []
    
    def __initialize_parameters(self):
    
        for i in range(1,self.__layers+1):
            self.__parameters['W'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
            self.__parameters['b'+str(i)] = np.zeros((layer_dims[i],1))
            self.__parameters['VdW' + str(i)] = np.zeros((layer_dims[i],1))
            self.__parameters['Vdb' + str(i)] = np.zeros((layer_dims[i],1))
            #print(parameters['W'+str(i)].shape,",",parameters['b'+str(i)].shape )
    
    def __compute_cost(self,A,Y):
        loss = 0
        
        temp = 0
        for i in range(1,self.__layers+1):
            temp += (self.__rparam /2) * np.sum(np.sum(self.__parameters['W' + str(i)] ** 2, 
                                                       axis = 1, keepdims = True))
        m = Y.shape[1]
        if(self.__loss_type == "logistic"):
            temp = temp / m
            loss = np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),(1-Y))) * -1/m
            loss += temp
        elif(self.__loss_type == "cross entropy"):
            loss = np.sum(np.sum(np.multiply(np.log(A),Y),axis = 1,keepdims = True)) * -1/m
            loss += temp/m
        elif(self.__loss_type == "mse"):
            loss = np.sum((Y-A)**2) * 1/m
            loss += temp/m
        return loss
    
    def __activation(self,X,func = "tanh"):
        val = 0
        if(func == "sigmoid"):
            t = 1 + np.exp(-1 * X)
            val = 1/t
        elif(func == "softmax"):
            t1 = np.sum(np.exp(X),axis = 0,keepdims = True)
            val = np.exp(X) / t1
        elif(func == "relu"):
            val = np.maximum(0,X)
        else:
            val = np.tanh(X)
        return val

    def __activation_derivative(self,X,func = "tanh"):
        val = 0
        if(func == "sigmoid"):
            val = np.multiply(X,1-X)
        elif func == "relu":
            val = X>0
        else:
            val = 1 - X**2
        return val
    
    def __forward_propagation(self,X):
        cache = {}
        fin = X
        for i in range(1,self.__layers+1):
            zi = np.matmul(self.__parameters["W"+str(i)],fin) + self.__parameters["b" + str(i)]
            ai = self.__activation(zi,self.__func[i-1])
            fin = ai
            cache["Z" + str(i)] = zi
            cache["A" + str(i)] = ai
        cache["A0"] = X
    
        return fin,cache
    
    def __backward_propagation(self,Y,A,cache):
        m = Y.shape[1]
        grads = {}
    
        dA = -1 * (np.divide(Y,A) - np.divide((1-Y),(1-A)))
    
        for i in range(self.__layers,0,-1):
            if(self.__func[i-1] == "softmax"):
                dzi = A-Y
            else:
                dzi = np.multiply(dA,self.__activation_derivative(cache["A" + str(i)],self.__func[i-1]))
        
            dwi = (self.__rparam * self.__parameters['W'+str(i)]) + np.matmul(dzi,cache["A" + str(i-1)].T) * 1/m 
            Vdwi = self.__beta * (self.__parameters['VdW' + str(i)]) + (1-self.__beta) * dwi
            dbi = np.sum(dzi,axis = 1,keepdims = True) * 1/m
            Vdbi = self.__beta * (self.__parameters['Vdb' + str(i)]) + (1-self.__beta) * dbi

        
            dA = np.matmul(self.__parameters["W" + str(i)].T,dzi)
        
            grads["VdW" + str(i)] = Vdwi
            grads["Vdb" + str(i)] = Vdbi
    
        return grads
    
    def __update_parameters(self,grads):
        for i in range(1,self.__layers+1):
            self.__parameters["W" + str(i)] -= (self.__alpha * grads["VdW" + str(i)]) 
            self.__parameters["b" + str(i)] -= (self.__alpha * grads["Vdb" + str(i)])
            self.__parameters['VdW' + str(i)] = grads["VdW" + str(i)]
            self.__parameters['Vdb' + str(i)] = grads["Vdb" + str(i)]
            
    def fit (self,X,Y,X_val,Y_val,print_cost = True):
        self.__initialize_parameters()
        
    
        for i in range(self.__epochs):
            A,cache = self.__forward_propagation(X)
            Aval,val_cache = self.__forward_propagation(X_val)
        
            cost = self.__compute_cost(A,Y)
            val_cost = self.__compute_cost(Aval,Y_val)
            self.__losses.append(cost)
            self._val_losses.append(val_cost)
        
            grads = self.__backward_propagation(Y,A,cache)
        
            self.__update_parameters(grads)
        
            if(print_cost and i%1000 == 0): 
                print("epoch {0}: loss {1},validation_loss {2}".format(i,cost,val_cost))
    
    def Weights(self):
        return self.__parameters
    
    def predict(self,X,chances = False):
        A,cache = self.__forward_propagation(X)

        pred = []
        if(func[self.__layers-1] == "softmax"):
            pred = [int(a[1] > a[0]) for a in A.T]
        else:
            pred = [int(i>0.5) for i in A[0]]
        if(chances):
            return np.array(pred),A
        return np.array(pred)
    
    def training_losses(self):
        return self.__losses
    def validation_losses(self):
        return self._val_losses
    def __to_numeric(self,Y):
        Y_n = []
        for i in Y:
            Y_n.append(np.argmax(i))
        return Y_n
    
    def confusion_matrix(self,X,Y,classes,show_table = True):
        pred = self.predict(X)
        Y_n = Y[0]
        if(Y.shape[0] > 1):
            Y_n = self.__to_numeric(Y.T)
    
        cmat = np.zeros((classes,classes))
        for j in range(len(pred)):
            cmat[pred[j]][Y_n[j]]+=1
    
        if(show_table):
            fig,ax = plt.subplots(1,1,figsize = (6,4))
            sns.heatmap(cmat,annot = True,xticklabels = range(classes),yticklabels = range(classes),ax = ax);
        else:
            return cmat
    def score(self,X,Y):
        cmat = self.confusion_matrix(X,Y,2,False)

        recall = cmat[1,1] / (cmat[1,1] + cmat[1,0])
        precision = cmat[1,1] / (cmat[0,1] + cmat[1,1])
        f1 = (2*recall*precision) / (recall + precision)
        print(f1,precision,recall)
def heart_disease_classifier(X_orig,Y_orig,t_size,layer_dims,learning_rate,C,epochs,beta,func):
    
    
    X = (X_orig - np.mean(X_orig,axis = 0,keepdims = True))/np.std(X_orig,axis= 0,keepdims = True)
    Y = np.zeros((Y_orig.shape[1],2))

    for i in range(Y_orig.shape[1]):
         Y[i,Y_orig[0,i]] = 1

    X_temp,X_test,Y_temp,Y_test = train_test_split(X,Y,stratify = Y,random_state = 42,test_size = t_size)
    X_train,X_val,Y_train,Y_val = train_test_split(X_temp,Y_temp,stratify = Y_temp,random_state = 42,test_size = 0.015)
       
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    X_val = X_val.T
    Y_val = Y_val.T
   
    layer_dims.insert(0,X_train.shape[0])
    layer_dims.append(Y_train.shape[0])
   
    model = MLP(layer_dims,func,'cross entropy',epochs,C,learning_rate,beta)
    model.fit(X_train,Y_train,X_val,Y_val,print_cost = False)
    
    #model.confusion_matrix(X_test,Y_test,2,True)
    
    model.score(X_test,Y_test)
    return model,X_test,Y_test

Y_orig = df["target"].values.reshape(1,-1)
df = df.drop(columns = 'target')

for i in range(5,15):
    X_orig = np.array(df)
    selector = SelectKBest(mutual_info_classif,k = i)
    selector.fit(X_orig,Y_orig[0])
    X_new = selector.transform(X_orig)
    col = df.columns
    temp= pd.DataFrame(selector.inverse_transform(X_new),columns = col)
    selected_features = temp.columns[temp[col].var() != 0]
    X_orig = X_new
    #print(selected_features)
    test_size = 0.05
    layer_dims = [17,8]
    learning_rate = 0.015
    C = 0.001
    epochs = 10000
    beta = 0.2
    func = ["relu","relu","softmax"]
    print('for K = {0}, score is :'.format(i),end = ' ')
    model,Xtest,Ytest = heart_disease_classifier(X_orig,Y_orig,test_size,layer_dims,learning_rate,C,epochs,beta,func)
    
#Y_orig = df["target"].values.reshape(1,-1)
#df = df.drop(columns = 'target')
for i in range(5,15):
    X_orig = np.array(df)
    selector = SelectKBest(mutual_info_classif,k = i)
    selector.fit(X_orig,Y_orig[0])
    X_new = selector.transform(X_orig)
    col = df.columns
    temp= pd.DataFrame(selector.inverse_transform(X_new),columns = col)
    selected_features = temp.columns[temp[col].var() != 0]
    print(selected_features)
    X_orig = X_new
    X = (X_orig - np.mean(X_orig,axis = 0,keepdims = True))/np.std(X_orig,axis= 0,keepdims = True)

    Y = Y_orig.T

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify = Y,random_state = 42,test_size = 0.025)
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    
    
    model2 = LogisticRegression().fit(X_train.T,Y_train[0])
    print('Logistic Regression,K = {0}, score = {1}'.format(i,model2.score(X_test.T,Y_test[0])))
    model3 = RandomForestClassifier().fit(X_train.T,Y_train[0])
    
    print('Random Forest,K = {0}, score = {1}'.format(i,model3.score(X_test.T,Y_test[0])))
    model4 = MLPClassifier((16,8),'relu',max_iter = 10000).fit(X_train.T,Y_train[0])
    print('Neural Network,K = {0}, score = {1}'.format(i,model4.score(X_test.T,Y_test[0])))
fig,ax = plt.subplots(1,2,figsize = (20,5))
ax[0].plot(np.arange(len(model.training_losses())),model.training_losses(),'r')
ax[1].plot(np.arange(len(model.validation_losses())),model.validation_losses(),'b')
print(model.predict(X_test))
print(Y_test[0])
print(model2.predict(X_test.T))
model2.score(X_test.T,Y_test[0]) * 100

chances = [((str(i[0] * 100) +","+ str(i[1]*100)).format('%0.4f') + '%') for i in predict(X_test,3,parameters,func,True)[1].T]
print(chances)
model.predict(X_test,True)
print(Y_test.T)
import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(heart_disease_classifier, "heart.py")
open("./Multi-LayerNeuralNetwork.py",'r').read()