from keras.models import Sequential
from keras.layers import Dense 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from keras.utils import to_categorical
import random

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
import copy
def decode(datum):
    return np.argmax(datum)
df=pd.read_excel('../input/mango-mlp/Traindata2.xlsx',sheet_name='New')

transformer = RobustScaler().fit(df)
df_scaled=pd.DataFrame(transformer.transform(df))
df_scaled.columns=df.columns
df_scaled.Label=df.Label



# df = df_scaled
X = df_scaled.loc[:, df.columns != 'Label']
y = df_scaled.loc[:, 'Label']




y=to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.2)

def CostFunction(Feature_index,train_data,target):
    # print(Feature_index)
    
    'Select Feature'
    X_s=train_data.T[Feature_index>0]
    
    num_selected=X_s.shape[0]
    
    ratio=num_selected/train_data.shape[1]
    
    w_train=0.8
    w_test=1-w_train
    

    beta=0.01
    alpha=1-beta

    
    score_test=Create_ANN(X_s,target)
        # print(scores_train,score_test)
    #z=w_train*scores_train[1]+w_test*score_test[1]
    z=alpha*(1-score_test[1])+beta*num_selected/len(X_s)
   
    
   # scores_train,score_test=Create_ANN(X_s,target)
        # print(scores_train,score_test)
    #z=score_test[1]
    
    
    return z
def Create_ANN(X_s,target):
    
    X_train, X_test, y_train, y_test = train_test_split(X_s.T, target, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.2)
    
    model=Sequential()

    model.add(Dense(30,activation='relu',input_dim=len(X_s)))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(X_train,y_train,epochs=30, validation_data=(X_val, y_val),verbose=0)
    
    #scores_train=model.evaluate(X_train,y_train,verbose=0)
    
    score_test=model.evaluate(X_test,y_test,verbose=0)
    
    return np.array(score_test)
def initialization(num_searchagent, Ub, Lb):
    Positions=np.zeros((num_searchagent, dim))
    #dim=len(Lb);
    for i in range(num_searchagent):
        for j in range(dim):
            Positions[i][j]=np.round(np.random.rand())
    return Positions


def Sigmoid(X):
    return 1/(1+np.exp(-X))

def GWO_Sigmoid(X):
    return 1/(1+np.exp(-10*(X-0.5)))

def GWO(SearchAgents_no,Max_iter,ub,lb,dim,Func,X,y):
    
    Alpha_pos=np.zeros(dim)
    Alpha_score=np.inf
    
    Beta_pos=np.zeros(dim)
    Beta_score=np.inf
    
    Delta_pos=np.zeros(dim)
    Delta_score=np.inf
    
    Positions=initialization(SearchAgents_no,ub,lb)

    l=0
    while l<Max_iter:
        for i in range(0,10):
            #Flag4ub=Positions[i]>ub
            #Flag4lb=Positions[i]<lb
           # Positions[i]=(Positions[i]*(~(Flag4ub+Flag4lb)))+ub*Flag4ub+lb*Flag4lb
#            print(Positions[i])
            fitness=Func(Positions[i],X,y)
            if fitness<Alpha_score:
                Alpha_score=fitness
                Alpha_pos=Positions[i].copy()
                
            if ((fitness>Alpha_score) and (fitness<Beta_score)):
                Beta_score=fitness
                Beta_pos=Positions[i].copy()
                
            if (fitness>Alpha_score) and (fitness>Beta_score) and (fitness<Delta_score):
                Delta_score=fitness
                Delta_pos=Positions[i].copy()
                
        a=2-l*((2)/Max_iter)
        
        for i in range(0,SearchAgents_no):
            for j in range(len(Positions[0])):
                r1=random.random()
                r2=random.random()
                
                A1=2*a*r1-a
                C1=2*r2
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i][j])
                X1=Alpha_pos[j]-A1*D_alpha
                
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a
                C2=2*r2
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i][j])
                X2=Beta_pos[j]-A2*D_beta
                
                r1=random.random()
                r2=random.random()
                
                A3=2*a*r1-a
                C3=2*r2
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i][j])
                X3=Delta_pos[j]-A3*D_delta
                
                S=GWO_Sigmoid((X1+X2+X3)/3)
                
                r=np.random.rand()
                
                if S>=r:
                    Positions[i][j]=1
                else:
                    Positions[i][j]=0
               
        l+=1
     
       # print(Alpha_score)
    return  Alpha_pos



   


class Particle:
    
    def __init__(self):
        self.Position=-1
        self.Velocity=-1
        self.Obj_val=-1
        self.Personalbest_P=-1
        self.Personalbest_Value=-1
    
    def __repr__(self):
        return str(self.Position)

class Swarm:
    
    def __init__(self):
        self.Particle_list=[]
        self.Global_Best_Pos=[]
        self.Global_Best_Value=np.inf
    
    def Create_Swarm(self, no_P):
        for i in range(no_P):
            self.Particle_list.append(Particle())
        return self.Particle_list
    
    def Initialization(self,no_P):
            for i in range(no_P):
                self.Particle_list[i].Position=np.round((ub-lb)*np.random.rand(dim)+lb)
                self.Particle_list[i].Velocity=np.zeros(dim)
                self.Particle_list[i].Personalbest_P=np.zeros(dim)
                self.Particle_list[i].Personalbest_Value=np.inf
            self.Global_Best_Pos=np.zeros(dim)
            self.Global_Best_Value=np.inf
            return self.Particle_list, self.Global_Best_Pos, self.Global_Best_Value

def main(Cost_fun,X,y):
    
    CC=np.zeros(maxIter)
    for i in range(maxIter):
        for k in range(noP):
            currentX=swarm.Particle_list[k].Position.copy()
            swarm.Particle_list[k].Obj_val=Cost_fun(currentX,X,y)
            if swarm.Particle_list[k].Obj_val<swarm.Particle_list[k].Personalbest_Value:
                swarm.Particle_list[k].Personalbest_P=copy.deepcopy(currentX)
                swarm.Particle_list[k].Personalbest_Value=swarm.Particle_list[k].Obj_val
            if swarm.Particle_list[k].Obj_val<swarm.Global_Best_Value:
                swarm.Global_Best_Pos=copy.deepcopy(currentX)
                swarm.Global_Best_Value=swarm.Particle_list[k].Obj_val
        'Update'
        w=(maxIter-i)*(wMax-wMin)/maxIter+wMin
        print('Iteration:',i,'--Fitness:',swarm.Global_Best_Value,'-- Num. Feature:',np.sum(swarm.Global_Best_Pos))
        
        for k in range(noP):
            c1=1.2-swarm.Particle_list[k].Obj_val/swarm.Global_Best_Value
            c2=0.5+swarm.Particle_list[k].Obj_val/swarm.Global_Best_Value
        
            swarm.Particle_list[k].Velocity=w*swarm.Particle_list[k].Velocity\
        + c1*np.random.rand(dim)*(swarm.Particle_list[k].Personalbest_P-swarm.Particle_list[k].Position)\
        + c2*np.random.rand(dim)*(swarm.Global_Best_Pos-swarm.Particle_list[k].Position)
            'Check velocity'
            index1 = swarm.Particle_list[k].Velocity > vMax
            index2 = swarm.Particle_list[k].Velocity < vMin
            swarm.Particle_list[k].Velocity[index1] = vMax[index1]
            swarm.Particle_list[k].Velocity[index2] = vMin[index2]
            'Update Position'
            s=Sigmoid(swarm.Particle_list[k].Velocity)
            
            for d in range(dim):
                ran=np.random.rand()
                if ran<s[d]:
                    swarm.Particle_list[k].Position[d]=1
                else:
                    swarm.Particle_list[k].Position[d]=0
            
        ran=np.random.rand()
        if ran<0.5:
            swarm.Particle_list[k].Position=copy.deepcopy(GWO(20,10,ub,lb,dim,Cost_fun,X,y))
        
        
        CC[i]=swarm.Global_Best_Value
        
    return swarm.Global_Best_Pos,swarm.Global_Best_Value,CC
Searchagent_no=10
dim=X_train.shape[1]
ub=np.array([1]*dim)
lb=np.array([0]*dim)

noP = 15
maxIter = 20
wMax = 0.9
wMin = 0.2
#c1 = 2
#c2 = 2
vMax = (ub - lb) * 0.2
vMin  = -vMax

ub=1
lb=0

swarm=Swarm()
swarm.Create_Swarm(noP)
swarm.Initialization(noP)


print("Optimizing...")
Best_pos,Best_score,Cg=main(CostFunction,X,y)

#GWO(20,10,ub,lb,dim,CostFunction,X,y)
plt.plot(Cg)
plt.show()
Selected=np.round(Best_pos)
print(np.round(Selected))
X_s=X.T[Selected>0]
print(X_s.shape,len(X_s))
#Retrain with FS
X_s=X.T[Selected>0]
print(X_s.shape)

X_train, X_test, y_train, y_test = train_test_split(X_s.T, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.2)


model=Sequential()

model.add(Dense(30,activation='relu',input_dim=X_train.shape[1]))
#model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=100, validation_data=(X_val, y_val))

scores_train=model.evaluate(X_train,y_train)
print('Train Acc:',scores_train[1])
score_test=model.evaluate(X_test,y_test)
print('Test Acc:',score_test[1])



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
Xs=X.T[Selected>0]


model.evaluate(Xs.T,y)
y_pred_test=model.predict_classes(X_test)
print(len(y_pred_test),len(y_test))
print(type(y_pred_test))
print(y_pred_test[:])
decoded_datum=[]
for i in range(y_test.shape[0]):
    datum = y_test[i]
    
    decoded_datum.append(decode(y_test[i]))
print(np.array(decoded_datum))


matrix = confusion_matrix(np.array(decoded_datum), y_pred_test)
print(classification_report(np.array(decoded_datum), y_pred_test))
print(matrix)
model.summary()
model.save('MLP_APGWO_51.h5')