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
import random
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
print(df.head())
from sklearn.preprocessing import StandardScaler #used for the scal the data
input_Data = df.drop(columns=['id','diagnosis','Unnamed: 32']) 
label=df['diagnosis'].map({'M':0,'B':1})

input_scaled= preprocessing.scale(input_Data)
df_scaled = pd.DataFrame(input_scaled, columns=input_Data.columns)

df_scaled['label']=label
print(df_scaled.head())

X = df_scaled.loc[:, df_scaled.columns != 'label']
y = df_scaled.loc[:, 'label']
def CostFunction(Feature_index,train_data,target):
    # print(Feature_index)
    
    'Select Feature'
    X_s=train_data.T[Feature_index>0]
    
    num_selected=X_s.shape[0]
    
    ratio=num_selected/train_data.shape[1]
    
    #w_train=0.8
    #w_test=1-w_train
    
       
    beta=0.01
    alpha=1-beta

    
    score_test=Create_ANN(X_s,target)
        # print(scores_train,score_test)
    #z=w_train*scores_train[1]+w_test*score_test[1]
    z=alpha*(1-score_test[1])+beta*num_selected/len(X_s)
    
    return z
def Create_ANN(X_s,target):
    
    X_train, X_test, y_train, y_test = train_test_split(X_s.T, target, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.2)
    
    model=Sequential()

    model.add(Dense(30,activation='relu',input_dim=len(X_s)))
    model.add(Dense(15,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    model.fit(X_train,y_train,epochs=20, validation_data=(X_val, y_val),verbose=0)
    
    #scores_train=model.evaluate(X_train,y_train,verbose=0)
    
    score_test=model.evaluate(X_test,y_test,verbose=0)
    
    return np.array(score_test)
def initialization(num_searchagent, dim):
    Positions=np.zeros((num_searchagent, dim))
    
    for i in range(num_searchagent):
        for j in range(dim):
            Positions[i][j]=round(np.random.uniform(low=0,high=1))
    return Positions



def Sigmoid(X):
    return 1/(1+np.exp(-10*(X-0.5)))



def Bstep(X):
    bstep=0
    Cstep=Sigmoid(X)
    ran=np.random.rand()
    if Cstep>=ran:
        bstep=1
    else:
        bstep=0
        
    return bstep

def Update_pos(X,bstep):
    Y=0
    if (X+bstep)>=1:
        Y=1
    else:
        Y=0
    return Y

def Crossover(x1,x2,x3):
    Y=0
    r=np.random.rand()
    if r<1/3:
        Y=x1
    elif r<2/3 and r>=1/3:
        Y=x2
    else:
        Y=x3
    return Y 
def GWO(SearchAgents_no,Max_iter,ub,lb,dim,Cost_fun,X,y):
    
    Alpha_pos=np.zeros(dim)
    Alpha_score=np.inf
    
    Beta_pos=np.zeros(dim)
    Beta_score=np.inf
    
    Delta_pos=np.zeros(dim)
    Delta_score=np.inf
    
    Positions=initialization(SearchAgents_no,dim)
    # print(Positions)
    Y1=np.zeros(dim)
    Y2=np.zeros(dim)
    Y3=np.zeros(dim)
    
	
    Convergence_curve=np.zeros(Max_iter)
    l=0
    while l<Max_iter:
        for i in range(0,SearchAgents_no):
            Flag4ub=Positions[i]>ub
            Flag4lb=Positions[i]<lb
            Positions[i]=(Positions[i]*(~(Flag4ub+Flag4lb)))+ub*Flag4ub+lb*Flag4lb
#            print(Positions[i])
            fitness=Cost_fun(Positions[i],X,y)
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
             
                                
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a
                C2=2*r2
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i][j])
                
                
                r1=random.random()
            
                
                r2=random.random()
                
                A3=2*a*r1-a
                C3=2*r2
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i][j])
                
                Bstep1=Bstep(A1*D_alpha)
                Bstep2=Bstep(A2*D_beta)
                Bstep3=Bstep(A3*D_delta)
                
                X1=Update_pos(Alpha_pos[j],Bstep1)
                X2=Update_pos(Beta_pos[j],Bstep2)                
                X3=Update_pos(Delta_pos[j],Bstep3)
                
                Positions[i][j]=Crossover(X1,X2,X3)
                
        Convergence_curve[l]=abs(Alpha_score)
        l+=1
        
        
        print('Iteration',l,'--',Alpha_score)
    return Alpha_score, Alpha_pos, Convergence_curve
Agents=30
MaxIter=20
ub=1
lb=0
dim=X.shape[1]

Best_score,Best_pos,Cg=GWO(Agents,MaxIter,1,0,dim,CostFunction,X,y)
plt.plot(Cg)
plt.show()
Selected=Best_pos
print(Selected, np.sum(Selected))

#Retrain with FS
X_s=X.T[Selected>0]
print(X_s.shape)

X_train, X_test, y_train, y_test = train_test_split(X_s.T, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.2)



model=Sequential()

model.add(Dense(30,activation='relu',input_dim=X_train.shape[1]))
# model.add(Dense(25,activation='relu'))
# model.add(Dense(10,activation='relu'))
model.add(Dense(15,activation='relu'))
# model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=20, validation_data=(X_val, y_val))

scores_train=model.evaluate(X_train,y_train)

print('Train Acc:',scores_train[1])
score_test=model.evaluate(X_val,y_val)
print('Test Acc:',score_test[1])
score_all=model.evaluate(X_s.T,y)
print('All acc:',score_all[1])

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