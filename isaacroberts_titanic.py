
import numpy as np # linear algebra
import sklearn
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
import tensorflow as tf
class Score():
    def __init__(self,score,lr=-1,ep_ct=-1):
        self.score=score
        self.lr=lr
        self.ep_ct=ep_ct
        
    def __str__(self):
        coords=""
        if self.lr!=-1:
            coords+="α="+str(self.lr)+","
        if self.ep_ct!=-1:
            coords+="Ep="+str(self.ep_ct)+","
        
        coords+=": "+str(self.score)
        return coords
    
def data_to_num(data):
    #['PassengerId', 'Survived', 'Pclass', 'Name',
    #'Sex', 'Age', 'SibSp',
    #  'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    """
    PassengerId                          1
    Survived                             0
    Pclass                               3
    Name           Braund, Mr. Owen Harris
    Sex                               male
    Age                                 22
    SibSp                                1
    Parch                                0
    Ticket                       A/5 21171
    Fare                              7.25
    Cabin                                0
    Embarked                             S
    """
    cont=['Age','SibSp','Parch','Fare']
    catg=['Sex','Pclass','Embarked']
    catv=[['male','female'],[1,2,3],['Q','S','C']]
    misc=['Cabin']
    strn=['Name','Ticket']

    feed=data.loc[:,cont].fillna(0)
    
    for c in feed:#Normalize Columns
        feed[c]=((feed[c]-feed[c].min())/(feed[c].max()-feed[c].min()))
    n=0
    for c in catg:
        dummies=pd.get_dummies(data[c].astype('category',categories=catv[n]),prefix=c+'=')
        feed=pd.concat([feed,dummies],axis=1,copy=False)
        n+=1
        
    last_init=data['Name'].str[0]
    commas=data['Name'].str.find('.')+2
    print (commas)
    first_init=data['Name'].str[commas]
    print (first_init)
    first_init=first_init.upper()
    firstlet=pd.get_dummies(first_init.astype('category', categories=list(string.ascii_lowercase)))
    feed=pd.concat([feed,firstlet],axis=1,copy=False)
    
    #print (feed)
    return feed

def prepare_inputs_and_labels(inputs,labels=None,startix=0,endix=-1):
    if endix==-1:
        endix=inputs.shape[0]
    inputs=data_to_num(inputs.iloc[startix:endix,:])
    
    if labels is not None:
        labels=pd.get_dummies(labels.iloc[startix:endix])
    return inputs,labels


from keras.layers import Dense
from keras.engine.input_layer import Input

def setupModel(lr,l_input,l_out):
    model=keras.models.Model(l_input,l_out)
    #optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
    optimizer=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer,loss='mean_absolute_error')
    return model

def trainModel(model,train,epoch_ct,batch_size,verb=0):

    trainCt=791
    
    #Train model
    inputs,labels=prepare_inputs_and_labels(train,train['Survived'],0,trainCt)
    model.fit(x=inputs,y=labels,batch_size=batch_size,epochs=epoch_ct,verbose=verb)

    #Test model
    #test=pd.read_csv('../input/test.csv')
    inputs,labels=prepare_inputs_and_labels(train,train['Survived'],trainCt,-1)
    if verb==2:
        pred=model.predict(x=inputs,batch_size=batch_size)
        pred[pred>.5]=1
        pred[pred<1]=0
        corr=pred[:,1]==labels.values[:,1]

        inputs['Surv']=labels.values[:,1].astype(int)
        inputs['Pred']=pred[:,1].astype(int)
        inputs['Corr']=corr.astype(int)
        
        print (inputs.to_string())
        
        loss=1-(corr.sum()/len(corr))
    else:
        loss=model.evaluate(x=inputs,y=labels,batch_size=batch_size,verbose=verb)
        
    
    if verb==3:
        inputs,labels=prepare_inputs_and_labels(train,train['Survived'])
        pred=model.predict(x=inputs,batch_size=batch_size)
        pred[pred>.5]=1
        pred[pred<1]=0
        corr=pred[:,1]==labels.values[:,1]

        inputs['Surv']=labels.values[:,1].astype(int)
        inputs['Pred']=pred[:,1].astype(int)
        inputs['Corr']=corr.astype(int)
        
        print (inputs.to_string())
        
    #print (epoch_ct,"Epochs Score = ",score)
    return Score(loss,lr=lr,ep_ct=epoch_ct)
    
train=pd.read_csv("../input/train.csv")
print (train.shape)

#Variables
#input_dim=inputs.shape[1]
input_dim=12+26#Its just easier to manually change this
batch_size=16

#Setup Layers
l_input=Input(shape=(input_dim,),name='Input_Layer')
l_1=Dense(input_dim,activation='tanh',name='Layer_1_')(l_input)
l_1=Dense(input_dim,activation='tanh',name='Layer_2_')(l_1)
l_out=Dense(2,activation='softmax',name='Softmax')(l_1)

#Setup model 
scores=[]
verb=2
TestMode=False
for lr in [.001] if not TestMode else [.0001,.001,.005,.01,.05,.1]:
    if verb==0:
        print ('lr=',lr)
    model=setupModel(lr,l_input,l_out)
    for epoch_ct in [10] if not TestMode else [1,2,3,4,5,7,9,10,12,15,16,18,20,25,30] :
        score=trainModel(model,train,epoch_ct,batch_size,verb)
        scores.append(score)

if TestMode:
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print ("Final scores for all epochs:")

    out=open ("out.txt",'w')
    for sc in scores:
        print (str(sc))
        out.write(str(sc.score))

    out.close()
else:
    print ("Score:",scores[0].score)
#Submission Version

train=pd.read_csv("../input/train.csv")
print (train.shape)

#Variables
#input_dim=inputs.shape[1]
input_dim=12+26#Its just easier to manually change this
batch_size=16

#Setup Layers
l_input=Input(shape=(input_dim,))
l_1=Dense(input_dim,activation='tanh')(l_input)
l_1=Dense(input_dim,activation='tanh')(l_1)
l_out=Dense(2,activation='softmax')(l_1)

#Setup model 
lr=.001
ep_ct=10
model=setupModel(lr,l_input,l_out)
trainModel(model,train,ep_ct,batch_size)

#Test model
test=pd.read_csv('../input/test.csv')
inputs,_=prepare_inputs_and_labels(test)
predictions=model.predict(x=inputs,batch_size=batch_size)
predictions[predictions>=.5]=1
predictions[predictions<.5]=0
print (predictions)
pred=pd.DataFrame(predictions)
pred.to_csv("submission.csv",index=False)
#print (epoch_ct,"Epochs Score = ",score)



