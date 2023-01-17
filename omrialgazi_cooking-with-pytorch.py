import pandas as pd 
import torch
from torch import nn
from torch import utils
from torch.utils.data import DataLoader
import numpy as np
import json
from torch.autograd import Variable
import matplotlib.pyplot as plt
import csv

#Custom weight init
def initi(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        

#Simple json load
with open('../input/train.json') as f:
    train = json.load(f)
        
with open('../input/test.json') as f:
    test = json.load(f)
    
print ('imported')
#Randomise the arrangment of the training set in case it is biased
# i.e: Starts with all italian recepies, then mexican, etc. 
# It's useful because we split into training and validation sets later in the code
from random import shuffle
shuffle(train)

xx=[d['ingredients'] for d in train]
yy=[d['cuisine'] for d in train]

#Converting into One Hot Vector:
# representation of array like [0,0,0,0,1,0,0...0] 
#  where 1 rerpresents the existance of an ingredient out of the entire ingredient list

#Make the list of ingredients
word_list=set()
for x in xx:
    word_list|=set(tuple(x))
word_list=list(word_list)
#Make the list of cuisines 

cat_list=list(set(yy))
            
'''
The more an ingredient is showing up in a single category, the bigger it's weight is
The more an ingredient is showing up in multiple categories, the smaller it's weight is
Lets start with only second term.

ing_recep_count=[]
for ing in word_list:
    cuis_list=[]
    for recep in train:
        if ing in recep['ingredients']:
            if recep['cuisine'] not in cuis_list:
                cuis_list.append(recep['cuisine'])
    ing_recep_count.append(len(cuis_list))
#This just doesn't work :(
'''



#new_word_list=[ing for ing in word_list if ing_recep_count[word_list.index(ing)]!=20]
#print (len (new_word_list))
#Make the one hot vector
newx=[]
for num,rec in enumerate(xx):
    newx.append(np.zeros(len(word_list)))
    for ing in rec:
        if ing in word_list:
            ing_location=word_list.index(ing)
            #newx[num][ing_location]=1/(ing_recep_count[ing_location])
            newx[num][ing_location]=1
newx=np.array(newx)


#Convert the cuisines to numerical representation
newy2=[]
for num,rec in enumerate(yy):
    newy2.append(cat_list.index(rec))
newy2=np.array(newy2)

print ('First 3 recepies of one hot vectors')
print (newx[0:5])

print ('First 5 cuisines in numerical')
print (newy2[0:5])
def stats(prevloss):
    train_loss_history.append(loss.data[0])
    #How does it look in our validation set?
    valpred=model(txval)
    valloss=loss_fn(valpred,tyval)
    val_loss_history.append(valloss.data[0])
    actualy=tyval.cpu().data.numpy()
    predy=valpred.cpu().data.numpy()
    val_accuracy=(np.argmax(predy,axis=1)==actualy).sum()/len(predy)
    val_accuracy_history.append(val_accuracy)
    if t%10==0:
        print ('Iteration # %s'%t)
        print ('Current valdiation loss')
        print (round(valloss.data[0],2))
        print ('Previous validation loss')
        print (round(prevloss,2))
        print ('Validation accuracy: {} %'.format(round(100*val_accuracy,3)))    
    '''if prevloss<valloss.data[0]:
        if steps_back<2:
            steps_back+=1
            print ('loss increasing')
            print (steps_back)
        else:
            print ('im breaking!')
            #break
    else:
        steps_back=0
    '''
    #if np.abs(prevloss-valloss.data[0])<0.000001:
    #    print ('breaking beacuse {} and {} are very similar'.format(round(prevloss,6),round(valloss.data[0],6)))
    #    break
    prevloss=valloss.data[0]
    return (prevloss)

    #Save validation loss

dtype = torch.cuda.FloatTensor
SPLIT = 0.8
#Splitting into training and validation sets with SPLIT amount
tv=int(np.ceil(len(newx)*SPLIT))
xtrain=newx[0:tv]
ytrain=newy2[0:tv]
xval=newx[tv:]
yval=newy2[tv:]

'''
xtrain=newx
ytrain=newy2
'''

#Convert to tensors and variables
txval=Variable(torch.from_numpy(xval.astype(float)).float()).type(dtype)
tyval=Variable(torch.from_numpy(yval.astype(float)).long()).cuda()
tx=Variable(torch.from_numpy(xtrain.astype(float)).float()).type(dtype)
ty=Variable(torch.from_numpy(ytrain.astype(float)).long()).cuda()
#tx=Variable(loader_train.__iter__().next()[0]).type(dtype)
#tx=tx.cuda()
print (tx.shape)

batch_size=50

#Our "deep" model
model= nn.Sequential(
    nn.Linear(6714,20),
    nn.LeakyReLU(),
)
model.cuda()
model.zero_grad()
model.apply(initi)

#The loss
loss_fn=nn.CrossEntropyLoss()
print ('built new model')

lr=1e-4
#The optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.99))
#optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,
#                                eps=1e-08, weight_decay=0.000001, momentum=0, centered=False)
optimizer.zero_grad()
print ('made new optimizer')


epochs=170
itt=0
train_loss_history=[]
val_loss_history=[]
val_accuracy_history=[]
prevloss=90
steps_back=0
for t in range(epochs):
    itt=0   
    for curr in range(batch_size,len(tx),batch_size):
        #Split
        x=tx[itt:curr]
        y=ty[itt:curr]

        #Predict
        pred=model(x)

        #How well we predicted
        loss=loss_fn(pred,y)

        optimizer.zero_grad()
        #What should we do next to be better
        loss.backward()

        #Try to be better next time
        optimizer.step()
        itt+=batch_size
        
    prevloss=stats(prevloss)

    #Save train loss


print ('Trained!')


curcurr=range(len(val_loss_history))
plt.plot(curcurr,train_loss_history,'.r',label='Train loss')
plt.plot(curcurr,val_loss_history,'+b',label='Validation loss')
plt.plot(curcurr,val_accuracy_history,'*g',label='Validation accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

#Load test set
xx=[d['ingredients'] for d in test]

#One hot that boi
testx=[]
for num,rec in enumerate(xx):
    testx.append(np.zeros(len(word_list)))
    for ing in rec:
        if ing in word_list:
            testx[num][word_list.index(ing)]=1
testx=np.array(testx)

#Make tensor and predict
testx=torch.from_numpy(testx.astype(float)).float()
testx=Variable(testx).type(dtype)
pred=model(testx)

results1=pred.data.cpu().numpy()
#Convert to format
answers=[]
for num,val in enumerate(test):
    answers.append(str(val['id'])+','+cat_list[int(np.argmax(results1[num]))])

answers.insert(0,'id,cuisine')
print ('predicted!')
print (answers[17])

np.savetxt("one_layer_1000_epoch.csv", answers, delimiter=",", fmt='%s')