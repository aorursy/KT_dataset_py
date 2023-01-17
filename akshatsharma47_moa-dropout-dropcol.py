import os 

import pandas as pd

import torch

from torch import nn as nn

import torchvision

import time

import pickle

import torch.optim as opt

import matplotlib.pyplot as plt

import random

from sklearn.model_selection import train_test_split
data_folder='../input/lish-moa'

train=data_folder+str('/train_features.csv')

label=data_folder+str('/train_targets_scored.csv')

test=data_folder+str('/test_features.csv')



df=pd.read_csv(train)

training_labels=pd.read_csv(label)

df_gen=pd.read_csv(test)



df=df.drop(columns={'sig_id','cp_type','cp_time','cp_dose'},axis=1)

training_labels=training_labels.drop('sig_id',axis=1)

names=df_gen['sig_id']

df_gen=df_gen.drop(['sig_id','cp_type','cp_time','cp_dose'],axis=1)
df_train,df_test,training_labels,validation_labels=train_test_split(df,training_labels,test_size=0.25)
class Dataset(torch.utils.data.Dataset):

  def __init__(self,df,t):

    self.df=df

    self.t=t



  def __len__(self):

    return len(self.df)



  def __getitem__(self,idx):

    if(torch.is_tensor(idx)):

      idx=idx.tolist()

    x=torch.Tensor(self.df.iloc[idx].values)

    y=torch.Tensor(self.t.iloc[idx].values)

    return x,y
trainset=Dataset(df_train,training_labels)

testset=Dataset(df_test,validation_labels)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=100)

testloader=torch.utils.data.DataLoader(testset,batch_size=100)
class model(nn.Module):

  def __init__(self):

    super(model,self).__init__()

    self.l1=nn.Linear(872,527)

    self.l2=nn.Linear(527,330)

    self.l3=nn.Linear(330,206)

    self.bn1=nn.BatchNorm1d(872)

    self.bn2=nn.BatchNorm1d(527)

    self.d1=nn.Dropout(0.5)

    self.d2=nn.Dropout(0.5)

    self.sfmx=nn.Softmax(dim=1)



    nn.init.kaiming_normal_(self.l1.weight)

    nn.init.kaiming_normal_(self.l2.weight)

    nn.init.kaiming_normal_(self.l3.weight)



  def forward(self,x):

    x=self.d1(x)

    x=self.bn1(x)

    x=nn.ReLU()(self.l1(x))

    



    x=self.d2(x)

    x=self.bn2(x)

    x=nn.ReLU()(self.l2(x))

    



    x=self.l3(x)

    x=self.sfmx(x)



    return x



m=model()

m=m.cuda()
training_epochs=300

op=opt.SGD(m.parameters(),lr=0.05,momentum=0.9)

criterion=nn.BCELoss()



training_loss={}

validation_loss={}

l1={'weight': [],'bias': []}

l2={'weight': [],'bias': []}

l3={'weight': [],'bias': []}



early_stopping_tolerance=4

path='model_dropout_sig_drop.pt'

torch.save({'episode': 1,'model_state_dict': m.state_dict(),'op': op.state_dict()}, path)
def validationloss(m,testloader):

  m.eval()

  loss=0

  with torch.no_grad():

    for x,y in testloader:

      x=x.cuda()

      y=y.cuda()

      out=m(x)

      l=criterion(out,y)

      loss+=len(x)*l.item()

  loss/=len(testset)

  m.train()

  return loss
early_stopping_buffer=0

best_val_loss=100



check=torch.load(path)

episode=check['episode']

if(episode!=1):

  m.load_state_dict(check['model_state_dict'])

  op.load_state_dict(check['op'])



while episode <= training_epochs:

  start=time.time()

  running_loss=0

  

  for x,y in trainloader:

    x=x.cuda()

    y=y.cuda()

    out=m(x)

    loss=criterion(out,y)

    op.zero_grad()

    loss.backward()

    op.step()

    running_loss+=len(x)*loss.item()



  loss=running_loss/len(trainset)

  training_loss[episode]=loss

  valloss=validationloss(m,testloader)

  validation_loss[episode]=valloss



  #early stopping section 

  if(valloss<best_val_loss):

    best_val_loss=valloss

    early_stopping_buffer=0

    torch.save({'episode': episode,'model_state_dict': m.state_dict(),'op': op.state_dict()}, path)

  

  if(valloss>=best_val_loss):

    early_stopping_buffer=early_stopping_buffer+1



  print('Done {} in time {}'.format(episode,time.time()-start))

  

  if(early_stopping_buffer>early_stopping_tolerance):

    print('stopped on episode {}'.format(episode))

    break



  #tracking gradients 

  if(episode%5==0):

    l1['weight'].append(m.l1.weight.grad.norm(2).item())

    l2['weight'].append(m.l2.weight.grad.norm(2).item())

    l3['weight'].append(m.l3.weight.grad.norm(2).item())

    l1['bias'].append(m.l1.bias.grad.norm(2).item())

    l2['bias'].append(m.l2.bias.grad.norm(2).item())

    l3['bias'].append(m.l3.bias.grad.norm(2).item())

  

  episode=episode+1
training_file='training_loss_history_dp_s_drp.pickle'

f=open(training_file,'wb')

pickle.dump(training_loss,f)

validation_file='validation_loss_history_dp_s_drp.pickle'

p=open(validation_file,'wb')

pickle.dump(validation_loss,p)
print(pickle.load(open('validation_loss_history_dp_s_drp.pickle','rb')))
print(validation_loss)
plt.plot(list(training_loss.keys()),list(training_loss.values()),label='Training loss')

plt.plot(list(validation_loss.keys()),list(validation_loss.values()),label='Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
print(validation_loss[len(validation_loss)-1])
#plot gradients

plt.plot(range(len(l1['weight'])),l1['weight'],label='l1')

plt.plot(range(len(l2['weight'])),l2['weight'],label='l2')

plt.plot(range(len(l3['weight'])),l3['weight'],label='l3')

plt.xlabel('Gradient Norm')

plt.xlabel('Epochs/5')

plt.legend()

plt.show()
m_test=model()

m_test=m_test.cuda()

checkpoint=torch.load(path)

m_test.load_state_dict(checkpoint['model_state_dict'])
data=torch.Tensor(df_gen.values)

data=data.cuda()

m_test.eval()

output=m_test(data).cpu().detach().numpy()



submission=pd.DataFrame(data=output,columns=training_labels.columns)

submission=pd.concat([names,submission],axis=1)
print(submission)
submission.to_csv('submission.csv',index=False)