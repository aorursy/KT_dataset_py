import os
# Data agg
import pandas as pd
import numpy as np

# Viz
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True  

# metrics
from sklearn import metrics

# Data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
full_data = pd.concat([train,test],ignore_index=False)
missing_age_map = full_data.groupby(['Sex','Pclass','Parch','Embarked'],as_index=False).agg({'Age':'mean'})

train_age_merge = train.loc[train.Age.isnull(),:].drop(['Age'],axis=1).merge(missing_age_map,on=['Sex','Pclass','Parch','Embarked'],how='inner')
train_age_merge.index = train.loc[train.Age.isnull(),:].index
train.loc[train.Age.isnull(),'Age'] = train_age_merge['Age']

test_age_merge = test.loc[test.Age.isnull(),:].drop(['Age'],axis=1).merge(missing_age_map,on=['Sex','Pclass','Parch','Embarked'],how='inner')
test_age_merge.index = test.loc[test.Age.isnull(),:].index
test.loc[test.Age.isnull(),'Age'] = test_age_merge['Age']

train.Age.fillna(full_data.Age.mean(),inplace=True)
test.Age.fillna(full_data.Age.mean(),inplace=True)

train['Embarked'].fillna('S',inplace=True)
test['Embarked'].fillna('S',inplace=True)
train.columns = train.columns.str.lower().str.replace('/','_').str.replace(' ','_')
test.columns = test.columns.str.lower().str.replace('/','_').str.replace(' ','_')

# to drop columns
to_drop = ['name','ticket','cabin']
train = train.drop(to_drop + ['passengerid'],axis=1)
test = test.drop(to_drop,axis=1)

train.embarked = train.embarked.replace({'S': 0, 'C': 1, 'Q': 2})
test.embarked = test.embarked.replace({'S': 0, 'C': 1, 'Q': 2})


train.sex = train.sex.replace({'male':1,'female':0})
test.sex = test.sex.replace({'male':1,'female':0})
scaler = StandardScaler()
target = 'survived'

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare','embarked']
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])
train_,test_ = train_test_split(train,test_size=0.33,random_state=42,stratify=train[target])
x_train = torch.from_numpy(train_[features].values).type(torch.FloatTensor)
y_train = torch.from_numpy(train_[target].values).type(torch.LongTensor)

x_test = torch.from_numpy(test_[features].values).type(torch.FloatTensor)
y_test = torch.from_numpy(test_[target].values).type(torch.LongTensor)
### Here is our neural network
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        # takes an imput of 6 features, and spits out a vector of size 256
        self.lin1 = nn.Linear(in_features=7,out_features=256,bias=True)
        # the second layer takes the 256 vector and process it into a vector of size 64
        self.lin2 = nn.Linear(in_features=256,out_features=64,bias=True)
        # the last layer takes the 64 size vector and returns the output vector which 2 == number of classes
        self.lin3 = nn.Linear(in_features=64,out_features=2,bias=True)
    
    # here we take the input data and pass it through the chain of layers
    def forward(self,input):
        x = self.lin1(input)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

# instance our model
model = Net()
# set the number of epochs
epochs = 100
# criterion aka loss function -> find more on pytorch docs
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# create 3 lists to store the losses and accuracy at each epoch
train_losses, test_losses, accuracy = [0]*epochs, [0]*epochs,[0]*epochs


# in this current case we don't use batches for training and we pass the whole data at each epoch
for e in range(epochs):
    optimizer.zero_grad()

    # Comput train loss
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    loss.backward()

    optimizer.step()

    # store train loss
    train_losses[e] = loss.item()
    
    # Compute the test stats
    with torch.no_grad():
        # Turn on all the nodes
        model.eval()
        
        # Comput test loss
        ps = model(x_test)
        loss = criterion(ps, y_test)

        # store test loss
        test_losses[e] = loss.item()
        
        # Compute accuracy
        top_p, top_class = ps.topk(1, dim=1)
    
        equals = (top_class == y_test.view(*top_class.shape))
        
        # store accuracy
        accuracy[e] = torch.mean(equals.type(torch.FloatTensor))

# Print the final information
print(f'Accuracy  : {100*accuracy[-1].item():0.2f}%')
print(f'Train loss: {train_losses[-1]}')
print(f'Test loss : {test_losses[-1]}')
    
# Plot the results
fig,ax = plt.subplots(1,2,figsize=(20,5))
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_title('Model Accuracy')
ax[0].plot(accuracy)

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_title('Train/Test Losses')
ax[1].plot(train_losses, label='train')
ax[1].plot(test_losses, label='test')
ax[1].legend()   

plt.tight_layout()
print(metrics.classification_report(test_[target],top_class.numpy().ravel()))
sns.heatmap(metrics.confusion_matrix(test_[target],top_class.numpy().ravel()),fmt='d',annot=True);
# predict test data
sub = model(torch.from_numpy(test[features].values).type(torch.FloatTensor))
# extract the predicted class
sub_p, sub_class = sub.topk(1, dim=1)

# copy test data frame
s = test.copy()
# new columnns with prediction
s[target] = sub_class.numpy().ravel()

# rename the columns for submission
s = s.rename(columns={
    target:'Survived',
    'passengerid':'PassengerId'
})

# save the submission file
s[['Survived','PassengerId']].to_csv('submission.csv',index=False)