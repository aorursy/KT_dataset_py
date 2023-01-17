# Load numpy, pandas, sklearn, torch, etc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import *
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.cross_validation import train_test_split

from dateutil.parser import parse



# Load already parsed data again
train_df = pd.read_csv('../input/train.csv',encoding ='latin1')
test_df = pd.read_csv('../input/test.csv',encoding ='latin1')
#make copy of original df
X_train_1=train_df
X_test_1=test_df

# Drop useless variables 
X_train_1.drop(labels = ['Unnamed: 0','Unfalldatum'], axis = 1, inplace = True)
X_test_1.drop(labels = ['Unnamed: 0','Unfalldatum'], axis = 1, inplace = True)
X_train_1.head()
#parse 24h and extract hour
X_train_1['Zeit (24h)']=X_train_1['Zeit (24h)'].apply(lambda x: '{0:0>4}'.format(x))
X_test_1['Zeit (24h)']=X_test_1['Zeit (24h)'].apply(lambda x: '{0:0>4}'.format(x))
X_test_1['Zeit (24h)'] = pd.to_datetime(X_test_1['Zeit (24h)'], format = '%H%M')
X_test_1['Zeit (24h)'] = X_test_1['Zeit (24h)'].dt.hour
X_train_1['Zeit (24h)'] = pd.to_datetime(X_train_1['Zeit (24h)'], format = '%H%M')
X_train_1['Zeit (24h)'] = X_train_1['Zeit (24h)'].dt.hour
## month is a cyclic feature. hence some cyclic feature engineering, see https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
m_per_year = 12

X_train_1['sin Monat'] = np.sin(2*np.pi*X_train_1['Monat']/m_per_year)
X_train_1['cos Monat'] = np.cos(2*np.pi*X_train_1['Monat']/m_per_year)
X_test_1['sin Monat'] = np.sin(2*np.pi*X_test_1['Monat']/m_per_year)
X_test_1['cos Monat'] = np.cos(2*np.pi*X_test_1['Monat']/m_per_year)
le=LabelEncoder()

columns = [
 'Strassenklasse',
 'Unfallklasse',
 'Lichtverhältnisse',
 'Bodenbeschaffenheit',
 'Geschlecht',
 'Fahrzeugtyp',
 'Wetterlage']

for col in columns:

       if train_df[col].dtypes=='object':
        data=train_df[col].append(test_df[col])
        le.fit(data.values)
        train_df[col]=le.transform(train_df[col])
        test_df[col]=le.transform(test_df[col])

columns = ['Unfallschwere']

enc=OneHotEncoder(sparse=False)

for col in columns:
    data=X_train_1[[col]]
    enc.fit(data)

    temp = enc.transform(X_train_1[[col]])

    temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])

    temp=temp.set_index(X_train_1.index.values) 
    Y_train_1=pd.concat([temp],axis=1)
    

columns = ['Alter', 'Verletzte Personen',
 'Anzahl Fahrzeuge']

for col in columns:
    data=X_train_1[[col]].append(X_test_1[[col]])
    scaler = StandardScaler()

    scaler.fit(data)
    
    temp = scaler.transform(X_train_1[[col]])

    temp=pd.DataFrame(temp,columns=[(col+"_"+str('scaled'))])

    temp=temp.set_index(X_train_1.index.values)
       
    X_train_1=pd.concat([X_train_1,temp],axis=1)

    temp = scaler.transform(X_test_1[[col]])
       
    temp=pd.DataFrame(temp,columns=[(col+"_"+str('scaled'))])


    temp=temp.set_index(X_test_1.index.values)

    X_test_1=pd.concat([X_test_1,temp],axis=1)

Y_traintemp = Y_train_1
X_traintemp = X_train_1.drop(labels = ["Unfallschwere"],axis = 1)
categorical_features = ['Strassenklasse',

 'Unfallklasse',
 'Lichtverhältnisse',
 'Zeit (24h)',
 'Geschlecht',

 'Bodenbeschaffenheit',

 'Fahrzeugtyp',
 'Wetterlage']

cont_features = [
 'Verletzte Personen_scaled',
 'Anzahl Fahrzeuge_scaled',

 'Alter_scaled',
 'sin Monat',
 'cos Monat',

                       ]

#how many unique values are in training and test dataset per categorial feature, construct embedding matrix
tempc = pd.concat([X_traintemp,X_test_1])

cat_dims = [int(tempc[col].nunique()) for col in categorical_features]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

cat_dims, emb_dims
X_train, x_test, Y_train, y_test = train_test_split(X_traintemp, Y_traintemp, test_size=0.2, random_state=107)
datatemp = pd.concat([pd.Series.astype(X_train,dtype = np.float64),pd.Series.astype(Y_train,dtype = np.float64)],axis=1)

# Separate majority and minority classes
df_1 = datatemp[datatemp['Unfallschwere_1'].values==1]
df_2 = datatemp[datatemp['Unfallschwere_2'].values==1]
df_3 = datatemp[datatemp['Unfallschwere_3'].values==1]

 
# Upsample minority class 2
df_2_upsampled = resample(df_2, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_1.shape[0],    # to match majority class
                                 random_state=123) # reproducible results

# Upsample minority class 3
df_3_upsampled = resample(df_3, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_1.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_1, df_2_upsampled, df_3_upsampled])
 
# Display new class counts
df_upsampled[df_upsampled['Unfallschwere_1'].values==1].shape, df_upsampled[df_upsampled['Unfallschwere_3'].values==1].shape

## seperate features from labels again
Y_train = df_upsampled[["Unfallschwere_1","Unfallschwere_2","Unfallschwere_3"]]
X_train = df_upsampled.drop(labels = ["Unfallschwere_1","Unfallschwere_2","Unfallschwere_3"],axis = 1)
Y_train.shape,X_train.shape
#seperate continous and categorial features for dataloader
cont = X_train[cont_features]
cat = X_train[categorical_features]

#change datatype to torch tensor, create dataset and dataloader with batches

#prepare train data for pytorch (categorial features are int, while continous are float)
Y_tensor_train  = torch.tensor(np.asarray(Y_train.values,dtype=np.float32))
X_tensor_train  = torch.tensor(np.asarray(cat.values,dtype=np.int64))
cont_data = torch.tensor(np.asarray(cont.values,dtype=np.float32))

dataset = TensorDataset(X_tensor_train, Y_tensor_train, cont_data)
dataloader = DataLoader(dataset, batch_size=100,
                        shuffle=True, num_workers=4)

#prepare validation data
cont_val = x_test[cont_features]
cat_val = x_test[categorical_features]
X_tensor_val  = torch.tensor(np.asarray(cat_val.values,dtype=np.int64))
Y_tensor_val  = torch.tensor(np.asarray(y_test.values,dtype=np.float32))
cont_data_val = torch.tensor(np.asarray(cont_val.values,dtype=np.float32))  

#prepare test data
cont_test = X_test_1[cont_features]
cat_test = X_test_1[categorical_features]
X_tensor_test  = torch.tensor(np.asarray(cat_test.values,dtype=np.int64))
cont_data_test = torch.tensor(np.asarray(cont_test.values,dtype=np.float32))  


N_FEATURES =  X_train_1.shape[1]
LR = 0.001
#different dropout for different layers, more dropout for later layers
dropout = torch.nn.Dropout(p=1 - (0.5))
dropout1 = torch.nn.Dropout(p=1 - (0.9))
dropout2 = torch.nn.Dropout(p=1 - (0.75))
no_of_cont = cont_data.shape[1]

N_LABELS = Y_train_1.shape[1]   #3 #n classes


hiddenLayer1Size=512
hiddenLayer2Size=int(hiddenLayer1Size/2)
hiddenLayer3Size=int(hiddenLayer1Size/4)
hiddenLayer4Size=int(hiddenLayer1Size/8)
hiddenLayer5Size=int(hiddenLayer1Size/16)

emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                     for x, y in emb_dims])
no_of_embs = sum([y for x, y in emb_dims])
bn1 = nn.BatchNorm1d(no_of_cont)

linear1=torch.nn.Linear(no_of_embs+no_of_cont, hiddenLayer1Size, bias=True) 
torch.nn.init.xavier_uniform(linear1.weight)

linear2=torch.nn.Linear(hiddenLayer1Size, hiddenLayer2Size)
torch.nn.init.xavier_uniform(linear2.weight)

linear3=torch.nn.Linear(hiddenLayer2Size, hiddenLayer3Size)
torch.nn.init.xavier_uniform(linear3.weight)

linear4=torch.nn.Linear(hiddenLayer3Size, hiddenLayer4Size)
torch.nn.init.xavier_uniform(linear4.weight)

linear5=torch.nn.Linear(hiddenLayer4Size, N_LABELS)
torch.nn.init.xavier_uniform(linear5.weight)



sigmoid = torch.nn.Sigmoid()
sftmx = torch.nn.Softmax()
tanh=torch.nn.Tanh()
leakyrelu=torch.nn.LeakyReLU()




#define classifier class, architecture of nn
class _classifier(nn.Module):
    def __init__(self):
        super(_classifier, self).__init__()
        self.emb_layers = emb_layers
        self.first_bn_layer = bn1
        self.main = nn.Sequential(

            linear1,leakyrelu,nn.BatchNorm1d(hiddenLayer1Size),dropout2,
            linear2,leakyrelu,nn.BatchNorm1d(hiddenLayer2Size),dropout,          
            linear3,leakyrelu,nn.BatchNorm1d(hiddenLayer3Size),dropout,
            linear4,leakyrelu,nn.BatchNorm1d(hiddenLayer4Size),dropout,
            linear5,sigmoid
            
        )
#define pytorch forward pass        
    def forward(self, cat_data, cont_data):
        x = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
        x = torch.cat(x, 1)
        x = dropout1(x)
        normalized_cont_data = self.first_bn_layer(cont_data)
        mainin = torch.cat([x, normalized_cont_data], 1) 
        
        return self.main(mainin)

classifier = _classifier().cuda()
#define optimizer and criterion, we dont use a LR sheduler for now
optimizer = optim.Adam(classifier.parameters())
#lr=LR,weight_decay=5e-3 learning rate and weight decay not implemented yet
criterion = nn.BCELoss()
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,100,150,400], gamma = 0.1)

#train network with n epochs and minibatches
epochs = 500
for epoch in range(epochs):
    losses = []
    tu = [] 
    
    for sample_batched, labels_batched, cont_data in dataloader:
          
        output = classifier(sample_batched.cuda(),cont_data.cuda()) # predict labels from input
        loss = criterion(output.cuda(), labels_batched.cuda()) #compute loss

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients
        losses.append(loss.data.mean())
    #scheduler.step() #apply scheduler after each epoch
        
    print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))

     
    if epoch % 10 == 0:
  #check validation log loss every 10 epoch  
        cl1 = classifier
        prediction = (cl1(X_tensor_val.cuda(),cont_data_val.cuda()).data > 0.5).float() # zero or one
        pred_y = prediction.cpu().numpy().squeeze()
        target_y = Y_tensor_val.cpu().data.numpy()
        tu.append(log_loss(target_y, pred_y))
        print('[%d/%d] Validation log loss: %.3f' % (epoch+1, epochs, np.mean(tu)))

cl1 = classifier

#prediction = (cl1(X_tensor_val).data).float() # probabilities 
prediction = (cl1(X_tensor_val.cuda(),cont_data_val.cuda()).data > 0.5).float() # zero or one

pred_y = prediction.cpu().numpy().squeeze()

target_y = Y_tensor_val.cpu().data.numpy()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
labels = ['0', '1','2']
cm = confusion_matrix(
    target_y.argmax(axis=1), pred_y.argmax(axis=1))
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
#cplt.show()
prediction_test = (cl1(X_tensor_test.cuda(),cont_data_test.cuda()).data > 0.5).float() # zero or one
pred_y_test = prediction_test.cpu().numpy().squeeze()

IDtest = pd.DataFrame(data=X_test_1.index.values,columns = ['Unfall_ID'])
pred = pd.Series(pred_y_test.argmax(axis=1), name="Unfallschwere")

results = pd.concat([IDtest,pred],axis=1)

results.to_csv("MLP_pytorch_embed.csv",index=False)
#labels instead of one-hot encoding
Y_train1 = Y_train.values
Y_train1  = np.argmax(Y_train1, axis=1)
Y_train1

# Try different models, combine them later in ensemble learning, use 4 folds, as kaggle runtime is limited
kfold = StratifiedKFold(n_splits=4)
# Gradient boosting
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,Y_train1)
GBC_best = gsGBC.best_estimator_
gsGBC.best_score_
pred_y = pd.Series(GBC_best.predict(X_train), name="GBC")
target_y = Y_train1

labels = ['0', '1','2']
cm = confusion_matrix(
    target_y, pred_y)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
pred_y = GBC_best.predict(x_test)
target_y = np.argmax(y_test.values, axis=1)
cm = confusion_matrix(
    target_y, pred_y)

print(cm)
labels = ['0', '1','2']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 7],
              "min_samples_split": [2, 3, 7],
              "min_samples_leaf": [1, 3, 7],
              "bootstrap": [False],
              "n_estimators" :[300,600],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(X_train,Y_train1)
ExtC_best = gsExtC.best_estimator_
gsExtC.best_score_
random_forest = RandomForestClassifier(n_estimators=100)
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 7],
              "min_samples_split": [2, 3, 7],
              "min_samples_leaf": [1, 3, 7],
              "bootstrap": [False],
              "n_estimators" :[300,600],
              "criterion": ["gini"]}
gsrandom_forest = GridSearchCV(random_forest,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsrandom_forest.fit(X_train,Y_train1)
# Best score
random_forest_best = gsrandom_forest.best_estimator_
gsrandom_forest.best_score_
pred_y = pd.Series(random_forest_best.predict(X_train), name="random_forest")
target_y = Y_train1
labels = ['0', '1','2']
cm = confusion_matrix(
    target_y, pred_y)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
pred_y = pd.Series(random_forest_best.predict(x_test), name="random_forest")
target_y = np.argmax(y_test.values, axis=1)
cm = confusion_matrix(
    target_y, pred_y)

print(cm)
labels = ['0', '1','2']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train,Y_train1)
adaDTC_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.1],
                  'C': [10,200]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(X_train,Y_train1)
SVMC_best = gsSVMC.best_estimator_
# Best score
gsSVMC.best_score_

test_Unfallschwere_AdaDTC = pd.Series(adaDTC_best.predict(X_test_1), name="AdaDTC")
test_Unfallschwere_ExtC = pd.Series(ExtC_best.predict(X_test_1), name="ExtC")
test_Unfallschwere_GBC = pd.Series(GBC_best.predict(X_test_1), name="GBC")
test_Unfallschwere_SVMC = pd.Series(SVMC_best.predict(X_test_1), name="SVMC")
test_Unfallschwere_random_forest = pd.Series(random_forest_best.predict(X_test_1), name="random_forest")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Unfallschwere_AdaDTC, test_Unfallschwere_ExtC, test_Unfallschwere_GBC,test_Unfallschwere_SVMC,test_Unfallschwere_random_forest],axis=1)

VotingPredictor = VotingClassifier(estimators=[('ExtC', ExtC_best), ('GBC',GBC_best),
('SVMC', SVMC_best), ('random_forest', random_forest_best)], voting='soft', n_jobs=4)
VotingPredictor = VotingPredictor.fit(X_train, Y_train1)
#Save prediciton from ensemble voting
IDtest = pd.DataFrame(data=X_test_1.index.values,columns = ['Unfall_ID'])
test = pd.Series(VotingPredictor.predict(X_test_1), name="Unfallschwere")

results = pd.concat([IDtest,test],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)



