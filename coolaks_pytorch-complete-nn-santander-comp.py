import torch

import numpy as np

from torch import nn, optim

import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import StratifiedKFold

import torch.nn.functional as F

import torchvision



from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import os

pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)

cuda = torch.cuda.is_available()

if cuda:

    device = "cuda"

    print("cuda available")

torch.cuda.get_device_name(0)    
os.listdir('../input')
%%time

df_train_raw = pd.read_csv('../input/train.csv')

df_test_raw = pd.read_csv('../input/test.csv')
%%time

df_train = df_train_raw.copy()

df_test = df_test_raw.copy()
train_cols = [col for col in df_train.columns if col not in ['ID_code', 'target']]

y_train = df_train['target']
df_train.shape
ss = StandardScaler()

rs = RobustScaler() 

df_train[train_cols] = ss.fit_transform(df_train[train_cols])

df_test[train_cols] = ss.fit_transform(df_test[train_cols])
interactions= {'var_81':['var_53','var_139','var_12','var_76'],

               'var_12':['var_139','var_26','var_22', 'var_53','var_110','var_13'],

               'var_139':['var_146','var_26','var_53', 'var_6', 'var_118'],

               'var_53':['var_110','var_6'],

              'var_26':['var_110','var_109','var_12'],

              'var_118':['var_156'],

              'var_9':['var_89'],

              'var_22':['var_28','var_99','var_26'],

              'var_166':['var_110'],

              'var_146':['var_40','var_0'],

              'var_80':['var_12']}
%%time

for col in train_cols:

        df_train[col+'_2'] = df_train[col] * df_train[col]

        df_train[col+'_3'] = df_train[col] * df_train[col]* df_train[col]

#         df_train[col+'_4'] = df_train[col] * df_train[col]* df_train[col]* df_train[col]

        df_test[col+'_2'] = df_test[col] * df_test[col]

        df_test[col+'_3'] = df_test[col] * df_test[col]* df_test[col]
%%time

for df in [df_train, df_test]:

    df['sum'] = df[train_cols].sum(axis=1)  

    df['min'] = df[train_cols].min(axis=1)

    df['max'] = df[train_cols].max(axis=1)

    df['mean'] = df[train_cols].mean(axis=1)

    df['std'] = df[train_cols].std(axis=1)

    df['skew'] = df[train_cols].skew(axis=1)

    df['kurt'] = df[train_cols].kurtosis(axis=1)

    df['med'] = df[train_cols].median(axis=1)
%%time

for key in interactions:

    for value in interactions[key]:

        df_train[key+'_'+value+'_mul'] = df_train[key]*df_train[value]

        df_train[key+'_'+value+'_div'] = df_train[key]/df_train[value]

        df_train[key+'_'+value+'_sum'] = df_train[key] + df_train[value]

        df_train[key+'_'+value+'_sub'] = df_train[key] - df_train[value]

        

        df_test[key+'_'+value+'_mul'] = df_test[key]*df_test[value]

        df_test[key+'_'+value+'_div'] = df_test[key]/df_test[value]

        df_test[key+'_'+value+'_sum'] = df_test[key] + df_test[value]

        df_test[key+'_'+value+'_sub'] = df_test[key] - df_test[value]
df_train_raw.columns
df_train['num_zero_rows'] = (df_train_raw[train_cols] == 0).astype(int).sum(axis=1)

df_test['num_zero_rows'] = (df_test_raw[train_cols] == 0).astype(int).sum(axis=1)
df_train.head()
all_train_columns = [col for col in df_train.columns if col not in ['ID_code', 'target']]
class classifier(nn.Module):

    

    def __init__(self,input_dim, hidden_dim, dropout = 0.4):

        super().__init__()

        

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(p = dropout)

    

    

    def forward(self,x):

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)

        

        return x   
model = classifier(len(all_train_columns),200)

print(model)

del model
folds = StratifiedKFold(n_splits = 5, shuffle = True)
#Definining the sigmoid function to calculate final results

def sigmoid(x):

    return 1/(1 + np.exp(-x))
%%time

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))



# Defining the parameters

batch_size = 1000

n_epochs = 10



#Loader for test dataset

test_x = torch.from_numpy(df_test[all_train_columns].values).float()#.cuda()

test_dataset = torch.utils.data.TensorDataset(test_x)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)



#Starting the tranining on each fold

for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, y_train)):

    running_train_loss, running_val_loss = [],[]

    val_loss_min = np.Inf

#     print(i, train_idx.shape, val_idx.shape)



    print("Fold number: ", n_fold+1)

    

    #Defining the train loader iterator

    train_x_fold = torch.from_numpy(df_train.iloc[train_idx][all_train_columns].values).float().cuda()

    train_y_fold = torch.from_numpy(y_train[train_idx].values).float().cuda()

    train_fold_dataset = torch.utils.data.TensorDataset(train_x_fold,train_y_fold)

    trainloader = torch.utils.data.DataLoader(train_fold_dataset, batch_size = batch_size, shuffle = True)

    

    #Defining the validation dataset loader iterator

    val_x_fold = torch.from_numpy(df_train.iloc[val_idx][all_train_columns].values).float().cuda()

    val_y_fold = torch.from_numpy(y_train[val_idx].values).float().cuda()

    val_fold_dataset = torch.utils.data.TensorDataset(val_x_fold,val_y_fold )

    valloader = torch.utils.data.DataLoader(val_fold_dataset, batch_size = batch_size, shuffle = False)

    

    #Initiating model, optimizer and loss function

    model = classifier(len(all_train_columns),200)

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = 0.004)

    # criterion = nn.CrossEntropyLoss() # also number of outputs should be 2

    criterion = nn.BCEWithLogitsLoss()

    

    #Starting the neural network training

    for epoch in range(n_epochs):

        train_loss = 0

        

        for train_x_batch, train_y_batch in trainloader:

            model.train()

            optimizer.zero_grad()

            output = model(train_x_batch)

            loss = criterion(output, train_y_batch.view(-1,1))

            loss.backward()

            optimizer.step()

            train_loss += loss.item()/len(trainloader)

        

        #Evaluating on validation dataset

        with torch.no_grad():

            val_loss = 0

            model.eval()

            val_preds = []

            val_true = []

            

            #Stratified Kfold is splitting the dataset a bit weird.. 

            #so have to introduce list methods to store validation and test predictions

            for i, (val_x_batch, val_y_batch) in enumerate(valloader):

                val_output = model(val_x_batch)

                val_loss += (criterion(val_output, val_y_batch.view(-1,1)).item())/len(valloader)

                batch_output = sigmoid(val_output.cpu().numpy().squeeze())

                try:

                    batch_output = list(batch_output)

                except TypeError:

                    batch_output =[batch_output]

                val_preds.extend(batch_output)

                

#                 batch_true = val_y_batch.cpu().numpy().squeeze()

#                 try:

#                     batch_true = list(batch_true)

#                 except TypeError:

#                     batch_true =[batch_true]

#                 val_true.extend(batch_true)

                

        running_train_loss.append(train_loss)

        running_val_loss.append(val_loss)

        

        

        print("Epoch: {}   Training loss: {:.6f}   Validation Loss: {:.6f}    Val_auc:{:.5f}".format(epoch+1,

                                                                              train_loss,

                                                                               val_loss,

                                                                               roc_auc_score(y_train[val_idx].values,

                                                                                             val_preds))

         )

        

        #Saving the model only if validation loss is going down in the epoch

        if val_loss <= val_loss_min:

            print("Validation loss decresed from {:.6f} ----> {:.6f} Saving Model".format(val_loss_min,val_loss))

            torch.save(model.state_dict(), "san_cust_tran_torch.pt")

            val_loss_min = val_loss

            

        

    oof[val_idx] = val_preds    

    print("Fold {} metrics:   Avg Training loss: {:.4f}   Avg Validation Loss: {:.4f}   Val_auc:{:.5f}".format(n_fold+1,

                                                                              np.mean(running_train_loss),

                                                                               np.mean(running_val_loss),

                                                                               roc_auc_score(y_train[val_idx].values,

                                                                                             oof[val_idx])))

    

    #Predicting on test set with the best model in the fold

    y_test_pred_fold = []

    print("Saving test results for best model")

    for (test_x_batch,) in testloader:

        model.load_state_dict(torch.load("san_cust_tran_torch.pt"))

        model.cpu()

        test_output = model(test_x_batch)

        test_batch_output = sigmoid(test_output.detach().numpy().squeeze())

        try:

            test_batch_output = list(test_batch_output)

        except TypeError:

            test_batch_output =[test_batch_output]

        y_test_pred_fold.extend(test_batch_output)

    predictions += np.array(y_test_pred_fold)/folds.n_splits 

    

    print("end of fold: ",n_fold+1,"\n")

        

        
plt.figure(figsize = (8,4))

plt.title("Train vs val loss on last epoch")

plt.plot(running_train_loss, label = "train")

plt.plot(running_val_loss, label = "val")

plt.legend()

plt.show()
sub = pd.DataFrame({'ID_code': df_test.ID_code.values,

                   'target': predictions})

sub.to_csv('sub_pytorch_simplenn.csv', index = False)