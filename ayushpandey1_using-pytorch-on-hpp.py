# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv' ,usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",

                                         "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()

df.head()
df.shape
df.head()
df.info()
for i in df.columns:

    print("Column names {} and Unique Values are {}".format(i,len(df[i].unique())))
import datetime

datetime.datetime.now().year
# We will create Derived Feature i.e Total Years ,we dont want YearBuilt 

df['Total Years']=datetime.datetime.now().year-df['YearBuilt']
df.drop('YearBuilt',axis=1,inplace=True)
df.columns
### Creating Categorical Variables

cat_feat=["MSSubClass","MSZoning","Street","LotShape"]

out_feat="SalePrice"
## Unique Values of MSSubClass now we will conert in categorical variable and label encoding



from sklearn.preprocessing import LabelEncoder

lbl_encoders={}

lbl_encoders["MSSubClass"]=LabelEncoder()

lbl_encoders["MSSubClass"].fit_transform(df["MSSubClass"])
lbl_encoders
from sklearn.preprocessing import LabelEncoder

lbl_encoders={}

for feature in cat_feat:

    lbl_encoders[feature]=LabelEncoder()

    df[feature]=lbl_encoders[feature].fit_transform(df[feature])
df
#Stacking and converting into Tensors

cat_feat=np.stack([df["MSSubClass"],df["MSZoning"],df["Street"],df["LotShape"]],1)

cat_feat
#Convert numpy to Tensors

# Categorical Features cannot be converted to Float

import torch

cat_feat= torch.tensor(cat_feat, dtype=torch.int64)

cat_feat
#### create continuous Variable

cont_feat=[]

for i in df.columns:

    if i in ["MSSubClass","MSZoning","Street","LotShape","SalePrice"]:

        pass

    else:

        cont_feat.append(i)

        
cont_feat
### Stacking continuous variable to a tensor

cont_values=np.stack([df[i].values for i in cont_feat],axis=1)

cont_values=torch.tensor(cont_values,dtype=torch.float)

cont_values
cont_values.dtype
### dependent Feature

y=torch.tensor(df['SalePrice'].values,dtype=torch.float).reshape(-1,1)   ##converting to 2D feature

y
df.info()
cat_feat.shape,cont_values.shape,y.shape
len(df['MSSubClass'].unique())
cat_dims=[len(df[col].unique()) for col in ["MSSubClass", "MSZoning", "Street", "LotShape"]]

cat_dims
#Thumb Rule says--Output dimension ahould be set based on the input variable 

#The formula is (min(50,featur_dimension/2))

embedding_dims=[(x,min(50,(x+1)//2)) for x in cat_dims]

embedding_dims
import torch 

import torch.nn as nn

import torch.nn.functional as F

embed_representation=nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dims])

embed_representation
cat_feat
cat_featz=cat_feat[:4]

cat_featz
pd.set_option('display.max_rows',500)

embedding_val=[]

for i,e in enumerate(embed_representation):               ## e is responsible for converting value to Vector

    

    embedding_val.append(e(cat_feat[:,i]))
embedding_val
# Stacking should be Column Wise So we will be using Concatination Operation using Embedding Value



z=torch.cat(embedding_val,1)            # So now all are stacked in one row

z
#We will apply Dropout layer which will help in avoiding Overfitting 

#After executing Some of the values become 0.So I am dropping 40% values

dropout=nn.Dropout(.4)
final_embed=dropout(z)

final_embed
### Create a Feed Forward Neural network



import torch 

import torch.nn as nn

import torch.nn.functional as F

class FeedForwardNN(nn.Module):

    

    def __init__(self,embedding_dims,n_cont,out_sz,layers,p=0.5):

        super().__init__()

        self.embeds = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dims])

        self.emb_drop = nn.Dropout(p)

        self.bn_cont = nn.BatchNorm1d(n_cont)

        

        layerlist=[]

        n_emb= sum(out for inp,out in embedding_dims)                    ### calculate the total dimension of embedding layer

        n_in= n_emb + n_cont

        

        for i in layers:

            layerlist.append(nn.Linear(n_in,i))

            layerlist.append(nn.ReLU(inplace=True))

            layerlist.append(nn.BatchNorm1d(i))

            layerlist.append(nn.Dropout(p))

            n_in=i

       

        layerlist.append(nn.Linear(layers[-1],out_sz))

        

        self.layers=nn.Sequential(*layerlist)

     

    def forward(self,x_cat,x_cont):

        embeddings=[]

        for i,e in enumerate(self.embeds):

            embeddings.append(e(x_cat[:,i]))

        x= torch.cat(embeddings,1)                      ## concatinating the embeddings and applying Dropout

        x= self.emb_drop(x)

    

        x_cont= self.bn_cont(x_cont)

        x= torch.cat([x,x_cont], 1)

        x= self.layers(x)

        return x
torch.manual_seed(100)

model=FeedForwardNN(embedding_dims, len(cont_feat),1,[100,50],p=0.1)
model
loss_func= nn.MSELoss()       ## Convert into RMSE later

optimizer= torch.optim.Adam(model.parameters(),lr=0.1)
df.shape
cont_values
cont_values.shape
# Train test split



batch_size=1200

test_size= int(batch_size*0.15)

train_categorical=  cat_feat[:batch_size-test_size]

test_categorical= cat_feat[batch_size-test_size:batch_size]

train_cont= cont_values[:batch_size-test_size]

test_cont= cont_values[batch_size-test_size:batch_size]

y_train= y[:batch_size-test_size]

y_test= y[batch_size-test_size:batch_size]
len(train_categorical),len(test_categorical),len(train_cont),len(test_cont),len(y_train),len(y_test)


epochs=5000

final_losses=[]

for i in range(epochs):

    i=i+1

    y_pred= model(train_categorical,train_cont)

    loss= torch.sqrt(loss_func(y_pred,y_train))     ## RMSE

    final_losses.append(loss)

    if i%10==1:

        print("Epoch number: {} and the Loss: {}".format(i,loss.item()))

    optimizer.zero_grad()

    loss.backward()         ##back propogation

    optimizer.step()

    
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(range(epochs),final_losses)

plt.ylabel('RMSE loss')

plt.xlabel('Epochs');
#### Validate the test data

y_pred=""

with torch.no_grad():

    y_pred= model(test_categorical,test_cont)

    loss=torch.sqrt(loss_func(y_pred,y_test))

    

print("RMSE: {}" .format(loss))
data_verify= pd.DataFrame(y_test.tolist(),columns=["test"])

data_predicted=pd.DataFrame(y_pred.tolist(),columns=["Prediction"])
data_predicted
final_output=pd.concat([data_verify,data_predicted],axis=1)

final_output["Difference"]= final_output['test']-final_output['Prediction']

final_output.head()
## Svaing the model

## Save the model

torch.save(model,'HousePrice.pt')
torch.save(model.state_dict,'HouseWeights.pt')           ## state_dict helps in saving Weights
## Loading the saved Model

emb_size=[(15,8),(5,3),(2,1),(4,2)]

model1= FeedForwardNN(emb_size,5,1,[100,50],p=0.4)

model1.eval