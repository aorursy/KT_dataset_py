import os # this makes better error tracing 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error

from torch import nn
import torch
from torch.nn  import functional as F
import torch.optim as  optim 
if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("gpu up")
else:  
  dev = "cpu"  
device = torch.device(dev)
ratings = pd.read_csv("/kaggle/input/anime-recommendations-database/rating.csv")
anime = pd.read_csv("/kaggle/input/anime-recommendations-database/anime.csv")
ratings
anime
anime.nunique() # number of unique values in each column
anime.isnull().sum()
anime.eq("Unknown").sum()
"""
we will replace null values from categorical columns with "NULL"(like it's a new category) 
and for the rating column we will use the mean
"""
anime.genre.fillna("NULL",inplace=True)
anime.type.fillna("NULL",inplace=True)
anime.rating.fillna(anime.rating.mean(),inplace=True)

"""
these are the animes that we don't have information about, they are just 3 and
they have only 10 combinations(which means they contribute with only 10 samples) with the users but they will produce null values  for other features  
we will drop them  
"""
badAnimes = [x  for x in ratings.anime_id.unique() if x not in anime.anime_id.unique() ] 
badIndecies = ratings[ratings.anime_id.isin(badAnimes) ].index
ratings.drop(badIndecies,inplace=True)
ratings.rating.eq(-1).sum() # -1 means the user didn't rate this anime
"""
on how weird this processing seems, but it had to be like that for memory efficiency

"""

ratingsMean = anime.rating.mean()
membersMean = anime.members.mean()
base = anime.copy() 

base.drop(["episodes","name"],axis=1,inplace=True) #dropped episodes  many null values 
# and will cause a false  relationship as there are many movies (1 episode)
base.rename(columns={"rating":"avgRating"},inplace=True)

base.genre = base.genre.apply(lambda x: " ".join(x.split(" ")).split(", ")) #to remove unnecessary spaces
genres = [] # generating the genres data
for i in range(len(base)):
    for g in base.genre[i]:
        if g in genres:
            base["genre_"+g][i] = 1
        else:
            base["genre_"+g] = 0
            base["genre_"+g][i] = 1
            genres.append(g)



base = pd.concat([base,pd.get_dummies(base['type'], prefix='type',dummy_na=True)],axis=1).drop(['type'],axis=1)


ratings =ratings[["rating","user_id","anime_id"]] # rearranging columns
base= pd.merge(ratings,base,how="left",on="anime_id")
ratings = None


base.avgRating.fillna(ratingsMean,inplace=True)
base.members.fillna(membersMean,inplace=True)
base.drop(["genre"],axis=1,inplace=True)


base.fillna(0,inplace=True) # for any 





import random
random.seed(42)

"""
we will set the productionData to the data where the users didn't rate the anime
and we will use 500000 rows for validation

"""


productionData= base[base.rating == -1]
base.drop(base[base.rating == -1].index,inplace=True)
base = base.sample(frac=1).reset_index(drop=True)
validation = base.iloc[:500000]
train = base.iloc[500000:]
base = None # for reducing memory usage (we won't need this df again)

membersEnc = MinMaxScaler()
avgRatingEnc = MinMaxScaler()

animeEncoder = LabelEncoder()

"""
since the anime ids are not consecutive numbers, we will have to label encode them for the embedding layer 
"""

anime["anime_id"] = animeEncoder.fit_transform(anime["anime_id"])



train["anime_id"] = animeEncoder.transform(train["anime_id"])
train["avgRating"] = avgRatingEnc.fit_transform(train["avgRating"].to_numpy().reshape(-1,1))[:,0]
train["members"] = membersEnc.fit_transform(train["members"].to_numpy().reshape(-1,1))[:,0]

validation["anime_id"] = animeEncoder.transform(validation["anime_id"])
validation["avgRating"] = avgRatingEnc.transform(validation["avgRating"].to_numpy().reshape(-1,1))[:,0]
validation["members"] = membersEnc.transform(validation["members"].to_numpy().reshape(-1,1))[:,0]


productionData["anime_id"] = animeEncoder.transform(productionData["anime_id"])
productionData["avgRating"] = avgRatingEnc.transform(productionData["avgRating"].to_numpy().reshape(-1,1))[:,0]
productionData["members"] = membersEnc.transform(productionData["members"].to_numpy().reshape(-1,1))[:,0]


"""
finally the neural network 

"""

class RecommendationNet(nn.Module):
    def __init__(self):
        super(RecommendationNet,self).__init__()
        self.users = nn.Embedding(73517,100) 
        self.animes = nn.Embedding(12294,100)
        self.linear1 = nn.Linear(100+100+54,128)
        self.linear2 = nn.Linear(128,32)
        self.linear3 = nn.Linear(32,1)
    def forward(self,x):
        user = x[:,0].long() # here am selecting the user and anime ids from the input 
        anime = x[:,1].long() 
        otherfeatures = x[:,2:]
        userVector = self.users(user)
        animeVector = self.animes(anime)
#         print(userVector.shape,animeVector.shape,otherfeatures.shape) was used for debugging 
        layer1 = torch.cat((userVector,animeVector,otherfeatures),1)# concatenating vectors
        layer2 = F.relu(self.linear1(layer1))
        layer3 = F.relu(self.linear2(layer2))
        out = torch.sigmoid(self.linear3(layer3)) 
        return out
    

myNN = RecommendationNet()
myNN.to(device)
        
        
        
        
optimizer = optim.Adagrad(myNN.parameters(),lr = 0.001)

batch_size = 128

npData = train.to_numpy()
npData[:,:1] = npData[:,:1]/10 # scaling the target variable
# traintrues = np.expm1(npData[:,4].reshape(-1,1)).reshape(-1)
def ceil(a,b):
    return -(-a//b)

n_samples = len(npData)
better_batch_size = ceil(n_samples, ceil(n_samples, batch_size))

for i in range(10):
#     preds=[]
    for i in range(ceil(n_samples, better_batch_size)):
        batch = npData[i * better_batch_size: (i+1) * better_batch_size]
        batch = torch.Tensor(batch).to(device)
        X = batch[:,1:]
        y = batch[:,:1]
        myNN.zero_grad()
#         print(i)
        pred = myNN(X)
#         preds.extend(np.expm1(pred.cpu().detach().numpy()).reshape(-1))
        err = F.mse_loss(pred,y)
        err.backward()
        optimizer.step()
    print(torch.sqrt(err))
    valpreds = myNN(torch.Tensor(validation.to_numpy()[:,1:]).to(device)).cpu().detach().numpy().reshape(-1)
    print(np.sqrt(mean_squared_error(validation.rating.to_numpy(),valpreds*10)),"Validation Error")

"""
we can use the productionData df(data where users didn't rate animes) to recommend animes for users
we will choose an arbitrary user id like 1 and will choose to show an arbitrary number also 
like the top 10 animes

"""

npRecommend = productionData[productionData.user_id == 1].to_numpy()
npRecommend[:,0] = myNN(torch.Tensor(npRecommend[:,1:]).to(device)).to(device).cpu().detach().numpy().reshape(-1)
indecies =  np.argsort(npRecommend[:,0])[-10:][::-1]
anime_ids = [npRecommend[i,2] for i in indecies]

recommendedAnimes = anime[anime.anime_id.isin(anime_ids)]
    
recommendedAnimes



