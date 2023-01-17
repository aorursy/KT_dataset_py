
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.learner import *
from fastai.column_data import *
from sklearn.decomposition import PCA
from plotnine import *
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))


path='../input/'
tmp_path='/kaggle/working/tmp/'
models_path='/kaggle/working/models/'
ratings = pd.read_csv(path+'rating_final.csv')
ratings
ratings.info()
places = pd.read_csv(path+'geoplaces2.csv')
places
len(ratings['rating'].isnull())
df = ratings['rating']
sns.countplot(df)
plt.title('Count of ratings given')
import plotly.graph_objs as go
df = places['country'].value_counts()

iplot([go.Choropleth(
locationmode='country names',
locations=df.index.values,
text=df.index,
z=df.values
)])
sns.countplot(places['country'])
plt.title('Count of countries')
sns.set()
columns = ['rating', 'food_rating','service_rating']
sns.pairplot(ratings[columns],height=5,kind='scatter')
plt.show()
fig = (
   ratings.loc[:,['rating', 'food_rating','service_rating']]
).corr()

sns.heatmap(fig, annot=True)
len(ratings['placeID'].unique())
len(ratings['userID'].unique())


ratings['userID'].value_counts().head(10).plot.bar( title='Users with the most reviews ')

ratings['placeID'].value_counts().head(10).plot.bar(title='Places with most reviews')

mean = ratings['placeID'].value_counts().mean()
mean

sns.boxplot(
   x='placeID',
    y='rating',
    data=ratings.head(5)
    
    
 )
ratings.isnull().any()
places.isnull().any()
places['country'] = places.country.apply(lambda x: x.replace('?','Mexico'))
places['country'] = places.country.apply(lambda x: x.replace('mexico country','Mexico'))
places['country'] = places.country.apply(lambda x: x.replace('mexico','Mexico'))
val_idxs = get_cv_idxs(len(ratings))
wd=2e-4
n_factors=50
cf = CollabFilterDataset.from_csv(path, 'rating_final.csv','userID','placeID','rating')
learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam,
                       tmp_name=tmp_path,models_name=models_path)

learn.fit(1e-2,2,wds=wd, cycle_len=1,cycle_mult=2)
math.sqrt(0.536)
learn.fit(1e-2,5,wds=wd, cycle_len=1,cycle_mult=2)
math.sqrt(0.516)
restaurant_names = places.set_index('placeID')['name'].to_dict()
g=ratings.groupby('placeID')['rating'].count()
topRestaurants = g.sort_values(ascending=False).index.values[:3000]
topRestIdx = np.array([cf.item2idx[o] for o in topRestaurants])
m=learn.model; m.cuda()
restaurant_bias = to_np(m.ib(V(topRestIdx)))
restaurant_bias
restaurant_ratings = [(b[0], restaurant_names[i] ) for i,b in zip(topRestaurants,restaurant_bias)]
sorted(restaurant_ratings, key=lambda o: o[0])[:15]
sorted(restaurant_ratings, key=lambda o: o[0], reverse=True)[:15]
rest_emb = to_np(m.i(V(topRestIdx)))
rest_emb.shape
pca = PCA(n_components=3)
rest_pca = pca.fit(rest_emb.T).components_
rest_pca.shape
fac0 = rest_pca[0]
rest_comp = [(f,restaurant_names[i]) for f,i in zip(fac0, topRestaurants)]
sorted(rest_comp, key=itemgetter(0), reverse=True)[:10]
sorted(rest_comp, key=itemgetter(0))[:10]
fac1 = rest_pca[1]
rest_comp= [(f,restaurant_names[i]) for f,i in zip(fac1, topRestaurants)]
sorted(rest_comp, key=itemgetter(0), reverse=True)[:10]
sorted(rest_comp, key=itemgetter(0))[:10]
idxs = np.random.choice(len(topRestaurants), 50, replace=False)
X = fac0[idxs]
Y = fac1[idxs]
plt.figure(figsize=(15,15))
plt.scatter(X,Y)
for i, x, y in zip(topRestaurants[idxs], X, Y):
    plt.text(x,y,restaurant_names[i],color=np.random.rand(3)*0.7, fontsize=11)
plt.show()
#Declear ttwo tensors
a = T([[1.,2],
      [3,4]])
b = T([[2.,2],
      [10,10]])
a,b
a*b
#this will allow it to run on the GPU
a*b.cuda()
(a*b).sum(1)
class DotProduct(nn.Module):
    def forward(self, u, m): return (u*m).sum(1)
model=DotProduct()
model(a,b)
unique_users = ratings.userID.unique()
user_to_idx = {o:i for i,o in enumerate(unique_users)}
ratings.userID = ratings.userID.apply(lambda x:user_to_idx[x])
unique_places = ratings.placeID.unique()
place_to_idx = {o:i for i,o in enumerate(unique_places)}
ratings.placeID = ratings.placeID.apply(lambda x:place_to_idx[x])
n_users=int(ratings.userID.nunique())
n_places=int(ratings.placeID.nunique())
class EmbeddingDot(nn.Module):
    def __init__(self, n_users, n_places):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_places, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
        
    def forward(self, cats, const):
        users,places = cats[:,0],cats[:,1]
        u,m = self.u(users),self.m(places)
        return (u*m).sum(1).view(-1,1)
x = ratings.drop(['rating'],axis=1)
y = ratings['rating'].astype(np.float32)
ratings['rating'] = ratings['rating'].astype(float)

ratings['userID'] = ratings.userID.apply(lambda x: x.replace('U',''))
data = ColumnarModelData.from_data_frame(path,val_idxs, x, y, ['userID','placeID'], 64)
#initialize optimization function
wd=1e-5
model = EmbeddingDot(n_users, n_places).cuda()
opt = optim.SGD(model.parameters(), 1e-1,weight_decay=wd,momentum=0.9)
fit(model, data, 3, opt, F.mse_loss)
set_lrs(opt, 0.01)
fit(model, data, 3, opt, F.mse_loss)
set_lrs(opt, 0.0001)
fit(model, data, 5, opt, F.mse_loss)
min_rating, max_rating =ratings.rating.min(), ratings.rating.max()
min_rating, max_rating
#1
def get_emb(ni,nf):
    e = nn.Embedding(ni,nf)
    e.weight.data.uniform_(-0.01,0.01)
    return e

class EmbeddingDotBias(nn.Module):
    def __init__(self,n_users, n_places):
        super().__init__()
        #2
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
            (n_users, n_factors),(n_places, n_factors), (n_users,1),(n_places,1)
        ]]
        
    #3
    def forward(self, cats, conts):
        users,places = cats[:,0],cats[:,1]
        um = (self.u(users)*self.m(places)).sum(1)
        res = um + self.ub(users).squeeze() + self.mb(places).squeeze()
        res= torch.sigmoid(res) * (max_rating-min_rating) + min_rating
        return res.view(-1,1)
wd=2e-4
model = EmbeddingDotBias(cf.n_users, cf.n_items).cuda()
opt = optim.SGD(model.parameters(), 1e-1,weight_decay=wd,momentum=0.9)

fit(model, data, 3, opt, F.mse_loss)

class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_places, nh=10, p1=0.05,p2=0.5):
        super().__init__()
        (self.u, self.m) = [get_emb(*o) for o in [
            (n_users,n_factors), (n_places,n_factors)
        ]]
        self.lin1 = nn.Linear(n_factors*2, nh)
        self.lin2 = nn.Linear(nh,1)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        
    def forward(self, cats, conts):
        users,places = cats[:,0],cats[:,1]
        x = self.drop1(torch.cat([self.u(users), self.m(places)],dim=1))
        x = self.drop2(F.relu(self.lin1(x)))
        return F.sigmoid(self.lin2(x)) * (max_rating-min_rating+1) + min_rating-0.5
    
wd=1e-5
model=EmbeddingNet(n_users,n_places).cuda()
opt=optim.Adam(model.parameters(), 1e-3,weight_decay=wd)
fit(model, data,3, opt, F.mse_loss)