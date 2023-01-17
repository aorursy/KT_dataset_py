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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
df =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 2_000_000, parse_dates=["pickup_datetime"])
df.head()
df['fare_amount'].describe()

from numpy import radians, cos, sin, arcsin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1.values)
    lat1 = np.radians(lat1.values)
    lon2 = np.radians(lon2.values)
    lat2 = np.radians(lat2.values)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    c = np.multiply(2, np.arcsin(np.sqrt(a)))
    r = 6371

    return c*r
def distance(s_lat, s_lng, e_lat, e_lng):

   # approximate radius of earth in km
   R = 6373.0

   s_lat = s_lat*np.pi/180.0                      
   s_lng = np.deg2rad(s_lng)     
   e_lat = np.deg2rad(e_lat)                       
   e_lng = np.deg2rad(e_lng)  

   d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

   return 2 * R * np.arcsin(np.sqrt(d))

# from haversine import haversine
df['dist_kmm'] = haversine(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
df['dist_km'] = distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])

df.head()

df.info()

# def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2):
#     fig, axs = plt.subplots(1, 2, figsize=(16,10))
#     axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
#     axs[0].set_xlim((BB[0], BB[1]))
#     axs[0].set_ylim((BB[2], BB[3]))
#     axs[0].set_title('Pickup locations')
#     axs[0].imshow(nyc_map, zorder=0, extent=BB)

#     axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c='r', s=s)
#     axs[1].set_xlim((BB[0], BB[1]))
#     axs[1].set_ylim((BB[2], BB[3]))
#     axs[1].set_title('Dropoff locations')
#     axs[1].imshow(nyc_map, zorder=0, extent=BB)
# BB = (-74.5, -72.8, 40.5, 41.8)
# nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png')
# plot_on_map(df, BB, nyc_map, s=1, alpha=0.3)

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

one_time = df['pickup_datetime'][0]

df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)

df['Hour'] = df['EDTdate'].dt.hour

df['AMPM'] = np.where(df['Hour']<12, 'am', 'pm')

df['Weekday'] = df['EDTdate'].dt.strftime("%a")
df['DoW'] = df['EDTdate'].dt.dayofweek
df.head()
cat_cols = ['Hour', 'AMPM', 'Weekday', 'DoW']
cont_cols = ['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount']

df.dtypes
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
df['AMPM'].cat.categories

hr = df['Hour'].cat.codes.values
ampm = df['AMPM'].cat.codes.values
wd = df['Weekday'].cat.codes.values
dw = df['DoW'].cat.codes.values
cats = np.stack([hr, ampm, wd, dw], axis=1)

cats = torch.tensor(cats, dtype=torch.int64)

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)
conts
y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1, 1)

cats_size = [len(df[col].cat.categories) for col in cat_cols]

embedding_size = [(size, min(50, (size+1)//2)) for size in cats_size]

catz = cats[:2]

selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])

# forward
embedding_z = []

for i, e in enumerate(selfembeds):
    embedding_z.append(e(catz[:,i]))
z = torch.cat(embedding_z, 1)

selfembeddingdrop = nn.Dropout(0.4)

z = selfembeddingdrop(z)

class TabularModel(nn.Module):
    
#     u can define the number of layers in this manner of build - flexibility
    def __init__(self, emb_size, n_cont, out_size, layers, p=0.5):
        
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layer_list = []
        n_emb = sum([nf for ni, nf in emb_size])
        n_in = n_emb + n_cont
        
        for i  in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            n_in = i
            
        layer_list.append(nn.Linear(layers[-1], out_size))
        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

torch.manual_seed(33)
model = TabularModel(embedding_size, conts.shape[1], 1, [200, 100], p=0.4)
# for classification problem, use class size 2 instead of 1
criterion = nn.MSELoss()
# for classification problem: use nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model
# train_test_split
batch_size = 60000
test_size = int(batch_size*0.2)
cat_train = cats[:batch_size-test_size] 
cat_test = cats[batch_size - test_size:batch_size]

con_train = conts[:batch_size-test_size]
con_test = conts[batch_size - test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size - test_size:batch_size]
import time
start_time = time.time()

epochs = 250

losses = []

for i in range(epochs):
    i+=1
    
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train))
    losses.append(loss)
    if i%25 == 1:
        print(f"epoch:{i} loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
duration = time.time() - start_time
print(f"training time: {duration/60}min")

# import matplotlib.pyplot as plt
plt.plot(losses)
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val,y_test))

for i in range(10):
    diff = np.abs(y_val[i].item()-y_test[i].item())
    print(f"{i}predicted {y_val[i].item():8.2f} True:{y_test[i].item():8.2f} DIFF: {diff:8.2f}")
torch.save(model.state_dict(), 'taxi_model_kaggle_pytorch.pt')




test = pd.read_csv(r'../input/titanic/test.csv')
result = pd.read_csv(r'../input/titanic/gender_submission.csv')

X_test = test.drop(columns=['PassengerId','Name','Ticket'])
X_test['Cabin'] = X_test.Cabin.fillna('NA')
X_test['Cabin'] = X_test.Cabin.apply(lambda x : 'NA' if x == 'No' else 'Yes')

y_test = result['Survived'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
0.30487833502771516
predictions = np.abs(np.around(final_predictions))
predictions = predictions.astype(int)
passenger_id = list(test['PassengerId'])
prediction_submission = list(zip(passenger_id,predictions))
prediction_submission = pd.DataFrame(prediction_submission, columns = ('PassengerId','Survived'))
prediction_submission

