import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

import torchvision.models as models

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

from PIL import Image

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from collections import Counter

import ast

import os 



%matplotlib inline



import pdb
torch.__version__
torch.cuda.is_available()
!ls -c ../input/tmdb-box-office-prediction-posters/tmdb_box_office_prediction_posters/tmdb_box_office_prediction_posters
folder_posters = '../input/tmdb-box-office-prediction-posters/tmdb_box_office_prediction_posters/tmdb_box_office_prediction_posters'
!ls -c ../input/tmdb-box-office-prediction
folder_csv = '../input/tmdb-box-office-prediction'
# coming from an other kernel 

# will add the reference later

def clean(df):

    

    # Runtime na

    df.loc[df.id == 1335, 'runtime'] = 119

    df.loc[df.id == 1336, 'runtime'] = 130

    df.loc[df.id == 2302, 'runtime'] = 100

    df.loc[df.id == 2303, 'runtime'] = 81

    

    # Runtime 0

    df.loc[df.id == 391, 'runtime'] = 86

    df.loc[df.id == 592, 'runtime'] = 90

    df.loc[df.id == 925, 'runtime'] = 86

    df.loc[df.id == 978, 'runtime'] = 93

    df.loc[df.id == 1256, 'runtime'] = 92

    df.loc[df.id == 1542, 'runtime'] = 93

    df.loc[df.id == 1875, 'runtime'] = 86

    df.loc[df.id == 2151, 'runtime'] = 108

    df.loc[df.id == 2499, 'runtime'] = 86

    df.loc[df.id == 2646, 'runtime'] = 98

    df.loc[df.id == 2786, 'runtime'] = 111

    df.loc[df.id == 2866, 'runtime'] = 96

    

    df.loc[df.id == 3829, 'release_date'] = '6/1/00'

    df.loc[df['id'] == 16,'revenue'] = 192864          # Skinning

    df.loc[df['id'] == 90,'budget'] = 30000000         # Sommersby          

    df.loc[df['id'] == 118,'budget'] = 60000000        # Wild Hogs

    df.loc[df['id'] == 149,'budget'] = 18000000        # Beethoven

    df.loc[df['id'] == 313,'revenue'] = 12000000       # The Cookout 

    df.loc[df['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

    df.loc[df['id'] == 464,'budget'] = 20000000        # Parenthood

    df.loc[df['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

    df.loc[df['id'] == 513,'budget'] = 930000          # From Prada to Nada

    df.loc[df['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

    df.loc[df['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

    df.loc[df['id'] == 850,'budget'] = 90000000        # Modern Times

    df.loc[df['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

    df.loc[df['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

    df.loc[df['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

    df.loc[df['id'] == 1542,'budget'] = 1              # All at Once

    df.loc[df['id'] == 1542,'budget'] = 15800000       # Crocodile Dundee II

    df.loc[df['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

    df.loc[df['id'] == 1714,'budget'] = 46000000       # The Recruit

    df.loc[df['id'] == 1721,'budget'] = 17500000       # Cocoon

    df.loc[df['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

    df.loc[df['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

    df.loc[df['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

    df.loc[df['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

    df.loc[df['id'] == 2612,'budget'] = 15000000       # Field of Dreams

    df.loc[df['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

    df.loc[df['id'] == 2801,'budget'] = 10000000       # Fracture

    df.loc[df['id'] == 3889,'budget'] = 15000000       # Colossal

    df.loc[df['id'] == 6733,'budget'] = 5000000        # The Big Sick

    df.loc[df['id'] == 3197,'budget'] = 8000000        # High-Rise

    df.loc[df['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

    df.loc[df['id'] == 5704,'budget'] = 4300000        # French Connection II

    df.loc[df['id'] == 6109,'budget'] = 281756         # Dogtooth

    df.loc[df['id'] == 7242,'budget'] = 10000000       # Addams Family Values

    df.loc[df['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

    df.loc[df['id'] == 5591,'budget'] = 4000000        # The Orphanage

    df.loc[df['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee



    if 'revenue' in df.columns.values:

        power_six = df.id[df.budget > 1000][df.revenue < 100]



        for k in power_six :

            df.loc[df['id'] == k,'revenue'] =  df.loc[df['id'] == k,'revenue'] * 1000000

            

    return df
def get_features_data(df):

    # work on a copy 

    df = df.copy()

    

    # transform json

    jsons = [

        'crew', 

        'cast', 

        'Keywords',  

        'genres', 

        'belongs_to_collection', 

        'production_companies', 

        'production_countries', 

        'spoken_languages'

    ]

    for j in jsons: 

        df[j] = df[j].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))

        

    # release date year 

    release_date = pd.to_datetime(df.release_date, format='%m/%d/%y')

    df['release_date_year'] = release_date.dt.year.apply(lambda x: x-100 if x>2018 else x)

    df['release_date_month'] = release_date.dt.month

    df['release_date_day'] = release_date.dt.day

    df['release_date_quarter'] = release_date.dt.quarter

    df['release_date_weekday'] = release_date.dt.weekday

    df['release_date_weekofyear'] = release_date.dt.weekofyear

    

    # genres 

    df.genres = df.genres.apply(lambda x: [item['name'] for item in x])

    df['num_genres'] = df.genres.apply(lambda x: len(x))

    df.num_genres = df.num_genres.astype('float64')

    

    # one hot genre 

    genres = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Crime', 

              'Adventure', 'Horror', 'Science Fiction', 'Family', 

              'Fantasy', 'Mystery', 'Animation', 'History', 'Music', 'War', 

              'Documentary', 'Western', 'Foreign']

    

    genres_one_hot = np.zeros((len(df.genres),len(genres)))

    for i in range(len(df)):

        for j, genre in enumerate(genres):

            row = df.iloc[i]

            if genre in row['genres']:

                genres_one_hot[i,j] = 1 

                

    # cast

    cast = df.cast.apply( lambda x: ','.join([c['name'] for c in x] ))

    df['size_of_cast'] = cast.apply(lambda x: len(x.split(',')))

    

    # crew

    df['size_of_crew'] =  df['crew'].apply(lambda x: len(x))

    

    df['total_crew'] = df['size_of_crew'] + df['size_of_cast']

                

    # budget

    df['log_budget'] = np.log1p(df.budget)

    df['budget_by_runtime'] = df['budget']/df['runtime']

    df['budget_by_popularity'] = df['budget']/df['popularity']

    df['release_year_by_popularity'] = df['release_date_year']/df['popularity']

    df['popularity_by_release_year'] = df['popularity']/df['release_date_year']



    # scaled data 

    cols_to_scale = [

        'release_date_year',

        'release_date_month',

        'release_date_day',

        'release_date_quarter',

        'release_date_weekday',

        'release_date_weekofyear',

        'popularity',

        'budget', 

        'budget_by_runtime',

        'budget_by_popularity',

        'runtime', 

        'num_genres',

        'log_budget', 

        'release_year_by_popularity', 

        'popularity_by_release_year',

        'size_of_cast',

        'size_of_crew'

    ]

    # make sure it is float before 

    for col in cols_to_scale:

         df[col].astype('float64')

            

    scaler = StandardScaler()

    data = scaler.fit_transform(df[cols_to_scale])

    

    # add other columns not to be scaled 

    data = np.concatenate([data,genres_one_hot], axis=1)

    

    return data, scaler 
df = pd.read_csv(f"{os.path.join(folder_csv, 'train.csv')}")
df = clean(df)

test, scaler = get_features_data(df)
torch.from_numpy(test[0]).float().cuda()
df.columns.values
sample_img_path  = os.path.join(os.path.join(folder_posters, 'train'), f"{df.iloc[10].id}.jpeg")

plt.figure(figsize=(5,5))

plt.imshow(Image.open(sample_img_path))

plt.axis('off')

plt.show()
class MovieDataset(Dataset):

    def __init__(self, csv_file, img_folder, transform=None, idx=None):

        self.csv_file = csv_file

        self.img_folder = img_folder 

        self.transform = transform

        self.df = clean(pd.read_csv(csv_file))

        # missing poster, will drop data for now 

        self.df.drop(self.df[self.df.id == 2303].index, inplace=True)

        

        # create features from dataframe 

        self.data, self.scaler = get_features_data(self.df)

        self.fs = self.data.shape[1]

        

        if idx is not None:

            self.df = self.df.iloc[idx]

            

        self.cols = self.df.columns.values

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx): 

        features = torch.from_numpy(self.data[idx,:]).float()

        img_path = os.path.join(self.img_folder, f"{self.df.iloc[idx].id}.jpeg")

        target = None

        if 'revenue' in self.cols:

            target = np.log1p(self.df.iloc[idx].revenue)

        image = Image.open(img_path)   

        if self.transform:

            image = self.transform(image)

        return {'images': image, 'features': features, 'targets': target}
def get_dataset(idx=None):

    data_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, .456, 0.406], # imagenet normalization

                             std=[0.229, 0.224, 0.225])

    ])

    dataset = MovieDataset(csv_file=f"{os.path.join(folder_csv, 'train.csv')}", 

                           img_folder=f"{os.path.join(folder_posters, 'train')}", 

                           transform=data_transform, 

                           idx=idx)

    return dataset
idx = [i for i in range(len(df)-1)]

idx = np.random.permutation(idx)

train_idx = idx[:round(0.9*(len(idx)))]

valid_idx = idx[round(0.9*(len(idx))):]

len(valid_idx) / (len(valid_idx) + len(train_idx))
train_dataset = get_dataset(idx=train_idx)

valid_dataset = get_dataset(idx=valid_idx)

len(valid_dataset) / (len(train_dataset) + len(valid_dataset))
train_dataset.fs
dataloader = DataLoader(get_dataset(), batch_size=4, shuffle=True, num_workers=4)
for _, sample_batch in enumerate(dataloader):

    image = sample_batch['images']

    revenue = sample_batch['targets']

    batch_size = image.shape[0]

    fig = plt.figure(figsize=(20,20))

    for i in range(batch_size):

        ax = plt.subplot(1, batch_size, i + 1)

        plt.tight_layout()

        data = image[i].cpu().numpy().transpose((1, 2, 0))

        plt.imshow(np.interp(data, (data.min(), data.max()), (0, 1)))

        ax.axis('off')

        ax.set_title(f"Sample {i+1}, Revenue: {revenue[i]:.2f}")

    plt.show()

    break
class WithPosterEmbeddings(nn.Module):

    

    def __init__(self, features_size, dp=0.5):

        super().__init__()

        self.dp = dp

        self.img_emb_size = 10 # change image embedding size 

        self.features_size = features_size

        

        self.resnet18 = models.resnet18(pretrained=True)

        # freeze all layers

        for param in self.resnet18.parameters():

            param.requires_grad = False

            

        #bs, drp, linear, relu   

        self.resnet18.fc = nn.Sequential(

            nn.BatchNorm1d(512),

            nn.Dropout(self.dp),

            nn.Linear(512, 1000, bias=True),

            nn.ReLU(),

            nn.BatchNorm1d(1000),

            nn.Dropout(self.dp),

            nn.Linear(1000, self.img_emb_size, bias=True),

            nn.ReLU(),

            nn.Dropout(self.dp) # dropout on the poster embeddings

        )

        

        self.l1 = nn.Sequential(

            nn.BatchNorm1d(self.img_emb_size + self.features_size), 

            nn.Linear(self.img_emb_size + self.features_size,512), 

            nn.ReLU()

        )

        

        self.l2 = nn.Sequential(

            nn.BatchNorm1d(512),

            nn.Dropout(self.dp),

            nn.Linear(512,256),

            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(self.dp),

            nn.Linear(256,1) 

        )

        

    """ imgs: posters

        x: features

    """

    def forward(self, imgs, features, skip_cnn=False):

        x = self.resnet18(imgs)

        if skip_cnn: 

            x = torch.zeros(imgs.size(0),self.img_emb_size).float().cuda()

        x = torch.cat([x, features], dim=1)

        x = self.l2(self.l1(x))

        

        return x
fs = 36 # feature size



epochs =  50

wd = 0.001

lr = 1e-3

dropout=0.5

bs = 8
def calculate_rmse(model, dataloader, monitor=False, skip_cnn=False):

    criterion = nn.MSELoss()

    model.eval()

    losses = []

    for sample_batch in tqdm(dataloader, disable=(not monitor)):

        images = sample_batch['images'].cuda()

        features = sample_batch['features'].cuda()

        targets = sample_batch['targets'].float().cuda()

        

        preds = model(images, features, skip_cnn=skip_cnn)

        loss = criterion(preds,targets)

        

        losses.append( loss.item() * len(sample_batch))

        

    # set the model to train mode

    model.train()

    return np.sqrt(np.mean(losses))
# model & dataloader

model = WithPosterEmbeddings(features_size = fs, dp=dropout).cuda()

dataloader_train = DataLoader(get_dataset(idx=train_idx), batch_size=bs, shuffle=True, num_workers=4)

dataloader_valid = DataLoader(get_dataset(idx=valid_idx), batch_size=bs, shuffle=True, num_workers=4)



criterion = nn.MSELoss()

opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

lr_sch = lr_scheduler.ReduceLROnPlateau(opt,'min', factor=0.1, patience=10, verbose=True) # learning rate scheduler 



i = 0

running_losses = []

train_losses = []

valid_losses = []

for epoch_i in range(epochs):

    print(f"Epoch {epoch_i+1}/{epochs}")

    running_loss = 0

    for sample_batch in tqdm(dataloader_train):

        images = sample_batch['images'].cuda()

        features = sample_batch['features'].cuda()

        targets = sample_batch['targets'].float().cuda()

        

        preds = model(images, features)

        loss = criterion(preds,targets.unsqueeze(1))

        opt.zero_grad()

        loss.backward()

        opt.step()

        

        running_losses.append((i, loss.item() ))

        i+=1

        

    print('Calculating validation loss..')

    valid_losses.append((i, calculate_rmse(model, dataloader_valid)))

    lr_sch.step(valid_losses[-1][1])

    

    print('Calculating train loss..')

    train_losses.append((i, calculate_rmse(model, dataloader_train)))

        

    print(f"Loss: {train_losses[-1][1]:.3f} (train) {valid_losses[-1][1]:.3f} (valid)")
def plot_rmse(train_losses, valid_losses):

    plt.figure(figsize=(10,10))

    plt.xlabel('Iteration #')

    plt.ylabel('RMSE loss')

    

    it, loss = zip(*train_losses)

    plt.plot(it, loss, marker='o')

    

    it, loss = zip(*valid_losses)

    plt.plot(it, loss, marker='o')

    

    plt.show()
plot_rmse(train_losses, valid_losses)
no_poster = calculate_rmse(model, dataloader_valid, skip_cnn=True)

with_poster = calculate_rmse(model, dataloader_valid, skip_cnn=False)
print(f"Loss: {with_poster:.3f} (poster) {no_poster:.3f} (no poster)")