import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import trange
val_ratio = 0.1
batch_size = 512
num_epochs = 30
class MovieRatings(Dataset):
    def __init__(self, dat_file):
        dfr = pd.read_csv(dat_file, delimiter='::', header=None)
        dfr.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        dfr = dfr.drop(columns=['Timestamp'])
        self.samples = torch.from_numpy(dfr.values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # every sample is a tensor with 3 values: user ID, movie ID, and rating
        return self.samples[idx]
class Model(nn.Module):
    def __init__(self, num_users, num_movies, num_latent=30):
        super(Model, self).__init__()
        self.user_features = nn.Embedding(num_users, num_latent)
        self.movie_features = nn.Embedding(num_movies, num_latent)
        nn.init.normal_(self.user_features.weight, 0, 0.1)
        nn.init.normal_(self.movie_features.weight, 0, 0.1)

    def forward(self, user_ids, movie_ids):
        users_latent = self.user_features(user_ids)
        movies_latent = self.movie_features(movie_ids)
        ratings = (users_latent * movies_latent).sum(dim=1)
        return ratings
data = MovieRatings('../input/movielens-1m/ml-1m/ratings.dat')
val_size = int(len(data) * val_ratio)
train_size = len(data) - val_size
train, val = random_split(data, [train_size, val_size])

dataloaders = {
    'train': DataLoader(train, batch_size=batch_size, shuffle=True,
                        num_workers=4),
    'val': DataLoader(val, batch_size=batch_size, shuffle=False,
                      num_workers=4)
}
dataset_sizes = {'train': train_size, 'val': val_size}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(
    num_users=data[:, 0].max() + 1,
    num_movies=data[:, 1].max() + 1,
    num_latent=30
).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bar = trange(num_epochs)
epoch_loss = {'train': 0, 'val': 0}
dft = pd.DataFrame(columns=['train_rmse', 'val_rmse'])
for epoch in bar:
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        bar.set_description(f'Epoch {epoch} {phase}'.ljust(20))

        # Iterate over data.
        for batch in dataloaders[phase]:
            user_ids = batch[:, 0].to(device)
            movie_ids = batch[:, 1].to(device)
            ratings = batch[:, 2].float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(user_ids, movie_ids)
                preds = torch.round(outputs)
                loss = criterion(outputs, ratings)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * user_ids.size(0)

        epoch_loss[phase] = running_loss / dataset_sizes[phase]
        bar.set_postfix(train_loss=f'{epoch_loss["train"]:0.5f}',
                        val_loss=f'{epoch_loss["val"]:0.5f}')
        dft.loc[epoch, f'{phase}_rmse'] = epoch_loss[phase]
dft.plot(marker=".");
