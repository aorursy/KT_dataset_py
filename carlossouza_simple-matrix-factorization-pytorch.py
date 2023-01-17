import pandas as pd
import numpy as np
import torch
from tqdm import trange

latent_vectors = 30
num_epochs = 1000
%%time
dfr = pd.read_csv('../input/movielens-1m/ml-1m/ratings.dat', delimiter='::', header=None)
dfr.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
dfr = dfr.drop(columns=['Timestamp'])
dfr.head()
%%time
rating_matrix = dfr.pivot(index='UserID', columns='MovieID', values='Rating')
n_users, n_movies = rating_matrix.shape
# Scaling ratings to between 0 and 1, this helps our model by constraining predictions
min_rating, max_rating = dfr['Rating'].min(), dfr['Rating'].max()
rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)

sparcity = rating_matrix.notna().sum().sum() / (n_users * n_movies)
print(f'Sparcity: {sparcity:0.2%}')
class PMFLoss(torch.nn.Module):
    def __init__(self, lam_u=0.3, lam_v=0.3):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v
    
    def forward(self, matrix, u_features, v_features, non_zero_mask):
        predicted = torch.sigmoid(torch.matmul(u_features, v_features.t()))
        
        diff = (matrix - predicted)**2
        prediction_error = torch.sum(diff*non_zero_mask)

        u_regularization = self.lam_u * torch.sum(u_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(v_features.norm(dim=1))
        
        return prediction_error + u_regularization + v_regularization
# Actual training loop now
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Replacing missing ratings with -1 so we can filter them out later
rating_matrix[rating_matrix.isna()] = -1
rating_matrix = torch.from_numpy(rating_matrix.values).to(device)
non_zero_mask = (rating_matrix != -1)

user_features = torch.randn(n_users, latent_vectors, requires_grad=True, device=device)
user_features.data.mul_(0.01)
movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True, device=device)
movie_features.data.mul_(0.01)

pmferror = PMFLoss(lam_u=0.05, lam_v=0.05)
optimizer = torch.optim.Adam([user_features, movie_features], lr=0.01)

bar = trange(num_epochs)
for epoch in bar:
    optimizer.zero_grad()
    loss = pmferror(rating_matrix, user_features, movie_features, non_zero_mask)
    loss.backward()
    optimizer.step()
    bar.set_postfix(loss=f'{loss:,.3f}')

# Checking if our model can reproduce the true user ratings
user_idx = 7
user_ratings = rating_matrix[user_idx, :]
true_ratings = user_ratings != -1
with torch.no_grad():
    predictions = torch.sigmoid(torch.mm(user_features[user_idx, :].view(1, -1), movie_features.t()))
predicted_ratings = (predictions.squeeze()[true_ratings]*(max_rating - min_rating) + min_rating).round()
actual_ratings = (user_ratings[true_ratings]*(max_rating - min_rating) + min_rating).round()

print("Predictions: \n", predicted_ratings)
print("Truth: \n", actual_ratings)
