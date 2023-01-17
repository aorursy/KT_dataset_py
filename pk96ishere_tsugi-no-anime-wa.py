# import required libraries
import numpy as np
import pandas as pd
# load in the data
# the anime dataset
anime = pd.read_csv("../input/anime-recommendations-database/anime.csv")

# the users rating dataset
user_ratings = pd.read_csv("../input/anime-recommendations-database/rating.csv")
print(anime.shape); anime.info()
print(user_ratings.shape); user_ratings.info()
# replaing the -1's in user_rating.rating with np.nan
user_ratings.loc[user_ratings.rating == -1, "rating"] = np.nan
# number of nulls in user_rating
user_ratings.isnull().mean() # about 19% of the values in user_rating.rating are missing.
anime.isnull().mean()
# merging anime and user_ratings
user_ratings = pd.merge(user_ratings, anime, on = "anime_id")

# dropping the unnecessary columns
user_ratings.drop(["genre", "type", "episodes", "rating_y", "members"], axis = 1, inplace = True)

# renaming rating_x to rating
user_ratings.rename(columns = {"rating_x": "rating"}, inplace = True)
# filtering out the first 5000 users
user_ratings = user_ratings[user_ratings.user_id <= 1000]
user_ratings.head()
# getting the rating matrix
rating_matrix = user_ratings.pivot_table(values = "rating", index = "user_id", columns = "anime_id")
rating_matrix.shape
rating_matrix.fillna(0, inplace = True)
# setting to raise exceptions
np.seterr(all = "raise")
class MF():
    
    def __init__(self, rating_matrix, learning_rate = 0.01, reg_coef = 0.02, n_factors = 10,
                 n_epochs = 5):
        self.R = rating_matrix
        self.alpha = learning_rate
        self.reg_coef = reg_coef
        self.k = n_factors
        self.n_epochs = n_epochs
        self.n_users, self.n_items = self.R.shape
        
    def getTrainset(self):
        trainset = [(u, i, self.R[u, i]) for u in range(self.n_users) for i in range(self.n_items) if self.R[u, i] > 0]
        self.trainset =  trainset
        
    def fit(self):
        self.getTrainset()
        
        self.p = np.random.normal(0, 0.1, size = (self.n_users, self.k))
        self.q = np.random.normal(0, 0.1, size = (self.n_items, self.k))
        
        training_errors = []
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(self.trainset)
            
            self.sgd()
            
            mse = self.mse()
            
            training_errors.append((epoch + 1, mse))
            
        self.training_errors = training_errors
        
    def mse(self):
        xs, ys = self.R.nonzero()
            
        predicted = self.getPredictedMatrix()
            
        err = 0
            
        for x, y in zip(xs, ys):
            err += ((self.R[x, y] - predicted[x, y]) ** 2)
                
        return np.sqrt(err)
        
    def sgd(self):
        for u, i, r_ui in self.trainset:
                
            prediction = np.dot(self.p[u, :], self.q[i, :].T)
            err = r_ui - prediction
                
                # updates
            self.p[u, :] += self.alpha * (err * self.q[i, :] - self.reg_coef * self.p[u, :])
            self.q[i, :] += self.alpha * (err * self.p[u, :] - self.reg_coef * self.q[i, :])
                
    def getPrediction(self, u, i):
        return np.dot(self.p[u, :], self.q[i, :].T)
        
    def getPredictedMatrix(self):
        return np.dot(self.p, self.q.T)
# converting the rating matrix into a numpy array
R = np.array(rating_matrix)
mf = MF(rating_matrix = R, n_epochs = 10)
mf.fit()
mf.training_errors