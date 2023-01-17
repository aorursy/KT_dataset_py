# Import libraries
%matplotlib inline
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading ratings file
ratings = pd.read_csv('../input/train.csv', sep=',', encoding='latin-1', usecols=['userId', 'movieId', 'rating'])
ratings_tr = ratings
max_userid = ratings['userId'].drop_duplicates().max()
max_movieid = ratings['movieId'].drop_duplicates().max()
movies = pd.read_csv('../input/movies.csv', sep=',', encoding='latin-1', usecols=['movieId', 'title', 'genres'])
# Process ratings dataframe for Keras Deep Learning model
# Add user_emb_id column whose values == user_id - 1
ratings['user_emb_id'] = ratings['userId'] - 1
# Add movie_emb_id column whose values == movie_id - 1
ratings['movie_emb_id'] = ratings['movieId'] - 1
print (len(ratings), 'ratings loaded')
# Create training set
shuffled_ratings = ratings.sample(frac=1., random_state=32456)

# Shuffling users
Users = shuffled_ratings['user_emb_id'].values
print ('Users:', Users, ', shape =', Users.shape)

# Shuffling movies
Movies = shuffled_ratings['movie_emb_id'].values
print ('Movies:', Movies, ', shape =', Movies.shape)

# Shuffling ratings
Ratings = shuffled_ratings['rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)
# Import Keras libraries

# !!! Works with Keras version = 1.2.2

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
from keras.layers import Embedding, Reshape, Merge
from keras.models import Sequential

class CFModel(Sequential):

    # The constructor for the class
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        # P is the embedding layer that creates an User by latent factors matrix.
        # If the intput is a user_id, P returns the latent factor vector for that user.
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))

        # Q is the embedding layer that creates a Movie by latent factors matrix.
        # If the input is a movie_id, Q returns the latent factor vector for that movie.
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))

        super(CFModel, self).__init__(**kwargs)
        
        # The Merge layer takes the dot product of user and movie latent factor vectors to return the corresponding rating.
        self.add(Merge([P, Q], mode='dot', dot_axes=1))
        
    # The rate function to predict user's rating of unrated items
    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]
# Define constants
K_FACTORS = 100 # The number of dimensional embeddings for movies and users
TEST_USER = 2000 # A random test user (user_id = 2000)
# Define model
model = CFModel(max_userid, max_movieid, K_FACTORS)
# Compile the model using MSE as the loss function and the AdaMax learning algorithm
model.compile(loss='mse', optimizer='adamax')
# Callbacks monitor the validation loss. Model fiting will stop if val_loss increased
# Save the model weights each time the validation loss has improved
# 
callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint('weights.h5', save_best_only=True)]

# Use 30 epochs, 90% training data, 10% validation data 
# For better performance change 'batch_size' parametr
history = model.fit([Users, Movies], Ratings, nb_epoch=30, validation_split=.1, verbose=2, callbacks=callbacks, batch_size=12192)
# Evaluate RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

# Use the pre-trained model
trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
# Load weights
trained_model.load_weights('weights.h5')
# Save Model to file.
model.save('model1.h5')
# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    return trained_model.rate(user_id - 1, movie_id - 1)
# Get list of uniq users in train dataset
users_in_ratings = sorted(ratings['userId'])
users_unic = set ()
for user in users_in_ratings:
    if user not in users_unic:
        users_unic.add (user)
users_lst = list(users_unic)  
# TopRecomender
from collections import Counter
class TopRecommender(object):
    def fit(self, train_data):
        counts = Counter(ratings_tr['movieId'])
        self.predictions = counts.most_common()
        
    def predict(self, user_id, n_recommendations=10):
        dat = self.predictions[:n_recommendations]
        #mid, ra = zip(*dat)
        #return mid
        return [movie_id for movie_id, frequency in self.predictions[:n_recommendations]]
# Fit Top Recomender
tr = TopRecommender()
tr.fit(ratings_tr)
# test Top Recomender
tr.predict(49526,10)
# Prediction
def keras_predict(userId ):
    # Check if user have ratings
    if userId in users_lst:
        # Get list of User ratings
        user_ratings = ratings[ratings['userId'] == userId][['userId', 'movieId', 'rating']]
        # Filter out allready recommeded movies
        recommendations = ratings[ratings['movieId'].isin(user_ratings['movieId']) == False].groupby(ratings.movieId).first()
        # Predict ratings for User
        recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(userId, x['movieId']), axis=1)
        # Sort ratings and get top 10
        top_pred = recommendations.sort_values(by='prediction', ascending=False).merge(movies, on='movieId', how='inner', suffixes=['_u', '_m']).head(10)
        pred_lst = list(top_pred['movieId'])
        return pred_lst
    # Cold start prediction
    else:
        return tr.predict(userId)

# Get test users list
with open('../input/test_user_id.list', 'r') as file:
    test_user_id = file.read()
test_user_id = list(map(int,test_user_id.split(',')))
# Prediction is slow. Create Results Dict for fast prediction of repited users
results = {}
# Write to file
with open('submit.csv', 'w') as f:
    f.write('userId,movieId\n')
    for user_id in test_user_id:
        if user_id not in results:
            recommendations = keras_predict(user_id)
            for rec in recommendations:
                f.write(str(user_id) + ',' + str(int(rec)) + '\n')
            results.update({user_id:recommendations})
        else:
            for rec in results.get(user_id):               
                f.write(str(user_id) + ',' + str(int(rec)) + '\n')
            qr += 1
            
print('Отлично! Время загрузить файл submit.csv на kaggle!  Уникальных записей:', len(results))
    
