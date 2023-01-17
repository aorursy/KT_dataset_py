# Some typical imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from sklearn.model_selection import train_test_split

from numba import jit # Compile some functions when performance is critical

import keras

from keras.initializers import RandomNormal

from keras.models import Model, load_model, save_model

from keras.layers import Embedding, Input, Dense, Concatenate, Multiply, Flatten

from keras.optimizers import Adam

import tensorflow as tf

if tf.test.gpu_device_name():

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

    print("No GPU")
# Check our data structure

anime = pd.read_csv("../input/anime-recommendations-database/anime.csv")

anime.head(10)
anime = anime[anime['type'] == 'TV']
# Copy the column

anime['features_genre'] = anime['genre']



# Cast None to an empty string

anime['features_genre'] = anime['features_genre'].fillna('') 

# Split genres into a list of strings

anime['features_genre'] = anime['features_genre'].map(lambda x: x.split(', '))



# Create a set of all genres

all_genres = set()

for row in anime['features_genre']:

    # Union of sets is declared with the | operator

    all_genres = all_genres | set(row)

all_genres.remove('') # Drop the empty genre



def invert_dict(d):

    return {value: key for key, value in d.items()}



all_genres = sorted(list(all_genres)) # We convert it to a list to enforce alphabetic ordering

ngenres = len(all_genres)



idx2genre = dict(enumerate(all_genres)) # Create a mapping dictionary from index to dict

genre2idx = invert_dict(idx2genre) # Inverse dict



genre2idx
def encode_genres(genres):

    out = np.zeros(ngenres)

    for genre in genres:

        if genre == '':

            pass

        else:

            out[genre2idx[genre]] = 1

    return out.tolist()

anime['features_genre'] = anime['features_genre'].map(encode_genres)

anime['features_genre'] # See how the encoded features look
anime['features_episodes'] = anime['episodes'].replace({'Unknown' : 1}).astype(np.int32)

sb.distplot(anime['features_episodes']);

# This feature is heavily unbalanced! Let's apply a quantile transformation to it
ep_discretizer = QuantileTransformer(n_quantiles = 100)

feats_ep = anime['features_episodes'].apply(np.log).to_numpy().reshape(-1, 1)

feats_ep = ep_discretizer.fit_transform(feats_ep).flatten().tolist()

anime['features_episodes'] = feats_ep

sb.distplot(anime['features_episodes']);
# Check our data structure

rating = pd.read_csv("../input/anime-recommendations-database/rating.csv")

rating.head(10)
rating = rating[rating['user_id'] <= 10000] # Can comment this line

rating = rating[rating['anime_id'].isin(anime['anime_id'])] # Don't comment this one though!
print(rating['rating'].replace({-1: np.nan}).dropna().describe())

sb.distplot(rating['rating'], kde = False);
user_median = rating.groupby('user_id').median()['rating']

sb.distplot(user_median, kde = False);

overall_median = user_median.median()

print("Median of all users' medians: ", overall_median)

user_median = dict(user_median.replace({-1 : overall_median}))
user_medians = rating['user_id'].apply(lambda x: user_median[x])

rating['rating'] = rating['rating'].replace({-1 : np.nan}).fillna(user_medians)

rating['rating'] = rating['rating'] / rating['rating'].max() # Divide by the max to normalize!
# Resulting histogram

sb.distplot(rating['rating'], kde = False);
num_neg = 4

user2n_anime = dict(rating.groupby('user_id').count()['anime_id'])
all_users = np.sort(rating['user_id'].unique())

all_anime = np.sort(rating['anime_id'].unique())

n_anime = len(all_anime)

n_users = len(all_users)



@jit

def choice_w_exclusions(array, exclude, samples):

    max_samples = len(array)-len(exclude)

    final_samples = min(samples, max_samples)

    possible = np.array(list(set(array) - set(exclude)))

    return np.random.choice(possible, size = final_samples, replace = False)

@jit

def flat(l):

    return [item for sublist in l for item in sublist]
%%time

#This part takes about 10 minutes with a full dataset. Time for coffee!

neg_user_id = []

neg_anime_id = []

neg_rating = []



for user in all_users:

    exclude = list(rating[rating['user_id'] == user]['anime_id'])

    sampled_anime_id = choice_w_exclusions(all_anime, exclude, len(exclude) * num_neg)

    

    neg_user_id.append([user] * len(sampled_anime_id))

    neg_anime_id.append(sampled_anime_id)

    neg_rating.append([0.] * len(sampled_anime_id))

    

neg_user_id = flat(neg_user_id)

neg_anime_id = flat(neg_anime_id)

neg_rating = flat(neg_rating)
negatives = pd.DataFrame({'user_id': neg_user_id,

                          'anime_id': neg_anime_id,

                          'rating': neg_rating})

data = pd.concat([rating, negatives], ignore_index = True)
anime['features'] = anime['features_genre'] + anime['features_episodes'].apply(lambda x: [x])

anime['features'] = anime['features'].apply(np.array)

n_feats = len(anime['features'].iloc[0])

data = data.join(anime['features'], on = 'anime_id').dropna()
anime2item_dict = dict(zip(np.sort(all_anime), list(range(n_anime))))

item2anime_dict = {v: k for k, v in anime2item_dict.items()}



def anime2item(a_id):

    return anime2item_dict[a_id]



def item2anime(i_id):

    return item2anime_dict[i_id]

                       

data['item_id'] = data['anime_id'].apply(anime2item)
x0 = data['user_id'].to_numpy()

x1 =data['item_id'].to_numpy()

x2 = np.stack(data['features'].to_numpy())

y = data['rating'].to_numpy()



(x0_train, x0_val,

 x1_train, x1_val,

 x2_train, x2_val,

 y_train, y_val) = train_test_split(x0, x1, x2, y,

                                    test_size = 0.1,

                                    random_state = 42)



x_train = [x0_train, x1_train, x2_train]

x_val = [x0_val, x1_val, x2_val]
def get_model(num_users, num_items, num_item_feats, mf_dim, layers = [64, 32, 16, 8]):

    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')

    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    feats_input = Input(shape=(num_item_feats,), dtype='float32', name = 'feats_input')



    # User&Item Embeddings for Matrix Factorization

    MF_Embedding_User = Embedding(input_dim = num_users + 1, output_dim = mf_dim,

                                  name = 'user_embedding',

                                  embeddings_initializer = RandomNormal(stddev=0.001),

                                  input_length = 1)

    MF_Embedding_Item = Embedding(input_dim = num_items + 1, output_dim = mf_dim,

                                  name = 'item_embedding',

                                  embeddings_initializer = RandomNormal(stddev=0.001),

                                  input_length = 1)

    

    # User&Item Embeddings for MLP part

    MLP_Embedding_User = Embedding(input_dim = num_users + 1, output_dim = int(layers[0] / 2),

                                   name = 'mlp_embedding_user',

                                   embeddings_initializer = RandomNormal(stddev=0.001),

                                   input_length = 1)

    MLP_Embedding_Item = Embedding(input_dim = num_items + 1, output_dim = int(layers[0] / 2),

                                   name = 'mlp_embedding_item',

                                   embeddings_initializer = RandomNormal(stddev=0.001),

                                   input_length = 1) 

    

    mf_user_latent = Flatten()(MF_Embedding_User(user_input))

    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))

    mf_vector = Multiply()([mf_user_latent, mf_item_latent])



    # MLP part with item features

    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))

    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))

    

    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent, feats_input])

    for l in layers:

        layer = Dense(l, activation='relu')

        mlp_vector = layer(mlp_vector)



    # Concatenate MF and MLP parts

    predict_vector = Concatenate()([mf_vector, mlp_vector])

    

    # Final prediction layer

    prediction = Dense(1, activation = 'sigmoid',

                       kernel_initializer = 'lecun_uniform',

                       name = 'prediction')(predict_vector)

    

    model = Model(input = [user_input, item_input, feats_input], output = prediction)

    return model
learning_rate = 0.001

batch_size = 256

n_epochs = 3

mf_dim = 15

layers = [128, 64, 32, 16, 8]
model = get_model(n_users, n_anime, n_feats, mf_dim, layers)

model.compile(optimizer = Adam(lr = learning_rate), loss = 'mean_squared_logarithmic_error')
hist = model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val),

                 batch_size = batch_size, epochs = n_epochs, verbose = True, shuffle = True)
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Eval'], loc = 'upper right')

plt.show()
indexed_anime = anime.set_index('anime_id')



def explore(user_id, top = 5):

    sub = rating[rating['user_id'] == user_id]

    watched_animes = sub['anime_id']

    ratings = sub['rating']

    names = indexed_anime.loc[watched_animes]['name']

    genres = indexed_anime.loc[watched_animes]['genre']

    rating_info = pd.DataFrame(zip(watched_animes, names,

                                   genres, ratings * 10),

                               columns = ['anime_id', 'name',

                                          'genre', 'rating']).set_index('anime_id')

    return rating_info.sort_values(by = 'rating', ascending = False).iloc[:top]



def recommend(user_id, recommendations = 5):

    watched_animes = rating[rating['user_id'] == user_id]['anime_id']

    

    test_anime = np.array(list(set(all_anime) - set(watched_animes)))

    test_user = np.array([user_id] * len(test_anime))

    test_items = np.array([anime2item(a) for a in test_anime])

    sub_anime = indexed_anime.loc[test_anime]

    test_features = np.stack(sub_anime['features'].to_numpy())

    test = [test_user, test_items, test_features]

    preds = model.predict(test).flatten()

    results = pd.DataFrame(zip(sub_anime['name'], test_anime,  sub_anime['genre'], preds * 10),

                           columns = ['name', 'anime_id',

                                      'genre', 'score']).set_index('anime_id')

    return results.sort_values(by = 'score', ascending = False).iloc[:recommendations]
explore(444) # Action Sports study
recommend(444)
explore(999) # Action Fantasy case study
recommend(999)
explore(111) # Techno study
recommend(111)