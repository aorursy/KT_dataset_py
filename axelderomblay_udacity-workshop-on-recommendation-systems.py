import pandas as pd
import numpy as np
df = pd.read_csv("../input/dataset.csv")
df.head()
df["timestamp"] = pd.to_datetime(df.timestamp)
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

print("Number of unique users : " + str(n_users))
print("Number of unique items : " + str(n_items))
df.item_category.value_counts()
df.user_age.describe()
df.groupby("item_category").user_age.mean().sort_values()
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(20,10))
plt.title("Popularity of items (long tail)")
plt.plot(range(n_items), df.item_id.value_counts())
plt.xlabel("Items")
plt.ylabel("Number of views")
df.item_id.value_counts().head(5)
%matplotlib inline
plt.figure(figsize=(20,10))
plt.title("Activity of the users (long tail)")
plt.plot(range(n_users), df.user_id.value_counts())
plt.xlabel("Users")
plt.ylabel("Activity of users")
df.user_id.value_counts().head(10)
target = "nb_views"
inter = df.groupby(["user_id","user_age" ,"item_id", "item_category"]).count().rename(columns={"timestamp":target}).reset_index()
inter.sample(10)
inter.shape
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *

X_train, X_test = train_test_split(inter, test_size=0.2, random_state = 2018)
print("Number of USERS in the TRAIN set: " + str(X_train.user_id.nunique()))
print("Number of ITEMS in the TRAIN set: " + str(X_train.item_id.nunique()))
print("Number of USERS in the VALIDATION set: " + str(X_test.user_id.nunique()))
print("Number of ITEMS in the VALIDATION set: " + str(X_test.item_id.nunique()))
X_train.shape, X_test.shape
def apk(actual, predicted, k=10): 

    if len(predicted)>k: 
        predicted = predicted[:k] 
        
    score = 0.0 
    num_hits = 0.0 
 
    for i,p in enumerate(predicted): 
        if p in actual and p not in predicted[:i]: 
            num_hits += 1.0 
            score += num_hits / (i+1.0) 
 
    if not actual: 
        return 0.0 
 
    return score / min(len(actual), k) 
model = dict(X_train.groupby("item_category").nb_views.mean())
model
print("MAE (train) : " + str(mean_absolute_error(X_train[target], 
                                                 X_train["item_category"].apply(lambda x: model[x]))))
print("MAE (test) : " + str(mean_absolute_error(X_test[target], 
                                                X_test["item_category"].apply(lambda x: model[x]))))
#R = pd.DataFrame([], index=df.user_id.unique(), columns=df.item_id.unique())
#for (user, item, inter) in X_train[["user_id","item_id", target]].values:
    #R.loc[user,item] = inter
def runALS(A, R, n_factors, n_iterations, lambda_):
    '''
    Runs Alternating Least Squares algorithm in order to calculate matrix.
    :param A: User-Item Matrix with ratings
    :param R: User-Item Matrix with 1 if there is a rating or 0 if not
    :param n_factors: How many factors each of user and item matrix will consider
    :param n_iterations: How many times to run algorithm
    :param lambda_: Regularization parameter
    :return:
    '''
    (n, m) = A.shape
    Users = 5 * np.random.rand(n, n_factors)
    Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MSE_List = []

    print("Starting Iterations")
    for iter in range(n_iterations):
        print(iter)
        
        print("solving user matrix")
        for i, Ri in enumerate(R):
            print(i)
            Users[i] = np.linalg.solve(np.dot(Items, np.dot(np.diag(Ri), Items.T)) + lambda_ * np.eye(n_factors),
                                       np.dot(Items, np.dot(np.diag(Ri), A[i].T))).T

        print("solving item matrix")
        for j, Rj in enumerate(R.T):
            print(j)
            Items[:,j] = np.linalg.solve(np.dot(Users.T, np.dot(np.diag(Rj), Users)) + lambda_ * np.eye(n_factors),
                                     np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])))
        
    return Users, Items
#(Users, Items) = runALS(R.fillna(0).values, (R>0).applymap(int).values, n_factors=3, n_iterations=1, lambda_=0.1)
from keras.layers.core import Dense, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Input
from keras.models import Model
item_map = {}
for i, item_id in enumerate(inter.item_id.unique()):
    item_map[item_id] = i
    
user_map = {}
for i, user_id in enumerate(inter.user_id.unique()):
    user_map[user_id] = i
from sklearn.preprocessing import LabelEncoder
enc_cat = LabelEncoder()
enc_cat.fit(inter.item_category)
# parameters
n_emb_user = 10
n_emb_item = 30
units = 10
drop = 0.2

n_epochs = 5
batch = 256

# input layers
user_input = Input(shape=(1,))
user_age_input = Input(shape=(1,))
item_input = Input(shape=(1,))
item_cat_input = Input(shape=(1,))

inputs  = [user_input, user_age_input, item_input, item_cat_input]

# embedding layers
user_emb = Embedding(input_dim=n_users, output_dim=n_emb_user, input_length=1)(user_input)
user_emb = Reshape(target_shape=(n_emb_user,))(user_emb)

item_emb = Embedding(input_dim=n_items, output_dim=n_emb_item, input_length=1)(item_input)
item_emb = Reshape(target_shape=(n_emb_item,))(item_emb)

emb_layer = concatenate([user_emb, item_emb])

# dense layers
lay = Dense(units, activation='relu')(emb_layer)
lay = Dropout(drop)(lay)

# output layer
outputs = Dense(1, kernel_initializer='normal')(lay)

# create the NN
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()
model.fit([X_train.user_id.apply(lambda x: user_map[x]).values, 
           X_train.user_age.values,
           X_train.item_id.apply(lambda x: item_map[x]).values, 
           enc_cat.transform(X_train.item_category), 
          ], 
          X_train[target].values, 
          epochs=n_epochs, 
          batch_size=batch, 
          verbose=1,
          validation_data=([X_test.user_id.apply(lambda x: user_map[x]).values, 
                            X_test.user_age.values,
                            X_test.item_id.apply(lambda x: item_map[x]).values, 
                            enc_cat.transform(X_test.item_category), 
                           ], 
                           X_test[target].values)
          )
def NN_predict(X):

    pred = model.predict([X.user_id.apply(lambda x: user_map[x]).values, 
                          X.user_age.values,
                          X.item_id.apply(lambda x: item_map[x]).values,
                          enc_cat.transform(X.item_category)
                         ])

    results = pd.DataFrame([X.user_id.values, 
                            X.user_age.values,
                            X.item_id.values, 
                            X.item_category.values, 
                            X[target].values, 
                            pred[:,0]]).T
    
    results.columns = ["user_id", "user_age", "item_id", "item_category", target, "predictions"]
    
    return results
results_test = NN_predict(X_test)
results_train = NN_predict(X_train)
results_test.sample(10)
print("MAE (train) : " + str(mean_absolute_error(results_train[target], results_train.predictions)))
print("MAE (test) : " + str(mean_absolute_error(results_test[target], results_test.predictions)))
model.fit([inter.user_id.apply(lambda x: user_map[x]).values, 
           inter.user_age.values,
           inter.item_id.apply(lambda x: item_map[x]).values, 
           enc_cat.transform(inter.item_category), 
          ], 
          inter[target].values, 
          epochs=n_epochs, 
          batch_size=batch, 
          verbose=0,
          )
user_emb = pd.DataFrame(model.get_weights()[0], index = inter.user_id.unique())
user_emb.head()
user_emb.shape
from sklearn.neighbors import NearestNeighbors

k = 5
knn = NearestNeighbors(k+1)
knn.fit(user_emb)
import random

np.random.seed(8)
user_id = np.random.choice(df.user_id.unique(),1)[0]
inter[inter.user_id==user_id]
similar_users = user_emb.index[knn.kneighbors([user_emb.loc[user_id]])[1][0]]
similar_users
i = 4
inter[inter.user_id==similar_users[i]]
from sklearn.decomposition import PCA
proj = PCA(n_components=2)
user_emb_proj = pd.DataFrame(proj.fit_transform(user_emb), columns = ["proj1", "proj2"], index=user_emb.index)

plt.figure(figsize=(20,10))
plt.plot(user_emb_proj.values[:,0], user_emb_proj.values[:,1],'o',color='b')
item_emb = pd.DataFrame(model.get_weights()[1], index = inter.item_id.unique())
item_emb.head()
item_emb.shape
k = 5
knn = NearestNeighbors(k+1)
knn.fit(item_emb)
np.random.seed(23)
item_id = np.random.choice(df.item_id.unique(),1)[0]
inter[inter.item_id==item_id]
similar_items = item_emb.index[knn.kneighbors([item_emb.loc[item_id]])[1][0]]
similar_items
i = 2
inter[inter.item_id==similar_items[i]]
from sklearn.decomposition import PCA
proj = PCA(n_components=2)
item_emb_proj = pd.DataFrame(proj.fit_transform(item_emb), columns = ["proj1", "proj2"], index=item_emb.index)

plt.figure(figsize=(20,10))
plt.plot(item_emb_proj.values[:,0], item_emb_proj.values[:,1],'o', color='r')
