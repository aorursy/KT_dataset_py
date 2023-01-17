import numpy as np

import pandas as pd

# Look! No scikit learn!
df_train = pd.read_json(open("../input/train.json", "r"))

df_train.set_index("listing_id", inplace=True)

df_test  = pd.read_json(open("../input/test.json", "r"))

df_test.set_index("listing_id", inplace=True)

# We will work with a concatenation of the two, then split after the scaling.

df = pd.concat([df_train, df_test])
df["num_photos"] = df["photos"].apply(len)

df["num_features"] = df["features"].apply(len)

df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

df["created"] = pd.to_datetime(df["created"])

#df["created_year"] = df["created"].dt.year

df["created_month"] = df["created"].dt.month

df["created_day"] = df["created"].dt.day
df["logprice"] = np.log(df.price)
df.loc[df.bathrooms == 112, "bathrooms"] = 1
numeric_feat = ["bathrooms", "bedrooms", "latitude", "longitude", "logprice",

             "num_photos", "num_features", "num_description_words",

              "created_month", "created_day"]

for col in numeric_feat:

    df[col] -= df[col].min()

    df[col] /= df[col].max()
X_train = df.loc[df.interest_level.notnull(), numeric_feat]

y_train = pd.get_dummies(df_train[["interest_level"]], prefix="")

y_train = y_train[["_high", "_medium", "_low"]]  # Set the order according to submission

X_test  = df.loc[df.interest_level.isnull(), numeric_feat]
del df

del df_train

del df_test
## A dead simple neural network class in Python+Numpy. Plain SGD, and no regularization.

def sigmoid(X):

    return 1.0 / ( 1.0 + np.exp(-X) )



def softmax(X):

    _sum = np.exp(X).sum()

    return np.exp(X) / _sum



class neuralnet(object):

    def __init__(self, num_input, num_hidden, num_output):

        self._W1 = (np.random.random_sample((num_input, num_hidden)) - 0.5).astype(np.float32)

        self._b1 = np.zeros((1, num_hidden)).astype(np.float32)

        self._W2 = (np.random.random_sample((num_hidden, num_output)) - 0.5).astype(np.float32)

        self._b2 = np.zeros((1, num_output)).astype(np.float32)



    def forward(self,X):

        net1 = np.matmul( X, self._W1 ) + self._b1

        y = sigmoid(net1)

        net2 = np.matmul( y, self._W2 ) + self._b2

        z = softmax(net2)

        return z,y



    def backpropagation(self, X, target, eta):

        z, y = self.forward(X)

        d2 = (z - target)

        d1 = y*(1.0-y) * np.matmul(d2, self._W2.T)

        # The updates are done within this method. This more or less implies

        # utpdates with Stochastic Gradient Decent. Let's fix that later.

        # TODO: Support for full batch and mini-batches etc.

        self._W2 -= eta * np.matmul(y.T,d2)

        self._W1 -= eta * np.matmul(X.reshape((-1,1)),d1)

        self._b2 -= eta * d2

        self._b1 -= eta * d1
# Some hyper-parameters to tune.

num_hidden = 17    # I think I get about 1 epoch/sec with this size on the docker instance

n_epochs   = 100

eta        = 0.01
nn = neuralnet( X_train.shape[1], num_hidden, y_train.shape[1])
def logloss( nn, X, Y ):

    err = 0

    for apartment, target in zip( X, Y ):

        probs = nn.forward( np.array(apartment, dtype=np.float32))[0][0]

        err += sum(target*np.log(probs))

    return -err/X.shape[0]
# It's much faster to convert the dataframes to numpy arrays and then iterate.

X = np.array(X_train, dtype=np.float32)

Y = np.array(y_train, dtype=np.float32)

for epoch in range(n_epochs):

    print("Epoch: {:3d} train-error: {}".format(epoch, logloss(nn, X, Y)), end='\r')    

    for apartment, target in zip(X,Y):

        nn.backpropagation( apartment, target, eta)
with open('submission-{}-hidden.csv'.format(num_hidden), 'w') as f:

    f.write("listing_id,high,medium,low\n")

    for index, apartment in X_test.iterrows():

        probs = nn.forward( np.array(apartment, dtype=np.float32))[0][0]

        f.write("{},{},{},{}\n".format(index, *probs))