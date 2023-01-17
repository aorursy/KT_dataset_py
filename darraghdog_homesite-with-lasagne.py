%matplotlib inline
import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy, binary_accuracy
import theano

train = pd.read_csv('../input/train.csv')

y = train.QuoteConversion_Flag.values
encoder = LabelEncoder()
y = encoder.fit_transform(y).astype(np.int32)


train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

# Lets take out some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)
train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek
train = train.drop('Date', axis=1)

# we fill the NA's and encode categories
train = train.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        # print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))


# Now we prep the data for a neural net
X = train
num_classes = len(encoder.classes_)
num_features = X.shape[1]

# Convert to np.array to make lasagne happy
X = np.array(X)
X = X.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Take the first 200K to train, rest to validate
split = 200000 
epochs = 10
val_auc = np.zeros(epochs)
# Comment out second layer for run time.
layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           #('dense1', DenseLayer)
           #('dropout1', DropoutLayer),
           ('output', DenseLayer)
           ]
           
net1 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=200, # 512, - reduce num units to make faster
                 dropout0_p=0.1,
                 # dense1_num_units=256,
                 # dropout1_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=0.04,
                 eval_size=0.0,
              
                 # objective_loss_function = binary_accuracy,
                 verbose=1,
                 max_epochs=1)
for i in range(epochs):
    net1.fit(X[:split], y[:split])
    pred = net1.predict_proba(X[split:])[:,1]
    val_auc[i] = roc_auc_score(y[split:],pred)

from matplotlib import pyplot
pyplot.plot(val_auc, linewidth=3, label="first attempt")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("validation auc")
pyplot.show()