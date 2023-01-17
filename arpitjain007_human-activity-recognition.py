import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from keras.models import Sequential

from keras.layers import LSTM ,Dense, Dropout

from keras.optimizers import SGD, Adam

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
train_path = "../input/human-activity-recognition/uci_har_dataset/UCI_HAR_Dataset/train/"

test_path = "../input/human-activity-recognition/uci_har_dataset/UCI_HAR_Dataset/test/"

features_path = "../input/human-activity-recognition/uci_har_dataset/UCI_HAR_Dataset/features.txt"
features = []

with open(features_path) as f:

    features = [line.split()[1] for line in f.readlines()]

print('No of Features: {}'.format(len(features)))

print("No. of unique features:{}".format(len(set(features))))
#LABELS

labels = {1: 'WALKING', 

          2:'WALKING_UPSTAIRS',

          3:'WALKING_DOWNSTAIRS',

          4:'SITTING',

          5:'STANDING',

          6:'LAYING'}
re=[]

for i , f in enumerate(features):

    for j in range(i+1 , len(features)):

        if features[i]==features[j] and features[i] not in re:

            re.append(features[i])
for i , f in enumerate(features):

    features[i] = ''.join(e for e in f if e not in ['(',')' , '-' , ',']) 
train = pd.read_csv(train_path + "X_train.txt" , delim_whitespace=True ,header=None)

train.columns = features

train['subject'] = pd.read_csv(train_path + 'subject_train.txt' , header=None , squeeze=True)

test = pd.read_csv(test_path + "X_test.txt" , delim_whitespace=True ,header=None)

test.columns = features

test['subject'] = pd.read_csv(test_path + 'subject_test.txt' , header=None , squeeze=True)
train.head()
test.head()
y_train = pd.read_csv(train_path + 'y_train.txt' , names=['Activity'] , squeeze=True)

y_test = pd.read_csv(test_path + 'y_test.txt' , names=['Activity'] , squeeze=True)
train['Activity']= y_train

test['Activity'] = y_test

train['ActivityName'] = y_train.map(labels)

test['ActivityName']  = y_test.map(labels)
print("The number of missing values in Training Data:" , train.isnull().values.sum())

print("The number of missing values in Testing Data:" , test.isnull().values.sum())
print("The number of duplicate values in Training Data:" , train.duplicated().sum())

print("The number of duplicate values in Testing Data:" , test.duplicated().sum())
plt.figure(figsize=(10,5))

plt.title('Subject Wise Data Distribution')

sns.countplot(x='subject' , data=train )

plt.show()
plt.figure(figsize=(25,10))

plt.title('Activity based Subject Distribution')

sns.countplot(x='subject' , hue='ActivityName', data=train )

plt.show()
accFeat=[]

for feat in features:

    if feat.find('BodyAcc') != -1 and feat.find('Magmean') !=-1 and feat.find('Freq')==-1:

        accFeat.append(feat)
def plotFacetGrid(feature, height):

    

    plt.figure(figsize=(10,10))

    facetgrid=sns.FacetGrid(train , hue='ActivityName',height=height,aspect=3)

    facetgrid.map(sns.distplot ,feature, hist=False).add_legend()

    plt.show()
for f in accFeat:

    plotFacetGrid(f,3) 
def boxplot(feature , ylabel):

    

    plt.figure(figsize=(5,5))

    sns.boxplot(x='ActivityName', y=feature, data=train , showfliers=False )

    plt.ylabel(ylabel)

    plt.axhline(y=-0.8, xmin=0.1, xmax=0.9,dashes=(5,5), c='g') #line separating both type of activities

    plt.xticks(rotation=90)
for f in accFeat:

    boxplot(f , f[5:])
from sklearn.manifold import TSNE
def plotTsne(X,y,perplexity):

    

    #performing dim reduction

    X_reduce = TSNE(verbose=2, perplexity=perplexity).fit_transform(X)

    

    print('Creating plot for this t-sne visualization..')

    data={'x':X_reduce[:,0],

          'y':X_reduce[:,1],

         'label':y}

    #preparing dataframe from reduced data

    df = pd.DataFrame(data)

    

    #draw the plot

    sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, height=8,\

                   palette="Set1",markers=['^','v','s','o', '1','2'])

    

    plt.title("perplexity : {}".format(perplexity))

    img_name = 'TSNE_perp_{}.png'.format(perplexity)

    print('saving this plot as image in present working directory...')

    plt.savefig(img_name)

    plt.show()

    print('Done')

    
X= train.drop(['ActivityName'],axis=1)

y= train['ActivityName']

perplexity=[2,5,10]
for p in perplexity:

    plotTsne(X,y,perplexity=p)
# Activities are the class labels

# It is a 6 class classification

ACTIVITIES = {

    0: 'WALKING',

    1: 'WALKING_UPSTAIRS',

    2: 'WALKING_DOWNSTAIRS',

    3: 'SITTING',

    4: 'STANDING',

    5: 'LAYING',

}



"-----------------------RAW DATA--------------------------------------"



# Raw data signals

# Signals are from Accelerometer and Gyroscope

# The signals are in x,y,z directions

# Sensor signals are filtered to have only body acceleration

# excluding the acceleration due to gravity

# Triaxial acceleration from the accelerometer is total acceleration

SIGNALS = [

    "body_acc_x",

    "body_acc_y",

    "body_acc_z",

    "body_gyro_x",

    "body_gyro_y",

    "body_gyro_z",

    "total_acc_x",

    "total_acc_y",

    "total_acc_z"

]
path= "../input/human-activity-recognition/uci_har_dataset/UCI_HAR_Dataset/"
# Utility function to read the data from csv file

def _read_csv(filename):

    return pd.read_csv(filename, delim_whitespace=True, header=None)



"----------------------------LOAD SIGNAL---------------------------------------------"



# Utility function to load the load

def load_signals(subset):

    signals_data = []



    for signal in SIGNALS:

        filename = path+subset+'/Inertial Signals/'+signal+'_'+subset+'.txt'

        signals_data.append(

            _read_csv(filename).values

        ) 



    # Transpose is used to change the dimensionality of the output,

    # aggregating the signals by combination of sample/timestep.

    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)

    return np.transpose(signals_data, (1, 2, 0))



"-------------------------CONFUSION MATRIX----------------------------------------------"



# Utility function to print the confusion matrix

def confusion_matrix(Y_true, Y_pred):

    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])

    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])



    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])





"----------------------------LOAD Y-------------------------------------------------------"







def load_y(subset):

    """

    The objective that we are trying to predict is a integer, from 1 to 6,

    that represents a human activity. We return a binary representation of 

    every sample objective as a 6 bits vector using One Hot Encoding

    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

    """

    filename = path+subset+'/y_'+subset+'.txt'

    y = _read_csv(filename)[0]



    return pd.get_dummies(y).values







"---------------------------------LOAD DATA---------------------------------------------"





def load_data():

    """

    Obtain the dataset from multiple files.

    Returns: X_train, X_test, y_train, y_test

    """

    X_train, X_test = load_signals('train'), load_signals('test')

    y_train, y_test = load_y('train'), load_y('test')



    return X_train, X_test, y_train, y_test





"---------------------------------COUNT CLASSES--------------------------------------------"



# Utility function to count the number of classes

def _count_classes(y):

    return len(set([tuple(category) for category in y]))
# Initializing parameters

epochs = 30

batch_size = 16

n_hidden = 32
# Loading the train and test data

X_train, X_test, Y_train, Y_test = load_data()
timesteps = len(X_train[0])

input_dim = len(X_train[0][0])

n_classes = _count_classes(Y_train)



print(timesteps)

print(input_dim)

print(len(X_train))
# Initiliazing the sequential model

model = Sequential()

# Configuring the parameters

model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))

# Adding a dropout layer

model.add(Dropout(0.5))

# Adding a dense output layer with sigmoid activation

model.add(Dense(n_classes, activation='sigmoid'))

model.summary()
# Compiling the model

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
# Training the model

model.fit(X_train,

          Y_train,

          batch_size=batch_size,

          validation_data=(X_test, Y_test),

          epochs=epochs)