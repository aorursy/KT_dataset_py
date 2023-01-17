import math
import numpy as np
import pandas as pd

import re
import time

np.random.seed(1)

from matplotlib import pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
# load data from csv files
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# view the first few rows of the training data
train.head()
# view some basic statistics about the training data
train.describe()
# analyze for null values
train.isnull().sum()
# making a copy of the data to use for analysis
df_analysis = train.copy()
sns.set()
sns.set_palette("muted")

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12,4))

sns.countplot(x="Pclass", hue="Survived", data=df_analysis, ax=ax[0])
ax[0].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
ax[0].set_title('Survival by Class')

sns.countplot(x="Sex", hue="Survived", data=df_analysis, ax=ax[1])
ax[1].set_title('Survival by Gender')
ax[1].set_ylabel('')

sns.countplot(x="Embarked", hue="Survived", data=df_analysis, ax=ax[2])
ax[2].set_title('Survival by Port')
ax[2].set_ylabel('')
# get dataset for analysis of ages
df_analysis = train.copy()
df_analysis = df_analysis[['Survived', 'Sex', 'Age']]

# split ages into groups
df_analysis['AgeGroup'] = pd.cut(df_analysis['Age'].fillna(-1), bins=(-10, 0, 10, 20, 55, 150), labels=['unknown','child','teen','adult','senior'])
df_analysis.head()
sns.set_style("whitegrid")

fig, ax = plt.subplots()

g = sns.countplot(y="AgeGroup", data=df_analysis, palette='muted', ax=ax)
ax.set_title('Breakdown of Age Groups')
ax.set_ylabel('Age Group')

g = sns.factorplot(y="AgeGroup", hue="Survived", col="Sex", data=df_analysis, kind="count", size=4, aspect=1.5)
g.set_axis_labels("count", "Age Group")
df_analysis = train.copy()
df_analysis['Cabin'].str.extract('(?P<cabin>[A-Z])\d*', expand=True).fillna('NA').head()
# function to extract adjusted titles (with some grouped)
def get_adjusted_title(name):
    srch = re.search('([A-Za-z]+)\.', name)
    
    if srch == None:
        title = 'None'
    else:
        title = srch.groups(0)[0]
        
    if title in ['Mr', 'Mrs', 'Miss', 'Ms', 'Mme', 'Mlle']:
        title = 'Gender'
    elif title in ['Master']:
        title = 'Master'
    elif title in ['Don', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer']:
        title = "Rare"
    else:
        title = "Other"
        
    return title
    
    
# this preprocesses the data in a format that is human-friendly so that I can inspect it
def preprocessData(data):
    x = data.copy()
    
    # break Fare into groups
    x['FareGroup'] = pd.cut(x['Fare'].fillna(-1), bins=[-1, 8, 14.5, 31, 1000]).cat.codes
          
    # encode Sex
    x['Sex'] = x['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
      
    # extract title from name and break into groups
    x['TitleGroup'] = x['Name'].apply(lambda n: get_adjusted_title(n))
    
    # break Age into groups
    x['Age'].fillna(0, inplace=True)
    x["AgeGroup"] = pd.cut(x['Age'].fillna(-1), bins=[-10, 0, 10, 20, 55, 150], labels=['unknown','child','teen','adult','senior'])
          
    # extract Cabin letter
    x['Cabin-Letter'] = x['Cabin'].str.extract('(?P<cabin>[A-Z])\d*', expand=True).fillna('NA')
    x['HasCabin'] = x['Cabin-Letter'] != 'NA'
    
    # Embarked
    x['Embarked'].fillna('UNK', inplace=True)
    
    # family size
    x['FamilySize'] = x['SibSp'] + x['Parch']
    x['IsAlone'] = x['FamilySize'] == 0
    
    if 'Survived' in data.columns:
        y = data['Survived']
    else:
        y = None
    
    return x, y


# this preprocesses the data in a format that is better to feed to a neural network 
def preprocessDataForModel(data):
    
    x, y = preprocessData(data)
    
    def oneHotEncode(dataSet, encodedColumn, validValues, prefix):
        for val in validValues:
            dataSet["{0}_{1}".format(prefix,val)] = (val == dataSet[encodedColumn]).astype(int)
    
    x['IsAlone'] = x['IsAlone'].astype(int)
    #x['HasCabin'] = x['HasCabin'].astype(int)
    
    oneHotEncode(x, 'FareGroup', validValues=range(4), prefix='f')
    oneHotEncode(x, 'AgeGroup', validValues=['unknown','child','teen','adult','senior'], prefix='a')
    oneHotEncode(x, 'TitleGroup', validValues=['Master','Rare','Other'], prefix='t')
    oneHotEncode(x, 'Embarked', validValues=['S','C','Q'], prefix='e')
    
    x = x.drop(columns=['Name','Age','AgeGroup','Cabin','Cabin-Letter','HasCabin','SibSp','Parch','Ticket','TitleGroup','Fare','FareGroup','Embarked'])
    if 'Survived' in x.columns:
        x = x.drop(columns=['Survived'])
    
    x = x.set_index('PassengerId')
    
    return x, y

# plot results of training (we'll use this after training the model)
def displayModelHistory(model_history):
    graph_history = model_history
    
    fig = plt.figure(figsize=(12,5))
    fig.suptitle('Training - Accuracy and Loss', fontsize=12)

    plt.subplot(121)
    line1, = plt.plot(range(1, len(graph_history['acc'])+1), graph_history['acc'], label='training')
    line2, = plt.plot(range(1, len(graph_history['val_acc'])+1), graph_history['val_acc'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(handles=[line1, line2])
    plt.grid(True)

    plt.subplot(122)
    line1, = plt.plot(range(1, len(graph_history['loss'])+1), graph_history['loss'], label='training')
    line2, = plt.plot(range(1, len(graph_history['val_loss'])+1), graph_history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(handles=[line1, line2])
    plt.grid(True)
# human friendly data
x_temp, _ = preprocessData(train)

# NN friendly data
x_train, y_train = preprocessDataForModel(train)
x_test, _ = preprocessDataForModel(test)

assert len(x_train.columns) == len(x_test.columns), 'Number of columns in x_train and x_test do not match'

# displaying the more human-readable format.  The one-hot encoded data will be fed to the NN
features = ['PassengerId','Survived','Pclass','Sex','Embarked','FareGroup','TitleGroup','AgeGroup','HasCabin','FamilySize','IsAlone']
x_temp = x_temp[features]
x_temp.head()
x_train.head()
num_epochs = 40
batch_size = 32
models = []
log = []

# create and train 5 models
for i in range(5):
    
    # create model
    model = Sequential()

    model.add(Dense(25, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # fit/train model
    hist = model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.3, verbose=0)

    log.append(hist.history)
    models.append(model)
    
    # evaluate model
    loss, acc = model.evaluate(x_train, y_train)
    print("loss: %s, accuracy: %s" % (loss, acc))
# display training accuracy and loss for one of the models
displayModelHistory(log[0])
# get all predictions from all trained models
y_hat = []
for m in models:
    y_hat.append(m.predict(x_test))
# helper method to aggregate results of different models 
# ensemble result can average the results of each model or let each model vote
def getEnsemblePrediction(predictions, method='average'):
    p = np.asarray(predictions)
    
    if p.ndim == 2:
        p = p.round()
    elif p.ndim == 3:
        if method == 'vote':
            p = p.round()
        p = np.average(p, axis=0)
        p = p.round()

    return np.array(p, dtype=int).reshape(-1,)
# get final predictions
y_pred = getEnsemblePrediction(y_hat, 'average')
out_df = pd.DataFrame(data={'PassengerId': test.PassengerId.values, 'Survived': y_pred})
out_df.to_csv('my_predictions_{0}.csv'.format(time.strftime('%Y%m%d-%H%M%S')), index=False)
