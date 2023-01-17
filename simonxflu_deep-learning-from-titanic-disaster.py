import pandas as pd

import numpy as np

from sklearn.preprocessing import RobustScaler

import keras

from keras import backend as K

from keras.models import Sequential

from keras.models import Model

from keras.layers.core import Dense, Dropout, Activation, Reshape

from keras.layers import Input

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# fix random seed for reproducibility

RANDOM_STATE = 7

np.random.seed(RANDOM_STATE)
train = pd.read_csv('../input/train.csv',dtype={'Age': np.float32, 'Fare': np.float32})

# get rid of the useless cols

train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)



train.info()

train.head()
def add_titles(dataFrame):

    # we extract the title from each name

    dataFrame['Title'] = dataFrame['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles

    Title_Dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Lady" :      "Royalty",

                        "Mr" :        "Mr",

                        "Mme":        "Mrs",

                        "Ms":         "Mrs",

                        "Mrs" :       "Mrs",

                        "Mlle":       "Miss",

                        "Miss" :      "Miss",

                        "Master" :    "Master"

                        }    

    # we map each title

    dataFrame['Title'] = dataFrame.Title.map(Title_Dictionary)



add_titles(train)

train.drop(['Name'], axis=1, inplace=True)

train.head()
train['Age'].isnull().sum()

# there are 173 other missing ages, fill with random int

#age_mean = train['Age'].mean()

#age_std = train['Age'].std()

#filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)

#train.loc[train['Age'].isnull(), 'Age'] = filling

#nan_num = train['Age'].isnull().sum()



#look into the age col

s = sns.FacetGrid(train,hue='Survived',aspect=3)

s.map(sns.kdeplot,'Age',shade=True)

s.set(xlim=(0,train['Age'].max()))

s.add_legend()
# Combine Sibsp and Parch features to Family feature

# check

print(train['SibSp'].value_counts(dropna=False))

print(train['Parch'].value_counts(dropna=False))



sns.factorplot('SibSp','Survived',data=train,size=5)

sns.factorplot('Parch','Survived',data=train,size=5)



'''through the plot, we suggest that with more family member, the survival rate will drop, we can create the new col

add up the parch and sibsp to check our theory''' 

train['Family'] = train['SibSp'] + train['Parch']

sns.factorplot('Family','Survived',data=train,size=5)



train.drop(['SibSp','Parch'],axis=1,inplace=True)
# fare research

train.Fare.isnull().sum()

sns.factorplot('Survived','Fare',data=train,size=5)

#according to the plot, smaller fare has higher survival rate
#Cabin feature research

# checking missing val, 687 out of 891 are missing, drop this col

train.Cabin.value_counts(dropna=False)

train.drop('Cabin',axis=1,inplace=True)
#Embark feature search

# 2 missing value

train.Embarked.value_counts(dropna=False)

# fill the majority val,'s', into missing val col

train['Embarked'].fillna('S',inplace=True)



sns.factorplot('Embarked','Survived',data=train,size=6)

# c has higher survival rate
# Define a few feature preparation helpers

def normalize(df, column):

    series = df[column]

    min = series.min()

    max = series.max()

    if (min==max):

        print(series.name + ' has only one value and should be removed.')

    scaler = RobustScaler()

    x = scaler.fit_transform(series.values.reshape(-1,1)).reshape(-1)

    df[column] = pd.Series(x)

    return scaler



def transform(df, column, scaler):

    series = df[column]

    x = scaler.transform(series.values.reshape(-1,1)).reshape(-1)

    df[column] = pd.Series(x)



def encode_one_hot(df, column, axis=1):

    x = df.join(pd.get_dummies(df[column], prefix=column, sparse=True))

    x.drop(column, axis=axis, inplace=True)

    return x



def FeaturesImportances(X_train, y_train, featureNames):

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.feature_selection import SelectFromModel

    rfr = RandomForestClassifier()

    rfr.fit(X_train, y_train)



    sfm = SelectFromModel(rfr, prefit=True, threshold=0)

    selected = sfm.get_support()

    names = featureNames[selected]

    scores = rfr.feature_importances_[selected]

    importances = pd.DataFrame({'feature':names,'importance':np.round(scores,5)})

    importances = importances.sort_values('importance').set_index('feature')

    #importances.to_csv("importances.csv")

    #print("Selected {} features".format(len(names)))

    #print(importances)

    #importances.plot.bar()

    importances.plot(kind='barh', figsize=(20, 20))

    #plt.show()

    return sfm
#Encoding training features



print("Encoding Pclass categorical features...")

train = encode_one_hot(train, 'Pclass')

print("Encoding Sex categorical features...")

train = encode_one_hot(train, 'Sex')

print("Encoding Embarked categorical features...")

train = encode_one_hot(train, 'Embarked')

print("Encoding Title categorical features...")

train = encode_one_hot(train, 'Title')



#scale only numeric features

print("scaling numeric features...")

#ageScaler = normalize(train, "Age")

fareScaler = normalize(train, "Fare")

familyScaler = normalize(train, "Family")
# Train age prediction model to predict missing ages

def getAgePredictionModel(dataFrame):

    def CreateModel(X):

        ip = Input(shape=(X.shape[1],))

        x_list = [ip]



        x = Dense(128, use_bias=False)(ip)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Dropout(0.5)(x)



        x_list.append(x)

        x = keras.layers.concatenate(x_list)    

        x = Dense(64, use_bias=False)(x)    

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Dropout(0.5)(x)



        x_list.append(x)

        x = keras.layers.concatenate(x_list)    

        x = Dense(32, use_bias=False)(x)    

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Dropout(0.5)(x)



        op = Dense(1)(x)



        model = Model(inputs=ip, outputs=op)

        adam = Adam(lr=0.05,)

        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['MAE'])

        return model



    trainCpy = dataFrame[dataFrame['Age'].isnull()==False]

    age = trainCpy['Age']

    trainCpy.drop(['Age'], axis=1, inplace=True)

    if 'Survived' in trainCpy.columns:

        trainCpy.drop('Survived', axis=1, inplace=True)

    trainCpy.head()



    model = CreateModel(trainCpy)



    print("Training age model...")

    model.fit(trainCpy.as_matrix(), age.values, epochs=500, batch_size=32, verbose=0)

    return model



def fillMissingAges(dataFrame, agePredictionModel):

    #predict missing ages

    tmp = dataFrame[dataFrame['Age'].isnull()==True]

    tmp.drop('Age', axis=1, inplace=True)

    if 'Survived' in tmp.columns:

        tmp.drop('Survived', axis=1, inplace=True)

    age_pred = agePredictionModel.predict(tmp.as_matrix())

    #fill missing ages

    dataFrame.loc[dataFrame['Age'].isnull(), 'Age'] = age_pred

    return dataFrame



agePredictionModel = getAgePredictionModel(train)

fillMissingAges(train, agePredictionModel)



#look into the age col

s = sns.FacetGrid(train, hue='Survived', aspect=3)

s.map(sns.kdeplot,'Age',shade=True)

s.set(xlim=(0,train['Age'].max()))

s.add_legend()



#nomalize age

ageScaler = normalize(train, "Age")



y = train['Survived']

train.drop('Survived', axis=1, inplace=True)

X = train
# Check feature importances, I keep all the features in X because it has only

# a few feautures.

FeaturesImportances(X, y, X.columns.values)
def DenseNet(X_train):

    ip = Input(shape=(X_train.shape[1],))

    x_list = [ip]

    

    x = Dense(128, use_bias=False)(ip)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(128, use_bias=False)(x)    

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(64, use_bias=False)(x)    

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(64, use_bias=False)(x)    

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(32, use_bias=False)(x)    

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(32, use_bias=False)(x)    

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(16, use_bias=False)(x)    

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

    

    x_list.append(x)

    x = keras.layers.concatenate(x_list)    

    x = Dense(16, use_bias=False)(ip)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)    

    

    op = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=ip, outputs=op)

    adam = Adam(lr=0.05,)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model



model = DenseNet(X)
# Training plots

from IPython import display

plt.rcParams['figure.figsize'] = (10, 10)

class PlotTraining(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.accs = []

        self.val_accs = []



        f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        self.fig = f

        self.ax1 = ax1

        self.ax2 = ax2



    def on_epoch_end(self, epoch, logs={}):        

        if (self.i%100==0):

            self.x.append(self.i)

            self.losses.append(logs.get('loss'))

            self.val_losses.append(logs.get('val_loss'))

            self.accs.append(logs.get('acc'))

            self.val_accs.append(logs.get('val_acc'))



            display.clear_output(wait=True)

            self.ax1.clear()

            self.ax1.plot(self.x, self.losses, label="Train")

            self.ax1.plot(self.x, self.val_losses, label="Validation")

            self.ax1.set_ylabel('Loss')

            self.ax1.legend()



            self.ax2.clear()

            self.ax2.plot(self.x, self.accs, label="Train")

            self.ax2.plot(self.x, self.val_accs, label="Validation")

            self.ax2.set_ylabel('Accuracy')

            self.ax2.set_xlabel('Epoch')

            self.ax2.legend()

            display.display(plt.gcf())

        self.i += 1



trainCallback = PlotTraining()
# Train the model with 40000 epochs

EPOCHS = 40000

BATCH_SIZE = 64

print("Training..., it may take 1 hour or 2.")

model.fit(X.as_matrix(), y.values, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,

          validation_split=0.1,

          callbacks=[trainCallback])

# evaluate the model

scores = model.evaluate(X.as_matrix(), y.values, verbose=0)

print("%s: %.3f%%" % (model.metrics_names[1], scores[1]*100))
# Predicting with test data

test = pd.read_csv('../input/test.csv',dtype={'Age': np.float32,'Fare': np.float32})

passengerId = test['PassengerId']



#add title

add_titles(test)

#test.drop(['Name'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)



#dealing missing fare

test['Fare'].fillna(test['Fare'].median(), inplace=True)



#create Family feature

test['Family'] = test['SibSp'] + test['Parch']

test.drop(['SibSp','Parch'],axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)



#dealing missing Embarked

test['Embarked'].fillna('S',inplace=True)



# encoding test features

test = encode_one_hot(test, 'Pclass')

test = encode_one_hot(test, 'Sex')

test = encode_one_hot(test, 'Embarked')

test = encode_one_hot(test, 'Title')

transform(test, "Fare", fareScaler)

transform(test, "Family", familyScaler)



# dealing the missing age

agePredictionModel = getAgePredictionModel(test)

fillMissingAges(test, agePredictionModel)

test['Age'].isnull().sum()

transform(test, "Age", ageScaler)



X_test = test

# Make sure X_test and X have the same dimentions in same sequence so that X_test fits the input of model

X.columns == X_test.columns



y_pred = model.predict(X_test.as_matrix())

y_pred = (y_pred > 0.5).astype('int32')
#submit

submission = pd.DataFrame({

        "PassengerId": passengerId,

        "Survived": y_pred.reshape(-1)

    })

print(submission['Survived'].value_counts(dropna=False))

submission.to_csv("prediction.csv", index=False)

print("Submitted.")