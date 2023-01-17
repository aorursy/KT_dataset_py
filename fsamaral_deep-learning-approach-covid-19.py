# base libs

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

# model libs

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder

from scipy import stats
# dataframe information

def data_info(df):

    

    null_series = df.isnull().sum()

    print(f"######### MISSING VALUES:")

    print(f"{null_series[null_series > 0]}\n")

    

    print(f"######### LABEL BALANCE:")

    print(f"{pd.value_counts(df['label'])}\n")

    

    print(f"######### DESCRIBE:")

    print(f"{df.describe()}\n")

    

    for column in df:

        print('######### COLUMN NAME:', column)

        print('TYPE:', df[column].dtypes)

        print('LEN:', len(df[column]))

        print('NUNIQUE:', df[column].nunique())

        print('NaN:', df[column].isnull().values.sum())

        print('')

        

    print(msno.bar(df, figsize=(16, 4)))  

    



# prepare neural network input

def df_to_nnetwork(df, normalization=True, verbose=False):

    

    # dataframe to neural network input format

    dataset = df.values

    nb_features = len(chosen_feat)



    X = dataset[:,1:nb_features].astype(float)

    Y = dataset[:,0]

    if verbose:

        print(f"Raw data Stats:")

        print(f"{stats.describe(X)}\n")



    # data normalization

    if normalization:

        X = (X - X.min()) / (X.max() - X.min())

        if verbose:

            print(f"Mean Normalized data Stats:")

            print(f"{stats.describe(X)}\n")



    # encode class values as integers

    encoder = LabelEncoder()

    encoder.fit(Y)

    encoded_Y = encoder.transform(Y)



    # convert integers to dummy variables (i.e. one hot encoded)

    Y = to_categorical(encoded_Y)

    if verbose:

        print(f"######### NEURAL NETWORK INPUT FORMAT:")

        print(f"Data Shape: {X.shape}")

        print(f"Labels Shape: {Y.shape}")

        

    return X, Y





# model build    

def build_model(X, Y, verbose=False):

    

    # definitions

    neurons = 16

    input_dim = len(X[0])

    depth = 3



    # define baseline model

    def baseline_model():



        # create model

        model = Sequential()

        for i in range(depth):

            model.add(Dense(neurons, input_dim=input_dim, activation='relu'))

        model.add(Dense(2, activation='softmax'))

        # Compile model

        model.compile(loss='categorical_crossentropy', 

                      optimizer='adam', 

                      metrics=['accuracy'])



        return model

        

    model = baseline_model() 

    if verbose:

        print(model.summary())

           

    return model  





# quantitative model metrics

def result_metrics(data, labels, model):



    y_pred = model.predict(data)

    y_pred = (y_pred > 0.5)

    print('Accuracy: %.2f%%' % (accuracy_score(labels, y_pred)*100))



    cm = confusion_matrix(labels.argmax(axis=1), 

                          y_pred.argmax(axis=1))

    print('\nConfusion Matrix:\n',cm)



    print("\nMetrics:\n", classification_report(labels, y_pred))

    print("")

    



# learning curves visualization    

def model_plots(history):

     

    plt.figure(figsize=(18, 6))

    

    # summarize history for accuracy

    plt.subplot(1, 2, 1)

    plt.plot(history['accuracy'])

    plt.title(f"model accuracy")

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='best')



    # summarize history for loss

    plt.subplot(1, 2, 2)

    plt.plot(history['loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper right')



    plt.show()

# definitions

label = 'SARS-Cov-2 exam result'



# load data from excel

path = '/kaggle/input/covid19/'

df = pd.read_excel(path+"dataset.xlsx")



print(f"######### DATASET FEATURES:")

feat_pool = []

for column in df:

    if column != label:

        feat_pool.append(column)

print(feat_pool)  



# shuffle the rows reseting indexs

df = df.sample(frac=1, random_state=42).reset_index(drop=True)



# show dataset balance

print(f"\n######### LABELS BALANCING:")

df[label].value_counts().plot.barh()
# data preprocess config

balance_data = True

drop_nan = True



# feature list

chosen_feat = [df['SARS-Cov-2 exam result'], 

               df['Leukocytes'], 

               df['Platelets'], 

               df['Eosinophils'],

               df['Monocytes'],

               #df['Neutrophils'],

               #df['Hematocrit'],

               #df['Hemoglobin']

              ]

keys=['label', 

      'Leukocytes', 

      'Platelets', 

      'Eosinophils',

      'Monocytes',

      #'Neutrophils',

      #'Hematocrit',

      #'Hemoglobin'

     ]



# create a reduced dataframe with keys features 

df = pd.concat(chosen_feat, axis=1, keys=keys)



# drop NaN rows 

if drop_nan:

    df = df.dropna(axis=0).reset_index(drop=True) 



# balancing data (same samples of positive and negatives)    

if balance_data:

    df_pos = df[df["label"]=="positive"]

    df_neg = df[df["label"]=="negative"]

    df_neg = df_neg.sample(n=len(df_pos), random_state=42)

    # append a dataframe from another one

    df = df_pos.append(df_neg, ignore_index=True)

    df = df.sample(n=len(df), random_state=42).reset_index(drop=True)

else:

    df = df.sample(n=len(df), random_state=42).reset_index(drop=True)

    

# general information about the data

data_info(df)    
# plot correlations between best features

feats_set = keys[1:]

print("Features Set Correlations:", feats_set)

sns.pairplot(df, hue='label')
# heat map

print("Features Heat Map:")

plt.figure(figsize=(10,8))

myBasicCorr = df[feats_set].corr('spearman')

sns.heatmap(myBasicCorr, annot = True)
# prepare neural network input

data, labels = df_to_nnetwork(df, normalization=True, verbose=True)
# train paramters

batch_size = 5

epochs = 300



# build the deep learning model

model = build_model(data, labels, verbose=False)  



# run!!

History = model.fit(data, 

                    labels, 

                    batch_size=batch_size, 

                    epochs=epochs)
# visualization of learning curves

print("\nLearning Curves:")

model_plots(History.history)



# quantitative metrics

print(f"Features: {keys[1:]}\n")

result_metrics(data, labels, model)