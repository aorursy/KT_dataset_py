import os.path



import numpy as np

import pandas as pd

from keras.layers import Dense, Input, Dropout

from keras.models import Model, load_model

from keras import optimizers

from keras import regularizers

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline

base_path = '/kaggle/input/titanic/'

model_file = 'mod.h5'

performance_file = base_path + 'perf.csv'

trainset = base_path + 'train.csv'

epochs = 3200

np.random.seed(1244)

def process_X(df):

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    X['Sex'] = X['Sex'].map({"male": 0, "female":1}) 

    # fill Embarked with the most common value

    X['Embarked'].fillna("S", inplace=True)

    X['Embarked'] = X['Embarked'].map({'S': 0, 'C':1, 'Q': 2})

    # Fill age with the negative value

    X['Age'].fillna(-1, inplace=True)

    # Extract title

    dataset_title = pd.Series([i.split(",")[1].split(".")[0].strip() for i in df["Name"]])

    le = preprocessing.LabelEncoder()

    le.fit(dataset_title)

    X["Title"] = le.transform(dataset_title)

    X['Title'].fillna(-1, inplace=True)

    

    X = (X - X.mean()) / X.std()

    return X

def read_dataset(filename):

    df = pd.read_csv(filename, low_memory=False)

    msk = np.random.rand(len(df)) < 0.85

    dev = df[~msk]

    train = df[msk]



    X_train = process_X(df)

    Y_train = df['Survived'].values



    return X_train, Y_train

X_train, Y_train = read_dataset(trainset)

print(X_train.columns)
def gen_model(input_shape):



    biasl1=0

    kernell1=0

    biasl2=0.03

    kernell2=0.03



    X_input = Input(shape=input_shape)



    X = Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2))(X_input)

    X = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2))(X)

    X = Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2))(X)

    X = Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2))(X)

    X = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2))(X)



    m = Model(inputs=X_input, outputs=X)

    m.summary()

    opt = optimizers.Adam(learning_rate=0.0001, decay=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    m.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])



    return m

model = None

if os.path.exists(model_file):

    print("Going to load model")

    model = load_model(model_file)

else:

    print("Going to generate a new model")

    model = gen_model(input_shape=(X_train.shape[1],))

model_result = model.fit(X_train, Y_train, batch_size=64, epochs=2500, validation_split=0.2, shuffle=True, verbose=2)

plt.figure(figsize=(30, 10))



plt.subplot(1, 2, 1)

plt.plot(model_result.history["loss"], label="training")

plt.plot(model_result.history["val_loss"], label="validation")

plt.axhline(0.55, c="red", linestyle="--")

plt.axhline(0.35, c="yellow", linestyle="--")

plt.axhline(0.15, c="green", linestyle="--")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(model_result.history["accuracy"], label="training")

plt.plot(model_result.history["val_accuracy"], label="validation")

plt.axhline(0.75, c="red", linestyle="--")

plt.axhline(0.80, c="green", linestyle="--")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()



plt.show()
def generateSubmission(m):

    df = pd.read_csv(base_path + 'test.csv', low_memory=False)

    X_test = process_X(df)

    Y = m.predict(X_test)

    ids = range(892, 1310)



    submission = open("submission.csv", "w+")

    submission.write("PassengerId,Survived\n")



    for i in range(0, len(ids)):

        id = ids[i]

        if Y[i] > 0.5:

            survived = 1

        else:

            survived = 0

        submission.write(str(id) + ',' + str(survived) + '\n')

generateSubmission(model)
model.save("mod.h5")