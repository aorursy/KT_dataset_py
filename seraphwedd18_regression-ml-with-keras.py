import numpy as np

import pandas as pd

import random

import time

import gc



from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



seed = random.randint(10, 10000)

seed = 2546 #From V21 of commit

print("This run's seed is:", seed)



np.random.seed(seed)

random.seed(seed)

def create_model(input_shape):

    model = Sequential()

    model.add(Dense(128, input_dim=input_shape, activation='sigmoid'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=input_shape, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(1))

    

    optimizer = Adam(lr=0.005, decay=0.)

    model.compile(optimizer=optimizer,

             loss='msle',

             metrics=['mse'])

    return model
def data_correction_numeric(keys):

    #We will access the global num_df for this

    for key in keys:

        mn, mx = abs(int(num_df[key].min())), abs(int(num_df[key].max()))

        if mx < mn:

            mn, mx = mx, mn

        print("Min:", mn, "Max:", mx)

        try:

            for suf in range(mn, mx, int((mx-mn)/3)):

                num_df[key+'>'+str(suf)] = num_df[key].map(lambda x: x>suf)

                num_df[key+'<'+str(suf)] = num_df[key].map(lambda x: x<suf)

        except:

            print("ERROR for %s" %key)

        x_val = num_df[key].median()

        num_df[key] = num_df[key].fillna(x_val)

        num_df[key] = num_df[key]-x_val



def data_correction_category(df, keys):

    for key in keys:

        x_val = 0#df[key].value_counts().median()

        df[key] = df[key].fillna(x_val)

    return df
#Read the input data

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

print(train.shape, test.shape)

train = train.set_index("Id")

test = test.set_index("Id")



test_index = test.index

#Clean the train and test data

combined = pd.concat([train, test], axis=0, sort=False)

print(combined.columns)



#Free some memory

del train, test



#Get the log(y) to minimize values

Y = combined[combined["SalePrice"].notnull()]["SalePrice"].sort_index().values

log_Y = np.log(Y)



del Y

gc.collect()
numeric_val_list = ["OverallQual", "GrLivArea", "YearBuilt", "MSSubClass", "OverallCond",

                    "GarageCars", "LotArea", "Fireplaces", "LotFrontage", "TotRmsAbvGrd",

                    "KitchenAbvGr", "FullBath"]

categorical_val_list = ["BsmtExposure", "BsmtFinType1", "Neighborhood", "BsmtQual", "MSZoning", "BsmtCond",

                        "Exterior1st", "KitchenQual", "Exterior2nd", "SaleCondition", "HouseStyle",

                        "LotConfig", "GarageFinish", "MasVnrType", "RoofStyle"]

num_df = combined[numeric_val_list]

cat_df = combined[categorical_val_list]



#Cleaning the data

data_correction_numeric(numeric_val_list)

cat_df = data_correction_category(cat_df, categorical_val_list)



cat_df = pd.get_dummies(cat_df)
num_df.columns
cat_df.columns
#Split Data to train and test

train_c = cat_df[cat_df.index <= 1460] 

test_c = cat_df[cat_df.index > 1460]

train_n = num_df[num_df.index <= 1460]

test_n = num_df[num_df.index > 1460]



del num_df, cat_df



scale = StandardScaler()



train_n = scale.fit_transform(train_n)

test_n = scale.fit_transform(test_n)



train = np.concatenate((train_n, train_c.values), axis=1)

test = np.concatenate((test_n, test_c.values), axis=1)



del train_c, train_n, test_c, test_n

gc.collect()
# summarize history for loss

import matplotlib.pyplot as plt

def plotter(history, n):

    plt.plot(history.history['mse'])

    plt.plot(history.history['val_mse'])

    plt.title('MODEL MSE #%i' %n)

    plt.ylabel('MSE')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper right')

    plt.ylim(top=.1, bottom=0.01)

    plt.savefig('history_mse_{}.png'.format(n))

    plt.show()
#Callbacks

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

lrr = ReduceLROnPlateau(monitor = 'val_mse',

                         patience = 200,

                         verbose = 1,

                         factor = 0.75,

                         min_lr = 1e-6)



es = EarlyStopping(monitor='val_loss',

                   mode='min',

                   verbose=1,

                   patience=1000,

                   restore_best_weights=True)



print("Shape of Train:", train.shape)

predictions = []



last_mse = []

folds = 10

for x in range(1, folds+1):

    #Separate train data into train and validation data

    X_train, X_val, Y_train, Y_val = train_test_split(train, log_Y, test_size=0.2, shuffle=True, random_state=seed)

    print("#"*72)

    print("Current RERUN: #%i" %(x))

    #Design the Model

    model = create_model(X_train.shape[1])

    #Start the training

    history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val),

                  epochs=10000, batch_size=128, verbose=0,

                 callbacks=[es, lrr])

    #Predicting

    predict=model.predict(test)

    try:

        predictions = np.concatenate([predictions, predict], axis=1)

    except:

        predictions = predict

    #Show the MSE Plot

    plotter(history, x)

    loss, mse = model.evaluate(X_val, Y_val)

    print("Loss:", loss, "\tMSE:", mse)

    last_mse.append(mse)

    #Clear some Memory

    del X_train, X_val, Y_train, Y_val, model, history, predict, loss, mse

    gc.collect()
def ensemble(preds, metrics):

    over = sum(metrics)

    n = len(metrics)

    return [sum((over - metrics[x])*preds[i,x]/((n-1)*over) for x in range(n)) for i in range(len(preds))]
print("Predicting the Test data...")

prediction = ensemble(predictions, last_mse)

prediction = np.exp(prediction)

submission = pd.DataFrame()

submission['Id'] = test_index

submission['SalePrice'] = prediction

print("Saving prediction to output...")

submission.to_csv("prediction_regression.csv", index=False)

print("Done.")



print(submission)



x = np.mean(last_mse)

print(x, x**.5)