import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras import models

from keras import layers

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



train = train.set_index("Id")

test = test.set_index("Id")



test_index = test.index



# concat train and test set so that I can clean them together

full = pd.concat([train, test], axis=0, sort=False)



y = full[full["SalePrice"].notnull()]["SalePrice"].sort_index().values

# Trans y to log(y)

log_y = np.log(y)



numeric_var_list = ["OverallQual", "GrLivArea", "YearBuilt", "MSSubClass", "OverallCond",

                    "GarageCars", "LotArea", "Fireplaces", "LotFrontage", "TotRmsAbvGrd",

                    "KitchenAbvGr", "FullBath"]

categorical_val_list = ["BsmtExposure", "BsmtFinType1", "Neighborhood", "BsmtQual", "MSZoning", "BsmtCond",

                        "Exterior1st", "KitchenQual", "Exterior2nd", "SaleCondition", "HouseStyle",

                        "LotConfig", "GarageFinish", "MasVnrType", "RoofStyle"]



numeric_df = full[numeric_var_list]

categorical_df = full[categorical_val_list]



for num in numeric_var_list:

    numeric_df[num] = numeric_df[num].fillna(numeric_df[num].median())



# numeric_df = numeric_df.fillna(numeric.median())



for cate in categorical_val_list:

    categorical_df[cate] = categorical_df[cate].fillna(categorical_df[cate].value_counts().idxmax())

    

categorical_df = pd.get_dummies(categorical_df)



train_c = categorical_df[categorical_df.index <= 1460]

test_c = categorical_df[categorical_df.index > 1460]



train_numeric_df = numeric_df[numeric_df.index <= 1460]

test_numeric_df = numeric_df[numeric_df.index > 1460]



scale = StandardScaler()



train_n = scale.fit_transform(train_numeric_df)

test_n = scale.fit_transform(test_numeric_df)



train = np.concatenate((train_n, train_c.values), axis=1)

test = np.concatenate((test_n, test_c.values), axis=1)



X_train, X_val, y_train, y_val = train_test_split(train, log_y, test_size=0.4)
# Create the Model

model = models.Sequential()

model.add(layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1))



# Compile the Model

model.compile(

    optimizer='adam', 

    loss='mean_squared_logarithmic_error', 

    metrics=["mae"]

)



history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=200, batch_size=32)
# Create the Model

model = models.Sequential()

model.add(layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1))



# Compile the Model

model.compile(

    optimizer='adam', 

    loss='mean_squared_logarithmic_error', 

    metrics=["mae"]

)



model.fit(train, log_y, epochs=130, batch_size=32)



pred = model.predict(test)

pred = np.exp(pred)

submission = pd.DataFrame()

submission['Id'] = test_index

submission['SalePrice'] = pred

submission.to_csv("submission3.csv", index=False)
def create_model(n=16, r=0.1, opt="adam"):

    model = models.Sequential()

    model.add(layers.Dense(n, input_dim=X_train.shape[1], activation="relu"))

    model.add(layers.Dropout(r))

    model.add(layers.Dense(1))

    model.compile(

        optimizer=opt, 

        loss="mean_squared_logarithmic_error",

        metrics=["accuracy"]   

    )

    return model



para_dict = {

    "batch_size": [8, 16, 32, 64, 128],

    "epochs": [10, 50, 100, 200],

    "n": [2, 4, 8, 16, 32, 64],

    "r": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    "opt": ["adam", "rmsprop"]

}



keras_model = KerasClassifier(build_fn=create_model, verbose=1)



grid = GridSearchCV(estimator=keras_model, param_grid=para_dict, n_jobs=-1, cv=3)

grid_result = grid.fit(train, log_y)
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
# Retrain the Model

model = models.Sequential()

model.add(layers.Dense(8, input_dim=X_train.shape[1], activation='relu'))

model.add(layers.Dense(1))



# Compile the Model

model.compile(

    optimizer='adam', 

    loss='mean_squared_logarithmic_error', 

    metrics=["mae"]

)



model.fit(train, log_y, epochs=50, batch_size=32)



pred = model.predict(test)

pred = np.exp(pred)

submission = pd.DataFrame()

submission['Id'] = test_index

submission['SalePrice'] = pred

submission.to_csv("submission4.csv", index=False)