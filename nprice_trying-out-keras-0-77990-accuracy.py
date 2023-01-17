import xgboost as xgb



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



# Simpel model that only has one Dense input layer and one output layer. 

def create_model_simple(input_size):

    model = Sequential([

        Dense(512, input_dim=input_size),

        Activation('relu'),

        Dense(1),

        Activation('sigmoid'),

    ])



    # For a binary classification problem

    model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

    

    return model



# Slightly more complex model with 1 hidden layer

def create_model_multiple_layers(input_size):

    model = Sequential([

        Dense(512, input_dim=input_size),

        Activation('relu'),

        Dense(128),

        Activation('relu'),

        Dense(1),

        Activation('sigmoid'),

    ])



    # For a binary classification problem

    model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

    

    return model



# Simple model with a dropout layer thrown in

def create_model_dropout(input_size):

    model = Sequential([

        Dense(512, input_dim=input_size),

        Activation('relu'),

        Dropout(0.5),

        Dense(1),

        Activation('sigmoid'),

    ])



    # For a binary classification problem

    model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

    

    return model



# Slightly more complex model with 2 hidden layers and a dropout layer

def create_model_complex(input_size):

    model = Sequential([

        Dense(512, input_dim=input_size),

        Activation('relu'),

        Dense(256),

        Activation('relu'),

        Dense(128),

        Dropout(0.5),

        Activation('relu'),

        Dense(64),

        Dropout(0.5),

        Activation('relu'),

        Dense(1),

        Activation('sigmoid'),

    ])



    # For a binary classification problem

    model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

    

    return model
def format_data(dataframe):

    # drop unnecessary columns

    # PassengerId is always different for each passenger, not helpful

    # Name is different for each passenger, not helpful (maybe last names would be helpful?)

    # Ticket information is different for each passenger, not helpful

    # Embarked does not have any strong correlation for survival rate. 

    # Cabin data is very sparse, not helpful

    dataframe = dataframe.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1)



    # Instead of having two columns Parch & SibSp, 

    # we can have only one column represent if the passenger had any family member aboard or not,

    # Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

    dataframe['Family'] = dataframe["Parch"] + dataframe["SibSp"]

    dataframe['Family'].loc[dataframe['Family'] > 0] = 1

    dataframe['Family'].loc[dataframe['Family'] == 0] = 0



    # drop Parch & SibSp

    dataframe = dataframe.drop(['SibSp','Parch'], axis=1)



    # get average, std, and number of NaN values in titanic_df

    average_age_titanic   = dataframe["Age"].mean()

    std_age_titanic       = dataframe["Age"].std()

    count_nan_age_titanic = dataframe["Age"].isnull().sum()



    # generate random numbers between (mean - std) & (mean + std)

    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, 

                               size = count_nan_age_titanic)



    dataframe["Age"][np.isnan(dataframe["Age"])] = rand_1

    

    return dataframe



def string_to_numbers(data, dataframe, encoder):

    # assign labels for all the non-numeric fields

    headings = list(dataframe.columns.values)

    for heading_index in range(len(headings)):

        dataframe_type = dataframe[headings[heading_index]].dtype

        column = data[:,heading_index]

        if dataframe_type == np.int64 or dataframe_type == np.float64:

            data[:,heading_index] = column.astype(float)

        else :

            data[:,heading_index] = encoder.fit(column).transform(column).astype(float)

            

    return data
import pandas

import numpy as np

import sklearn.preprocessing as preprocessing



from sklearn.preprocessing import LabelEncoder



# load dataset

titanic_df = pandas.read_csv('../input/train.csv')

# format the data

titanic_df = format_data(titanic_df)



# attempt to remove all of the "outliers", things like high class female passengers who died (likely to live) or 

# low class males surviving (likely to die)

#titanic_df = titanic_df.drop(titanic_df[(titanic_df["Pclass"] == 1) & (titanic_df["Survived"] == 0)].index)

#titanic_df = titanic_df.drop(titanic_df[(titanic_df["Pclass"] == 3) & (titanic_df["Survived"] == 1)].index)



# pull out the correct answers (survived or not)

Y_train = titanic_df["Survived"].values

# drop the survived column for the training data

titanic_df = titanic_df.drop("Survived",axis=1)

X_train = titanic_df.values



# assign labels for all the non-numeric fields

encoder = LabelEncoder()

X_train = string_to_numbers(X_train, titanic_df, encoder)

        

# Extract a small validation set

validation_set_size = 200

random_indices = np.random.randint(low=0,high=len(X_train)-1,size=validation_set_size)

X_valid = X_train[random_indices]

Y_valid = Y_train[random_indices]

X_train = np.delete(X_train, random_indices, axis=0)

Y_train = np.delete(Y_train, random_indices, axis=0)

               

# normalize the data

preprocessing.scale(X_train)

preprocessing.scale(X_valid)
# Train the model, iterating on the data in batches of 64 samples

model = create_model_simple(len(X_train[0]))

#model = create_model_multiple_layers(len(X_train[0]))

#model = create_model_dropout(len(X_train[0]))

#model = create_model_complex(len(X_train[0]))



model.optimizer.lr = 0.01

model.fit(X_train, Y_train, epochs=100, batch_size=64)
F_train = model.predict(X_train, batch_size=64)

F_val = model.predict(X_valid, batch_size=64)



dTrain = xgb.DMatrix(F_train, label=Y_train)

dVal = xgb.DMatrix(F_val, label=Y_valid)



xgb_params = {

    'objective': 'binary:logistic',

    'booster': 'gbtree',

    'eval_metric': 'logloss',

    'eta': 0.1, 

    'max_depth': 9,

    'subsample': 0.9,

    'colsample_bytree': 1 / F_train.shape[1]**0.5,

    'min_child_weight': 5,

    'silent': 1

}

best = xgb.train(xgb_params, dTrain, 1000,  [(dTrain,'train'), (dVal,'val')], 

                verbose_eval=10, early_stopping_rounds=10)
# run predictions on our validation data (the small subset we removed before training)

train_preds = best.predict(dVal, ntree_limit=best.best_ntree_limit)

rounded_preds = np.round(train_preds).astype(int).flatten()

correct_preds = np.where(rounded_preds==Y_valid)[0]

print("Accuracy: {}%".format(float(len(correct_preds))/float(len(rounded_preds))*100))
import matplotlib.pyplot as plt



def render_value_frequency(dataframe, title):

    fig, ax = plt.subplots()

    dataframe.value_counts().plot(ax=ax, title=title, kind='bar')

    plt.show()

    

def render_plots(dataframes):

    headings = dataframes.columns.values

    for heading in headings:

        data_type = dataframes[heading].dtype

        if data_type == np.int64 or data_type == np.float64:

            dataframes[heading].plot(kind='hist',title=heading)

            plt.show()

        else:

            render_value_frequency(dataframes[heading],heading)
correct = np.where(rounded_preds==Y_valid)[0]

print("Found {} correct labels".format(len(correct)))

render_plots(titanic_df.iloc[correct])
incorrect = np.where(rounded_preds!=Y_valid)[0]

print("Found {} incorrect labels".format(len(incorrect)))

render_plots(titanic_df.iloc[incorrect])
confident_survived_correct = np.where((rounded_preds==1) & (rounded_preds==Y_valid))[0]

print("Found {} confident correct survived labels".format(len(confident_survived_correct)))

render_plots(titanic_df.iloc[confident_survived_correct])
confident_died_correct = np.where((rounded_preds==0) & (rounded_preds==Y_valid))[0]

print("Found {} confident correct died labels".format(len(confident_died_correct)))

render_plots(titanic_df.iloc[confident_died_correct])
confident_survived_incorrect = np.where((rounded_preds==1) & (rounded_preds!=Y_valid))[0]

print("Found {} confident incorrect survived labels".format(len(confident_survived_incorrect)))

render_plots(titanic_df.iloc[confident_survived_incorrect])
confident_died_incorrect = np.where((rounded_preds==0) & (rounded_preds!=Y_valid))[0]

print("Found {} confident incorrect died labels".format(len(confident_died_incorrect)))

render_plots(titanic_df.iloc[confident_died_incorrect])
most_uncertain = np.argsort(np.abs(train_preds.flatten()-0.5))[:10]

render_plots(titanic_df.iloc[most_uncertain])
# load test dataset

test_df = pandas.read_csv('../input/test.csv')

# get the passenger IDs

passenger_ids = test_df['PassengerId'].values

# format the data

test_df = format_data(test_df)



# only for test_df, since there is a missing "Fare" value

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



X_test = test_df.values



# assign labels for all the non-numeric fields

encoder = LabelEncoder()

X_test = string_to_numbers(X_test, test_df, encoder)

               

#normalize the data

#preprocessing.normalize(dataset)

preprocessing.scale(X_test)



F_test = model.predict(X_test, batch_size=64)



dTest = xgb.DMatrix(F_test)



preds = best.predict(dTest, ntree_limit=best.best_ntree_limit)
preds = np.round(preds).astype(int).flatten()

    

submission = pandas.DataFrame({

        "PassengerId": passenger_ids,

        "Survived": preds

    })

submission.to_csv('titanic.csv', index=False)