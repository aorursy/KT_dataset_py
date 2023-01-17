import pandas as pd

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler



from keras.layers import Input 

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Model 
def preprocess_data(test_data=False):

    def encode_one_categorical_feature(column):

        le = LabelEncoder()

        ohe = OneHotEncoder(sparse=False)

        num_encoded = le.fit_transform(column.fillna('unk'))

        oh_encoded = ohe.fit_transform(num_encoded.reshape(-1, 1))

        return oh_encoded

    data = pd.read_csv('../input/train.csv')

    target = ['SalePrice']

    features = data.drop(['Id'] + target, axis=1).columns

    

    dataset_types = pd.DataFrame(data[features].dtypes, columns=['datatype'])

    dataset_types.reset_index(inplace=True)



    numeric_features = dataset_types.rename(columns={"index" : "feature"}).feature[(dataset_types.datatype == 'float64') | (dataset_types.datatype == 'int64')]

    num_data = data[numeric_features]

    num_features = num_data.fillna(num_data.mean()).values

    scaler = StandardScaler()

    num_features_scaled = scaler.fit_transform(num_features)



    categorical_features = dataset_types.rename(columns={"index" : "feature"}).feature[(dataset_types.datatype == 'object')]

    cat_data = data[categorical_features]

    cat_features = np.hstack([encode_one_categorical_feature(data[column]) for column in cat_data.columns])

    

    print("Of the {} features in this dataset".format(len(data.columns)))

    print("{} features are numeric".format(len(numeric_features)))

    print("and {} features are categorical.".format(len(categorical_features)))

    print("The last two are the target, which is numeric, and the id column.")

    

    X = np.hstack((num_features_scaled, cat_features))

    if test_data == True:

        return X

    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=606)

    return X_train, X_test, y_train, y_test



def plot_history(history):

    plt.plot(history.history['loss'], 'b')

    plt.plot(history.history['val_loss'], 'r')

    plt.title('model accuracy') 

    plt.ylabel('loss') 

    plt.xlabel('epoch') 

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    

def keras_model(X_train, X_test, y_train, y_test):

    NUM_EPOCHS = 50

    BATCH_SIZE = 128

    

    inputs = Input(shape=(304, ))

    x = Dropout(0.2)(inputs)

    

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = Dropout(0.2)(x)

    

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = Dropout(0.4)(x)

    

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = Dropout(0.4)(x)

    

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = Dropout(0.4)(x)

    

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = Dropout(0.4)(x)

    

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = Dropout(0.4)(x)

        

    predictions = Dense(1)(x)



    model = Model(inputs=[inputs], outputs=[predictions])



    model.compile(loss="mse", optimizer="adam")

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2, verbose=0)

    

    plot_history(history)

    

    score = model.evaluate(X_test, y_test, verbose=0)

    print("Test MSE is {:.2e}".format(score))

    return history, model
X_train, X_test, y_train, y_test = preprocess_data()
model, history = keras_model(X_train, X_test, y_train, y_test)
predicted = model.model.predict(X_test)
plt.plot(y_test - predicted)
test_data = preprocess_data(test_data=True)