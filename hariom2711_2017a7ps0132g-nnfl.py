import numpy as np

import pandas as pd



from keras.layers import Dense, Dropout, Activation, BatchNormalization

from keras.models import Sequential

import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.regularizers import l2



from sklearn.metrics import mean_absolute_error
raw_df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv")
none_val = np.nan



def notNumToNeg(s):

    if s.isdigit():

        return int(s)

    else:

        return none_val

    

def trimTrainData(raw_df):

    df = raw_df.copy()

    change_to_int = ['Number of Quantities', 'Number of Insignificant Quantities', 'Total Number of Words', 'Number of Special Characters']



    # change all floating to floats

    df['Difficulty'] = raw_df['Difficulty'].replace({

        '?': none_val

    })

    df['Difficulty'] = df['Difficulty'].astype(float)

    

    df['Difficulty'] = df['Difficulty'].replace({

        -1: none_val

    })

    

    for item in change_to_int:

        df[item] = df[item].apply(notNumToNeg)



    # change Size Attribute

    df['Size'] = df['Size'].replace({

        'Small': 0,

        'Medium': 1,

        'Big': 2,

        '?': -1

    })

    df['Size'] = df['Size'].astype(int)

    df['Size'] = df['Size'].replace({

        -1: none_val

    })

    return df

df = trimTrainData(raw_df)
def getMissingFill(train_df):

    df = train_df.copy()

    # for size fill with maximum

    vals = {}

    vals["Size"] = df["Size"].value_counts().idxmax()

    # for the others use mean

    for col in df.columns:

        if col != "Size":

            vals[col] = (df[col].mean(skipna = True))

    return vals

            

global_missing_fill = getMissingFill(df)

def getMissingFilled(raw_df):

    df = raw_df.copy()

    for col in df.columns:

        df[col].fillna(global_missing_fill[col], inplace=True)

    return df



data = getMissingFilled(df)

# data is final

from sklearn.model_selection import train_test_split



X = data.copy()

X.drop(['Class'], axis=1, inplace=True)



Y = data['Class'].copy()



x_train_unscaled, x_test_unscaled, y_train_without_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scale_cols = ["ID",

              "Total Number of Words",

              "Total Number of Characters",

              "Number of Special Characters",

              "Number of Sentences",

              "First Index",

              "Second Index",

              "Difficulty",

              "Score"

             ]



def getTrainedScaler(df):

    features = df[scale_cols]

    return StandardScaler().fit(features)



trainedScaler = getTrainedScaler(x_train_unscaled)



def getScaled(raw_df, scaler=trainedScaler):

    df = raw_df.copy()

    features = df[scale_cols]

    features = scaler.transform(features.values)

    df[scale_cols] = features

    return df

x_train = getScaled(x_train_unscaled)

x_test = getScaled(x_test_unscaled)

y_train = y_train_without_val

y_test = y_test
num_nodes = 30

def build_model4(learning_rate=0.01,  reg1=0.5, reg2=0.5, input_dim=x_train.shape[1]):

    model = Sequential()

    

    model.add(Dense(num_nodes, input_dim=input_dim, activation="relu"))



    model.add(Dropout(0.3))

    

    model.add(BatchNormalization())

    model.add(Dense(num_nodes,activation="tanh"))   

    

    model.add(Dense(num_nodes,activation="relu"))   

    

    model.add(Dropout(0.1))

    model.add(Dense(num_nodes,activation="tanh"))

    

    model.add(Dense(num_nodes,activation="relu"))

    

#     model.add(Dense(num_nodes,activation="tanh"))

    

#     model.add(Dense(num_nodes,activation="elu"))

    

    model.add(Dense(6, activation="softmax"))

    

    optimizer = keras.optimizers.adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

md = build_model4(0.01)

cb = [EarlyStopping(monitor='val_loss', patience=40),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]



md.fit(x_train, y_train, epochs=200, validation_split=0.3, batch_size=8, callbacks=cb)

md.evaluate(x_test, y_test, batch_size=8)
raw_df_test = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')



# change size

df_test = getMissingFilled(raw_df_test)

df_test['Size'] = df_test['Size'].replace({

        'Small': 0,

        'Medium': 1,

        'Big': 2,

        '?': -1

    })

df_test['Size'] = df_test['Size'].astype(int)

df_test['Size'] = df_test['Size'].replace({

    -1: np.nan

})



scaled_df_test = getScaled(df_test)

pred = md.predict(scaled_df_test, batch_size=5)



class_result=np.argmax(pred,axis=-1)

class_result
ans = pd.DataFrame()

ans['ID'] = df_test['ID']

ans['Class'] = class_result

ans.to_csv('./res2.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(ans)