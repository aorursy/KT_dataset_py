import numpy as np 

import pandas as pd
default_path = '../input/'



df_train = pd.read_csv(default_path+'train.csv')

df_test  = pd.read_csv(default_path+'test.csv')

df_struct = pd.read_csv(default_path+'structures.csv')

unique_type = list(set(df_test['type']))



elements = []

for i in unique_type: 

    elements.append(i[2])

    elements.append(i[3])

unique_elements = list(set(elements))
# Gives a normalized list of values for each of the TYPE of coupling

def give_uniqueId(lis): 

    l = np.size(lis)

    fin = np.zeros((l,1))

    for i in range(0,l):

        fin[i] = (unique_type.index(lis[i])+1)

    return fin



def give_uniqueElement(lis):

    element = np.frompyfunc(lambda x:x[3:4],1,1)(lis)

    uniqueID = np.frompyfunc(lambda x:unique_elements.index(x)-1,1,1)(element)

    return uniqueID

def giveFirst(lis):

    return np.frompyfunc(lambda x:int(x[0:1])-1,1,1)(lis)

    

# Returns dataframe merging xyz values

def map_atom_info(df, atom_idx):

    df = pd.merge(df, df_struct, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df
# Merging the x,y,z values into the train and test dataframes

train_df = map_atom_info(map_atom_info(df_train,0),1)

test_df = map_atom_info(map_atom_info(df_test,0),1)
#First parameter of the type

train_df['typeval']= giveFirst(train_df['type'].values)

test_df['typeval']= giveFirst(test_df['type'].values)

# Giving last 2nd element as a parameter as first element is always hydrogen

train_df['elem'] = give_uniqueElement(train_df['type'].values)

test_df['elem']  = give_uniqueElement(test_df['type'].values)

# Assigning unique numerical values to the type 

train_df['type'] = give_uniqueId(train_df['type'].values)

test_df['type'] = give_uniqueId(test_df['type'].values)
train_df.head(4)
test_df.head(3)
train_df['dx'] = train_df['x_0']-train_df['x_1']

train_df['dy'] = train_df['y_0']-train_df['y_1']

train_df['dz'] = train_df['z_0']-train_df['z_1']

test_df['dx'] = test_df['x_0']-test_df['x_1']

test_df['dy'] = test_df['y_0']-test_df['y_1']

test_df['dz'] = test_df['z_0']-test_df['z_1']
train_df.head(4)
# Assigning features and lables

X_train = train_df[['x_0','y_0','z_0','x_1','y_1','z_1','dx','dy','dz','elem','typeval','type']].values

y_train = train_df['scalar_coupling_constant'].values

X_test = test_df[['x_0','y_0','z_0','x_1','y_1','z_1','dx','dy','dz','elem','typeval','type']].values
X_train
# Preprocessing : Making the mean of features to 0

from sklearn import preprocessing

X_train = preprocessing.scale(X_train)

X_test = preprocessing.scale(X_test)
# Importing requirements for dnn

import tensorflow as tf
n_cols = X_train.shape[1]

model = tf.keras.models.Sequential([

     

    tf.keras.layers.Dense(5000, activation='relu', input_shape=(n_cols,)),

    tf.keras.layers.Dropout(0.5),  # Since the network is dense and big, using dropout

    tf.keras.layers.Dense(1000, activation='relu'),

    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Dense(500, activation='relu'),

    tf.keras.layers.Dense(500, activation='relu'),

    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Dense(200, activation='relu'),

    tf.keras.layers.Dense(1, activation='linear'),

    ])
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(X_train, y_train,epochs=7)# not using validation set as regression will usually not have exact answer
pred = model.predict(X_test)
def submit(predictions):

    submit = pd.read_csv(default_path+'sample_submission.csv')

    print(len(submit), len(predictions))   

    submit["scalar_coupling_constant"] = predictions

    submit.to_csv("submission.csv", index=False)
submit(pred)