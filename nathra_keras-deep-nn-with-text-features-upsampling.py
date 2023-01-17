import numpy as np
import pandas as pd
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from keras import regularizers
from matplotlib import pyplot as plt
import seaborn as sns
descriptions = {
    1: 'Cathedral family - Rock outcrop complex, extremely stony.',
    2: 'Vanet - Ratake families complex, very stony.',
    3: 'Haploborolis - Rock outcrop complex, rubbly.',
    4: 'Ratake family - Rock outcrop complex, rubbly.',
    5: 'Vanet family - Rock outcrop complex complex, rubbly.',
    6: 'Vanet - Wetmore families - Rock outcrop complex, stony.',
    7: 'Gothic family.',
    8: 'Supervisor - Limber families complex.',
    9: 'Troutville family, very stony.',
    10: 'Bullwark - Catamount families - Rock outcrop complex, rubbly.',
    11: 'Bullwark - Catamount families - Rock land complex, rubbly.',
    12: 'Legault family - Rock land complex, stony.',
    13: 'Catamount family - Rock land - Bullwark family complex, rubbly.',
    14: 'Pachic Argiborolis - Aquolis complex.',
    15: 'unspecified in the USFS Soil and ELU Survey.',
    16: 'Cryaquolis - Cryoborolis complex.',
    17: 'Gateview family - Cryaquolis complex.',
    18: 'Rogert family, very stony.',
    19: 'Typic Cryaquolis - Borohemists complex.',
    20: 'Typic Cryaquepts - Typic Cryaquolls complex.',
    21: 'Typic Cryaquolls - Leighcan family, till substratum complex.',
    22: 'Leighcan family, till substratum, extremely stony.',
    23: 'Leighcan family, till substratum - Typic Cryaquolls complex.',
    24: 'Leighcan family, extremely stony.',
    25: 'Leighcan family, warm, extremely stony.',
    26: 'Granile - Catamount families complex, very stony.',
    27: 'Leighcan family, warm - Rock outcrop complex, extremely stony.',
    28: 'Leighcan family - Rock outcrop complex, extremely stony.',
    29: 'Como - Legault families complex, extremely stony.',
    30: 'Como family - Rock land - Legault family complex, extremely stony.',
    31: 'Leighcan - Catamount families complex, extremely stony.',
    32: 'Catamount family - Rock outcrop - Leighcan family complex, extremely stony.',
    33: 'Leighcan - Catamount families - Rock outcrop complex, extremely stony.',
    34: 'Cryorthents - Rock land complex, extremely stony.',
    35: 'Cryumbrepts - Rock outcrop - Cryaquepts complex.',
    36: 'Bross family - Rock land - Cryumbrepts complex, extremely stony.',
    37: 'Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.',
    38: 'Leighcan - Moran families - Cryaquolls complex, extremely stony.',
    39: 'Moran family - Cryorthents - Leighcan family complex, extremely stony.',
    40: 'Moran family - Cryorthents - Rock land complex, extremely stony.',
}

words = sum([desc.split(' ') for _, desc in descriptions.items()], [])
freq_dict = {word: words.count(word) for word in set(words)}
freq_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
for word, count in freq_dict:
    if count > 2:
        print(count, word)
keywords = [
    'Rock outcrop complex',
    'Rock land complex',
    'Rock',
    'extremely stony',
    'very stony',
    'stony',
    'rubbly',
    'till substratum',
    'Bullwark',
    'Catamount',
    'Cryaquolis',
    'Cryorthents',
    'Cryumbrepts',
    'Leighcan',
    'Legault',
    'Moran',
    'Ratake',
    'Typic',
    'Vanet',
    ]
keyword_dict = {}
for soil_id, desc in descriptions.items():
    keyword_dict[soil_id] = np.array([keyword in desc for keyword in keywords])
print(keyword_dict[1])
def feature_engineering(df):
    # Numeric features
    df['Euclidean_Distance_To_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    df['Hillshade_Total'] = df.filter(like='Hillshade').sum(axis=1)
    df['Hillshade_Slope'] = (df['Hillshade_3pm'] - df['Hillshade_9am']) / df['Hillshade_9am'].apply(lambda x: 1 if x == 0 else x)
    
    # Text features
    soils = df.filter(like='Soil_Type')
    for col in soils.columns:
        id = int(col[9:])
        soils.loc[:, col] = soils.loc[:, col] * id
    soils = soils.sum(axis=1)
    keyword_array = np.stack([keyword_dict[x] for x in soils])
    
    keyword_df = pd.DataFrame(keyword_array, index=soils.index, columns=['keyword_{}'.format(k) for k in keywords])
    df = pd.concat([df, keyword_df], axis=1)
    
    return df

def expand(df):
    target = df.pop('Cover_Type')
    target = pd.get_dummies(target, prefix='Cover_Type')
    df = pd.concat([df, target], axis=1)
    return df

print('processing train data...')
df = pd.read_csv('../input/train.csv')
df = df.sample(frac=1) # Shuffling is necessary for the training set as the dataset is not shuffled.

train_df = feature_engineering(expand(df))
print('\tdone.')
train_df.info()

print('processing test data...')
df = pd.read_csv('../input/test.csv')

test_df = feature_engineering(df)
print('\tdone.')
print('computing mean and std...')
mean_std_df = test_df.agg(['mean', 'std'])
print('\tdone.')
norm_columns = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Euclidean_Distance_To_Hydrology',
    'Hillshade_Total',
    'Hillshade_Slope',
]

mean_std_df = mean_std_df[norm_columns]

def normalize(df):
    df[norm_columns] = (df[norm_columns] - mean_std_df.loc['mean', :]) / mean_std_df.loc['std', :].apply(lambda x: 1 if x == 0 else x)
    no_variance = mean_std_df.loc['std', mean_std_df.loc['std', :] == 0].index.tolist()
    print('Dropped no_variance columns: {}'.format(len(no_variance)))
    df.drop(no_variance, axis=1, inplace=True)
    return df

train_df = normalize(train_df)
test_df = normalize(test_df)
def load_data(df, val_ids=None):
    print('building test set...')
    if val_ids is None:
        X_test = df.iloc[:1512, :]
        val_ids = X_test['Id']
    else:
        X_test = df[df['Id'].isin(val_ids)]
    X_test.drop('Id', axis=1, inplace=True)
    data = df.drop('Id', axis=1)
    y_test = X_test.filter(like='Cover_Type')
    X_test.drop(y_test.columns, axis=1, inplace=True)
    data.drop(X_test.index, axis=0, inplace=True)

    print('building training set...')
    X_train = data
    y_train = X_train.filter(like='Cover_Type')
    X_train.drop(y_train.columns, axis=1, inplace=True)

    return X_train, y_train, X_test, y_test, X_train.shape[1], val_ids

X_train, y_train, X_test, y_test, input_size, val_ids = load_data(train_df)

def make_nn(size='big'):
    inputs = Input(shape=(input_size,))
    if size=='big':
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(7, activation='softmax')(x)
        
    else:
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(7, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def fit_nn(model, lr, bs, num_epochs):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr),
                  metrics=['accuracy'])
    clr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_delta=0.5, cooldown=2, min_lr=1e-7, verbose=1)
    checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    model.fit(
        X_train,
        y_train,
        batch_size=bs,
        epochs=num_epochs,
        callbacks=[clr, checkpoint],
        shuffle=True,
        validation_data=(X_test, y_test)
        )

    return model

fit_nn(make_nn('big'), 1e-3, 32, 30)
model = load_model('checkpoint.h5')
test_id_col = test_df.pop('Id')

y = model.predict(test_df)
y = [np.argmax(x)+1 for x in y]
y = pd.Series(y)

sns.countplot(y)
class_proportions = y.value_counts(normalize=True)
print(class_proportions)

class_fracs = class_proportions / class_proportions.iloc[-1] - 1
for ct in class_fracs.index[:-1]:
    class_samples = train_df[train_df['Cover_Type_{}'.format(ct)] == 1]
    train_df = pd.concat([train_df, class_samples.sample(frac=class_fracs[ct], replace=True)])
print(train_df.shape)
X_train, y_train, X_test, y_test, _, _ = load_data(train_df, val_ids)

y_test_classes = pd.Series([np.argmax(x)+1 for x in y_test])
print(y_test_classes.value_counts(normalize=True))

fit_nn(make_nn('small'), 1e-3, 256, 10)
model = load_model('checkpoint.h5')
y = model.predict(test_df)
y = pd.Series([np.argmax(x)+1 for x in y])

sns.countplot(y)
print(y.value_counts(normalize=True))

submit = pd.DataFrame({'Id': test_id_col, 'Cover_Type': y})

submit.to_csv('submission.csv', index=False)