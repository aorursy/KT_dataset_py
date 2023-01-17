# Linear algebra
import numpy as np
# Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
# Import keras deep learning library
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Flatten,Dropout
from keras.optimizers import SGD,Adam,Adamax,Nadam,Adadelta
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical
from keras.utils import np_utils
from keras import backend
# Globals constants
input_neurons=142
output_neurons=5
# Fix random seed for reproducibility
np.random.seed(7)
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# Inspect the dataset, as you can see some columns have an empty value
train_df.info()
# Columns type object for categorical
train_df.loc[:, train_df.dtypes == object].head()
# See the head of the train dataset
train_df.head()
# Train Data Frame
xtrain_df = train_df.drop(['Id','idhogar','dependency','edjefe','edjefa','Target',
                          'v18q1', 'rez_esc','meaneduc'], axis=1)
ytrain_df = train_df.Target
ytrain_df= to_categorical(ytrain_df,output_neurons)
# Test Data Frame 
xtest_df = test_df.drop(['Id','idhogar','dependency','edjefe','edjefa',
                        'v18q1', 'rez_esc','meaneduc'], axis=1)
# Free some space
del train_df
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms']
    
    df['age'] += np.random.randint(2) + np.random.randint(2) - 2
    df['SQBage'] = df['age'] ** 2
    df['hogar_total'] += np.random.randint(3) - 1
    df['SQBhogar_total'] = df['hogar_total'] ** 2
    df['v2a1'] += np.random.randint(10) * 1000 - 5000
    
    df['child_weight'] = (df['hogar_nin'] + df['hogar_mayor']) / df['hogar_total']
    df['child_weight3'] = df['r4t1'] / df['r4t3']
    df['SQBworker'] = df['hogar_adul'] ** 2
    df['rooms_per_person'] = df['rooms'] / (df['tamviv'])
    df['female_weight'] = df['r4m3'] / df['r4t3']

extract_features(xtest_df)
extract_features(xtrain_df)
# Quantity columns for train dataset
len(xtrain_df.columns)
# Quantity columns for test dataset
len(xtest_df.columns)
#Model #3
model = Sequential()
#Base Model
model.add(Dense(64, input_dim=input_neurons, kernel_initializer ='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization(moving_mean_initializer='zeros',momentum=0.9))
model.add(Dense(32 ,kernel_initializer ='uniform', activation='relu'))
model.add(BatchNormalization(moving_mean_initializer='zeros',momentum=0.9))
model.add(Dropout(0.2))
model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
model.add(BatchNormalization(moving_mean_initializer='zeros',momentum=0.9))
model.add(Dropout(0.2))
model.add(Dense(output_neurons, kernel_initializer ='uniform', activation='softmax'))
adam=Adam(lr=1e-3, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
    xtrain_df,
    ytrain_df,
    epochs=50,
#     validation_split=0.1,
    shuffle=True,
    batch_size=100,
    verbose=1
)
scores = model.evaluate(xtrain_df, ytrain_df)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict_classes(xtest_df,verbose=0)
predictions=predictions.flatten()
results = pd.Series(predictions,name="Target")
submission = pd.concat([pd.Series(test_df.Id,name = "Id"),results],axis = 1)
submission.to_csv("costa_rican_household_poverty_datagen.csv",index=False)
# Clear error in tensorflow for session
backend.clear_session()