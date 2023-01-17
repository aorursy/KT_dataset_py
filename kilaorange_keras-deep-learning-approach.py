import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')



seed = 123456

np.random.seed(seed)
target_variable = 'saleprice'

df = (

    pd.read_csv('input/train.csv') # change this to run on kaggle

    #pd.read_csv('../input/train.csv')



    # Rename columns to lowercase and underscores

    .pipe(lambda d: d.rename(columns={

        k: v for k, v in zip(

            d.columns,

            [c.lower().replace(' ', '_') for c in d.columns]

        )

    }))

    # Switch categorical classes to integers

    #.assign(**{target_variable: lambda r: r[target_variable].astype('category').cat.codes})

)

print('Done')
df['bedroomabvgr'].head()
# Categorical / Ordinal

# using binary coding according to this artical 

# to encode categoricals 

# http://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html

# class 4 = x0100 = 0 | 1 | 0 | 0 in columns

# use this encoder https://github.com/scikit-learn-contrib/categorical-encoding

categorical_vars = ['MSZoning'

                    ,'Street'

                    ,'Alley'

                    ,'LotShape'

                    ,'LandContour'

                    ,'Utilities'

                    ,'LotConfig'

                    ,'LandSlope'

                    ,'Neighborhood'

                    ,'Condition1'

                    ,'Condition2'

                    ,'BldgType'

                    ,'HouseStyle'

                    ,'RoofStyle'

                    ,'RoofMatl'

                    ,'Exterior1st'

                    ,'Exterior2nd'

                    ,'MasVnrType'

                    ,'ExterQual'

                    ,'ExterCond'

                    ,'Foundation'

                    ,'BsmtQual'

                    ,'BsmtCond'

                    ,'BsmtExposure'

                    ,'BsmtFinType1'

                    ,'BsmtFinSF1'

                    ,'BsmtFinType2'

                    ,'Heating'

                    ,'HeatingQC'

                    ,'CentralAir'

                    ,'Electrical'

                    ,'KitchenQual'

                    ,'Functional'

                    ,'FireplaceQu'

                    ,'GarageType'

                    ,'GarageFinish'

                    ,'GarageQual'

                    ,'GarageCond'

                    ,'PavedDrive'

                    ,'PoolQC'

                    ,'Fence'

                    ,'MiscFeature'

                    ,'SaleType'

                    ,'SaleCondition'

]



# Nominal:

# Could try binary encoding these too

nominal_vars = ['OverallQual'

                ,'OverallCond'

                ,'YearBuilt'

                ,'YearRemodAdd'

                ,'BsmtFullBath'

                ,'BsmtHalfBath'

                ,'FullBath'

                ,'HalfBath'

                ,'Bedroomabvgr'

                ,'KitchenAbvGr'

                ,'TotRmsAbvGrd'

                ,'Fireplaces'

                ,'GarageYrBlt'

                ,'GarageCars'

                ,'MoSold'

                ,'YrSold'

                ,'MSSubClass'

                ]



bin_enc = categorical_vars + nominal_vars

# make the list lowercase

bin_enc = [x.lower() for x in bin_enc]





# Continuous 

# Anything not mentioned above







# Feature engineering

# (1stFlrSF + 2ndFlrSF) / number levels # avg level size

# 2ndFlrSF / 1stFlrSF  # how big is 1st floor compared to 2nd floor

# LotArea / LotFrontage # lot shape and ratio

# TotalBsmtSF / GrLivArea # basement size compared to living space

# LowQualFinSF / GrLivArea # ratio of Low quality finished to total

# FullBath / GrLivArea # bathrooms per living space

# HalfBath / GrLivArea # half bathrooms per living space

# Bedroom / GrLivArea # bedroom per living space

# Kitchen / GrLivArea # kitchen per living space

# TotRmsAbvGrd / GrLivArea # rooms per living space

# GarageCars / GarageArea # number of cars per garage area

# (WoodDeckSF+OpenPorchSF+EnclosedPorch+3SsnPorch+ScreenPorch+PoolArea) / GrLivArea # ratio of entertaining area vs living space

# Log saleprice

df['saleprice'] = np.log(df['saleprice'])
import sklearn.preprocessing as preprocessing

import seaborn as sns
# Encode the categorical features as numbers

import category_encoders as ce

def number_encode_features(df):

    result = df.copy()

    encoders = {}

    for column in result.columns:

        #print(column)

        #print(result.dtypes[column])

        if (result.dtypes[column] == np.int64 or result.dtypes[column] == np.int32 or result.dtypes[column] == np.float64):

            # impute missing values in column

            print('Imputing...')

            

        if result.dtypes[column] == np.object:

            encoders[column] = preprocessing.LabelEncoder()

            # if there are NaN's in the categorical data fill it with 'None' which becomes another category

            result[column] = encoders[column].fit_transform(result[column].fillna(value='None'))

            #encoder = ce.BinaryEncoder(cols=bin_enc)

    return result, encoders



# METHOD 1: NUMERICAL ENCODING

#encoded_data, _ = number_encode_features(df)



# METHOD 2: BINARY ENCODING

target_variable = 'saleprice'

encoder = ce.BinaryEncoder(cols=bin_enc)

encoder.fit(X=df.drop(target_variable, axis=1))

encoded_data = encoder.transform(df.drop(target_variable, axis=1)) 



print('Done')
df.yrsold.unique()
encoded_data.head()
# Data is now in dataframe "encoded_data"



y = df[target_variable].values

X = encoded_data.fillna(0).as_matrix()



# UPDATE THIS AN THE ONE ON THE SUBMISSION SET

# min / max scaling should only occur on continuous variables

# (

#     # Drop target variable

#     #encoded_data.drop(target_variable, axis=1)

#     # Min-max-scaling (only needed for the DL model)

#     encoded_data.pipe(lambda d: (d-d.min())/d.max()).fillna(0)

#     .as_matrix()

# )

print('Done')
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score



test_size = 0.2



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=test_size, random_state=seed

)

print('Done')
from keras.models import Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.layers import Dense, Activation, Dropout

from keras.layers.advanced_activations import  LeakyReLU  

from keras import optimizers

from keras import initializers

from keras.layers.normalization import BatchNormalization

print('Done')
epochs = 100

learn_rate = 0.01

batch_size= 256

dropout = 0.5

init_mean = 0.0

init_stdev = 0.05
# setup callback so you can look at tensorboard later

tbCallBack = TensorBoard(log_dir='./Graph'

                         ,histogram_freq=0

                         #,write_graph=True

                         ,write_images=True

                         #,write_grads=True

                        )

print('Done')
m = Sequential()

m.add(Dense(512, input_shape=(X.shape[1],)

            ,kernel_initializer=initializers.TruncatedNormal(mean=init_mean

                                                             ,stddev=init_stdev

                                                             ,seed=seed)

            ,bias_initializer='zeros'))

m.add(BatchNormalization())

m.add(LeakyReLU())  # helps to stop disappearing gradient

m.add(Dropout(dropout))



m.add(Dense(512

            ,kernel_initializer=initializers.TruncatedNormal(mean=init_mean

                                                             ,stddev=init_stdev

                                                             ,seed=seed)

            ,bias_initializer='zeros'))

m.add(BatchNormalization())

m.add(LeakyReLU())

m.add(Dropout(dropout))



m.add(Dense(128

             ,kernel_initializer=initializers.TruncatedNormal(mean=init_mean

                                                             ,stddev=init_stdev

                                                             ,seed=seed)

            ,bias_initializer='zeros'))

m.add(BatchNormalization())

m.add(LeakyReLU())

m.add(Dropout(dropout))



m.add(Dense(1, activation=None))  # linear activation for regression



m.compile(

    optimizer=optimizers.Adam(lr=learn_rate),

    loss='mean_squared_error',

    #metrics=[log_rmse]

)



print('Starting training....')



m.fit(

    # Feature matrix

    X_train,

    # Target class one-hot-encoded

    y_train,

    # Iterations to be run if not stopped by EarlyStopping

    epochs=epochs,

    callbacks=[

        # Stop iterations when validation loss has not improved

        EarlyStopping(monitor='val_loss', patience=25),

        # Nice for keeping the last model before overfitting occurs

        ModelCheckpoint(

            'best.model',

            monitor='val_loss',

            save_best_only=True,

            verbose=1

        ),

        # Tensorboards

        #tbCallBack

    ],

    verbose=0,

    validation_split=0.1,

    batch_size=batch_size,

)

print('Done')

y_test_preds = m.predict(X_test) 



r = np.sqrt( np.mean((y_test_preds - y_test)**2) ) 

print(r)
# RUN ME FOR RMSE
#plt.scatter( y_test_preds, y_test)



y_real = y_test

y_pred = y_test_preds.reshape(146,)



print(y_real.shape, y_pred.shape )



fig, ax = plt.subplots()

fit = np.polyfit(y_pred, y_real, deg=1)

ax.plot(y_pred, fit[0] * y_pred + fit[1], color='red')

ax.scatter(y_pred, y_real)



fig.show()
df = (

    pd.read_csv('input/test.csv') # change this to run on kaggle

    #pd.read_csv('../input/train.csv')



    # Rename columns to lowercase and underscores

    .pipe(lambda d: d.rename(columns={

        k: v for k, v in zip(

            d.columns,

            [c.lower().replace(' ', '_') for c in d.columns]

        )

    }))

    # Switch categorical classes to integers

    #.assign(**{target_variable: lambda r: r[target_variable].astype('category').cat.codes})

)

print('Done')
# make the dummy columns for categoricals

#encoded_data, _ = number_encode_features(df)



encoded_data = encoder.transform(X=df)



X_sub = encoded_data.fillna(0).as_matrix()



# RESTRICT TO ONLY ON CONTINUOUS VARS

# (# Min-max-scaling (only needed for the DL model)

#     encoded_data.pipe(lambda d: (d-d.min())/d.max()).fillna(0)

#     .as_matrix()

# )



print('Done')
# Export dataframes to csv for inspection

pd.DataFrame(X_train).to_csv('X_train.csv', index = False)



pd.DataFrame(X_sub).to_csv('X_sub.csv', index = False)
## Save to CSV with Image name and results

# Run the model

y_sub_preds = np.exp( m.predict(X_sub) ) # bring them back to sales prices using exponential

pred = pd.DataFrame(data=y_sub_preds) 



print("Here is a sample...")



result = pd.concat([df['id'], pred], axis=1)

result.columns = ['Id','SalePrice'] 

print(result[0:10])



# Header: [image ALB BET DOL LAG NoF OTHER   SHARK   YFT]

result.to_csv('submission.csv', index = False)



print('Done')