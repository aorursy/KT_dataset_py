!pip install tensorflow_addons

!pip install scikit-learn==0.22
import pandas as pd

import pandas_profiling

import numpy as np

import gc

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv', low_memory=False)

df = df[~df[df.columns[df.isnull().any()]].isnull().any(axis=1)]

df = df.reset_index(drop=True)

df.shape
df = df.sort_values(by=['City', 'IntersectionId'])
targets = [

'TotalTimeStopped_p20', 

#'TotalTimeStopped_p40',

'TotalTimeStopped_p50',

#'TotalTimeStopped_p60', 

'TotalTimeStopped_p80',

#'TimeFromFirstStop_p20', 

#'TimeFromFirstStop_p40',

#'TimeFromFirstStop_p50', 

#'TimeFromFirstStop_p60',

#'TimeFromFirstStop_p80',

'DistanceToFirstStop_p20',

#'DistanceToFirstStop_p40', 

'DistanceToFirstStop_p50',

#'DistanceToFirstStop_p60', 

'DistanceToFirstStop_p80'

]



to_drop = [

    'DistanceToFirstStop_p60', 

    'DistanceToFirstStop_p40',

    'TotalTimeStopped_p40',

    'TotalTimeStopped_p60', 

    'TimeFromFirstStop_p20', 

    'TimeFromFirstStop_p40',

    'TimeFromFirstStop_p50', 

    'TimeFromFirstStop_p60',

    'TimeFromFirstStop_p80',

]
Y = df[targets]

X = df.drop(columns=targets + to_drop)
X, X_valid, Y, Y_valid = train_test_split(X, Y, test_size=0.1, shuffle=False, random_state=42)

X = X.reset_index(drop=True)

X_valid = X_valid.reset_index(drop=True)

Y = Y.reset_index(drop=True)

Y_valid = Y_valid.reset_index(drop=True)
del df
monthly_av = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78, 'Atlanta9': 73,

              'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68,

              'Boston7': 74, 'Boston8': 73, 'Boston9': 66, 'Boston10': 55,'Boston11': 45, 'Boston12': 35,

              'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,

              'Chicago10': 56,  'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,

              'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,

              'Philadelphia11': 49, 'Philadelphia12': 40}

monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09,

              'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,

              'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73,

              'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,

              'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88,

              'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 'Philadelphia10': 2.75,

              'Philadelphia11': 3.16, 'Philadelphia12': 3.31}

monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0, 'Atlanta9': 0,

              'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0,

              'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0,'Boston11': 1.3, 'Boston12': 9.0,

              'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0,

              'Chicago10': 0,  'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5, 'Philadelphia5': 0,

              'Philadelphia6': 0, 'Philadelphia7': 0, 'Philadelphia8': 0, 'Philadelphia9':0 , 'Philadelphia10': 0,

              'Philadelphia11': 0.3, 'Philadelphia12': 3.4}



monthly_daylight = {'Atlanta1': 10, 'Atlanta5': 14, 'Atlanta6': 14, 'Atlanta7': 14, 'Atlanta8': 13, 'Atlanta9': 12,

              'Atlanta10': 11, 'Atlanta11': 10, 'Atlanta12': 10, 'Boston1': 9, 'Boston5': 15, 'Boston6': 15,

              'Boston7': 15, 'Boston8': 14, 'Boston9': 12, 'Boston10': 11,'Boston11': 10, 'Boston12': 9,

              'Chicago1': 10, 'Chicago5': 15, 'Chicago6': 15, 'Chicago7': 15, 'Chicago8': 14, 'Chicago9': 12,

              'Chicago10': 11,  'Chicago11': 10, 'Chicago12': 9, 'Philadelphia1': 10, 'Philadelphia5': 14,

              'Philadelphia6': 15, 'Philadelphia7': 15, 'Philadelphia8': 14, 'Philadelphia9':12 , 'Philadelphia10': 11,

              'Philadelphia11': 10, 'Philadelphia12': 9}



monthly_sunshine = {'Atlanta1': 5.3, 'Atlanta5': 9.3, 'Atlanta6': 9.5, 'Atlanta7': 8.8, 'Atlanta8': 8.3, 'Atlanta9': 7.6,

              'Atlanta10': 7.7, 'Atlanta11': 6.2, 'Atlanta12': 5.3, 'Boston1': 5.3, 'Boston5': 8.6, 'Boston6': 9.6,

              'Boston7': 9.7, 'Boston8': 8.9, 'Boston9': 7.9, 'Boston10': 6.7,'Boston11': 4.8, 'Boston12': 4.6,

              'Chicago1': 4.4, 'Chicago5': 9.1, 'Chicago6': 10.4, 'Chicago7': 10.3, 'Chicago8': 9.1, 'Chicago9': 7.6,

              'Chicago10': 6.2,  'Chicago11': 3.6, 'Chicago12': 3.4, 'Philadelphia1': 5.0, 'Philadelphia5': 7.9,

              'Philadelphia6': 9.0, 'Philadelphia7': 8.9, 'Philadelphia8': 8.4, 'Philadelphia9':7.9 , 'Philadelphia10': 6.6,

              'Philadelphia11': 5.2, 'Philadelphia12': 4.4}



center_latitude = {"Atlanta":33.753746,

                             "Boston":42.361145,

                             "Chicago":41.881832,

                             "Philadelphia":39.952583

                  }

center_longitude = {"Atlanta":-84.386330,

                             "Boston": -71.057083,

                             "Chicago": -87.623177,

                             "Philadelphia":-75.165222

                   }



directions = {

    'N': 0,

    'NE': np.pi/4,

    'E': np.pi/2,

    'SE': 3*np.pi/4,

    'S': np.pi,

    'SW': -3*np.pi/4,

    'W': -np.pi/2,

    'NW': -np.pi/4

}
def pre_process(X, Y=None):

    X['IntersectionId'] = X['IntersectionId'].astype('str') + X['City']

    X['city_month'] = X["City"] + X["Month"].astype(str)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature

    X["average_temp"] = X['city_month'].map(monthly_av)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall

    X["average_rainfall"] = X['city_month'].map(monthly_rainfall)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall

    X["average_snowfall"] = X['city_month'].map(monthly_snowfall)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly daylight

    X["average_daylight"] = X['city_month'].map(monthly_daylight)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly sunshine

    X["average_sunshine"] = X['city_month'].map(monthly_sunshine)

    

    

    X["Center_Latitude"] = X['City'].map(center_latitude)

    X["Center_Longitude"] = X['City'].map(center_longitude)

    X["CenterDistance"] = np.sqrt((X['Latitude'] - X["Center_Latitude"]) ** 2 + (X['Center_Longitude'] - X["Longitude"]) ** 2)

    

    X['SameStreet'] = X['EntryStreetName'] ==  X['ExitStreetName']

    X['SameHeading'] = X['EntryHeading'] ==  X['ExitHeading']

    X['Vector'] = X['EntryHeading'] + X['ExitHeading']

    X['Hour_x'] = np.cos(X['Hour'] * np.pi/12.)

    X['Hour_y'] = np.sin(X['Hour'] * np.pi/12.)

    X['Month_x'] = np.cos(X['Month'] * np.pi/6.)

    X['Month_y'] = np.sin(X['Month'] * np.pi/6.)

    X['is_day'] = 0

    X.iloc[X[(X['Hour'] > 5) & (X['Hour'] < 20)].index, X.columns.get_loc('is_day')] = 1 

    

    for street_dir in ['Entry', 'Exit']:

        data = np.char.lower(X[street_dir + 'Heading'].values.astype('str'))

        # N => Y +1

        # S => Y -1

        # E => X +1

        # W => X -1

        X['NS_' + street_dir] = np.where(np.char.rfind(data, 'N') > -1, 1, 0)

        X['NS_' + street_dir] = np.where(np.char.rfind(data, 'S') > -1, -1, X['NS_' + street_dir].values)

        X['EW_' + street_dir] = np.where(np.char.rfind(data, 'E') > -1, 1, 0)

        X['EW_' + street_dir] = np.where(np.char.rfind(data, 'W') > -1, -1, X['EW_' + street_dir].values)

        X[street_dir + '_Angle'] = X[street_dir + 'Heading'].map(directions)



    X['Angle'] = X['Exit_Angle'] - X['Entry_Angle'] 

    X['x_Angle'] = np.cos(X['Angle'].values)

    X['y_Angle'] = np.sin(X['Angle'].values)



    X['NS'] = X['NS_Exit'] - X['NS_Entry'] 

    X['EW'] = X['EW_Exit'] - X['EW_Entry']

    

    for street_dir in ['Entry', 'Exit']:

        data = np.char.lower(X[street_dir + 'StreetName'].values.astype('str'))

        for type_cat in ['road', 'way', 'street', 'avenue', 'boulevard', 'lane', 'drive', 'terrace', 'place', 'court', 'plaza', 'square']:

            X['Is' + street_dir + type_cat] = np.char.rfind(data, type_cat) > -1

            

    #X = X.drop(columns=['IntersectionId', 'Center_Latitude', 'Center_Longitude', 'city_month', 'Latitude', 'Longitude', 'CenterDistance' ])

    #X = X.drop(columns=['EntryStreetName', 'ExitStreetName' ])



    road_type = []

    for street_dir in ['Entry', 'Exit']:

        for type_cat in ['road', 'way', 'street', 'avenue', 'boulevard', 'lane', 'drive', 'terrace', 'place', 'court', 'plaza', 'square']:

            road_type.append('Is' + street_dir + type_cat)

    

    return X[[

        'CenterDistance',

        'EntryHeading',

        'ExitHeading',

        'NS_Entry',

        'EW_Entry',

        'NS_Exit',

        'EW_Exit',

        'Entry_Angle',

        'Exit_Angle',

        'NS',

        'EW',

        'Angle',

        'x_Angle',

        'y_Angle',

        'is_day',

        'SameStreet',

        'SameHeading',

        'Vector',

        'Hour_x',

        'Hour_y',

        'Month_x',

        'Month_y',

        'City',

        'average_temp',

        'average_rainfall',

        'average_snowfall',

        'average_daylight',

        'average_sunshine',

        *road_type

    ]]
class custom_column_selector:

    def __init__(self, *, type_select, min_nunique=1, max_nunique=None, unicity_ratio=0.7, reverse=False):

        self.type_select = type_select

        self.min_nunique = min_nunique

        self.max_nunique = max_nunique

        self.unicity_ratio = unicity_ratio

        self.reverse = reverse



    def __call__(self, df):

        if not hasattr(df, 'iloc'):

            raise ValueError("make_column_selector can only be applied to "

                             "pandas dataframes")

        df_row = df.iloc[:1]

        df_row = df_row.select_dtypes(include=self.type_select)

        cols = df_row.columns.tolist()

        min_cols = df_row.columns[df[cols].nunique() > self.min_nunique].tolist()

        max_cols = cols

        if self.max_nunique is not None:

            max_cols = df_row.columns[df[cols].nunique() < self.max_nunique].tolist()



        return list(set(min_cols).intersection(set(max_cols)))      
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder, FunctionTransformer, QuantileTransformer

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
ct = ColumnTransformer(

    [

        (

            'one-hot', 

            OneHotEncoder(sparse=False, handle_unknown='ignore'),

            custom_column_selector(type_select=['object', 'int64', 'float64'], max_nunique=100)

        ),

        (

            'label-encode', 

            OrdinalEncoder(),

            custom_column_selector(type_select=['object', 'int64', 'float64'], min_nunique=100, max_nunique=1000)

        ),

        (

            'identity', 

            'passthrough',

            custom_column_selector(type_select=['int64', 'float64'], min_nunique=1000)

        )

    ], 

    remainder='drop', 

    sparse_threshold=0, 

    n_jobs=None, 

    #transformer_weights=None, 

    verbose=True)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

from sklearn.base import BaseEstimator



from tensorflow_addons.activations import gelu
def create_model(grid_params, in_dim, out_dim, patience=20, loss='rmse', activation='sigmoid'):

    

    mul_input = grid_params['mul_input']

    n_layer = grid_params['n_layer']

    

    first_layer_size = int(in_dim*mul_input)

    hidden_layers = []

    for i_layer in range(n_layer, 0, -1):

        layer_size = int(((first_layer_size - out_dim) / n_layer) * i_layer + out_dim)

        hidden_layers.append(layer_size)



    print("Input dim:" + str(in_dim))

    print("Hidden Layers:" + str(hidden_layers))

    print("Output dim:" + str(out_dim))



    model = Sequential()

    

    model.add(Dense(in_dim,input_shape=[in_dim],activation=gelu))

    #model.add(BatchNormalization())

    model.add(Dropout(.5))

    

    for layer in hidden_layers:

        model.add(Dense(layer,activation=gelu))

        #model.add(BatchNormalization())

        model.add(Dropout(.5))

    

    model.add(Dense(out_dim, activation=activation))

    

    radam = RectifiedAdam()

    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)

    optimizer = ranger#Adam(learning_rate=0.001)

    

    es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=patience, restore_best_weights=True)

    es.set_model(model)



    model.compile(optimizer=optimizer, loss=[loss], metrics=[])

    

    return model, [ es ]

class KerasModel(BaseEstimator):



    def __init__(

        self, 

        n_layer=1, 

        mul_input=1.75, 

        patience=5,

        batch_size=32,

        loss='msle',

        activation='sigmoid'

        ):

        self._estimator_type = 'reg' 

        self.n_layer = n_layer

        self.mul_input = mul_input

        self.patience = patience

        self.loss = loss

        self.activation = activation

        self.batch_size = batch_size

        #self.__name__ = self._wrapped_obj.__class__.__name__ + "PredictWrapper"



    def __repr__(self):

        if not hasattr(self, 'model'):

            return "Empty"

        return self.model.__repr__()



    def __str__(self):

        if not hasattr(self, 'model'):

            return "Empty"

        return self.model.__str__()

        

    def fit(self, X, Y):

        model, cbs = create_model(

            self.get_params(),

            X.shape[1],

            Y.shape[1],

            patience=self.patience,

            loss=self.loss,

            activation=self.activation

        )

        X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

        self.model = model

        self.model.fit(X_train,y_train, batch_size=self.batch_size,epochs=10000, validation_data=[X_valid,y_valid], verbose=2, callbacks=cbs)

        return self



    def predict(self, *args, **kwargs):

        return self.model.predict(*args, **kwargs)

model = KerasModel(n_layer=3, mul_input=8, batch_size=1024, patience=10, activation=None, loss='mse')
pipeline = Pipeline(steps=[

    ('feature-engineering', FunctionTransformer(pre_process)), # First, we build more features

    ('data-prep', ct), # Then, we apply columns transformations

    ('robust-scaler', RobustScaler()), # Then, we standard-scale the whole dataset

    #('pca', PCA(.9999)), # Should I use PCA to reduce dimension ?

    ('model', model)

], verbose= True)
%%time

# Fitting the whole pipeline

pipeline.fit(X, Y.values)
#raise Exception('Stop HERE')
# Forcing memory cleaning (needed for XGBppst or LGBM)

del X, Y

gc.collect()
%%time

Y_pred = pipeline.predict(X_valid)
from sklearn.metrics import mean_squared_error

mean_squared_error(Y_valid, Y_pred, squared=False)
del X_valid, Y_valid, Y_pred

gc.collect()
X_test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv', low_memory=False)
Y_test = pipeline.predict(X_test)

res_df = pd.DataFrame(data=Y_test, columns=targets)

res_df['RowId'] = X_test['RowId']

del X_test

gc.collect()
res_map = {

    'TotalTimeStopped_p20':'0',

    'TotalTimeStopped_p50':'1',

    'TotalTimeStopped_p80':'2',

    'DistanceToFirstStop_p20':'3',

    'DistanceToFirstStop_p50':'4',

    'DistanceToFirstStop_p80':'5'

}

final_df = pd.DataFrame()

final_df['RowId'] = res_df['RowId']

for key, value in res_map.items():

    final_df[value] = res_df[key]

final_df = pd.melt(final_df, id_vars=['RowId'], value_vars=['0','1','2','3','4','5'], var_name='target', value_name='result')
final_df['RowId'] = final_df['RowId'].astype('str')

final_df['target'] = final_df['target'].astype('str')

final_df['RowId'] = final_df['RowId'] + '_' + final_df['target']

final_df = final_df.rename(columns={

    'RowId': 'TargetId',

    'result': 'Target'

})

final_df = final_df.drop(columns=['target'])

final_df.to_csv('final_res.csv', index=False, sep=',', encoding='utf-8')
del res_df

gc.collect()