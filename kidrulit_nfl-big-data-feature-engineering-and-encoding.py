import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import datetime

from kaggle.competitions import nflrush

import tqdm

import re

from string import punctuation

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin



from tensorflow.keras import Sequential

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Add, Embedding, Concatenate



from sys import getsizeof

from sklearn.model_selection import KFold, GroupKFold



sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]
DATA_FOLDER = '../input/nfl-big-data-bowl-2020'

ARRAYS_FOLDER = '../files/ndarrays'

MODEL_FOLDER = '../files/models'
train_df = pd.read_csv(DATA_FOLDER+'/train.csv', low_memory = False)
MAX_DATE = '2018-12-31'



outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 

           'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']



indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',

                 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']



indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

dome_open     = ['Domed, Open', 'Domed, open']



rain = ['Rainy', 'Rain Chance 40%', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']



overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',

            'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',

            'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',

            'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',

            'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',

            'Partly Cloudy', 'Cloudy']



clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',

        'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',

        'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',

        'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',

        'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',

        'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy']



snow  = ['Heavy lake effect snow', 'Snow']



none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']





north = ['N','From S','North']

south = ['S','From N','South','s']

west = ['W','From E','West']

east = ['E','From W','from W','EAST','East']

north_east = ['FROM SW','FROM SSW','FROM WSW','NE','NORTH EAST','North East','East North East','NorthEast','Northeast','ENE','From WSW','From SW']

north_west = ['E','From ESE','NW','NORTHWEST','N-NE','NNE','North/Northwest','W-NW','WNW','West Northwest','Northwest','NNW','From SSE']

south_east = ['E','From WNW','SE','SOUTHEAST','South Southeast','East Southeast','Southeast','SSE','From SSW','ESE','From NNW']

south_west = ['E','From ENE','SW','SOUTHWEST','W-SW','South Southwest','West-Southwest','WSW','SouthWest','Southwest','SSW','From NNE']

no_wind = ['clear','Calm']

nan = ['1','8','13']



natural_grass = ['natural grass','Naturall Grass','Natural Grass']

grass = ['Grass']

fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

artificial = ['Artificial','Artifical']





def feature_engineering(data_frame, features_to_convert, convert_null_variables=True):

    df = data_frame.copy()

    

    def strtoseconds(txt):

        txt = txt.split(':')

        ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

        return ans

    

    def convert_jn_to_position(df):

        jn = df['JerseyNumber']

        if jn < 10:

            df['QB'] = 1

            df['KP'] = 1

        elif jn >= 10 and jn < 20:

            df['QB'] = 1

            df['WR'] = 1

            df['KP'] = 1

        elif jn >= 20 and jn < 40:

            df['RB'] = 1

            df['DB'] = 1

        elif jn >= 40 and jn < 50:

            df['RB'] = 1

            df['LB'] = 1

            df['DB'] = 1

            df['TE'] = 1

        elif jn >= 50 and jn < 60:

            df['OL'] = 1

            df['DL'] = 1

            df['LB'] = 1

        elif jn >= 60 and jn < 80:

            df['OL'] = 1

            df['DL'] = 1

        elif jn >= 80 and jn < 90:

            df['WR'] = 1

            df['TE'] = 1

        elif jn >= 90 and jn < 100:

            df['DL'] = 1

            df['LB'] = 1



        return df

    

    def clean_wind_speed(windspeed):

        nan = ['nan','E','SE','Calm','SSW', 'SSE']

        

        def avg_list(temp_list):

            int_temp_list = [int(x) for x in temp_list]

            return sum(int_temp_list)/len(temp_list)    



        ws = str(windspeed)

        if ws in nan:

            return np.nan

        else:

            matches = re.findall('(\d+)', ws, re.DOTALL)

            return avg_list(matches)

    

    #event data

    if 'TimeHandoff' in features_to_convert:

        df['Month'] = df['TimeHandoff'].apply(lambda x : int(x[5:7]))

        df['Year'] = df['TimeHandoff'].apply(lambda x : int(x[0:4]))

        df['Morning'] = df['TimeHandoff'].apply(lambda x : 1 if (int(x[11:13]) >=0 and int(x[11:13]) <12) else 0)

        df['Afternoon'] = df['TimeHandoff'].apply(lambda x : 1 if (int(x[11:13]) <18 and int(x[11:13]) >=12) else 0)

        df['Evening'] = df['TimeHandoff'].apply(lambda x : 1 if (int(x[11:13]) >= 18 and int(x[11:13]) < 24) else 0)

        df['Month_Snap'] = df['TimeSnap'].apply(lambda x : int(x[5:7]))

        df['Year_Snap'] = df['TimeSnap'].apply(lambda x : int(x[0:4]))

        df.drop(['TimeHandoff'], axis=1,inplace=True)



    if 'TimeSnap' in features_to_convert:

        df['Morning_Snap'] = df['TimeSnap'].apply(lambda x : 1 if (int(x[11:13]) >=0 and int(x[11:13]) <12) else 0)

        df['Afternoon_Snap'] = df['TimeSnap'].apply(lambda x : 1 if (int(x[11:13]) <18 and int(x[11:13]) >=12) else 0)

        df['Evening_Snap'] = df['TimeSnap'].apply(lambda x : 1 if (int(x[11:13]) >= 18 and int(x[11:13]) < 24) else 0)

        df['GameClock'] = df['GameClock'].apply(strtoseconds)

        df.drop(['TimeSnap'], axis=1, inplace=True)

    

    if 'JerseyNumber' in features_to_convert:

        l = ['QB', 'KP', 'WR', 'RB', 'DB', 'LB', 'DB', 'TE', 'OL', 'DL']

        d = dict.fromkeys(l, 0)

        df = df.assign(**d).apply(convert_jn_to_position, axis=1)

        df.drop(['JerseyNumber'], axis=1, inplace=True)

    

    if 'PlayerHeight' in features_to_convert:

        df["HeightFt"] = df["PlayerHeight"].str.split('-', expand=True)[0].astype(int)

        df["HeightIn"] = df["PlayerHeight"].str.split('-', expand=True)[1].astype(int)

        df["HeightCm"] = df["HeightFt"]*30.48 + df["HeightIn"]*2.54

        df.drop(['PlayerHeight','HeightIn', 'HeightFt'], axis=1,inplace=True)



    if 'PlayerWeight' in features_to_convert:

        df["WeightKg"] = df["PlayerWeight"]*0.45359237

        df.drop(['PlayerWeight'], axis=1,inplace=True)

    

    if 'PlayerBirthDate' in features_to_convert:

        df['BirthDate'] = df['PlayerBirthDate'].astype('datetime64[ns]')

        df['Age'] = round((pd.to_datetime(MAX_DATE) - df['BirthDate'])/np.timedelta64(1,'D')/365.25,1)

        df.drop(['BirthDate', 'PlayerBirthDate'], axis = 1, inplace=True)

    

    if 'StadiumType' in features_to_convert:

        #stadium data

        df['StadiumType'] = df['StadiumType'].replace(outdoor,'outdoor')

        df['StadiumType'] = df['StadiumType'].replace(indoor_closed,'indoor_closed')

        df['StadiumType'] = df['StadiumType'].replace(indoor_open,'indoor_open')

        df['StadiumType'] = df['StadiumType'].replace(dome_closed,'dome_closed')

        df['StadiumType'] = df['StadiumType'].replace(dome_open,'dome_open')

        df['StadiumType'] = df['StadiumType'].replace(np.nan,'no_data')



        df['GameWeather'] = df['GameWeather'].replace(outdoor,'outdoor')

        df['GameWeather'] = df['GameWeather'].replace(indoor_closed,'indoor_closed')

        df['GameWeather'] = df['GameWeather'].replace(indoor_open,'indoor_open')

        df['GameWeather'] = df['GameWeather'].replace(dome_closed,'dome_closed')

        df['GameWeather'] = df['GameWeather'].replace(dome_open,'dome_open')

        df['Turf'] = df['Turf'].replace(natural_grass,'natural_grass')

        df['Turf'] = df['Turf'].replace(grass,'grass')

        df['Turf'] = df['Turf'].replace(fieldturf,'fieldturf')

        df['Turf'] = df['Turf'].replace(artificial,'artificial')

    

    if 'WindSpeed' in features_to_convert:

        #weather speed

        df['WindSpeed'] = df['WindSpeed'].apply(clean_wind_speed)

    

    if 'WindDirection' in features_to_convert:

        df['WindDirection'] = df['WindDirection'].replace(north,'north')

        df['WindDirection'] = df['WindDirection'].replace(south,'south')

        df['WindDirection'] = df['WindDirection'].replace(west,'west')

        df['WindDirection'] = df['WindDirection'].replace(east,'east')

        df['WindDirection'] = df['WindDirection'].replace(north_east,'north_east')

        df['WindDirection'] = df['WindDirection'].replace(north_west,'north_west')

        df['WindDirection'] = df['WindDirection'].replace(south_east,'clear')

        df['WindDirection'] = df['WindDirection'].replace(south_west,'south_west')

        df['WindDirection'] = df['WindDirection'].replace(no_wind,'no_wind')

        df['WindDirection'] = df['WindDirection'].replace(nan,np.nan)

    

    df['IsRusher'] = (df['NflId'] == df['NflIdRusher']).astype(int)

    

    if 'OffensePersonnel' in features_to_convert:

        df = pd.concat([df, df['OffensePersonnel'].str.get_dummies(sep=',').add_prefix('Offence_')], axis=1)

        df.drop(['OffensePersonnel'], axis=1, inplace=True)

    if 'DefensePersonnel' in features_to_convert:

        df = pd.concat([df, df['DefensePersonnel'].str.get_dummies(sep=',').add_prefix('Defence_')], axis=1)

        df.drop(['DefensePersonnel'], axis=1, inplace=True)

    

    if convert_null_variables:

        null_columns = df.columns[df.isna().any()].tolist()

        num_null_columns = df[null_columns].select_dtypes(exclude='object')

        object_null_cloumns = df[null_columns].select_dtypes(include='object')

        for column in num_null_columns:

            df[column].fillna((df[column].median()), inplace=True)



        for column in object_null_cloumns:

            df[column].fillna(('NaN'), inplace=True)

            

    return df
def columns(df, treshold):

    cat_cols = []

    dense_cols = []

    cat_cols_to_shrink = []



    for i, column in enumerate(df.columns):

        if (str(df[column].dtype)=="object" or str(df[column].dtype)=="category") :

            cat_cols.append(column)

            if df[column].nunique()>treshold:

                cat_cols_to_shrink.append(column)   

        else:

            dense_cols.append(column)



    return (cat_cols, dense_cols, cat_cols_to_shrink)
class TargetEncoder(BaseEstimator, TransformerMixin):

    """Target encoder.

    

    Replaces categorical column(s) with the mean target value for

    each category.



    """

    

    def __init__(self, cols=None):

        """Target encoder

        

        Parameters

        ----------

        cols : list of str

            Columns to target encode.  Default is to target 

            encode all categorical columns in the DataFrame.

        """

        if isinstance(cols, str):

            self.cols = [cols]

        else:

            self.cols = cols

        

        

    def fit(self, X, y):

        """Fit target encoder to X and y

        

        Parameters

        ----------

        X : pandas DataFrame, shape [n_samples, n_columns]

            DataFrame containing columns to encode

        y : pandas Series, shape = [n_samples]

            Target values.

            

        Returns

        -------

        self : encoder

            Returns self.

        """

        

        # Encode all categorical cols by default

        if self.cols is None:

            self.cols = [col for col in X 

                         if str(X[col].dtype)=='object']



        # Check columns are in X

        for col in self.cols:

            if col not in X:

                raise ValueError('Column \''+col+'\' not in X')



        # Encode each element of each column

        self.maps = dict() #dict to store map for each column

        for col in self.cols:

            tmap = dict()

            uniques = X[col].unique()

            for unique in uniques:

                target_mean = y[X[col]==unique].mean()

                tmap[unique] = target_mean

            self.maps[col] = tmap

            

        return self



        

    def transform(self, X, y=None):

        """Perform the target encoding transformation.

        

        Parameters

        ----------

        X : pandas DataFrame, shape [n_samples, n_columns]

            DataFrame containing columns to encode

            

        Returns

        -------

        pandas DataFrame

            Input DataFrame with transformed columns

        """

        Xo = X.copy()

        for col, tmap in self.maps.items():

            vals = np.full(X.shape[0], np.nan)

            for val, mean_target in tmap.items():

                vals[X[col]==val] = mean_target

            Xo[col] = vals

        return Xo

            

            

    def fit_transform(self, X, y=None):

        """Fit and transform the data via target encoding.

        

        Parameters

        ----------

        X : pandas DataFrame, shape [n_samples, n_columns]

            DataFrame containing columns to encode

        y : pandas Series, shape = [n_samples]

            Target values (required!).



        Returns

        -------

        pandas DataFrame

            Input DataFrame with transformed columns

        """

        return self.fit(X, y).transform(X, y)
from sklearn.preprocessing import LabelEncoder



class MultiColumnLabelEncoder:

    """Multicolumns label encoder.

    

    Wrapper for Label Encoder over multiple columns



    """

    def __init__(self,columns = None):

        """Target encoder

        

        Parameters

        ----------

        columns : list of str

            Columns to label encode.  Default is to target 

            encode all categorical columns in the DataFrame.

        """

        self.columns = columns # array of column names to encode

        

    def fit(self,X,y):

        return self

    



    def transform(self,X):

        """Perform the label encoding transformation for multiple columns.

        

        Parameters

        ----------

        X : pandas DataFrame, shape [n_samples, n_columns]

            DataFrame containing columns to encode

            

        Returns

        -------

        pandas DataFrame

            Input DataFrame with transformed columns

        """

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        """Fit and transform the data via label encoding.

        

        Parameters

        ----------

        X : pandas DataFrame, shape [n_samples, n_columns]

            DataFrame containing columns to encode

        y : pandas Series, shape = [n_samples]

            Target values (required!).



        Returns

        -------

        pandas DataFrame

            Input DataFrame with transformed columns

        """

        return self.fit(X,y).transform(X)

    

class NeuralNetworkEncoder:

    """Cat2vec encoder.

    

    Deep embedding's for categorical variables



    """

    

    def __init__(self, cat_cols, treshold):

        """Cat2vec encoder

        

        Parameters

        ----------

        cat_cls : list of str

            Columns to label encode.  Default is WIP

        """

        self.columns = cat_cols

        self.max_emb_size = treshold



    def fit(self, X, y):

        X_cat = [X[col].values for col in self.columns]

        y_ = np.zeros((y.shape[0], 199))

        for idx, val in enumerate(list(y)):

            y_[idx][99 + val] = 1



        inputs = []

        embeddings = []



        for col in self.columns:

            #as we itereate over columns so the shape = (1,) 

            input_ = Input(shape=(1,))   



            no_of_unique_cat = X[col].nunique()

            embedding_size = int(min(np.ceil((no_of_unique_cat+1)/2), self.max_emb_size))

            input_emb_dim = no_of_unique_cat

            emb_input_length = 1



            embedding = Embedding(input_emb_dim, embedding_size, input_length=emb_input_length)(input_)

            embedding = Reshape(target_shape=(embedding_size,))(embedding)

            inputs.append(input_)

            embeddings.append(embedding)





        x = Concatenate()(embeddings)

        x = Dense(256, activation='relu')(x)

        x = Dense(128, activation='relu')(x)

        x = Dropout(0.5)(x)

        output = Dense(199, activation='softmax')(x)

        model = Model(inputs, output)

        model.compile(loss='binary_crossentropy', optimizer='adam')



        model.fit(X_cat, y_)

        self.model = model

        return self

    

    def transform(self, X):

        """Perform the cat2vec encoding transformation for multiple columns.

        

        Parameters

        ----------

        X : pandas DataFrame, shape [n_samples, n_columns]

            DataFrame containing columns to encode

            

        Returns

        -------

        pandas DataFrame

            Input DataFrame with transformed columns

        """

        df = X.copy()

        trained_embedings = self.model.layers[len(self.columns):2*len(self.columns)]



        for i,cat in enumerate(self.columns):

            cat_emb_df = pd.DataFrame(trained_embedings[i].get_weights()[0])

            cat_emb_df.columns = [cat + '_' + str(col) + '_emb' for col in cat_emb_df.columns]

            X = X.merge(cat_emb_df, left_on = cat, right_index=True)

            X.drop(cat, axis=1, inplace=True)

    

        return df

    

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)
#implicitly specify the columns we to transform

features_to_transform = ['TimeHandoff', 'TimeSnap', 'JerseyNumber', 'PlayerHeight', \

                       'PlayerWeight', 'PlayerBirthDate', 'StadiumType' \

                      'OffensePersonnel', 'DefensePersonnel']





class DataFrameEncoder:

    

    def __init__(self, features_to_convert, shrink_treshold):

        """Dataframe encoder

        

        Parameters

        ----------

        features_to_convert : list 

            Columns to encode.  Default is not set yet

        shrink_treshold : int

            A threshold that specify N the top-N value counts columns

        """

        

        self.features_to_convert = features_to_convert

        self.shrink_treshold = shrink_treshold

    

    def fit_transform(self, df, target):

        df_ = df.copy()

        df_ = feature_engineering(df_, self.features_to_convert, convert_null_variables=True)

        self.cat_cols, self.dense_cols, self.cat_cols_to_shrink = columns(df_, self.shrink_treshold)



        #define encoders

        if len(self.cat_cols_to_shrink) != 0:

            print("---Start target encoding---")

            self.te = TargetEncoder(cols = self.cat_cols_to_shrink)

            df_ = self.te.fit_transform(df_, target)

            self.dense_cols = self.dense_cols + self.cat_cols_to_shrink

            print("---Successfully finished target encoding---")

        

        print("---Start label encoding---")

        self.mle = MultiColumnLabelEncoder(columns = self.cat_cols)

        df_ = self.mle.fit_transform(df_)

        print("---Successfully finished label encoding---")



        print("---Start neural network encoding---")

        self.nne = NeuralNetworkEncoder(cat_cols = self.cat_cols, treshold = self.shrink_treshold)

        

        df_ = self.nne.fit_transform(df_, target)

        print("---Successfully finished network encoding---")



        self.columns = df_.columns

    

        return df_

    

    def transform(self, df):

        df_ = df.copy()

        df_ = feature_engineering(df_, self.features_to_convert, convert_null_variables=True)

        

        df_ = self.te.transform(df_)

        df_ = self.mle.transform(df_)

        df_ = self.nne.transform(df_)

        

        #if some of the train columns doesn't exist in test

        not_exsit_in_test_columns = list(set(self.columns) - set(df_.columns))

        d = dict.fromkeys(not_exsit_in_test_columns, 0)

        df_ = df_.assign(**d)

        

        not_exist_in_train_columns = list(set(df_.columns)-set(self.columns))

        df_.drop(not_exist_in_train_columns, axis=1, inplace=True)

        return df_
df_transfromer = DataFrameEncoder(features_to_transform, 70)

df_train_transformed = df_transfromer.fit_transform(train_df.drop(['Yards'], axis=1), train_df['Yards'])
# train_df_transformed_ohe = train_df_transformed.where(train_df_transformed.apply(lambda x: x.map(x.value_counts()))>=200, "other")

# cat_train_ohe = np.array(pd.get_dummies(train_df_transformed_ohe[cat_cols],sparse=True))
import lightgbm as lgb

from sklearn.metrics import mean_squared_error,mean_absolute_error
X_train = df_train_transformed.drop(['GameId', 'PlayId'], axis=1)

y_train = train_df['Yards']
best_params_lgb = {'lambda_l1': 0.13413394854686794, 

'lambda_l2': 0.0009122197743451751, 

'num_leaves': 44, 

'feature_fraction': 0.4271070738920401, 

'bagging_fraction': 0.9999128827046064, 

'bagging_freq': 3, 

"learning_rate": 0.005,

'min_child_samples': 43, 

'objective': 'regression', 

'metric': 'mae', 

'verbosity': -1, 

'boosting_type': 'gbdt', 

"boost_from_average" : False,

'random_state': 42}
from sklearn.model_selection import KFold

from sklearn.model_selection import GroupKFold

nfold = 5

folds = KFold(n_splits=nfold, shuffle=False, random_state=42)



groups = df_train_transformed['PlayId']

gkf = GroupKFold(n_splits=nfold)



print('-'*20)

print(str(nfold) + ' Folds training...')

print('-'*20)



oof = np.zeros(len(X_train))

#y_valid_pred = np.zeros(X_train.shape[0])

feature_importance_df = pd.DataFrame()



tr_mae = []

val_mae = []

models = []
for fold_, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    

    

    X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]

    train_y, y_val = y_train[trn_idx], y_train[val_idx]

    

    model = lgb.LGBMRegressor(**best_params_lgb, n_estimators = 300, n_jobs = -1,early_stopping_rounds = 100)

    model.fit(X_tr, 

              train_y, 

              eval_set=[(X_tr, train_y), (X_val, y_val)], 

              eval_metric='mae',

              verbose=10

              )

    oof[val_idx] = model.predict(X_val)

    val_score = mean_absolute_error(y_val, oof[val_idx])

    val_mae.append(val_score)

    tr_score = mean_absolute_error(train_y, model.predict(X_tr))

    tr_mae.append(tr_score)

    models.append(model)

    

    

    # Feature importance

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = X_tr.columns

    fold_importance_df["importance"] = model.feature_importances_[:len(X_tr.columns)]

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
mean_mae_tr = np.mean(tr_mae)

std_mae_tr =  np.std(tr_mae)



mean_mae_val =  np.mean(val_mae)

std_mae_val =  np.std(val_mae)



all_mae = mean_absolute_error(oof,y_train)



print('-'*20)

print("Train's Score")

print('-'*20,'\n')

print("Mean mae: %.5f, std: %.5f." % (mean_mae_tr, std_mae_tr),'\n')



print('-'*20)

print("Validation's Score")

print('-'*20,'\n')

print("Mean mae: %.5f, std: %.5f." % (mean_mae_val, std_mae_val),'\n')



print("All mae: %.5f." % (all_mae))
cols_imp = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:50].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols_imp)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
env = nflrush.make_env()
pd.options.mode.chained_assignment = None

for (df_test, sample_prediction_df) in tqdm.tqdm(env.iter_test()):

    df_test_transfromed = df_transfromer.transform(df_test)

    df_test_transfromed.drop(['GameId', 'PlayId'], axis=1, inplace=True)

    

    y_pred = np.zeros(199)        

    y_pred_p = np.mean([model.predict(df_test_transfromed)[0] for model in models])

    y_pred_p += 99

    for j in range(199):

        if j>=y_pred_p+10:

            y_pred[j]=1.0

        elif j>=y_pred_p-10:

            y_pred[j]=(j+10-y_pred_p)*0.05

            

    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction_df.columns))

env.write_submission_file()