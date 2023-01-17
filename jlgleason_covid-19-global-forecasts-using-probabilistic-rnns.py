# install gluonts package

!pip install gluonts

!pip install sentence_transformers
# load and clean data

import pandas as pd

import numpy as np

train_all = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")



def preprocess(

    frame: pd.DataFrame,

    log_transform: bool = False

):

    

    # set index

    new_frame = frame.set_index('Date')



    # fill 'NaN' Province/State values with Country/Region values

    new_frame['Province/State'] = new_frame['Province/State'].fillna(new_frame['Country/Region'])

    

    # convert target values to log scale

    if log_transform:

        new_frame[['ConfirmedCases', 'Fatalities']] = np.log1p(

            new_frame[['ConfirmedCases', 'Fatalities']].values

    )

    

    return new_frame



def split(

    df: pd.DataFrame, 

    date: str = '2020-03-12'

):

    train = df.loc[train_all.index < date] 

    test = df.loc[train_all.index >= date]

    return train, test



train_all = preprocess(train_all)

train, test = split(train_all)
# plot confirmed cases and fatalities in train

import matplotlib.pyplot as plt

from gluonts.dataset.util import to_pandas

from gluonts.dataset.common import ListDataset



cum_train = train.groupby('Date').sum()

cum_test = test.groupby('Date').sum()



def plot_observations(

    target: str = 'ConfirmedCases'

):

    fig = plt.figure(figsize=(15, 6.1), facecolor="white",  edgecolor='k')



    train_ds = ListDataset(

        [{"start": cum_train.index[0], "target": cum_train[target].values}],

        freq = "D",

    )

    test_ds = ListDataset(

        [{"start": cum_test.index[0], "target": cum_test[target].values}],

        freq = "D",

    )

    

    for tr, te in zip(train_ds, test_ds):

        tr = to_pandas(tr)

        te = to_pandas(te)

        tr.plot(linewidth=2, label = f'train {target}')

        tr[-1:].append(te).plot(linewidth=2, label = f'test {target}')

    

    plt.axvline(cum_train.index[-1], color='purple') # end of train dataset

    plt.title(f'Cumulative global number of {target}', fontsize=16)

    plt.legend(fontsize=16)

    plt.grid(which="both")

    plt.show()

    

plot_observations('ConfirmedCases')

plot_observations('Fatalities')
from sentence_transformers import SentenceTransformer



model = SentenceTransformer('bert-base-nli-mean-tokens')

country_embeddings = model.encode(list(train['Country/Region'].unique()))

embedding_dim = len(country_embeddings[0])

embed_df = pd.DataFrame(np.concatenate([np.array(list(train['Country/Region'].unique())).reshape(-1,1),country_embeddings],axis=1))

embed_df.columns=['Country/Region']+list(range(embedding_dim))
from sklearn.preprocessing import OrdinalEncoder



#pop_df = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')



country_map = {

    "Congo (Brazzaville)": "Congo",

    "Congo (Kinshasa)": "Congo",

    "Cote d'Ivoire": "Côte d'Ivoire",

    "Czechia": "Czech Republic (Czechia)",

    "Gambia, The": "Gambia",

    "Guernsey": "Channel Islands",

    "Jersey": "Channel Islands",

    "Korea, South": "South Korea",

    "Republic of the Congo": "DR Congo",

    "Reunion": "Réunion",

    "Saint Vincent and the Grenadines": "St. Vincent & Grenadines",

    "Taiwan*": "Taiwan",

    "The Bahamas": "Bahamas",

    "The Gambia": "Gambia",

    "US": "United States"

}



def clean_pop_data(

    pop_df: pd.DataFrame

):

    """ bootstrapped from https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions"""

        

    # remove the % character from Urban Pop, World Share, and Yearly Change values 

    pop_df['Yearly Change'] = pop_df['Yearly Change'].str.rstrip('%')

    pop_df['Urban Pop %'] = pop_df['Urban Pop %'].str.rstrip('%')

    pop_df['World Share'] = pop_df['World Share'].str.rstrip('%')

    

    # Replace Urban Pop, Fertility Rate, and Med Age "N.A" by their respective modes

    pop_df.loc[pop_df['Urban Pop %']=='N.A.', 'Urban Pop %'] = pop_df.loc[pop_df['Urban Pop %']!='N.A.', 'Urban Pop %'].mode()[0]

    pop_df.loc[pop_df['Med. Age']=='N.A.', 'Med. Age'] = pop_df.loc[pop_df['Med. Age']!='N.A.', 'Med. Age'].mode()[0]    

    pop_df.loc[pop_df['Fert. Rate']=='N.A.', 'Fert. Rate'] = pop_df.loc[pop_df['Fert. Rate']!='N.A.', 'Fert. Rate'].mode()[0]    

    

    # replace empty migration rows with 0s

    pop_df['Migrants (net)'] =  pop_df['Migrants (net)'].fillna(0)

    

    # cast to types

    pop_df = pop_df.astype({

        "Population (2020)": int,

        "Yearly Change": float,

        "Net Change": int,

        "Density (P/Km²)": int,

        "Land Area (Km²)": int,

        "Migrants (net)": float,

        "Fert. Rate": float,

        "Med. Age": int,

        "Urban Pop %": int,

        "World Share": float,

    })   



    return pop_df



def join(

    df: pd.DataFrame,

    pop_df: pd.DataFrame

):

    

    # add join column with country mapping

    df['join_col'] = [

        val if val not in country_map.keys() 

        else country_map[val] 

        for val in df['Country/Region'].values

    ]

    

    # join, delete merge columns

    new_df = df.reset_index().merge(

        pop_df,

        left_on = 'join_col',

        right_on = 'Country (or dependency)',

        how = 'left'

    ).set_index('Date')

    new_df = new_df.drop(columns=['join_col', 'Country (or dependency)'])

    

    # replace columns that weren't matched in join with mean

    new_df = new_df.fillna(new_df.mean())

    

    return new_df



def join_with_embeddings(

    df: pd.DataFrame,

    embed_df: pd.DataFrame

):

    

    # join, delete merge columns

    new_df = df.reset_index().merge(

        embed_df,

        left_on = 'Country/Region',

        right_on = 'Country/Region',

        how = 'left'

    ).set_index('Date')

    

    # make sure no NaN in dataframe

    assert new_df.isnull().sum().sum()==0

    return new_df



def encode(

    df: pd.DataFrame

):

    """ encode 'Province/State' and 'Country/Region' categorical variables as numerical ordinals"""

    

    enc = OrdinalEncoder()

    df[['Province/State', 'Country/Region']] = enc.fit_transform(

        df[['Province/State', 'Country/Region']].values

    )

    return df, enc



# pop_df = clean_pop_data(pop_df)

join_df = join_with_embeddings(train_all, embed_df)



all_df, enc = encode(join_df)

train_df, test_df = split(all_df)

_, val_df = split(all_df, date = '2020-02-28')

from gluonts.dataset.common import ListDataset

from gluonts.dataset.field_names import FieldName

import typing



REAL_VARS = [

    'Population (2020)', 

    'Yearly Change', 

    'Net Change', 

    'Density (P/Km²)', 

    'Land Area (Km²)',

    'Migrants (net)',

    'Fert. Rate',

    'Med. Age',

    'Urban Pop %',

    'World Share'

]



EMBED_VARS = list(range(embedding_dim))



def build_dataset(

    frame: pd.DataFrame,

    target: str = 'Fatalities',

    cat_vars: typing.List[str] = ['Province/State', 'Country/Region'],

    real_vars: typing.List[str] = EMBED_VARS

):

    return ListDataset(

        [

            {

                FieldName.START: df.index[0], 

                FieldName.TARGET: df[target].values,

                FieldName.FEAT_STATIC_CAT: df[cat_vars].values[0],

                FieldName.FEAT_STATIC_REAL: df[real_vars].values[0]

            }

            for g, df in frame.groupby(by=['Province/State', 'Country/Region'])

        ],

        freq = "D",

    )



training_data_fatalities = build_dataset(train_df)

training_data_cases = build_dataset(train_df, target = 'ConfirmedCases')

training_data_fatalities_all = build_dataset(all_df)

training_data_cases_all = build_dataset(all_df, target = 'ConfirmedCases')

val_data_fatalities = build_dataset(val_df)

val_data_cases = build_dataset(val_df, target = 'ConfirmedCases')
from gluonts.model.deepar import DeepAREstimator

from gluonts.trainer import Trainer

from gluonts.distribution import NegativeBinomialOutput

import mxnet as mx

import numpy as np



# set random seeds for reproducibility

mx.random.seed(42)

np.random.seed(42)



def fit(

    training_data: ListDataset,

    validation_data: ListDataset = None,

    use_real_vars: bool = False,

    pred_length: int = 14,

    epochs: int = 10,

    weight_decay: float = 5e-5,

):

    estimator = DeepAREstimator(

        freq="D", 

        prediction_length=pred_length,

        use_feat_static_real = use_real_vars,

        use_feat_static_cat = True,

        cardinality = [train['Province/State'].nunique(), train['Country/Region'].nunique()],

        distr_output=NegativeBinomialOutput(),

        trainer=Trainer(

            epochs=epochs,

            learning_rate=0.005, 

            batch_size=64,

            weight_decay=weight_decay,

            learning_rate_decay_factor=0.1,

            patience=15,

        ),

    )

    predictor = estimator.train(training_data = training_data, validation_data = validation_data)

    

    return predictor



predictor_fatalities = fit(training_data_fatalities)

predictor_cases = fit(training_data_cases, epochs=20)

predictor_fatalities_all = fit(training_data_fatalities_all, pred_length = 30, epochs=20)

predictor_cases_all = fit(training_data_cases_all, pred_length = 30, epochs=40)
from gluonts.dataset.util import to_pandas

import matplotlib.pyplot as plt

from typing import List



def plot_forecast(

    predictor,

    location: List[str] = ['Italy', 'Italy'],

    target: str = 'Fatalities',

    cat_vars: typing.List[str] = ['Province/State', 'Country/Region'],

    real_vars: typing.List[str] = EMBED_VARS,

    log_preds: bool = False,

    fontsize: int = 16

):

    fig = plt.figure(figsize=(15, 6.1), facecolor="white",  edgecolor='k')



    # plot train observations, true observations from public test set, and forecasts

    location_tr = enc.transform(np.array(location).reshape(1,-1))

    tr_df = train_df[np.all((train_df[['Province/State', 'Country/Region']].values == location_tr), axis=1)]

    train_obs = ListDataset(

        [{

            FieldName.START: tr_df.index[0], 

            FieldName.TARGET: tr_df[target].values,

            FieldName.FEAT_STATIC_CAT: tr_df[cat_vars].values[0],

            FieldName.FEAT_STATIC_REAL: tr_df[real_vars].values[0]

        }],

        freq = "D",

    )

    te_df = test_df[np.all((test_df[['Province/State', 'Country/Region']].values == location_tr), axis=1)]

    test_gt = ListDataset(

        [{"start": te_df.index[0], "target": te_df[target].values}],

        freq = "D",

    )

    for train_series, gt, forecast in zip(train_obs, test_gt, predictor.predict(train_obs)):

        

        train_series = to_pandas(train_series)

        gt = to_pandas(gt)

        

        if log_preds:

            train_series = np.expm1(train_series)

            gt = np.expm1(gt)

            forecast.samples = np.expm1(forecast.samples)

        

        train_series.plot(linewidth=2, label = 'train series')

        gt.plot(linewidth=2, label = 'test ground truth')

        forecast.plot(color='g', prediction_intervals=[50.0, 90.0])

        

    plt.title(f'Cumulative number of {target} in {location[0]}', fontsize=fontsize)

    plt.legend(fontsize = fontsize)

    plt.grid(which='both')

    plt.show()

    

plot_forecast(predictor_fatalities, ['Italy', 'Italy'])

plot_forecast(predictor_fatalities, ['California', 'US'])

plot_forecast(predictor_fatalities, ['Korea, South', 'Korea, South'])

plot_forecast(predictor_fatalities, ['Hubei', 'China'])



from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.evaluation import Evaluator

import json



all_data = build_dataset(all_df)



# evaluate fatalities predictor

forecast_iterable, ts_iterable = make_evaluation_predictions(

    dataset=all_data,

    predictor=predictor_fatalities,

    num_samples=100

)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

agg_metrics, item_metrics = evaluator(ts_iterable, forecast_iterable, num_series=len(all_data))

print('Fatalities Predictor Metrics: ')

print(json.dumps(agg_metrics, indent=4))



# evaluate confirmed cases predictor

forecast_iterable, ts_iterable = make_evaluation_predictions(

    dataset=all_data,

    predictor=predictor_cases,

    num_samples=100

)

agg_metrics, item_metrics = evaluator(ts_iterable, forecast_iterable, num_series=len(all_data))

print('Confirmed Cases Predictor Metrics: ')

print(json.dumps(agg_metrics, indent=4))
# generate submission csv



# aggregate fatalities

fatalities = []

for public_forecast, private_forecast in zip(

    predictor_fatalities.predict(training_data_fatalities),

    predictor_fatalities_all.predict(training_data_fatalities_all)

):

    # offset by 1 because last training date is March 24, want to start predicting at March 26

    fatalities.append(np.concatenate((public_forecast.median, private_forecast.median[1:])))



# aggregate cases

cases = []

for public_forecast, private_forecast in zip(

    predictor_cases.predict(training_data_cases),

    predictor_cases_all.predict(training_data_cases_all)

):

    # offset by 1 because last training date is March 24, want to start predicting at March 26

    cases.append(np.concatenate((public_forecast.median, private_forecast.median[1:])))



# load test csv 

sub_df = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")



# fill 'NaN' Province/State values with Country/Region values

sub_df['Province/State'] = sub_df['Province/State'].fillna(sub_df['Country/Region'])



# get forecast ids

ids = []

for _, df in sub_df.groupby(by=['Province/State', 'Country/Region']):

    ids.append(df['ForecastId'].values)



# create submission df

submission = pd.DataFrame(

    list(zip(

        np.array(ids).flatten(),

        np.array(cases).flatten(),

        np.array(fatalities).flatten()

    )), 

    columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']

)

submission.to_csv('submission.csv', index=False)