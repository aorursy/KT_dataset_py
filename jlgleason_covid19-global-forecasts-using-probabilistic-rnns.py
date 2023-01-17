# install gluonts package

!pip install gluonts

!pip install sentence_transformers
# load and clean data

import pandas as pd

import numpy as np



train_all = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

LOG_TRANSFORM = False



def preprocess(

    df: pd.DataFrame,

    log_transform: bool = LOG_TRANSFORM

):

    

    # set index

    df = df.set_index('Date')



    # fill 'NaN' Province/State values with Country/Region values

    df['Province_State'] = df['Province_State'].fillna(df['Country_Region'])



    # take difference of fatalities and cases

    df[['ConfirmedCases', 'Fatalities']] = df[['ConfirmedCases', 'Fatalities']].diff()

    df = df.fillna(0)

    

    df._get_numeric_data()[df._get_numeric_data() < 0] = 0

    assert df.isnull().sum().all() == 0

    

    # convert target values to log scale

    if log_transform:

        df[['ConfirmedCases', 'Fatalities']] = np.log1p(

            df[['ConfirmedCases', 'Fatalities']].values

    )

    

    return df



def split(

    df: pd.DataFrame, 

    date: str = '2020-03-19', 

):



    train = df.loc[df.index < date] 

    test = df.loc[df.index >= date]

    return train, test



train_all = preprocess(train_all)

train, test = split(train_all)
# plot confirmed cases and fatalities in train

import matplotlib.pyplot as plt

from gluonts.dataset.util import to_pandas

from gluonts.dataset.common import ListDataset



def plot_observations(

    target: str = 'ConfirmedCases',

    cumulative: bool = False,

    log_transform: bool = LOG_TRANSFORM

):

    fig = plt.figure(figsize=(15, 6.1), facecolor="white",  edgecolor='k')

    

    local_train = train.copy()

    local_test = test.copy()

    if log_transform:

        local_train[['ConfirmedCases', 'Fatalities']] = np.expm1(

            local_train[['ConfirmedCases', 'Fatalities']].values

        )

        local_test[['ConfirmedCases', 'Fatalities']] = np.expm1(

            local_test[['ConfirmedCases', 'Fatalities']].values

        )

    

    if cumulative:

        cum_train = local_train.groupby(['Province_State', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].cumsum()

        cum_train = cum_train.groupby('Date').sum()

        cum_test = local_test.groupby(['Province_State', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].cumsum()

        cum_test = cum_test.groupby('Date').sum() + cum_train.tail(1).values

    else:

        cum_train = local_train.groupby('Date').sum()

        cum_test = local_test.groupby('Date').sum()



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

    type_string = 'Cumulative' if cumulative else 'Daily'

    plt.title(f'{type_string} number of {target} globally', fontsize=16)

    plt.legend(fontsize=16)

    plt.grid(which="both")

    plt.show()

    

plot_observations('ConfirmedCases')

plot_observations('Fatalities')

plot_observations('ConfirmedCases', cumulative = True)

plot_observations('Fatalities', cumulative = True)
# generate BERT embeddings

from sentence_transformers import SentenceTransformer



places = []

for idx,row in train.iterrows():

    if row['Province_State']!=row['Country_Region']:

        places.append(row['Province_State']+", "+row["Country_Region"])

    else:

        places.append(row['Country_Region'])

places = np.unique(places)



model = SentenceTransformer('bert-base-nli-mean-tokens')

country_embeddings = model.encode(list(places))

embedding_dim = len(country_embeddings[0])



province_state = [p.split(',')[0] if p!="Korea, South" else p for p in places] #error catching

country_region = [p.split(',')[1][1:] if (len(p.split(','))==2 and p!='Korea, South') else p for p in places]

embed_df = pd.DataFrame(

    np.concatenate(

        [

            np.array(province_state).reshape(-1,1),

            np.array(country_region).reshape(-1,1),

            country_embeddings

        ]

        ,axis=1)

)

embed_df.columns=['Province_State','Country_Region']+list(range(embedding_dim))
from sklearn.preprocessing import OrdinalEncoder



def join_with_embeddings(

    df: pd.DataFrame,

    embed_df: pd.DataFrame

):

    

    # join, delete merge columns

    new_df = df.reset_index().merge(

        embed_df,

        left_on = ["Province_State",'Country_Region'],

        right_on = ["Province_State",'Country_Region'],

        how = 'left'

    ).set_index('Date')

    

    #make sure no NaN in dataframe

    assert new_df.isnull().sum().sum()==0

    return new_df



def encode(

    df: pd.DataFrame

):

    """ encode 'Province/State' and 'Country/Region' categorical variables as numerical ordinals"""

    

    enc = OrdinalEncoder()

    df[['Province_State', 'Country_Region']] = enc.fit_transform(

        df[['Province_State', 'Country_Region']].values

    )

    return df, enc



join_df = join_with_embeddings(train_all, embed_df)

all_df, enc = encode(join_df)

train_df, test_df = split(all_df)

_, val_df = split(all_df, date = '2020-03-06')
from gluonts.dataset.common import ListDataset

from gluonts.dataset.field_names import FieldName

import typing



REAL_VARS = list(range(embedding_dim))



def build_dataset(

    frame: pd.DataFrame,

    target: str = 'Fatalities',

    cat_vars: typing.List[str] = ['Province_State', 'Country_Region'],

    real_vars: typing.List[int] = REAL_VARS

):

    return ListDataset(

        [

            {

                FieldName.START: df.index[0], 

                FieldName.TARGET: df[target].values,

                FieldName.FEAT_STATIC_CAT: df[cat_vars].values[0],

                FieldName.FEAT_STATIC_REAL: df[real_vars].values[0]

            }

            for g, df in frame.groupby(by=['Province_State', 'Country_Region'])

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

mx.random.seed(0)

np.random.seed(0)



def fit(

    training_data: ListDataset,

    validation_data: ListDataset = None,

    pred_length: int = 13,

    epochs: int = 30,

    weight_decay: float = 5e-5,

    log_preds: bool = LOG_TRANSFORM,

):

    estimator = DeepAREstimator(

        freq="D", 

        prediction_length=pred_length,

        context_length=pred_length//2,

        use_feat_static_cat = True,

        use_feat_static_real = True,

        cardinality = [train['Province_State'].nunique(), train['Country_Region'].nunique()],

        distr_output=NegativeBinomialOutput(),

        trainer=Trainer(

            epochs=epochs,

            learning_rate=0.005, 

            batch_size=64,

            weight_decay=weight_decay,

            learning_rate_decay_factor=0.1,

            patience=15

        ),

    )

    _, trained_net, predictor = estimator.train_model(

        training_data = training_data, 

        validation_data = validation_data

    )

    

    return predictor, trained_net



predictor_fatalities, net = fit(training_data_fatalities, val_data_fatalities)

predictor_cases, case_net = fit(training_data_cases, val_data_cases)

predictor_fatalities_all, all_net = fit(training_data_fatalities_all, pred_length=30, epochs=60)

predictor_cases_all, all_case_net = fit(training_data_cases_all, pred_length=30, epochs=60)
from gluonts.dataset.util import to_pandas

import matplotlib.pyplot as plt

from typing import List



## make it run sorted_samples code again!

def plot_forecast(

    predictor,

    train_df: pd.DataFrame,

    location: List[str] = ['Italy', 'Italy'],

    target: str = 'Fatalities',

    cat_vars: typing.List[str] = ['Province_State', 'Country_Region'],

    real_vars: typing.List[int] = REAL_VARS,

    cumulative: bool = True,

    log_preds: bool = LOG_TRANSFORM,

    show_gt: bool = True,

    start_offset: int = 0, 

    fontsize: int = 16,

    save: bool = False

):

    fig = plt.figure(figsize=(15, 6.1), facecolor="white",  edgecolor='k')



    # plot train observations, true observations from public test set, and forecasts

    location_tr = enc.transform(np.array(location).reshape(1,-1))

    tr_df = train_df[np.all((train_df[['Province_State', 'Country_Region']].values == location_tr), axis=1)]



    train_obs = ListDataset(

        [{

            FieldName.START: tr_df.index[0], 

            FieldName.TARGET: tr_df[target].values,

            FieldName.FEAT_STATIC_REAL: tr_df[real_vars].values[0],

            FieldName.FEAT_STATIC_CAT: tr_df[cat_vars].values[0],

        }],

        freq = "D",

    )

    te_df = test_df[np.all((test_df[['Province_State', 'Country_Region']].values == location_tr), axis=1)]



    test_gt = ListDataset(

        [{"start": te_df.index[0], "target": te_df[target].values}],

        freq = "D",

    )



    for train_series, gt, forecast in zip(train_obs, test_gt, predictor.predict(train_obs)):

        

        train_series = to_pandas(train_series)

        gt = to_pandas(gt)

        

        if start_offset:

            train_series = train_series[start_offset:]

        

        # connect train series visually (either to GT or to forecast)

        if show_gt:

            train_series[train_series.index[-1] + pd.DateOffset(1)] = gt.iloc[0]

        else:

            train_series[train_series.index[-1] + pd.DateOffset(1)] = forecast.median[:1][0]

            

        # log and/or cumulative transforms

        if log_preds:

            train_series = np.expm1(train_series)

            gt = np.expm1(gt)

            forecast.samples = np.expm1(forecast.samples) 

            forecast._sorted_samples_value = None

        if cumulative:

            train_series = train_series.cumsum()

            gt = gt.cumsum() + train_series.iloc[-2]

            forecast.samples = np.cumsum(forecast.samples, axis=1) + train_series.iloc[-2]

            forecast._sorted_samples_value = None



        # plot

        train_series.plot(linewidth=2, label = 'train series')

        if show_gt:

            gt.plot(linewidth=2, label = 'test ground truth')

            

        # plot layout

        type_string = 'Cumulative' if cumulative else 'Daily'

        plt.title(

            f'{len(forecast.median)} day forecast: {type_string} number of {target} in {location[0]}', 

            fontsize=fontsize

        )

        plt.legend(fontsize = fontsize)

        plt.grid(which='both')

        if save:

            forecast.plot(

                color='g', 

                prediction_intervals=[50.0, 90.0], 

                show_mean = True,

                output_file = f'{len(forecast.median)} day forecast {type_string} number of {target} in {location[0]}'

            )

        else:

            forecast.plot(color='g', prediction_intervals=[50.0, 90.0], show_mean = True)

        forecast._sorted_samples_value = None

        plt.show()

# plot public leaderboard case forecasts

plot_forecast(predictor_cases, train_df, ['Italy', 'Italy'], target='ConfirmedCases')

plot_forecast(predictor_cases, train_df, ['Iran', 'Iran'], target='ConfirmedCases')

plot_forecast(predictor_cases, train_df, ['Spain', 'Spain'], target='ConfirmedCases')

plot_forecast(predictor_cases, train_df, ['Korea, South', 'Korea, South'], target='ConfirmedCases')

plot_forecast(predictor_cases, train_df, ['New York', 'US'], target='ConfirmedCases')
# plot private leaderboard case forecasts

plot_forecast(predictor_cases_all, all_df, ['Italy', 'Italy'], show_gt = False, target='ConfirmedCases')

plot_forecast(predictor_cases_all, all_df, ['Iran', 'Iran'], show_gt = False, target='ConfirmedCases')

plot_forecast(predictor_cases_all, all_df, ['New York', 'US'], show_gt = False, target='ConfirmedCases')

plot_forecast(predictor_cases_all, all_df, ['Hubei', 'China'], show_gt = False, target='ConfirmedCases')

plot_forecast(predictor_cases_all, all_df, ['Korea, South', 'Korea, South'], show_gt = False, target='ConfirmedCases')

from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.model.forecast import Forecast

from gluonts.gluonts_tqdm import tqdm

from gluonts.dataset.util import to_pandas

import json

from typing import Dict, Union, Tuple, List



# copied from https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py

def extract_pred_target(

    time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast

) -> np.ndarray:

    

    assert forecast.index.intersection(time_series.index).equals(

        forecast.index

    ), (

        "Cannot extract prediction target since the index of forecast is outside the index of target\n"

        f"Index of forecast: {forecast.index}\n Index of target: {time_series.index}"

    )



    # cut the time series using the dates of the forecast object

    return np.atleast_1d(

        np.squeeze(time_series.loc[forecast.index].transpose())

    )



def msle(target, forecast):

    return np.mean(np.square(np.log1p(forecast) - np.log1p(target)))



# bootstrapped and edited from https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py

def get_metrics_per_ts(

    time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast, keys: List[str]

) -> Dict[str, Union[float, str, None]]:

    pred_target = np.array(extract_pred_target(time_series, forecast))

    

    try:

        mean_fcst = forecast.mean

    except:

        mean_fcst = None

    median_fcst = forecast.quantile(0.5)



    metrics = {

        "Province_State": keys[0],

        "Country_Region": keys[1],

        "MSLE_on_mean": msle(pred_target, mean_fcst)

        if mean_fcst is not None

        else None,

        "MSLE_on_median": msle(pred_target, median_fcst)

    }



    return metrics



# bootstrapped and edited from https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py

def get_aggregate_metrics(

    metric_per_ts: pd.DataFrame

) -> Tuple[Dict[str, float], pd.DataFrame]:

    agg_funs = {

        "MSLE_on_mean": "mean",

        "MSLE_on_median": "mean",

    }



    assert (

        set(metric_per_ts.columns) >= agg_funs.keys()

    ), "The some of the requested item metrics are missing."



    totals = {

        key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()

    }

    totals["RMSLE_on_mean"] = np.sqrt(totals["MSLE_on_mean"])

    totals["RMSLE_on_median"] = np.sqrt(totals["MSLE_on_median"])



    return totals, metric_per_ts



def evaluate(

    data_df: pd.DataFrame, 

    predictor_fatalities,

    predictor_cases,

    num_samples: int = 100,

    log_preds: bool = LOG_TRANSFORM,

):

    

    all_data_fat = build_dataset(all_df)

    all_data_case = build_dataset(all_df, target = 'ConfirmedCases')



    rows = []

    with tqdm(

        zip(training_data_fatalities, all_data_fat, predictor_fatalities.predict(training_data_fatalities)),

        total=len(training_data_fatalities),

        desc="Evaluating Fatalities Predictor",

    ) as it, np.errstate(invalid="ignore"):

        for train, ts, f in it:

            

            keys = enc.inverse_transform(ts['feat_static_cat'].reshape(1,-1))[0]

            train = to_pandas(train)

            ts = to_pandas(ts)

            

            # undo log

            if log_preds:

                train = np.expm1(train)

                ts = np.expm1(ts)

                f.samples = np.expm1(f.samples) 

                

            f.samples = np.cumsum(f.samples, axis=1) + train.cumsum().iloc[-1]

            rows.append(get_metrics_per_ts(ts.cumsum(), f, keys))

            

    metrics_per_ts_fat = pd.DataFrame(rows, dtype=np.float64)

    agg_metrics, metrics_per_ts = get_aggregate_metrics(metrics_per_ts_fat)

    print(json.dumps(agg_metrics, indent=4))



    rows = []

    with tqdm(

        zip(training_data_cases, all_data_case, predictor_cases.predict(training_data_cases)),

        total=len(all_data_case),

        desc="Evaluating Cases Predictor",

    ) as it, np.errstate(invalid="ignore"):

        for train, ts, f in it:

            

            keys = enc.inverse_transform(ts['feat_static_cat'].reshape(1,-1))[0]

            train = to_pandas(train)

            ts = to_pandas(ts)

            

            # undo log

            if log_preds:

                train = np.expm1(train)

                ts = np.expm1(ts)

                f.samples = np.expm1(f.samples) 

                 

            f.samples = np.cumsum(f.samples, axis=1) + train.cumsum().iloc[-1]

            rows.append(get_metrics_per_ts(ts.cumsum(), f, keys))

            

    metrics_per_ts_case = pd.DataFrame(rows, dtype=np.float64)

    agg_metrics, metrics_per_ts = get_aggregate_metrics(metrics_per_ts_case)

    print(json.dumps(agg_metrics, indent=4))

    

    return metrics_per_ts_fat, metrics_per_ts_case

    

metrics_per_ts_fat, metrics_per_ts_case = evaluate(all_df, predictor_fatalities, predictor_cases)
print('Best fit ConfirmedCases series (mean prediction, on validation set):\n')

print(metrics_per_ts_case.sort_values(by='MSLE_on_mean').head(10))

print('\nWorst fit ConfirmedCases series (mean prediction, on validation set):\n')

print(metrics_per_ts_case.sort_values(by='MSLE_on_mean', ascending=False).head(10))
from sklearn import manifold

from matplotlib.ticker import NullFormatter



# get list of countries with most fatalities

def visualize_embedding(

    trained_net,

    province_state = True,

    sort_key: str = 'Fatalities',

    top_n: int = 15

):

    group_key = 'Province_State' if province_state else 'Country_Region'

    countries = train_all.groupby(group_key).sum().sort_values(

        by=sort_key, 

        ascending = False

    ).head(top_n).index.tolist()



    # visualize 2-D projection of learned Country/Region embedding space with TSNE

    e_key = 0 if province_state else 1

    embedding = trained_net.collect_params()[f'{trained_net.name}_featureembedder0_cat_{e_key}_embedding_weight'].data()

    proj = manifold.TSNE(init='pca', random_state = 0).fit_transform(embedding.asnumpy())



    # plot

    fig = plt.figure(figsize=(15, 6.1), facecolor="white",  edgecolor='k')

    ax = plt.gca()

    for country in countries:

        idx = np.where(enc.categories_[e_key] == country)[0][0]

        plt.scatter(

            proj[idx, 0], 

            proj[idx, 1], 

            cmap=plt.cm.Spectral, 

            label = country

        )

        ax.annotate(

            country, 

            (proj[idx, 0], proj[idx, 1]), 

            fontsize=16,

        )

    ax.xaxis.set_major_formatter(NullFormatter())

    ax.yaxis.set_major_formatter(NullFormatter())

    title_string = 'Country/Region' if not province_state else 'Province/State'

    plt.title(f"TSNE Visualization of Learned {group_key} Embedding Space", fontsize=16)

    plt.show()



visualize_embedding(all_case_net, sort_key = 'ConfirmedCases')

visualize_embedding(all_case_net, sort_key = 'ConfirmedCases', province_state=False)
# generate submission csv



def aggregate(

    all_data: ListDataset,

    train_data: ListDataset, 

    train_data_all: ListDataset, 

    predictor,

    predictor_all,

    log_preds: bool = LOG_TRANSFORM,

    mean: bool = False,

):

    

    aggregates = []

    for train, train_all, public_forecast, private_forecast in zip(

        train_data,

        train_data_all,

        predictor.predict(train_data),

        predictor_all.predict(train_data_all)

    ):

        

        train = to_pandas(train)

        train_all = to_pandas(train_all)

        

        # undo log

        if log_preds:

            train = np.expm1(train)

            train_all = np.expm1(train_all)

            public_forecast.samples = np.expm1(public_forecast.samples) 

            private_forecast.samples = np.expm1(private_forecast.samples) 

            

        # accumulate

        ts = train.cumsum()

        ts_all = train_all.cumsum()

        public_forecast.samples = np.cumsum(public_forecast.samples, axis=1) + ts.iloc[-1]

        private_forecast.samples = np.cumsum(private_forecast.samples, axis=1) + ts_all.iloc[-1]

    

        # concatenate

        public_f = public_forecast.mean if mean else public_forecast.median

        private_f = private_forecast.mean if mean else private_forecast.median

        aggregates.append(np.concatenate((public_f, private_f)))  

    

    return aggregates



def submit(

    filename: str,

    mean: bool = False,

):

    

    # aggregate fatalities

    fatalities = aggregate(

        build_dataset(all_df), 

        training_data_fatalities,

        training_data_fatalities_all,

        predictor_fatalities,

        predictor_fatalities_all,

        mean = mean

    )



    # aggregate cases

    cases = aggregate(

        build_dataset(all_df, target = 'ConfirmedCases'), 

        training_data_cases,

        training_data_cases_all,

        predictor_cases,

        predictor_cases_all,

        mean = mean

    )



    # load test csv 

    sub_df = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")



    # fill 'NaN' Province/State values with Country/Region values

    sub_df['Province_State'] = sub_df['Province_State'].fillna(sub_df['Country_Region'])



    # get forecast ids

    ids = []

    for _, df in sub_df.groupby(by=['Province_State', 'Country_Region']):

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

    submission.to_csv(filename, index=False)



submission = submit('submission.csv', mean = True)