!pip install nb_black -q
%load_ext nb_black
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
avocado_price = pd.read_csv("../input/avocado-prices/avocado.csv")

avocado_price.drop(

    [

        "Unnamed: 0",

        "Total Volume",

        "4046",

        "4225",

        "4770",

        "Total Bags",

        "Small Bags",

        "Large Bags",

        "XLarge Bags",

        "year",

    ],

    inplace=True,

    axis=1,

)



avocado_price.Date = pd.to_datetime(avocado_price.Date)

avocado_price = avocado_price[avocado_price.type.isin(["conventional"])]

avocado_price = avocado_price.groupby(["Date"]).mean()

avocado_price["Date"] = avocado_price.index

avocado_price.reset_index(inplace=True, drop=True)

avocado_price.columns=['Price', 'Date']

avocado_price.head()

price_oil = pd.read_csv(

    "../input/brent-oil-prices/BrentOilPrices.csv", dtype={"Price": float}

)

print("price_oil shape ->", price_oil.shape)

price_oil.head()
def transform_dataset(df):

    df.Date = pd.to_datetime(df.Date)

    df.sort_values("Date", inplace=True)

    df.reset_index(inplace=True, drop=True)

    return df





price_oil = transform_dataset(price_oil)

price_avd = transform_dataset(avocado_price)
import plotly.graph_objects as go

import plotly.express as px





def analysi_basic_statistical(data, title):

    fig = go.Figure(

        data=[

            go.Table(

                header=dict(values=["Parameter", "Price ($)"], font=dict(size=20)),

                cells=dict(

                    values=[

                        list(data.describe().index),

                        list(data.describe().round(2)["Price"]),

                    ],

                    align="left",

                    height=30,

                    font=dict(size=15),

                ),

            )

        ]

    )

    fig.update_layout(

        width=600, showlegend=False, title_text=title,

    )

    fig.show()

    return None





analysi_basic_statistical(price_oil, "Statistical information for Oil")
analysi_basic_statistical(price_avd, "Statistical information for Avocado")
import plotly.graph_objects as go

import plotly.express as px





def analysi_historical(df, title):

    fig = px.line(df, x="Date", y="Price")

    fig.update_xaxes(

        rangeslider_visible=True,

        rangeselector=dict(

            buttons=list(

                [

                    dict(count=1, label="1 month", step="month", stepmode="backward"),

                    dict(count=3, label="3 months", step="month", stepmode="backward"),

                    dict(count=6, label="6 months", step="month", stepmode="backward"),

                    dict(count=1, label="1 year", step="year", stepmode="backward"),

                    dict(count=2, label="2 years", step="year", stepmode="backward"),

                    dict(count=4, label="4 years", step="year", stepmode="backward"),

                    dict(step="all"),

                ]

            )

        ),

    )

    fig.update_layout(title_text=title, title_font_size=20)

    fig.show()

    return None





analysi_historical(price_oil, "Historical oil price in american dollars ($)")
analysi_historical(price_avd, "Historical avocado in american dollars ($)")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib import pyplot as plt





def plot_correlations(serie):

    fig, ax = plt.subplots(figsize=(20, 5))

    _ = plot_acf(serie, ax=ax)

    fig, ax = plt.subplots(figsize=(20, 5))

    _ = plot_pacf(serie, ax=ax)
plot_correlations(price_oil["Price"])
plot_correlations(price_avd["Price"])
from statsmodels.tsa.seasonal import seasonal_decompose





def seasonal_decompose_and_graph(

    data,

    seasonal_type="additive",

    period=52,

    date_column="Date",

    value_column="Price",

    title="Seasonal analysis",

):

    res = seasonal_decompose(data[value_column], model=seasonal_type, period=period)

    df = {

        "observed": res.observed,

        "trend": res.trend,

        "seasonal": res.seasonal,

        "resid": res.resid,

    }

    res = pd.DataFrame(df)

    res.head()



    from plotly.subplots import make_subplots

    import plotly.graph_objects as go



    # Defining variables

    fig = make_subplots(shared_xaxes=True, rows=4, cols=1)

    x = data[date_column]

    y = res.observed

    z = res.trend

    k = res.seasonal

    w = res.resid



    # Ploting the lines

    fig.append_trace(go.Scatter(x=x, y=y,), row=1, col=1)

    fig.append_trace(go.Scatter(x=x, y=z,), row=2, col=1)

    fig.append_trace(go.Scatter(x=x, y=k), row=3, col=1)

    fig.append_trace(go.Scatter(x=x, y=w), row=4, col=1)



    # Update properties and descriptions

    fig.update_layout(height=700, width=1400, title_text=title)

    fig.update_xaxes(title_text="Date", row=4, col=1)

    fig.update_yaxes(title_text="Observed", row=1, col=1)

    fig.update_yaxes(title_text="Trend", row=2, col=1)

    fig.update_yaxes(title_text="Seasonal", row=3, col=1)

    fig.update_yaxes(title_text="Resid/Noise", row=4, col=1)

    fig.show()





seasonal_decompose_and_graph(

    price_oil, period=5, title="Seasonal decomposition for oil price."

)
seasonal_decompose_and_graph(

    price_avd, period=52, title="Seasonal decomposition for avocado price."

)
import plotly.graph_objects as go





def moving_averange(

    data, delay=51, date_column="Date", value_column="Price", title="Price"

):

    # Create figure

    fig = go.Figure()



    x = data[date_column]



    # Add traces, one for each slider step

    for step in np.arange(1, delay, 1):

        y = data[value_column].rolling(step).mean().values

        fig.add_trace(

            go.Scatter(

                visible=False,

                line=dict(color="#3C5074", width=1),

                name="WS=" + str(step),

                x=x,

                y=y,

            )

        )



    # Make 10th trace visible

    fig.data[5].visible = True



    # Create and add slider

    steps = []

    for i in range(len(fig.data)):

        step = dict(

            method="update",

            args=[

                {"visible": [False] * len(fig.data)},

                {

                    "title": "Moving average with window size: "

                    + str(i)

                    + ", for "

                    + title

                },

            ],  # layout attribute

        )

        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"

        steps.append(step)



    sliders = [

        dict(

            active=10,

            currentvalue={"prefix": "Window Size : "},

            pad={"t": 111},

            steps=steps,

        )

    ]



    fig.update_layout(sliders=sliders)



    fig.show()





moving_averange(price_oil, title="oil price")
moving_averange(price_avd, delay=7, title="avocado price")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fbprophet import Prophet





def format_to_prophet(serie_ds, serie_y):

    aux = pd.DataFrame()

    aux["ds"] = serie_ds

    aux["y"] = serie_y

    return aux





def train_predict(

    data, periods, kind, freq="W", plot=False, yearly_seasonality=False, cps=1

):

    model = Prophet(yearly_seasonality=yearly_seasonality, changepoint_prior_scale=cps)

    model.fit(data[:-periods])



    future = model.make_future_dataframe(

        periods=periods, freq=freq, include_history=True

    )

    forecast = model.predict(future)



    r2 = round(r2_score(data["y"], forecast["yhat"]), 3)

    mse = round(mean_squared_error(data["y"], forecast["yhat"]), 3)

    mae = round(mean_absolute_error(data["y"], forecast["yhat"]), 3)



    if plot:

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=forecast["ds"],

                y=forecast["yhat"],

                mode="lines",

                name="Predict Values",

            )

        )

        fig.add_trace(

            go.Scatter(

                x=forecast["ds"], y=forecast["trend"], mode="lines", name="Trend"

            )

        )

        fig.add_trace(

            go.Scatter(x=data["ds"], y=data["y"], mode="lines", name="Real Values",)

        )

        fig.update_layout(

            title_text=f"Comperating the real x predicted for car sales",

            yaxis_title=f"Sales",

            xaxis_title="Date",

        )



        fig.show()

        print("R2: ", r2)

        print("MSE: ", mse)

        print("MAE: ", mae)

    else:

        return {"CPS": cps, "R2": r2, "MSE": mse, "MAE": mae}





from joblib import Parallel, delayed



cps_options = [round(x, 1) for x in np.linspace(start=0.1, stop=10, num=50)]



prediction_size = 50

data_fb = format_to_prophet(price_oil.Date, price_oil.Price)



results = Parallel(n_jobs=-1, verbose=10)(

    delayed(train_predict)(

        data=data_fb,

        periods=prediction_size,

        freq="D",

        kind="Oil",

        plot=False,

        cps=i,

        yearly_seasonality=True,

    )

    for i in cps_options

)



results = pd.DataFrame(results)

results = results[results.R2.isin([max(results.R2)])]

results = results[results.MSE.isin([min(results.MSE)])]

results
forecast = train_predict(

    data=data_fb,

    periods=prediction_size,

    freq="D",

    kind="Oil",

    plot=True,

    cps=results.CPS.iloc[0],

    yearly_seasonality=True,

)
prediction_size = 10

data_fb = format_to_prophet(price_avd.Date, price_avd.Price)



results = Parallel(n_jobs=-1, verbose=10)(

    delayed(train_predict)(

        data=data_fb,

        periods=prediction_size,

        freq="W",

        kind="Oil",

        plot=False,

        cps=i,

        yearly_seasonality=True,

    )

    for i in cps_options

)



results = pd.DataFrame(results)

results = results[results.R2.isin([max(results.R2)])]

results = results[results.MSE.isin([min(results.MSE)])]

results
forecast = train_predict(

    data=data_fb,

    periods=prediction_size,

    freq="W",

    kind="Avocado",

    plot=True,

    cps=results.CPS.iloc[0],

    yearly_seasonality=True,

)