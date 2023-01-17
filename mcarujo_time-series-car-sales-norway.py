!pip install nb_black -q
%load_ext nb_black
import pandas as pd

import numpy as np



car_sales_monthly = pd.read_csv(

    "../input/newcarsalesnorway/norway_new_car_sales_by_month.csv"

)

print("Rows: {} and Columns: {}".format(*car_sales_monthly.shape))

print("Number of Nan {}".format(car_sales_monthly.isna().sum().sum()))

car_sales_monthly.fillna(0, inplace=True)

car_sales_monthly.head()
car_sales_monthly["Year"] = car_sales_monthly["Year"].astype(str)

car_sales_monthly["Month"] = car_sales_monthly["Month"].astype(str)



car_sales_monthly["Date"] = car_sales_monthly["Year"].str.cat(

    car_sales_monthly["Month"].str.zfill(2), sep="-"

)

car_sales_monthly["Year"] = car_sales_monthly["Year"].astype(int)

car_sales_monthly["Month"] = car_sales_monthly["Month"].astype(int)

car_sales_monthly["Quantity"] = car_sales_monthly["Quantity"].astype(int)



car_sales_monthly.Date[:5]
import plotly.express as px



fig = px.bar(

    car_sales_monthly,

    x="Date",

    y="Quantity",

    hover_data=["Year", "Month", "Quantity"],

    color="Quantity",

    text="Quantity",

    height=600,

    width=2400,

)

fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

fig.show()
import seaborn as sns

import matplotlib.pyplot as plt



aux_csm = car_sales_monthly[["Month", "Year", "Quantity"]]

plt.figure(figsize=(25, 10))

sns.heatmap(

    aux_csm.pivot("Month", "Year", "Quantity").fillna(0).astype(int),

    annot=True,

    fmt="d",

    linewidths=0.5,

    cmap="Reds",

)
import plotly.graph_objects as go





def plot_var_comp(df, column, title, xlabel, ylabel):

    df["d" + column] = df[column].diff()

    df["d2" + column] = df["d" + column].diff()



    # Create traces

    x = df.Date

    y = df[column]  # y

    dy = df["d" + column]  # dy'

    d2y = df["d2" + column]  # dy''

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=column))

    fig.add_trace(go.Scatter(x=x, y=dy, mode="lines", name=f"Increase of {column}"))

    fig.add_trace(go.Scatter(x=x, y=d2y, mode="lines", name="Increase's Acceleration"))

    fig.update_layout(

        title=title, xaxis_title=xlabel, yaxis_title=ylabel,

    )

    fig.show()





plot_var_comp(

    car_sales_monthly,

    "Quantity",

    "Sales and your components",

    "Date",

    "Number of sales",

)
from pandas.plotting import autocorrelation_plot



plt.figure(figsize=(12, 5))

plt.title("Autocorrelation of Quantity")

ax = autocorrelation_plot(car_sales_monthly["Quantity"])
fig = px.box(

    car_sales_monthly,

    y="Quantity",

    facet_col="Year",

    color="Year",

    boxmode="overlay",

    points="all",

)



fig.show()
from statsmodels.tsa.seasonal import seasonal_decompose



res = seasonal_decompose(car_sales_monthly.Quantity, period=12)

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

fig = make_subplots(shared_xaxes=True,rows=4, cols=1)

x = car_sales_monthly.Date

y = res.observed

z =res.trend

k =res.seasonal

w =res.resid



# Ploting the lines

fig.append_trace(go.Scatter(

    x=x,

    y=y,

), row=1, col=1)



fig.append_trace(go.Scatter(

    x=x,

    y=z,

), row=2, col=1)



fig.append_trace(go.Scatter(

    x=x,

    y=k

), row=3, col=1)



fig.append_trace(go.Scatter(

    x=x,

    y=w

), row=4, col=1)



# Update properties and descriptions

fig.update_layout(height=700, width=1400, title_text="Statsmodel")

fig.update_xaxes(title_text="Date", row=4, col=1)

fig.update_yaxes(title_text="Observed", row=1, col=1)

fig.update_yaxes(title_text="Trend", row=2, col=1)

fig.update_yaxes(title_text="Seasonal", row=3, col=1)

fig.update_yaxes(title_text="Resid/Noise", row=4, col=1)

fig.show()

car_sales_monthly["sma_Quantity"] = car_sales_monthly.Quantity.rolling(10).mean()

plot_var_comp(

    car_sales_monthly,

    "sma_Quantity",

    "Sales and your components with SMA",

    "Date",

    "SMA to Number of sales",

)
from fbprophet import Prophet

from joblib import Parallel, delayed

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error





def format_to_prophet(serie_ds, serie_y):

    aux = pd.DataFrame()

    aux["ds"] = serie_ds

    aux["y"] = serie_y

    return aux





X = car_sales_monthly.Date

Y = car_sales_monthly.Quantity

prophet_data = format_to_prophet(X, Y)





def train_and_plot(cps, prophet_data, plot=False):

    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=cps)

    model.fit(prophet_data)



    future = model.make_future_dataframe(periods=12, freq="M")

    forecast = model.predict(future)



    r2 = round(r2_score(prophet_data["y"], forecast["yhat"][:-12]), 3)

    mse = round(mean_squared_error(prophet_data["y"], forecast["yhat"][:-12]), 3)

    mae = round(mean_absolute_error(prophet_data["y"], forecast["yhat"][:-12]), 3)



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

            go.Scatter(

                x=prophet_data["ds"],

                y=prophet_data["y"],

                mode="lines",

                name="Real Values",

            )

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



cps_options = [round(x, 1) for x in np.linspace(start=0.1, stop=10, num=100)]



results = Parallel(n_jobs=-1, verbose=10)(

    delayed(train_and_plot)(i, prophet_data) for i in cps_options

)
results = pd.DataFrame(results)

results = results[results.R2.isin([max(results.R2)])]

results = results[results.MSE.isin([min(results.MSE)])]

results
train_and_plot(results.CPS.iloc[0], prophet_data, True)