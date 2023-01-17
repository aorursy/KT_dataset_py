import pandas as pd

import altair as alt



df = pd.read_csv("/kaggle/input/amex-nyse-nasdaq-stock-histories/full_history/GOOGL.csv")



base = alt.Chart(df).encode(

    alt.X(

        "date:T",

        axis=alt.Axis(

            title="Year",

            format="%Y",

            labelAngle=-45

        )

    )

).properties(

    title="History of Google stock value",

    width=600

)



historic = base.mark_line().encode(

    alt.Y(

        "adjclose:Q",

        axis=alt.Axis(

            title="Close price (adjusted)",

            format="$.0f"

        )

    ),

)



historic
import datetime as dt



def date_str2ord(date_str):

    date = dt.date.fromisoformat(date_str)

    date_ord = date.toordinal()



    return date_ord



df["date_ord"] = df["date"].map(date_str2ord)

print(df[["date","date_ord"]])
from sklearn.model_selection import train_test_split



train, test = train_test_split(df, test_size=0.3)
from sklearn.linear_model import LinearRegression



X = train["date_ord"].values.reshape(-1, 1)

y = train["adjclose"]



model = LinearRegression().fit(X, y)

coef, intercept = model.coef_[0], model.intercept_

sign = "+" if intercept >= 0 else "-"

print(f"price = {round(coef, 3)} * date {sign} {round(abs(intercept))}")
df["adjclose_predict"] = model.predict(df["date_ord"].values.reshape(-1, 1))



predictions = base.mark_line(color="red").encode(

    alt.Y("adjclose_predict:Q")

)



historic.mark_line(opacity=0.3) + predictions
r2 = model.score(test["date_ord"].values.reshape(-1, 1), test["adjclose"])

print(f"r2: {round(r2, 3)}")
import numpy as np

from tabulate import tabulate



future_dates = ["2021-01-01", "2022-01-01", "2025-01-01", "2030-01-01", "2050-01-01"]



X = np.array([date_str2ord(date_str) for date_str in future_dates]).reshape(-1, 1)

future_prices = model.predict(X)



print(tabulate(zip([date_str[:4] for date_str in future_dates], [f"${round(price, 2)}" for price in future_prices]), headers=["Year", "Price"]))