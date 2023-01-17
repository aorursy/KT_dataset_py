import scipy.stats as st

import numpy as np



def wald_confidence_interval(X, c):

    n = X.shape[0]

    

    p_hat = X.astype(int).sum() / n

    z_score = st.norm.ppf(1 - ((1 - c) / 2))



    additive_part = z_score * np.sqrt(p_hat * (1 - p_hat) / n)

    

    return (p_hat - additive_part, p_hat + additive_part)
import pandas as pd

sales = pd.read_csv("../input/Iowa_Liquor_Sales.csv")
_sales = (sales

     .head(1000000)

     .assign(n=0)

     .groupby('Date')

     .count()

     .n

     .to_frame()

     .reset_index()

     .pipe(lambda df: df.assign(Date=pd.to_datetime(df.Date)))

     .pipe(lambda df: df.assign(Day=df.Date.dt.dayofyear))

     .groupby('Day')

     .mean()

)



christmas_day_sales = _sales.loc[359, 'n']

all_sales = _sales.n.round().sum()
is_christmas_sale = np.array([True]*int(christmas_day_sales) + [False]*(1000000 - int(christmas_day_sales)))
wald_confidence_interval(is_christmas_sale, 0.95)
1/365