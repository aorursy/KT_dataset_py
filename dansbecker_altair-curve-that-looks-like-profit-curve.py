import altair as alt
import pandas as pd
import numpy as np

data = pd.DataFrame({'x': np.linspace(0,1,100)})
data['y'] = (data.x **.5 - data.x * 1.1) * 1000000

alt.themes.enable('fivethirtyeight')
alt.Chart(data).mark_line().encode(
    alt.X('x', title='Decision Threshold'),
    alt.Y('y', title='Profit')
)