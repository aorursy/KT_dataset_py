import pandas as pd

import numpy as np

from IPython.display import display, HTML
data = pd.read_csv("../input/fifa19/data.csv")

pd.set_option('display.max_columns', len(data.columns))

data
ratings = data[["Nationality", "Overall"]]

ratings.groupby(["Nationality"]).mean().sort_values(by="Overall", ascending=False).head(10)
ratings = data[["Nationality", "Overall"]]

ratings = ratings.groupby(["Nationality"]).agg(['mean', 'count'])

ratings.columns=["Overall", "Count"]

ratings[ratings.Count>100].sort_values(by="Overall", ascending=False).head(10)
atk = ['ST', 'CF', 'LW', 'RW', "RF", "LF", "RS", "LS", "LAM", "RAM"]

mid = ["CAM", "CM", "LM", "RM", "CDM", "RCM", "LCM", "LDM", "RDM"]

defe = ["CB", "LB", "RB", "LWB", "RWB", "RCB", "LCB", ]

gk = ['GK']

nan = ["nan"]



#Display dataframes side by side

#Credits https://stackoverflow.com/a/44923103



from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
atks = data[data.Position.isin(atk)]

atks = atks[["Nationality", "Overall"]].groupby(["Nationality"]).agg(['mean', 'count'])

atks.columns=["Overall ATK", "Count"]

atks = atks[atks.Count>25].sort_values(by="Overall ATK", ascending=False)



mids = data[data.Position.isin(mid)]

mids = mids[["Nationality", "Overall"]].groupby(["Nationality"]).agg(['mean', 'count'])

mids.columns=["Overall MID", "Count"]

mids = mids[mids.Count>25].sort_values(by="Overall MID", ascending=False)



defes = data[data.Position.isin(defe)]

defes = defes[["Nationality", "Overall"]].groupby(["Nationality"]).agg(['mean', 'count'])

defes.columns=["Overall DEF", "Count"]

defes = defes[defes.Count>25].sort_values(by="Overall DEF", ascending=False)



gks = data[data.Position.isin(gk)]

gks = gks[["Nationality", "Overall"]].groupby(["Nationality"]).agg(['mean', 'count'])

gks.columns=["Overall GK", "Count"]

gks = gks[gks.Count>25].sort_values(by="Overall GK", ascending=False)



display_side_by_side(gks.head(10), defes.head(10), mids.head(10), atks.head(10))