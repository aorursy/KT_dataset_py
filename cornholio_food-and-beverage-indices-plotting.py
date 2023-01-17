import matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

matplotlib.style.use('ggplot')
ITEMS_FILE = '../input/cu.item.csv' # contains type of indices and description

FNB_FILE = '../input/cu.data.11.USFoodBeverage.csv'

PERIODS_FILE = '../input/cu.period.csv' # need to extract data by month
periods = pd.read_csv(PERIODS_FILE, index_col='period')

items = pd.read_csv(ITEMS_FILE)

fnb = pd.read_csv(FNB_FILE)



fnb.drop('footnote_codes', axis=1, inplace=True)

fnb.series_id = [i[8:] for i in fnb.series_id] # no there are only codes
items.head()
fnb.head()
c = np.array([[c, items.loc[items['item_code'] == c]['item_name'].values[0]] for c in fnb.series_id.unique()])

c = pd.DataFrame(c[:, 1], columns=['Stands for'], index=c[:, 0])

c.head() # indices that is in Food'n'Beverage part
# merge year and period to use it as x on plots

fnb_m = fnb[fnb['period'].isin(periods.index[:12])].copy()

fnb_m.period = fnb_m.period.str.split('M').str[1]

fnb_m.year = fnb_m.year.astype(np.str)

fnb_m.year = fnb_m.year + '-' + fnb_m.period

fnb_m.drop('period', axis=1, inplace=True)

fnb_m.year = pd.to_datetime(fnb_m.year)



print(fnb_m.shape)

fnb_m.head()
def show_by_month(indices, figsize=(10, 8)):

    ax = None

    for i in indices:

        sdf = fnb_m.loc[fnb_m['series_id'] == i]

        sdf.columns = ['series_id', 'year', i]

        if not ax:

            ax = sdf.plot(x='year', y=i, figsize=figsize)

        else:

            sdf.plot(x='year', y=i, ax=ax, figsize=figsize)

    return ax





show_by_month(['SAF1', 'SAF11'])