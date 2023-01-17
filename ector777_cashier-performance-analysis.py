import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

%matplotlib inline
data = pd.read_csv('/kaggle/input/retail-sales/Sales.csv')

data.head()
data.shape
data['Cajero'].value_counts()
data.dtypes
shop = pd.DataFrame(data['Cajero']).astype('category')

shop.rename(columns={'Cajero': 'Cashier'}, inplace=True)

shop['Time'] = pd.to_datetime(data['Hora'])

shop['Total'] = data['Total']

shop['Change'] = data['Pago'] - data['Total']

shop.head()
shop.dtypes
shop.shape
def holiday_code(date):

    if date.month == 5 and date.day == 1:

        return 2

    if date.weekday() >= 5:

        return 1

    return 0



def add_duration_holidays(threshold_mins):

    cashier_last = {} # Time of the last check for every cashier 

    cashier_first = {} # Time of the first check for every cashier without breaks 

    duration_col = [] # Duration

    holiday_col = []  # Holiday

    nobreaktime_col = [] # NoBreakTime



    for index, row in shop.iterrows():

        cashier = row.Cashier

        time = row.Time

        holiday_col.append(holiday_code(time))

        time_last = cashier_last.get(cashier, None)

        cashier_last[cashier] = time  

        if time_last is None:  # Nothing to compare with

            duration_col.append(np.NaN)

            cashier_first[cashier] = None  # The non-break work sequence have broken 

        else:

            time_diff = time - time_last

            if time_diff.seconds > threshold_mins*60:  # Processing can't take that much time - it was a break or idle time

                duration_col.append(np.NaN)

                cashier_first[cashier] = None  # Sequence was broken

            else:

                duration_col.append(time_diff.seconds)

        time_first = cashier_first.get(cashier, None)

        if time_first is None:  # Nothing to compare with

            nobreaktime_col.append(np.NaN)

            cashier_first[cashier] = time

        else:

            time_diff = time - time_first

            if time_first.day != time.day:

                nobreaktime_col.append(np.NaN)

                cashier_first[cashier] = None  # The non-break work sequence have broken

            else:

                nobreaktime_col.append(time_diff.seconds/60)

    shop['Duration'] = duration_col

    shop['Holiday'] = holiday_col

    shop['NoBreakTime'] = nobreaktime_col



add_duration_holidays(30)
shop[shop['Duration'].between(1750, 1801)].head()
add_duration_holidays(20)

shop[shop['Duration'].between(1150, 1201)].head()
add_duration_holidays(30)

shop.nlargest(20, 'Total')
add_duration_holidays(120)

shop.nlargest(20, 'Total')
shop['Total'][shop['Total'].between(0, 2000)].hist(bins=30);
shop[shop['Total'] >= 1000.0].count()
shop[(shop['Total'] < 1000.0) & (shop['Duration'] > 900) & (shop['Total']*7 < shop['Duration'])]
shop = shop[shop['Total']>0]  # Remove negative Totals

shop['Duration'].mask(shop['Total'] == 4980.15, 1800, inplace=True)  # Fix the Jaqueline's check

shop.loc[66804]
# Clean the garbage

garbage = (shop['Total'] < 1000.0) & (shop['Duration'] > 900) & (shop['Total']*7 < shop['Duration'])

shop['Duration'].mask(garbage, inplace=True)
shopt = shop.set_index('Time')

shopt.head()
shopt=shopt.groupby(['Cashier']).resample('10Min').sum()

shopt.head()
shopt=shopt.unstack(level='Cashier')['Total']

shopt.head()
shopt.mask(lambda x: x==0.0, inplace=True)

grand=shopt.sum(axis=1)

cash_num = shopt.agg('notnull').sum(axis=1)

shopt.columns = shopt.columns.add_categories(['Grand Total', 'Cashiers Num'])

shopt['Grand Total'] = grand

shopt['Cashiers Num'] = cash_num

shopt.head()
import plotly.graph_objects as go
# I don't know why this doesn't show anything in Kaggle, it works in Jupyter notebook

fig = go.Figure()

for name in shopt.columns:

    if name != 'Cashiers Num' and name != 'Grand Total':

        fig.add_trace(go.Scatter(x=shopt.index, y=shopt[name], name=name))



fig.update_layout(title_text="Supermarket sales with a Rangeslider", xaxis_rangeslider_visible=True)

fig.update_yaxes(range=[0, 7500])

fig.show()
sns.set(rc={'figure.figsize':(18, 10)})

shopt.iloc[:,:-2]['2018-05-01'].plot();   # Substitute 2018-05-01 with your date here
shopt['Cashiers Num'].max()
shopt[shopt['Cashiers Num'] > 4].count()
def get_ratings(data, cashier):

    res = []

    cash_ratings = data[(data['Cashiers Num'] > 1) & data[cashier].notnull()]

    for _, row in cash_ratings.iterrows():

        percent = row[cashier] / row['Grand Total']

        res.append(percent - (1 / row['Cashiers Num']))

    return res



def get_plot_table(data):

    cashiers = data.columns[:-2]  # Ignore last two columns

    res = {}

    max_len = 0

    for cashier in cashiers:

        ratings = get_ratings(data, cashier)

        res[cashier] = ratings

        if len(ratings) > max_len:

            max_len = len(ratings)

    for cashier in cashiers:     # Add NaN's to the short columns

        rating = res[cashier]

        while len(rating) < max_len:

            rating.append(np.nan)

    return pd.DataFrame(res)
plot_table = get_plot_table(shopt)

plot_table.head()
plt.figure(figsize=(16, 7))

p = sns.boxplot(data=plot_table,

                order = sorted(plot_table.columns),

                orient="h")
shop_tmin = shop[shop['Duration'].notnull() & (shop['Change'] == 0.0)]

shop_tmin.shape
shop_tmin['Cashier'].value_counts()
min_table = shop_tmin.sort_values("Duration").groupby("Cashier").head(20)

min_table.head()
gb = min_table[['Cashier', 'Duration']].groupby('Cashier')

gb.mean()
gb.boxplot(subplots=False, figsize=(16,7), rot=90);
tb=shop_tmin.groupby('Cashier')['Total', 'Duration'].mean()

tb
tb['t_obr'] = tb['Duration']-gb['Duration'].mean()

tb['t_perf'] = tb['Duration'] / tb['t_obr']

tb
plt_data=tb.reset_index()

fig, ax = plt.subplots(figsize=(16,10))

sns.barplot(ax=ax, x='t_perf', y='Cashier', data=plt_data);
shop.dropna(subset=['Duration'], inplace=True)

shop['NoBreakTime'].fillna(0.0, inplace=True)
mari = shop[shop['Cashier']=='MARICRUZ']

data=mari[['Total', 'Change', 'Holiday', 'NoBreakTime']]

y = mari['Duration']

data.head()
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_scaled=scaler.fit_transform(data)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

gb_parms= {

    'loss': ['ls', 'lad', 'quantile'],

    'learning_rate': [0.05, 0.1, 0.5],

    'max_depth': [2, 3, 5]

}

gbr = GradientBoostingRegressor()

clf = GridSearchCV(gbr, gb_parms, cv=5)

clf.fit(X_train, y_train)
best_xgr = clf.best_estimator_

print(clf.best_params_)
print(clf.best_score_)
best_xgr.feature_importances_