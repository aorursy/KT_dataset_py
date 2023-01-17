import pandas as pd

import numpy as np

import random



import re

from zipfile import ZipFile

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline



import time



from sklearn.linear_model import LinearRegression
pd.set_option("display.max_columns", 25)
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
calendar_df['date'] = pd.to_datetime(calendar_df['date'])
calendar_df.tail()
sell_prices_df = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
sell_prices_df.head()
sell_prices_df.info()
temp_plot_df = (sell_prices_df

 .groupby('wm_yr_wk')

 .sell_price.agg({'mean', 'sem'})

)

fig, ax1 = plt.subplots(figsize=(10, 5))

temp_plot_df['mean'].sort_values().plot.bar(ax=ax1, ylim=[4.00,4.6], yerr=temp_plot_df['sem'], xticks=[])
temp_plot_df = (sell_prices_df

 .groupby('store_id')

 .sell_price.agg({'mean', 'sem'})

)

fig, ax1 = plt.subplots(figsize=(10, 5))

temp_plot_df['mean'].sort_values().plot.bar(ax=ax1, ylim=[4.35,4.48], yerr=temp_plot_df['sem'])
temp_plot_df = (sell_prices_df

                .groupby(['wm_yr_wk', 'store_id'])

                .sell_price.agg({'mean', 'sem'})

               ).reset_index(['store_id', 'wm_yr_wk'])

temp_cross_tab_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['store_id'], temp_plot_df['mean'], aggfunc='mean').plot(kind='line', subplots=True, grid=True, layout=(2, 5), sharex=True, sharey=True, figsize=(25,5), title='sell price evolution by store')
# Global price by state

temp_plot_df = (sell_prices_df

 .assign(state=lambda x: x.store_id.str[:2])

 .groupby('state')

 .sell_price.agg({'mean', 'sem'})

)

fig, ax1 = plt.subplots(figsize=(10, 5))

temp_plot_df['mean'].sort_values().plot.bar(ax=ax1, ylim=[4.35,4.48], yerr=temp_plot_df['sem']);
# we observe a différence in the price evolution between the states.

# Also, we can see a general 

temp_plot_df = (sell_prices_df

                .assign(state=lambda x: x.store_id.str[:2])

                .groupby(['wm_yr_wk', 'state'])

                .sell_price.agg({'count', 'mean', 'std'})

               ).reset_index(['state', 'wm_yr_wk'])

temp_cross_tab_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['state'], temp_plot_df['mean'], aggfunc='mean').plot(kind='line', subplots=True, grid=True, layout=(1, 3), sharex=True, sharey=True, figsize=(15,5), title='sell price evolution by state');
# Global price by state

temp_plot_df = (sell_prices_df

 .assign(category=lambda x: x.item_id.str[:4])

 .groupby('category')

 .sell_price.agg({'mean', 'sem'})

)

fig, ax1 = plt.subplots(figsize=(10, 5))

temp_plot_df['mean'].sort_values().plot.bar(ax=ax1, ylim=[3 ,6], yerr=temp_plot_df['sem']);
# collect the category from item id 

# compute aggregate by week and categories + error

temp_plot_df = (sell_prices_df

                .assign(category=lambda x: x.item_id.str[:4])

                .groupby(['wm_yr_wk', 'category'])

                .sell_price.agg({'mean', 'sem'})

               ).reset_index(['category', 'wm_yr_wk'])

# dataframe with index: week and column: cat.

temp_cross_tab_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['category'], temp_plot_df['mean'], aggfunc='mean').plot(kind='line', subplots=True, grid=True, layout=(1, 3), sharex=True, sharey=True, figsize=(15,5), title='sell price evolution by category');
temp_plot_df = (sell_prices_df

                .assign(cat_state=lambda x: x.item_id.str[:4]+ x.store_id.str[:2])

                .groupby(['wm_yr_wk', 'cat_state'])

                .sell_price.agg({'mean', 'sem'})

               ).reset_index(['cat_state', 'wm_yr_wk'])

# dataframe with index: week and column: cat. and state

temp_cross_tab_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['cat_state'], temp_plot_df['mean'], aggfunc='mean').plot(kind='line', subplots=True, grid=True, layout=(3, 3), sharex=True, sharey=True, figsize=(8,7), title='sell price evolution by category/state');
temp_plot_df = (sell_prices_df

                .assign(dept_id=lambda x: x.item_id.str.split(r"_",  expand=False).apply(lambda x: '_'.join(x[:2])))

                .groupby(['wm_yr_wk', 'dept_id'])

                .sell_price.agg({ 'mean', 'sem'})

               ).reset_index(['dept_id', 'wm_yr_wk'])

tem1_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['dept_id'], temp_plot_df['mean'], aggfunc='mean')

tem2_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['dept_id'], temp_plot_df['sem'], aggfunc='mean')

tem1_df.plot(kind='line', subplots=True, grid=True, layout=(1, 7), sharex=True, sharey=True, figsize=(20,4), yerr=tem2_df, title='sell price evolution by dept');
temp_plot_df = (sell_prices_df

                .assign(

                    dept_state=lambda x:  x.store_id.str[:2] + x.item_id.str.split(r"_",  expand=False).apply(lambda x: '_'.join(x[:2]))

                )

                .groupby(['wm_yr_wk', 'dept_state'])

                .sell_price.agg({ 'mean', 'sem'})

               ).reset_index(['dept_state', 'wm_yr_wk'])

tem1_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['dept_state'], temp_plot_df['mean'], aggfunc='mean')

#tem2_df = pd.crosstab(temp_plot_df['wm_yr_wk'], temp_plot_df['dept_state'], temp_plot_df['sem'], aggfunc='mean')
tem1_df.plot(kind='line', subplots=True, grid=True, layout=(3, 7), sharex=True, sharey=True, figsize=(25,9), title='sell price evolution by dept and state');
# number of items

sell_prices_df.item_id.nunique()
# we can see that the average price of each item is concentrated around 0.0 and 5.0

# var possible: 'state', 'cat', 'store_id'

var = 'dept'

temp1_df = (sell_prices_df

 .assign(dept=lambda x: x.item_id.str[:-4])

 .groupby([var, 'item_id'])

 .sell_price.mean()

 .reset_index([var, 'item_id'])

)#.hist(bins=50)

temp2_df = pd.crosstab(temp1_df[var], temp1_df['item_id'], temp1_df['sell_price'],aggfunc='mean').T
for c in temp2_df.columns:

    temp2_df[c].plot.kde(alpha=2, label=c, bw_method=0.3, xlim=[0, 20], figsize=(15,5), linewidth=0.8)

    plt.vlines(temp2_df[c].mean(), -.005, +.005)

plt.legend(list(temp2_df.columns))

plt.hlines(0, 0, 20)
pd.DataFrame([temp2_df.quantile(q=i/10) for i in range(11)]).round(2).T
k = 5

print("Here, the less expensive items against the most expensive.")

pd.DataFrame({

    'less': sell_prices_df.groupby('item_id').sell_price.mean().sort_values()[:k].index, 

    'more': sell_prices_df.groupby('item_id').sell_price.mean().sort_values(ascending=False)[:k].index

})
# Variance 

sell_prices_df.groupby(['item_id', 'wm_yr_wk']).sell_price.median().reset_index().assign(dept=lambda x: x.item_id.str[:-4]).groupby('dept').sell_price.agg({'var', 'max', 'min', 'median'}).sort_values(by='var')#.plot.bar()
sell_prices_df.assign(dept=lambda x: x.item_id.str[:-4]).groupby(['wm_yr_wk', 'dept']).var().unstack()['sell_price'].groupby(1+np.arange(282)//30).mean().plot.bar(subplots=True, sharex=True, layout=(1,7), figsize=(20, 4));
(sell_prices_df.groupby('item_id')

 .sell_price.std().reset_index()

 .assign(

     discret_var_price=lambda x: pd.qcut((x.sell_price * 10).round(), q=[0, 0.5, 0.75, 0.9, 0.95, 1], labels=[0, 1, 2, 3, 4])

 )

)
sell_prices_df.query("item_id == 'FOODS_1_002'").groupby('wm_yr_wk').sell_price.mean().plot()
stv_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
stv_df.head()
stv_df.info()
# plot: evolution of the total sales in days

temp_s = stv_df.select_dtypes('int64').sum()
temp_s.plot(alpha=0.8, figsize=(25,4))

temp_s.rolling(window=90).mean().plot(alpha=0.8)

plt.hlines(temp_s.mean(), 0, 2000, linestyles='--', colors='r');

plt.title("Evolution of the sales from day 0 to day 1913.");

plt.grid(True)
print("{} millions of sales in the whole period.".format(int(temp_s.sum()/1e6)))
def plot_by_period(s, period, ax=None, fig=None, color='blue', perso_title=None, fontsize=None):

    n_split = period 

    step = (len(s)//n_split)

    pd.DataFrame({'period_{p}'.format(p=i+1):  s.iloc[i*step:(i+1)*step].reset_index(drop=True) 

                  for i in range(n_split)}).sum().pct_change().plot.bar(

        xticks=[i*(n_split//5) for i in range(5)], ax=ax, fig=fig, color=color)

    ax.grid(True)

    ax.set_title('Sales for each {} days ({perso})'.format(step, perso=perso_title), fontsize=fontsize)
fig, axes = plt.subplots(nrows=1, ncols=4, sharey=False, sharex=False, figsize=(25,5))

plot_by_period(temp_s, 1913//365, ax=axes[0], color='blue', perso_title='figure 1'); # Per years

plot_by_period(temp_s, 1913//180, ax=axes[1], color='red', perso_title='figure 2'); # per seasons

plot_by_period(temp_s, 1913//30, ax=axes[2], color='green', perso_title='figure 3'); # per months

plot_by_period(temp_s, 1913//7, ax=axes[3], color='violet', perso_title='figure 4'); # per weeks
# by categories

temp_df = stv_df.groupby('cat_id').sum().T



fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(25,5))

((temp_df/temp_df.iloc[0]).plot(alpha=0.3, subplots=True, figsize=(15,5), title="Evolution of sales by cat", ax=axes));

((temp_df/temp_df.iloc[0]).rolling(window=60).mean().plot(subplots=True, figsize=(15,5), title="Evolution of sales by cat", ax=axes));



for ax in axes:

    ax.hlines(1, 0, 2000, linestyles='--', colors='r')

    ax.grid(True)
(temp_df.sum()/1e6).plot.bar()
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=False, figsize=(20,15))

for i, c in enumerate(['FOODS', 'HOBBIES', 'HOUSEHOLD']):

    plot_by_period(temp_df[c], 1913//365, ax=axes[i,0], color='blue', perso_title='figure 1, {c}'.format(c=c)); # Per years

    plot_by_period(temp_df[c], 1913//180, ax=axes[i,1], color='red', perso_title='figure 2, {c}'.format(c=c)); # per seasons

    plot_by_period(temp_df[c], 1913//30, ax=axes[i,2], color='green', perso_title='figure 3, {c}'.format(c=c)); # per months

fig.suptitle('Sales by periods for each category', fontsize=16);
temp_df = stv_df.groupby('dept_id').sum().T



fig, axes = plt.subplots(nrows=1, ncols=7, sharey=True, figsize=(25,4))

((temp_df/temp_df.iloc[0]).plot(alpha=0.3, subplots=True, title="Evolution of sales by dept", ax=axes));

((temp_df/temp_df.iloc[0]).rolling(window=60).mean().plot(subplots=True, title="Evolution of sales by dept", ax=axes));



for ax in axes:

    ax.hlines(1, 0, 2000, linestyles='--', colors='r')

    ax.grid(True)

    ax.set_ylim(0, 3)
# millions of sales for each departments

# The department with the most sales is FOODS_3.

(temp_df.sum()/1e6).round(3).plot.bar()
fig, axes = plt.subplots(nrows=3, ncols=7, sharey=True, sharex=False, figsize=(22,10))

for i, c in enumerate(list(temp_df.columns)):

    plot_by_period(temp_df[c], 1913//365, ax=axes[0, i], color='blue', perso_title='figure 1, {c}'.format(c=c), fontsize=7); # Per years

    plot_by_period(temp_df[c], 1913//180, ax=axes[1, i], color='red', perso_title='figure 2, {c}'.format(c=c), fontsize=7); # per seasons

    plot_by_period(temp_df[c], 1913//30, ax=axes[2, i], color='green', perso_title='figure 3, {c}'.format(c=c), fontsize=7); # per months

fig.suptitle('Sales by periods for each dept', fontsize=15);
# by state_id

temp_df = stv_df.groupby('state_id').sum().T



fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(25,5))

((temp_df/temp_df.iloc[0]).plot(alpha=0.3, subplots=True, figsize=(15,5), title="Evolution of sales by state", ax=axes));

((temp_df/temp_df.iloc[0]).rolling(window=60).mean().plot(subplots=True, figsize=(15,5), title="Evolution of sales by state", ax=axes));



for ax in axes:

    ax.hlines(1, 0, 2000, linestyles='--', colors='r')

    ax.grid(True)
(temp_df.sum()/1e6).round(3).plot.bar(title="volume of sales by state");
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=False, figsize=(20,10))

for i, c in enumerate(['CA', 'TX', 'WI']):

    plot_by_period(temp_df[c], 1913//365, ax=axes[i,0], color='blue', perso_title='figure 1, {c}'.format(c=c)); # Per years

    plot_by_period(temp_df[c], 1913//180, ax=axes[i,1], color='red', perso_title='figure 2, {c}'.format(c=c)); # per seasons

    plot_by_period(temp_df[c], 1913//30, ax=axes[i,2], color='green', perso_title='figure 3, {c}'.format(c=c)); # per months

fig.suptitle('Sales by periods for each category', fontsize=16);
temp_df = stv_df.groupby('store_id').sum().T



fig, axes = plt.subplots(nrows=2, ncols=5, sharey=True, figsize=(25,6))

((temp_df/temp_df.iloc[0]).plot(alpha=0.3, subplots=True, title="Evolution of sales by store_id", ax=axes));

((temp_df/temp_df.iloc[0]).rolling(window=90).mean().plot(alpha=10, subplots=True, title="Evolution of sales by store_id", ax=axes));



for c in axes:

    for ax in c:

        ax.hlines(1, 0, 2000, linestyles='--', colors='r')

        ax.grid(True)

        ax.set_ylim(0, 2.5)
(temp_df.sum()/1e6).round(3).plot.bar(title="volume of sales by store_id");
fig, axes = plt.subplots(nrows=2, ncols=10, sharey=True, sharex=False, figsize=(24,5))

for i, c in enumerate(list(temp_df.columns)):

    plot_by_period(temp_df[c], 1913//365, ax=axes[0, i], color='blue', perso_title='figure 1, {c}'.format(c=c), fontsize=6); # Per years

    plot_by_period(temp_df[c], 1913//180, ax=axes[1, i], color='red', perso_title='figure 2, {c}'.format(c=c), fontsize=6); # per seasons

fig.suptitle('Sales by periods for each store', fontsize=15);
temp_df = stv_df.groupby(['dept_id', 'state_id']).sum().T
fig, axes = plt.subplots(ncols=7, figsize=(30,4))

for i, col in enumerate(stv_df.dept_id.unique()):

    temp_df[col].rolling(56).mean().plot(ax=axes[i], title="sales for {}".format(col))
temp_df = stv_df.groupby(['dept_id', 'store_id']).sum().T
fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(30,15))

for i, cols in enumerate(stv_df.dept_id.unique()):

    temp_df[cols][[col for col in temp_df[cols].columns if 'CA' in col]].rolling(56).mean().plot(ax=axes[0][i], title="sales for {} in CA".format(cols), legend=[])

    temp_df[cols][[col for col in temp_df[cols].columns if 'TX' in col]].rolling(56).mean().plot(ax=axes[1][i], title="sales for {} in TX".format(cols), legend=[])

    temp_df[cols][[col for col in temp_df[cols].columns if 'WI' in col]].rolling(56).mean().plot(ax=axes[2][i], title="sales for {} in Wi".format(cols), legend=[])

    

fig.legend(labels=list(stv_df.store_id.unique()));
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,4))

stv_df.groupby('item_id').sum().loc['FOODS_3_090'].plot(ax=axes[0], title="Item with the larger std")

stv_df.groupby('item_id').sum().loc['HOBBIES_2_119'].plot(ax=axes[1], title="Item with the lower std")
sales = (stv_df

         .query('item_id == "HOBBIES_1_078" and store_id == "CA_3"')

         .select_dtypes('int64').sum().reset_index(drop=True))
n_start = 600

omg1 = (2*np.pi)/7

omg2 = (2*np.pi)/360

omg3 = (2*np.pi)/100

t = np.linspace(0, len(sales), len(sales))

# n sales

y = sales.values[n_start:]

# seasonality

x1 = np.cos(omg1*t).reshape(-1, 1)[n_start:]

x2 = np.sin(omg1*t).reshape(-1, 1)[n_start:]

x3 = np.cos(omg2*t).reshape(-1, 1)[n_start:]

x4 = np.sin(omg2*t).reshape(-1, 1)[n_start:]

x5 = np.cos(omg3*t).reshape(-1, 1)[n_start:]

x6 = np.sin(omg3*t).reshape(-1, 1)[n_start:]

# cte

cte = np.ones((len(sales), 1))[n_start:]

# lag vars

y_28 = sales.shift(28).values.reshape(-1, 1)[n_start:]

y_56 = sales.shift(56).values.reshape(-1, 1)[n_start:]

y_112 = sales.shift(112).values.reshape(-1, 1)[n_start:]

# linear trend

trend = t.reshape(-1, 1)[n_start:]

# non-linear trend

trend_nl = sales.shift(28).rolling(360).mean().values.reshape(-1, 1)[n_start:]

X = np.concatenate((y_28, y_56, y_112, x1, x2, x3, x4, x5, x6, cte, trend, trend_nl), axis=1)



# Training and test
X_train, X_test = X[:-28], X[-28:]

y_train, y_test = y[:-28], y[-28:]
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier
# Linear Reg.

# w.o diff

b1 = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)

pred1 = X_test.dot(b1)
# Gradient boosting

# No statistical data model

model = GradientBoostingRegressor(n_estimators=200,random_state=0)

model.fit(X_train,y_train)
# statistical data model

model2 = GradientBoostingRegressor(n_estimators=200, random_state=0)

#model2.fit(X_train,y_train-y_28[:-28].reshape(-1))
# pred

pred = model.predict(X_test)

#pred2 = model2.predict(X_test) + y_28[-28:].reshape(-1)
plt.bar(['y_28', 'y_56', 'y_112', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'cte', 'trend', 'trend_nl'], model.feature_importances_)

plt.xticks(rotation=45)
plt.figure(figsize=(25,5))

plt.plot(y, label='sales ')

#plt.plot(model2.predict(X)+ y_28.reshape(-1), label='sales prediction 2')

plt.plot(model.predict(X), label='sales prediction 1')

#plt.plot(X.dot(b1), label='sales prediction lr')

plt.xlim((1250, 1320))

plt.vlines(len(y_train), ymin=y.min(), ymax=y.max(), colors='r', linestyles='--')

plt.legend()
from sklearn.metrics import confusion_matrix
# logistic model predict 0 or not

clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=15, max_depth=30, random_state=0, class_weight='balanced', max_samples=0.75)

clf.fit(X_train, 1*(y_train==0), sample_weight=5+np.arange(len(X_train)))
confusion_matrix(clf.predict(X_test), 1*(y_test==0), normalize='true')
plt.figure(figsize=(25,5))

plt.plot(y>0, label='sales ')

plt.plot(-clf.predict(X)+1, label='sales prediction 1')

plt.xlim((1250, 1320))

plt.vlines(len(y_train), ymin=y.min(), ymax=y.max(), colors='r', linestyles='--')

plt.legend()
tem_s = stv_df.select_dtypes('int64').sum().reset_index(drop=True)

calendar_df['sales'] = tem_s
calendar_df = calendar_df.dropna(subset=['sales'])
# saisonality: weekend more sales/day than during the week

calendar_df.groupby('wday')['sales'].mean().plot.bar();
# week seasonality evolution is same for each day

calendar_df.groupby(['wday', 'year'])['sales'].mean().unstack().T.plot()
# There is a large dip in may 

calendar_df.groupby('month').sales.mean().plot.bar(ylim=[30000, None])
# We observe the large impact of SNAP

tem3_df = calendar_df.reset_index().assign(snap=lambda x: x['snap_CA']+x['snap_TX']+x['snap_WI']).groupby('snap')['sales'].agg({'mean', 'sem'})

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(15,4), ylim=[32000, None])
# The effect of SNAP is not the same for each weekdays

# In Saturday, Sunday low impact; 

# In  Monday and Friday middle impact;

# In Tuesday, Wednesday, Thursday large impact;

tem3_df = calendar_df.reset_index().assign(snap=lambda x: x['snap_CA']+x['snap_TX']+x['snap_WI'])[['wday', 'snap', 'sales']]

pd.crosstab(tem3_df['wday'], tem3_df['snap'], values=tem3_df['sales'], aggfunc='mean').T.plot.bar(grid=True, figsize=(20,5), subplots=True, layout=(2, 4), ylim=[28000, None]);
tem_df = stv_df.groupby('state_id').sum().T.reset_index(drop=True)

tem2_df = pd.merge(tem_df, calendar_df, how='outer', left_index=True, right_index=True).dropna(subset=['CA', 'TX', 'WI']).set_index('date')

tem2_df[['CA', 'TX', 'WI']] = tem2_df.groupby('date')[['CA', 'TX', 'WI']].sum()
tem2_df.head()
# The Sunday is the day is the highest number of sales in CA and TX but in WI it's the saturday (Sunday: 17.5% of the week's sales for TX and CA  against 16% for WI)

# Also there is a clear difference between the week end and the remains of the week.

tem3_df = tem2_df.groupby('wday')[['CA', 'TX', 'WI']].mean().plot.bar(grid=True, figsize=(20,4), subplots=True, layout=(1, 3), sharey=True)

#tem3_df.div(tem3_df.sum())
tem3_df = tem2_df.assign(total=lambda x: x[['CA', 'TX', 'WI']].sum(axis=1)).groupby('month')[['CA', 'TX', 'WI']].mean().plot.bar(grid=True, figsize=(20,4), subplots=True, layout=(1, 3), ylim=[5000, 16000])
tem3_df = tem2_df.groupby('year')[['CA', 'TX', 'WI']].mean().plot.bar(grid=True, figsize=(15,4), subplots=True, layout=(1, 3))

# ax = (tem3_df.div(tem3_df.sum()) /(1/6) ).plot(grid=True)

# ax.hlines(1, 2011, 2016, colors='r', linestyles='--', label='base')

# ax.legend()
# Effect of snap on the sales:

# The WI is the state with the greatest effect of the snap on the sale

tem3_df = pd.DataFrame([tem2_df.groupby('snap_{state}'.format(state=state))['{state}'.format(state=state)].mean() for state in ['CA', 'TX', 'WI']]).T

ax = (tem3_df/tem3_df.sum()).T.diff(axis=1)[1].plot.bar(stacked=True, title="Difference of the average sales by day between Snap/No Snap for each state")
tem_df = stv_df.groupby('cat_id').sum().T.reset_index(drop=True)

cats = ['FOODS','HOBBIES','HOUSEHOLD']

tem2_df = pd.merge(tem_df, calendar_df, how='outer', left_index=True, right_index=True).dropna(subset=cats).set_index('date')

tem2_df[cats] = tem2_df.groupby('date')[cats].sum().apply(lambda x: x[x.between(x.quantile(0.01), x.quantile(0.99))], axis=0)
tem2_df.head()
tem3_df = tem2_df.groupby('wday')[['FOODS','HOBBIES','HOUSEHOLD']].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,4), subplots=True, layout=(1, 3));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

axes[0][0].set_ylim(ymins[0]*.95, ymaxs[0]*1.05)

axes[0][1].set_ylim(ymins[1]*.95, ymaxs[1]*1.05)

axes[0][2].set_ylim(ymins[2]*.95, ymaxs[2]*1.05)
tem3_df = tem2_df.groupby('month')[['FOODS','HOBBIES','HOUSEHOLD']].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,4), subplots=True, layout=(1, 3))

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

axes[0][0].set_ylim(ymins[0]*.95, ymaxs[0]*1.05)

axes[0][1].set_ylim(ymins[1]*.95, ymaxs[1]*1.05)

axes[0][2].set_ylim(ymins[2]*.95, ymaxs[2]*1.05)
tem3_df = tem2_df.groupby('year')[['FOODS','HOBBIES','HOUSEHOLD']].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,4), subplots=True, layout=(1, 3))

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

axes[0][0].set_ylim(ymins[0]*.95, ymaxs[0]*1.05)

axes[0][1].set_ylim(ymins[1]*.95, ymaxs[1]*1.05)

axes[0][2].set_ylim(ymins[2]*.95, ymaxs[2]*1.05)
tem3_df = tem2_df.reset_index().assign(snap=lambda x: x['snap_CA']+x['snap_TX']+x['snap_WI']).groupby('snap')[['FOODS','HOBBIES','HOUSEHOLD']].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,4), subplots=True, layout=(1, 3))

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

axes[0][0].set_ylim(ymins[0]*.95, ymaxs[0]*1.05)

axes[0][1].set_ylim(ymins[1]*.95, ymaxs[1]*1.05)

axes[0][2].set_ylim(ymins[2]*.95, ymaxs[2]*1.05)
tem_df = stv_df.groupby('store_id').sum().T.reset_index(drop=True)

stores = list(tem_df.columns)

tem2_df = pd.merge(tem_df, calendar_df, how='outer', left_index=True, right_index=True).dropna(subset=stores).set_index('date')

tem2_df[stores] = tem2_df.groupby('date')[stores].sum().apply(lambda x: x[x.between(x.quantile(0.01), x.quantile(0.99))], axis=0)
tem2_df.head()
tem3_df = tem2_df.groupby('wday')[stores].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,5), subplots=True, layout=(2,5));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

        i+=1
tem3_df = tem2_df.groupby('month')[stores].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,5), subplots=True, layout=(2,5));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

        i+=1
tem3_df = tem2_df.groupby('year')[stores].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,5), subplots=True, layout=(2,5));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

        i+=1
# Effect of snap on the sales:

tem3_df = pd.DataFrame([tem2_df.groupby('snap_{state}'.format(state=store[:-2]))['{state}'.format(state=store)].mean() for store in stores]).T

ax = (tem3_df/tem3_df.sum()).T.diff(axis=1)[1].plot.bar(stacked=True, title="Difference of the average sales by day between Snap/No Snap for each store_id")
tem_df = stv_df.groupby('dept_id').sum().T.reset_index(drop=True)

depts = list(tem_df.columns)

tem2_df = pd.merge(tem_df, calendar_df, how='outer', left_index=True, right_index=True).dropna(subset=depts).set_index('date')

tem2_df[depts] = tem2_df.groupby('date')[depts].sum().apply(lambda x: x[x.between(x.quantile(0.01), x.quantile(0.99))], axis=0)
tem2_df.head()
tem3_df = tem2_df.groupby('wday')[depts].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,5), subplots=True, layout=(2,4));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        try:

            c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

            i+=1

        except: 'Done'
tem3_df = tem2_df.groupby('month')[depts].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,5), subplots=True, layout=(2,4));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        try:

            c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

            i+=1

        except: 'Done'
tem3_df = tem2_df.groupby('year')[depts].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(20,5), subplots=True, layout=(2,4));

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        try:

            c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

            i+=1

        except: 'Done'
tem3_df = tem2_df.reset_index().assign(snap=lambda x: x['snap_CA']+x['snap_TX']+x['snap_WI']).groupby('snap')[depts].agg({'mean', 'sem'}).swaplevel(axis=1)

axes = tem3_df['mean'].plot.bar(yerr=tem3_df['sem'], grid=True, figsize=(18,5), subplots=True, layout=(2, 4))

ymaxs = tem3_df['mean'].max(0).to_numpy(); ymins = tem3_df['mean'].min(0).to_numpy() 

i = 0

for r in axes:

    for c in r:

        try:

            c.set_ylim(ymins[i]*.95, ymaxs[i]*1.05)

            i+=1

        except: 'Done'
calendar_df.event_name_1.unique()
s = stv_df.select_dtypes('int64').sum().reset_index(drop=True)

s.plot(figsize=(15,3))

events = list((calendar_df.event_name_1 == 'StPatricksDay').where(lambda x: x == 1).dropna().index)

plt.vlines(events, s.min(), s.max(), linestyles='--', colors='r')
def events_week(x):

    if len(x)!=0:

        return np.unique(list(x))[0]

    else:

        return 0



calendar_df['sales'] = s

temp_s = (calendar_df

 .groupby('wm_yr_wk')

 .agg({'sales': 'mean', 'event_name_1':events_week})

 #.assign(event_name_1_neg=lambda x: x.event_name_1.shift(-4))

 .fillna({'event_name_1':'No event'})

 .assign(sum_lag_sales=lambda x: pd.DataFrame([x.sales.shift(i) for i in range(1, 3)]).mean(skipna=False))

 .groupby('event_name_1')

 .sum_lag_sales.mean()

).sort_values()
temp_s.drop('No event').plot.bar(figsize=(15,5), ylim=[30000, None])

plt.hlines(temp_s['No event'], -1, len(temp_s), 'r', '--')

plt.vlines(19, 0, 40000, 'r', '--')

plt.show()
s_cat = stv_df.groupby('cat_id').sum().T.reset_index(drop=True)

calendar_df[['FOODS', 'HOBBIES', 'HOUSEHOLD']] = s_cat
temp_s = (calendar_df

 .groupby('wm_yr_wk')

 .agg({'FOODS': 'mean', 'HOBBIES': 'mean', 'HOUSEHOLD': 'mean', 'event_name_1':events_week})

 #.assign(event_name_1_neg=lambda x: x.event_name_1.shift(-4))

 .fillna({'event_name_1':'No event'})

 .assign(

     sum_lag_FOODS=lambda x: pd.DataFrame([x.FOODS.shift(i) for i in range(1, 3)]).mean(skipna=False),

     sum_lag_HOBBIES=lambda x: pd.DataFrame([x.HOBBIES.shift(i) for i in range(1, 3)]).mean(skipna=False),

     sum_lag_HOUSEHOLD=lambda x: pd.DataFrame([x.HOUSEHOLD.shift(i) for i in range(1, 3)]).mean(skipna=False),     

 )

 .groupby('event_name_1')

 [['sum_lag_FOODS', 'sum_lag_HOBBIES', 'sum_lag_HOUSEHOLD']].mean()

)
cat_ = 'sum_lag_HOUSEHOLD'

temp_s[cat_].drop('No event').sort_values().plot.bar(figsize=(15,5), ylim=[2800, None])

plt.hlines(temp_s[cat_]['No event'], -1, len(temp_s), 'r', '--')

plt.vlines(19, 0, 40000, 'r', '--')

plt.show()
tem_df = stv_df.query("dept_id == 'HOBBIES_2' and store_id == 'CA_1'").groupby('item_id').sum().T.reset_index(drop=True)

items = list(tem_df.columns)

tem2_df = pd.merge(tem_df, calendar_df, how='outer', left_index=True, right_index=True).dropna(subset=items).set_index('date').groupby('wm_yr_wk')[items].sum()
tem3_df = sell_prices_df[sell_prices_df.item_id.apply(lambda x: 'HOBBIES_2' in x)].query("store_id == 'CA_1'").groupby(['wm_yr_wk', 'item_id']).sell_price.mean().unstack().fillna(method='bfill')
i = 5

labels = pd.cut(tem3_df.iloc[:, i].values, bins=4, labels=[0,1,2,3], retbins=True)

ax = tem2_df.iloc[:, i].plot(c='k', figsize=(25,5), title=tem2_df.iloc[:, i].name)

#ax2 = ax.twinx()

#ax2.plot(tem3_df.iloc[:, i])

colors = ['tab:red', 'tab:purple', 'tab:blue', 'tab:green']

for j, c in enumerate(colors):

    ax.fill_between(list(tem3_df.index), tem2_df.iloc[:, i].min(), tem2_df.iloc[:, i].max(), where= labels[0] == j, facecolor=c, alpha=0.3)
stv = stv_df.select_dtypes('int64').sum().reset_index(drop=True)
stv_total_df = stv_df.select_dtypes('int64').sum()

stv_state_df = stv_df.groupby("state_id").sum()

stv_store_df = stv_df.groupby("store_id").sum()
# Bottom-Up
def create_data(serie, lag=5):

    data = pd.DataFrame()

    # create features

    data['sales'] = serie.copy().reset_index(drop=True)

    data['constant'] = np.ones(len(serie))

    data['trend'] = np.arange(len(serie))

    omg1 = (2*np.pi)/7

    omg2 = (2*np.pi)/365

    data['cos_s1'] = np.cos(omg1*data['trend'])

    data['sin_s1'] = np.sin(omg1*data['trend'])

    data['cos_s2'] = np.cos(omg2*data['trend'])

    data['sin_s2'] = np.sin(omg2*data['trend'])

    for i in range(1, lag+1):

        data['lag_{}'.format(i)] = data['sales'].shift(i)

    return data.dropna()
def compute_ols(X, y):

    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
def predict(X, beta):

    return X.dot(beta)
# calculate forcast for all the series

forecast_vect = []

series_to_compute = [stv_total_df] + [stv_state_df.T[idx] for idx in stv_state_df.index] + [stv_store_df.T[idx] for idx in stv_store_df.index] 
p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = tuple(stv_store_df.sum(1)/stv_total_df.sum())
S = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 

              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])



P = np.array([[p1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 

              [p10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
for s in series_to_compute:

    data = create_data(s)

    X, y = data.drop(columns='sales').values, data['sales'].values

    beta = compute_ols(X, y)

    forecast_vect.append(predict(X, beta))
y_pred = S.dot(P).dot(np.array(forecast_vect))
data = create_data(stv_state_df.T['CA'], lag=5)

X, y = data.drop(columns='sales').values, data['sales'].values

y_pred_total = predict(X, compute_ols(X, y))
# plt.figure(figsize=(30,4))

y_true = stv_store_df.T['CA_1'].values 

plt.plot(y_pred[4])

plt.plot(forecast_vect[4])
np.abs(y_pred[4] - y_true[5:]).mean(), np.abs(forecast_vect[4] - y_true[5:]).mean()