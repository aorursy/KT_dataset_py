import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

%matplotlib inline
sns.set()
data = pd.read_csv('../input/avocado.csv')
data.head()
columns_for_drop = ['Unnamed: 0', 'year'] 

data = pd.read_csv('../input/avocado.csv', index_col='Date', parse_dates=True).drop(columns_for_drop, axis = 1)
data = data[data.region != 'TotalUS']

data['TB_minus'] = (data['Total Bags']-data['Small Bags']-data['Large Bags']-data['XLarge Bags']).round(2)
data['TV_minus'] = (data['Total Volume']-data['4046']-data['4225']-data['4770'] - data['Total Bags']).round(2)

data.head()
data_kinds = np.array([dtype.kind for dtype in data.dtypes])
num_cols = data.columns[data_kinds != 'O']
cat_cols = data.columns[data_kinds == 'O']
print('Numerical columns: {} pcs \n {}'.format(len(num_cols), num_cols.values))
print()
print('Categorical columns: {} pcs \n {}'.format(len(cat_cols), cat_cols.values))
for cat_col in cat_cols:
    print('Column: "{}"'.format(str(cat_col)))
    print(data[cat_col].value_counts().tail())
missing_mask = ((data['type'] == 'organic') & (data['region'] == 'WestTexNewMexico'))
standard_mask = ((data['type'] == 'organic') & (data['region'] == 'Boise'))

filter_mask = pd.Series(data[standard_mask].index).isin(pd.Series(data[missing_mask].index))

missing_index_values = (data[standard_mask].index)[~filter_mask]
print('''Missing observations:
type = "organic"
region = "WestTexNewMexico"
for the following dates:''')
for t in missing_index_values:
    print(t.strftime('%Y-%m-%d'))
data.describe().round(2)
data.sort_values(by = ['TV_minus'], ascending=False).head(5)
(
    data.assign(TV_percent = ((data['TV_minus'] / data['Total Volume'])*100).round(2))
    .sort_values(by = ['TV_percent'], ascending=False)
    .head(10)
)
data.drop(['TB_minus', 'TV_minus'], axis = 1, inplace=True)
data.head()
data.isna().sum()
data.head()
annot_customize_dict = dict(xycoords='data', 
                            textcoords='offset points', 
                            bbox=dict(boxstyle="round4,pad=.5", fc="0.9", color = 'k'),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color = 'k'))

fig, ax = plt.subplots(figsize = (12,6))
g = sns.lineplot(x = data[data.region == 'Albany'].index, 
                 y = 'AveragePrice', 
                 hue = 'type',
                 data=data[data.region == 'Albany'],
                 err_style=None)

ax.annotate(
    """ \"organic\" type is
usually more
expensive""",
    xy=('2016-04-1', 1.53),
    xytext=(-150, 0),
    **annot_customize_dict)

plt.title('Average price of avocado in the "Albany" region');
regions_chosen = ['Spokane', 'Denver']
types_chosen = ['organic', 'conventional']

data2 = data[data.region.isin(regions_chosen) & data.type.isin(types_chosen)]
data2.head()
fig, ax = plt.subplots(figsize = (12,8))

g = sns.lineplot(x = data2.index, 
                 y = 'AveragePrice', 
                 hue = 'region',
                 style = 'type',
                 data=data2,
                 err_style=None)

ax.annotate(
"""There is some consistency between regions. 
When in one region price rises we expect 
similar behavior in other regions""", 
            xy=('2016-10-1', 1.53), 
            xytext=(-330, 115), 
            **annot_customize_dict)

ax.annotate(
"""Maybe some consitency between 
'type' is reasonable to investigate?""", 
            xy=('2017-10-1', 1.2), 
            xytext=(-100, -110), 
            **annot_customize_dict);
ax.set_title('Comparision of "region" and "type" for two examples');
def create_filled_subplot(region, ax = None, legend = None):
    if ax == None:
        ax = plt.gca()
    data2 = data[data.region == region]
    pt = pd.pivot_table(data2, index=['Date'], columns=['type']).AveragePrice #no agg func!
    pt['orga_more_exp'] = (pt.conventional < pt.organic)
    
    g = sns.lineplot(x = data2.index, 
                 y = 'AveragePrice', 
                 hue = 'type',
                 data=data2,
                 err_style=None, 
                 legend = legend, ax = ax)

    ax.fill_between(x = pt.index, 
                    y1 = 0.4, 
                    y2 = 3.3, 
                    where = (pt.conventional<pt.organic), 
                    facecolor='red', alpha=0.4)

    ax.set_title('{} \n OME = {}'
                 .format(region, pt.orga_more_exp.mean().round(3)));
regions_chosen = ['West']
xticks=[pd.to_datetime('2015'), pd.to_datetime('2016'), pd.to_datetime('2017'), pd.to_datetime('2018')]
fig, ax = plt.subplots(figsize = (10,6), subplot_kw=dict(xticks=xticks), sharex=True, sharey=True)
create_filled_subplot(regions_chosen[0], ax = ax, legend = 'brief')
length = len(data.region.unique())
regions_chosen = data.region.unique()

fig, ax = plt.subplots(9, 6, figsize = (16,25), subplot_kw=dict(xticks=xticks), 
                       sharex=True, sharey=True, 
                       gridspec_kw=dict(wspace = 0.05, hspace = 0.4))

for i, axi in enumerate(ax.flat):
    try:
        create_filled_subplot(regions_chosen[i], ax = axi)
    except IndexError:
        pass
region = 'Albany'
data2 = data[data.region == region]
data2_organic = data2[data2.type == 'organic']
data2_conventional = data2[data2.type == 'conventional']

fig, (ax_1, ax_o, ax_c) = plt.subplots(3,1, figsize = (10,12), subplot_kw=dict(xticks=xticks))

for database, axi in zip([data2, data2_organic, data2_conventional], [ax_1, ax_o, ax_c]):
    sns.lineplot(x = database.index, y = 'Total Volume', hue = 'type', data=database, err_style=None, ax = axi)
    
fig.suptitle('"Total Volume" of avocado sold in "Albany" region', size = 20)
fig.subplots_adjust(top=0.92)
data.groupby('type')['Total Volume'].sum()
average_price_mean = pd.DataFrame(data.groupby(['region','type'])['AveragePrice'].mean())
total_volume_sum = pd.DataFrame(data.groupby(['region','type'])['Total Volume'].sum())
total_volume_sum.head()
sns.swarmplot(x = average_price_mean.index.droplevel(0), y = 'AveragePrice', data=average_price_mean)
plt.gca().set_title('Avocado price in different regions \n(average of 2015-2018)');
sns.swarmplot(x = total_volume_sum.index.droplevel(0), y = 'Total Volume', data=total_volume_sum)
plt.gca().set_title('Total volume of avocados sold \n in different regions \n (total sum of 2015-2018)');
data2_plu = data.pivot_table(['4046', '4225', '4770'], index=['region'], columns=['type']).stack(level = 0).round(2)
data2_plu = data2_plu.assign(PLU = data2_plu.index.droplevel(0))
data2_plu.head()
fig, ax = plt.subplots(figsize = (14,10))

max1 = 1000000
ax1 = plt.subplot(121)
sns.boxplot(x = 'PLU', y = 'conventional', data=data2_plu, ax=ax1)
ax1.set_ylim(0,max1)
ax1.set_title('Zoom (max = {})'.format(max1))

ax2 = plt.subplot(222)
sns.boxplot(x = 'PLU', y = 'conventional', data=data2_plu, ax=ax2)
ax2.set_title('Maximum Values (No zoom)')

max3 = 100000
ax3 = plt.subplot(224)
sns.boxplot(x = 'PLU', y = 'conventional', data=data2_plu, ax=ax3)
ax3.set_ylim(0, max3)
ax3.set_title('Zoom (max = {})'.format(max3))

plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
plt.suptitle(
"""Total Volume of avocados sold in every week:
average value of 2015-2018 for different regions (type - conventional)""", size = 15);
fig, ax = plt.subplots(figsize = (14,10))

max1 = 20000
ax1 = plt.subplot(121)
sns.boxplot(x = 'PLU', y = 'organic', data=data2_plu, ax=ax1)
ax1.set_ylim(0,max1)
ax1.set_title('Zoom (max = {})'.format(max1))

ax2 = plt.subplot(222)
sns.boxplot(x = 'PLU', y = 'organic', data=data2_plu, ax=ax2)
ax2.set_title('Maximum Values (No zoom)')

max3 = 800
ax3 = plt.subplot(224)
sns.boxplot(x = 'PLU', y = 'organic', data=data2_plu, ax=ax3)
ax3.set_ylim(0, max3)
ax3.set_title('Zoom (max = {})'.format(max3))

plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
plt.suptitle(
"""Total Volume of avocados sold in every week:
average value of 2015-2018 for different regions (type - organic)""", size = 15);
chosen_type = 'conventional'
chosen_region = 'California'

data3 = data[(data.region == chosen_region) & (data.type == chosen_type)]
data3 = data3[['AveragePrice', 'Total Volume']].sort_index()
data3.head()
sns.scatterplot(x = 'Total Volume', y = 'AveragePrice', data = data3);
plt.gca().set_title('"Total Volume" and "AveragePrice" seems to be corelated');
fig, ax = plt.subplots(2, figsize = (12,8))
sns.lineplot(x = data3.index.values, y = 'AveragePrice', data=data3, ax = ax[0])
sns.lineplot(x = data3.index.values, y = 'Total Volume', data=data3, ax = ax[1]);
data3_past = pd.concat([data3, data3.shift(1), data3.shift(2), data3.shift(4)], axis=1).dropna()
data3_past.columns = ['AveragePrice', 'Total Volume', 'AP_1', 'TV_1', 'AP_2', 'TV_2', 'AP_4', 'TV_4']
data3_past.head()
X = data3_past.iloc[:, 1:]
y = data3_past.iloc[:, 0][:, np.newaxis]

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
y_scaled = StandardScaler().fit_transform(y) #calculated only for the purpose of visualization
fig, ax = plt.subplots(figsize = (14,5))
sns.lineplot(x = data3_past.index.values, y = y_scaled.ravel(), label = 'AveragePrice')
sns.lineplot(x = data3_past.index.values, y = X_scaled[:, 0], label = 'Total Volume')
plt.legend()
ax.set_title('Scaled values for "AveragePrice" and "Total Volume" in {} region'.format(chosen_region), size = 15);
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state = 0)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_fit = model_lr.predict(X_test)

mae = mean_absolute_error(y_test, y_fit)
score = model_lr.score(X_test, y_test)

print('Mean absolute error (test set) = {}'.format(np.round(mae,4)))
print('Coefficient of determination R^2 (test set) = {}'.format(np.round(score, 4)))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def predict_prize(chosen_region, chosen_type, model_estimator, 
                  param_grid, test_size = 0.2, KFold_split = 4, 
                  to_print=True, random_state = 0, to_ravel_y = False):
    
    dataset = data[(data.region == chosen_region) & (data.type == chosen_type)]
    dataset = dataset[['AveragePrice', 'Total Volume']].sort_index()
    dataset_past = pd.concat([dataset, dataset.shift(1), dataset.shift(2), dataset.shift(4)], axis=1).dropna()
    dataset_past.columns = ['AveragePrice', 'Total Volume', 'AP_1', 'TV_1', 'AP_2', 'TV_2', 'AP_4', 'TV_4']
    
    X = dataset_past.iloc[:, 1:]
    y = dataset_past.iloc[:, 0][:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    if to_ravel_y:
        y_train = y_train.ravel()
        y_test = y_test.ravel()
    
    est1_step = ('est1', StandardScaler())
    est2_step = ('est2', model_estimator)
    steps = [est1_step, est2_step]
    pipe = Pipeline(steps)
    
    kf = KFold(n_splits=KFold_split, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv = kf)
    gs.fit(X_train, y_train);
    
    model = gs.best_estimator_
    y_predicted = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_predicted)
    score = model.score(X_test, y_test)
    
    if to_print:
        print(gs.best_params_)
        print('Mean absolute error (test set) = {}'.format(np.round(mae,4)))
        print('Coefficient of determination R^2 (test set) = {}'.format(np.round(score, 4)))
    return (model, mae, score)
from sklearn.linear_model import Ridge

param_grid = {
    'est2__alpha':[0, 0.1, 0.5, 1, 5, 10]
}
predict_prize('West', 'conventional', Ridge(), param_grid);
# from sklearn.linear_model import LinearRegression

param_grid = {}
predict_prize('West', 'conventional', LinearRegression(), param_grid);
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    'est2__n_estimators':[100, 500, 1000, 2000],
    'est2__learning_rate':[0.05, 0.1, 0.5],
    'est2__min_samples_split': [0.1, 0.5, 2]
}

predict_prize('West', 'conventional', GradientBoostingRegressor(), param_grid, to_ravel_y=True);
from sklearn.neighbors import KNeighborsRegressor

param_grid = {
    'est2__n_neighbors':[1, 2, 3, 4, 5],
    'est2__weights':['uniform', 'distance'],
    'est2__algorithm':['ball_tree', 'kd_tree', 'brute', 'auto'],
    'est2__leaf_size':[2, 5, 10, 20]
}

predict_prize('West', 'conventional', KNeighborsRegressor(), param_grid);
param_grid = {
    'est2__alpha':[0]
}
predict_prize('West', 'conventional', Ridge(), param_grid);
algoritm_chosen = LinearRegression()
types_chosen = data.type.unique()
regions_chosen = data.region.unique()

conventional_dict = {}
for region in regions_chosen:
    model, mae, score = predict_prize(region, 'conventional', algoritm_chosen, {}, to_print=False)
    conventional_dict[region] = (mae, score)
    
organic_dict = {}
for region in regions_chosen:
    model, mae, score = predict_prize(region, 'organic', algoritm_chosen, {}, to_print=False)
    organic_dict[region] = (mae, score)
conventional_df = pd.DataFrame(conventional_dict).T
conventional_df.columns = ['mae', 'score']
conventional_df['type'] = 'conventional'

organic_df = pd.DataFrame(organic_dict).T
organic_df.columns = ['mae', 'score']
organic_df.columns.names = ['organic']
organic_df['type'] = 'organic'

organic_df.head()
summary_df = pd.concat([conventional_df, organic_df], axis = 0).sort_index().reset_index()
summary_df.columns = ['region', 'mae', 'score', 'type']

marker_sizes = data.groupby(['region', 'type'], as_index=False)[['Total Volume']].mean()
summary_df = pd.merge(summary_df, marker_sizes, how='inner', on=['region', 'type'])
summary_df['log10(Total Volume)'] = np.log10(summary_df['Total Volume'])

summary_df.head()
fig, ax = plt.subplots(1, 2, figsize = (12,5))
sns.swarmplot(x = 'type', y = 'mae', data = summary_df, ax = ax[0])
sns.swarmplot(x = 'type', y = 'score', data = summary_df, ax = ax[1]);
ax[0].set_title('"Mean Absolute Error" in different "regions"')
ax[1].set_title('"Coefficient of determination R^2" in different "regions"');
x_lim = (0, 0.2)
y_lim = (0, 1)

fig, ax = plt.subplots(figsize = (8,8))
g = sns.scatterplot(x = 'mae', y = 'score', 
                hue = 'type', data = summary_df,
                sizes=(20, 200),
                ax = ax, size = 'log10(Total Volume)');
plt.legend(loc = 'lower right')
ax.set_ylim(y_lim)
ax.set_xlim(x_lim)
ax.set_title(
'''Relationship between "Mean Absolute Error" and "Coefficient of determination R^2" in different "regions"
discrimined between "type" and "Total Volume" of avocados sold in ''');
g = (sns.jointplot(x = 'mae', y = 'score', data = summary_df[summary_df.type == 'conventional'],
                   kind='scatter', height=8, xlim=x_lim, ylim=y_lim, 
                  marginal_kws = dict(bins=15)).plot_joint(sns.kdeplot, zorder=0, n_levels=5))

plt.suptitle('''Relationship between "mae" and "score" for  "conventional" type''', x = 0.5, y = 1.0, size = 15);
g = (sns.jointplot(x = 'mae', y = 'score', data = summary_df[summary_df.type == 'organic'],
                   kind='scatter', height=8, xlim=x_lim, ylim=y_lim, 
                  marginal_kws = dict(bins=20)).plot_joint(sns.kdeplot, zorder=0, n_levels=5))

plt.suptitle('''Relationship between "mae" and "score" for  "organic" type''', x = 0.5, y = 1.0, size = 15);