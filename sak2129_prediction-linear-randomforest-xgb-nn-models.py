# Basic Modules
import numpy as np
import pandas as pd

# Charting modules
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d, NumeralTickFormatter, Legend
from bokeh.plotting import output_notebook, figure, show
from bokeh.palettes import Category20
output_notebook()

# Statistics
from statsmodels.tsa.stattools import adfuller, kpss

# Pre-processing
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb
# Import sales
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
sales.head(3)
# Compute monthly values that can be added to the items dataframe
monthly_item_sales = sales.groupby(['date_block_num', 'item_id'])[['item_cnt_day']].sum().reset_index()

monthly_item_min = monthly_item_sales.groupby(['item_id'])[['item_cnt_day']].min().reset_index()
monthly_item_min.rename(columns={'item_cnt_day':'Monthly Minimum (Items)'}, inplace=True)
monthly_item_mean= monthly_item_sales.groupby(['item_id'])[['item_cnt_day']].mean().reset_index()
monthly_item_mean.rename(columns={'item_cnt_day':'Monthly Average (Items)'}, inplace=True)
monthly_item_max= monthly_item_sales.groupby(['item_id'])[['item_cnt_day']].max().reset_index()
monthly_item_max.rename(columns={'item_cnt_day':'Monthly Maximum (Items)'}, inplace=True)

monthly_items = pd.merge(monthly_item_min, monthly_item_mean, on='item_id', how='left', sort=False)
monthly_items = pd.merge(monthly_items, monthly_item_max, on='item_id', how='left', sort=False)
# Import items and add monthly minimum, mean and maximum values which will be useful later
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
items = pd.merge(items, monthly_items, on='item_id', how='left', sort=False)
items.drop(columns=['item_name'], inplace=True)
items.head(3)
# Import categories

# Since I have translated and stored the english language categories, we will use that file instead
categories = pd.read_csv('../input/futuresaleswithenglishtranslation/categories_english.csv')
categories['category']=categories['item_category_english'].str.split(' - ').str[0]
categories['subcategory']=categories['item_category_english'].str.split(' - ').str[1]
categories.drop(columns=['item_category_name', 'item_category_english'], inplace=True)

# Replace values that should be identical. I don't see specific sub-category items that can be combined
categories['category'].replace(to_replace='Movies', value='Movie', inplace=True)
categories['category'].replace(to_replace='Programs', value='Program', inplace=True)
categories['category'].replace(to_replace='Payment cards', value='Payment card', inplace=True)
categories['category'].replace(to_replace='Игры', value='Games', inplace=True)
categories.head(3)
# Import shops

# I have also translated the shops info, so we will use the English version instead
shops = pd.read_csv('../input/futuresaleswithenglishtranslation/shops_english.csv')
shops['shop_name_clean'] = shops['shop_name_english'].str.replace('! ', '')
shops['shop_name_clean'].replace(to_replace='Shop Online Emergencies', value='ShopOnline Emergencies', inplace=True)
shops['shop_name_clean'].replace(to_replace='St. Petersburg TK "Nevsky Center"', value='St.Petersburg', inplace=True)
shops['shop_name_clean'].replace(to_replace='St. Petersburg TK "Sennaya"', value='St.Petersburg', inplace=True)
shops['city']=shops['shop_name_clean'].str.split(' ').str[0]
shops.drop(columns=['shop_name', 'shop_name_english', 'shop_name_clean'], inplace=True)
# Compute monthly values that can be added to the items dataframe
monthly_shop_sales = sales.groupby(['date_block_num', 'shop_id'])[['item_cnt_day']].sum().reset_index()

monthly_shop_min = monthly_shop_sales.groupby(['shop_id'])[['item_cnt_day']].min().reset_index()
monthly_shop_min.rename(columns={'item_cnt_day':'Monthly Minimum (Shop)'}, inplace=True)
monthly_shop_mean= monthly_shop_sales.groupby(['shop_id'])[['item_cnt_day']].mean().reset_index()
monthly_shop_mean.rename(columns={'item_cnt_day':'Monthly Average (Shop)'}, inplace=True)
monthly_shop_max= monthly_shop_sales.groupby(['shop_id'])[['item_cnt_day']].max().reset_index()
monthly_shop_max.rename(columns={'item_cnt_day':'Monthly Maximum (Shop)'}, inplace=True)

monthly_shop = pd.merge(monthly_shop_min, monthly_shop_mean, on='shop_id', how='left', sort=False)
monthly_shop = pd.merge(monthly_shop, monthly_shop_max, on='shop_id', how='left', sort=False)

shops = pd.merge(shops, monthly_shop, on='shop_id', how='left', sort=False)
shops.head(3)
# Import test
test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
test.head(3)
# Print column names
print('categories columns:', list(categories.columns))
print('items columns:', list(items.columns))
print('sales columns:', list(sales.columns))
print('shops columns:', list(shops.columns))
print('test columns:', list(test.columns))
# Merge the dataframes to create a combined dataset
all_items = pd.merge(items, categories, how='left', on='item_category_id', sort=False)
all_sales = pd.merge(sales, shops, how='left', on='shop_id', sort=False)
data = pd.merge(all_sales, all_items, how='left', on='item_id', sort=False)
data['date'] = pd.to_datetime(data['date'])
data['YearMonth']=pd.to_datetime(data['date']).dt.to_period('m')
data['shop-item']= data['shop_id'].astype(str) + '-' + data['item_id'].astype(str)
data = data.sort_values(by=['date'])

data.head(3)
# Plotting the number of unique items sold on any given day
p = figure(plot_width=800, plot_height=400, title='# of Transactions',x_axis_label='Date', 
           y_axis_label='# of Transactions',x_axis_type='datetime')
vc = data['date'].value_counts()
p.vbar(x=vc.index, top=vc.values, width=0.5)
show(p)
# Monthly data

# first we aggregate the item count by month, then pivot on shop and item
monthlydata = data.groupby(['YearMonth', 'shop-item'])[['item_cnt_day']].sum()
monthlydata.reset_index(inplace=True)
monthlypivot = monthlydata.pivot(index='YearMonth', columns='shop-item', values='item_cnt_day').fillna(0)
# Monthly Data
monthly = monthlypivot.sum(axis=1)

x = list(monthly.index.astype(str))
top = monthly.values/1000
label = top.astype(int).astype(str)

source = ColumnDataSource(data=dict(x=x, top=top,label=label))

p = figure(plot_width=800, plot_height=450, x_range=x, title='Monthly Sales Quantity (in thousands)',x_axis_label='Time', 
           y_axis_label='Sales (in 1000s)', y_range=(0, 210))
p.vbar(x='x', top='top', width=0.8, source=source)
p.xaxis.major_label_orientation = 365

labels = LabelSet(x='x', y='top', text='label', level='glyph',x_offset=10, y_offset=2, source=source, render_mode='canvas', 
                  angle=3.14/2)
p.add_layout(labels)
p.xgrid.visible = False; p.ygrid.visible = False
show(p)
# Exploring data by category
cat_items = data.groupby(['YearMonth', 'category'])[['item_cnt_day']].sum().reset_index()

cat_pivot = cat_items.pivot(index='YearMonth', columns='category', values='item_cnt_day').fillna(0)
cat_pivot.columns.sort_values()
# Total items sold by catgegory for each month
cat_pivot.head(3)
# Compute the correlation matrix
corr = cat_pivot.corr()

#Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, center=0, cmap='RdYlGn')
# Top correlations positive and negative
corr_mtx = corr.reset_index()
corr_df = corr_mtx.melt(id_vars=['category'], var_name='category_name', value_name='Correlation')
corr_df = corr_df[corr_df['Correlation']<1].sort_values(by='Correlation', ascending=False)
coff_df_noduplicates = corr_df.drop_duplicates(subset='Correlation', keep='first', inplace=False)

# We can print the positive and negative correlations between the various product category names
coff_df_noduplicates.head(10)
coff_df_noduplicates.tail(10)
p = figure(plot_width=800, plot_height=400, title='Items sold by Category', x_axis_label='Item Category ID', 
           y_axis_label='Items Sold', x_axis_type='datetime')
p.yaxis.formatter=NumeralTickFormatter(format="0,0")
p.add_layout(Legend(), 'right')
p.legend.click_policy="hide"
p.legend.label_text_font_size='8pt'
for n,c in zip(range(len(cat_pivot.columns)),Category20[20]):
    x = cat_pivot.index
    y = cat_pivot.iloc[:,n]
    p.line(x=x,y=y, color=c, line_width=3, legend_label=cat_pivot.columns[n])

show(p)
# Add a row for the last month that will be used for predictions
monthlypivot.loc['2016-01']=0
monthlypivot.tail(3)
# Since a vast majority of items have not been sold recently, we identify items that were recently sold
retention_months = 6
sale_quantity = 100

sales2015 = monthlypivot.tail(retention_months).sum()
shop_items2015 = sales2015[sales2015>sale_quantity]
importantitems = shop_items2015.index.values

# Only retain shop-item IDs that are important
importantsales = monthlypivot[importantitems]
# Unpivot dataframe
melted = importantsales.reset_index()
melted = melted.melt(id_vars='YearMonth', var_name='shop_item', value_name='item_cnt_day')
melted['YearMonth']=melted['YearMonth'].astype('str')

# Separate out shop and item so other features can be added
melted['shop_id'] = melted['shop_item'].str.split('-').str[0].astype(int)
melted['item_id'] = melted['shop_item'].str.split('-').str[1].astype(int)
melted['Year']=melted['YearMonth'].str.split('-').str[0].astype(int)
melted['Month']=melted['YearMonth'].str.split('-').str[1].astype(int)

# Add other important information
items_and_categories = pd.merge(items, categories, how='left', on='item_category_id', sort=False)
melted = pd.merge(melted, items_and_categories, how='left', on='item_id', sort=False)
melted = pd.merge(melted, shops, how='left', on='shop_id', sort=False)
# Add lags of data
months = melted['YearMonth'].astype('str').unique()
products = melted['shop_item'].unique()

lag_cols = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10', 'lag11', 'lag12']
melted[lag_cols]=0
for l in range(len(lag_cols)):
    for p in range(len(products)):
        for m in range(len(months)-l-1):
            val = melted.loc[(melted['shop_item']==products[p]) & (melted['YearMonth']==months[m]), 'item_cnt_day'].values
            melted.loc[(melted['shop_item']==products[p]) & (melted['YearMonth']==months[m+l+1]), lag_cols[l]] = val
melted.head(3)
# Prepare final dataset
modeldata = melted.copy()
modeldata['YearMonth'] = modeldata['YearMonth'].astype(str)
modeldata.set_index(['YearMonth', 'shop_item'], inplace=True, drop=True)
modeldata = modeldata[modeldata['Year']>=2014]

modeldata.drop(columns=['shop_id', 'item_id', 'Year', 'Month', 'item_category_id'], inplace=True)
modeldata['category']=modeldata['category'].astype('category')
modeldata['subcategory']=modeldata['subcategory'].astype('category')
modeldata['city']=modeldata['city'].astype('category')
modeldata.info()
# Create dummy variables for categorical 
modeldata_dummies = pd.get_dummies(modeldata)
modeldata_dummies.head(3)
def export_results(results, outputfilename):
    ind = results.index
    shopid = ind.str.split('-').str[0].astype('int64')
    itemid = ind.str.split('-').str[1].astype('int64')

    output = pd.DataFrame({'shop_id':shopid, 'item_id':itemid, 'item_cnt_month':results.values})
    res = pd.merge(test, output, how='left', on=['shop_id', 'item_id'], sort=False)

    result = res['item_cnt_month'].fillna(0)
    result.index = res['ID']
    filename = outputfilename + '.csv'
    result.to_csv(filename, header=True)
def test_RMSE(y_pred, y_actual):
    error = y_pred-y_actual
    squared_error = error*error
    mean_squared_error = np.mean(squared_error)
    rmse = np.sqrt(mean_squared_error)
    print('RMSE value: ', round(rmse,2))
    return rmse
# Use the most recent month's data as the prediction for the upcoming month
y_pred = monthlypivot.iloc[-3,:]
y_actual=monthlypivot.iloc[-2,:]
rm = test_RMSE(y_pred, y_actual)
# The December 2015 prediction is the next prediction
results = monthlypivot.iloc[-2,:]
#export_results(results = results, outputfilename = 'naive')
train_data = monthlypivot.iloc[0:len(monthlypivot)-2,:]
y_pred = train_data.mean()
y_actual=monthlypivot.iloc[-2,:]
rm = test_RMSE(y_pred, y_actual)
results = monthlypivot.iloc[0:len(monthlypivot)-1,:].mean()
#export_results(results = results, outputfilename = 'meanmodel')
y_pred = monthlypivot.iloc[0]/3+monthlypivot.iloc[12]/3+monthlypivot.iloc[24]/3
y_actual=monthlypivot.iloc[-2,:]
rm = test_RMSE(y_pred, y_actual)
results = monthlypivot.iloc[0]/3+monthlypivot.iloc[12]/3+monthlypivot.iloc[24]/3
#export_results(results = results, outputfilename = 'pastjan')
# Undertake prediction on most sold items only; rest of the items will be predicted as zero
train_data = modeldata_dummies.drop(index=['2015-12','2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']
test_data = modeldata_dummies.loc[('2015-12'),:]
test_x = test_data.drop(columns=['item_cnt_day'])
test_y = test_data['item_cnt_day']
model = Lasso()
scaler=StandardScaler()

# Scale the x-variables
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

model.fit(train_x_scaled, train_y)
res = model.predict(test_x_scaled)

prediction = pd.Series(res, index=test_y.index)

y_pred = monthlypivot.iloc[-3,:]
y_pred.update(prediction)

y_actual = monthlypivot.iloc[-2,:]
rm = test_RMSE(y_pred, y_actual)
lasso_coefficients = pd.DataFrame({'Variable':train_x.columns, 'Coefficients':model.coef_}).sort_values(by='Coefficients',ascending=False)
lasso_coefficients.head(20)
# Lasso submission
train_data = modeldata_dummies.drop(index=['2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']

test_data = modeldata_dummies.loc[('2016-01'),:]
test_x = test_data.drop(columns=['item_cnt_day'])

model = Lasso()
scaler=StandardScaler()

# Scale the x-variables
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

model.fit(train_x_scaled, train_y)
res = model.predict(test_x_scaled)

prediction = pd.Series(res, index=test_y.index)

results = monthlypivot.iloc[-2,:]
results.update(prediction)
#export_results(results = results, outputfilename = 'Lasso')
# Undertake prediction on most sold items only; rest of the items will be predicted as zero
train_data = modeldata_dummies.drop(index=['2015-12','2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']

test_data = modeldata_dummies.loc[('2015-12'),:]
test_x = test_data.drop(columns=['item_cnt_day'])
test_y = test_data['item_cnt_day']

model = RandomForestRegressor()
model.fit(train_x, train_y)
res = model.predict(test_x)

prediction = pd.Series(res, index=test_y.index)

y_pred = monthlypivot.iloc[-3,:]
y_pred.update(prediction)

y_actual = monthlypivot.iloc[-2,:]
rm = test_RMSE(y_pred, y_actual)
# Random Forest submission
train_data = modeldata_dummies.drop(index=['2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']

test_data = modeldata_dummies.loc[('2016-01'),:]
test_x = test_data.drop(columns=['item_cnt_day'])

model = RandomForestRegressor()
scaler=StandardScaler()

# Scale the x-variables
model.fit(train_x, train_y)
res = model.predict(test_x)

prediction = pd.Series(res, index=test_y.index)

results = monthlypivot.iloc[-2,:]
results.update(prediction)
#export_results(results = results, outputfilename = 'RandomForest')
# Undertake prediction on most sold items only; rest of the items will be predicted as zero
train_data = modeldata_dummies.drop(index=['2015-12','2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']

test_data = modeldata_dummies.loc[('2015-12'),:]
test_x = test_data.drop(columns=['item_cnt_day'])
test_y = test_data['item_cnt_day']

model = xgb.XGBRegressor()
model.fit(train_x, train_y)
res = model.predict(test_x)

prediction = pd.Series(res, index=test_y.index)

y_pred = monthlypivot.iloc[-3,:]
y_pred.update(prediction)

y_actual = monthlypivot.iloc[-2,:]
rm = test_RMSE(y_pred, y_actual)
# XGB submission
train_data = modeldata_dummies.drop(index=['2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']

test_data = modeldata_dummies.loc[('2016-01'),:]
test_x = test_data.drop(columns=['item_cnt_day'])

model = RandomForestRegressor()
scaler=StandardScaler()

# Scale the x-variables
model.fit(train_x, train_y)
res = model.predict(test_x)

prediction = pd.Series(res, index=test_y.index)

results = monthlypivot.iloc[-2,:]
results.update(prediction)
#export_results(results = results, outputfilename = 'XGBoost')
# Prepare data using dataframes
train_data = modeldata_dummies.drop(index=['2015-12','2016-01'],level=0)
train_x = train_data.drop(columns=['item_cnt_day'])
train_y = train_data['item_cnt_day']

test_data = modeldata_dummies.loc[('2015-12'),:]
test_x = test_data.drop(columns=['item_cnt_day'])
test_y = test_data['item_cnt_day']
# Convert dataframes into numpy arrays that can be formulated as tensors
train_x_np = np.array(train_x)
train_y_np = np.array(train_y)

test_x_np = np.array(test_x)
test_y_np = np.array(test_y)
# Normalize the tran and test data
mean = train_x_np.mean(axis=0)
train_x_np -= mean

std = train_x_np.std(axis=0)
train_x_np /= std

test_x_np -= mean
test_x_np /= std
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_x_np.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
import numpy as np

k=4
num_val_samples = len(train_x_np) //k
num_epochs = 19
all_scores = []
train_loss_histories = []
val_loss_histories = []

for i in range(k):
    print('Processing batch#', i,'of', k-1)
    val_data = train_x_np[i*num_val_samples: (i+1)*num_val_samples]
    val_targets=train_y_np[i*num_val_samples: (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate([train_x_np[:i*num_val_samples],train_x_np[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_y_np[:i*num_val_samples], train_y_np[(i+1)*num_val_samples:]], axis=0)
    
    model=build_model()
    history=model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs,
                     batch_size=1, verbose=0)
    
    train_loss=history.history['loss']
    val_loss = history.history['val_loss']
    train_loss_histories.append(train_loss)
    val_loss_histories.append(val_loss)
avg_train_loss_history = [np.mean([x[i] for x in train_loss_histories]) for i in range(num_epochs)]
avg_val_loss_history = [np.mean([x[i] for x in val_loss_histories]) for i in range(num_epochs)]
p = figure(plot_width=950, plot_height=400, x_axis_label='Epochs', y_axis_label='Loss', title='Training & Validation Loss')
x = range(1,len(avg_train_loss_history)+1)
p.line(x, avg_train_loss_history, line_width=3, legend_label='Training Loss')
p.line(x, avg_val_loss_history, color='green', line_width=3, legend_label='Validation Loss')
show(p)
# Neural Network Submission
res = model.predict(test_x_np)

prediction = pd.Series(res[:,0], index=test_y.index)

results = monthlypivot.iloc[-2,:]
results.update(prediction)
export_results(results = results, outputfilename = 'NeuralNetwork')