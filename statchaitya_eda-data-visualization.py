import pandas as pd
import time
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize

input_dir = "../input/"

train = pd.read_csv(input_dir+"train.csv")
test = pd.read_csv(input_dir+"test.csv")

def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    json_df_dict = dict()
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        json_df_dict[column] = column_as_df
    
    json_df = pd.concat(json_df_dict.values(), axis=1)
    json_df = pd.merge(df, json_df, right_index=True, left_index=True)
    return json_df

start_time = time.time()
train_df = load_df(input_dir+"train.csv", nrows=10000000)
print("load_df took", (time.time() - start_time)/60, "minutes for TRAIN data")


#train_df.to_csv(input_dir+"train_new.csv",  index=False)

start_time = time.time()
test_df = load_df(input_dir+"test.csv", nrows=10000000)
print("load_df took", (time.time() - start_time)/60, "minutes for TEST data")
#test_df.to_csv(input_dir+"test_new.csv",  index=False)-
source_viz = train_df.groupby('source').size().sort_values(ascending=False).reset_index(name="num_sessions").head(25)

sns.set(style="darkgrid")
f, ax=plt.subplots(figsize=(15,7))
barplot_1 = sns.barplot(x='source', y='num_sessions', data=source_viz)
barplot_1.axes.set_title("Top 25 Traffic Sources",fontsize=20)
barplot_1.set_xlabel("Source",fontsize=15)
barplot_1.set_ylabel("Number of Sessions",fontsize=15)
barplot_1.tick_params(labelsize=12.5)
sns.despine(offset=15, trim=True)
for item in barplot_1.get_xticklabels():
    item.set_rotation(90)
import warnings
warnings.filterwarnings('ignore')

train_df['transactionRevenue'] = train_df.transactionRevenue.astype(float)
train_df['transactionRevenue'] = train_df['transactionRevenue'].fillna(0)

var='source'
source_logmeantr = train_df.groupby(var).mean()['transactionRevenue'].reset_index(name='TransRevenueMean')
source_logmeantr_t25 = source_logmeantr.loc[source_logmeantr['source'].isin(source_viz.iloc[:,0])]
source_logmeantr_t25['source'] = pd.Categorical(
    source_logmeantr_t25['source'], 
    categories=list(source_viz['source']), 
    ordered=True
)
source_logmeantr_t25.sort_values('source', inplace=True)

f, ax=plt.subplots(figsize=(15,7))
#sns.set(style="darkgrid")
barplot_3 = sns.barplot(x='source', y='TransRevenueMean', data=source_logmeantr_t25)
barplot_3.axes.set_title("Mean Transaction Revenue (Top 25 Sources)",fontsize=20)
barplot_3.set_xlabel("Source",fontsize=15)
barplot_3.set_ylabel("Mean Trans. Revenue",fontsize=15)
barplot_3.tick_params(labelsize=12.5)
sns.despine(offset=15, trim=True)
for item in barplot_3.get_xticklabels():
    item.set_rotation(90)
source_SessConvRate = (round(((train_df[train_df['transactionRevenue'] > 0].groupby(['source']).size())/train_df.groupby('source').size())*100,2)).reset_index(name="sessionConversionRate")
source_SessConvRate_t25 = source_SessConvRate.loc[source_SessConvRate['source'].isin(source_viz.iloc[:,0])]
source_SessConvRate_t25['source'] = pd.Categorical(
    source_SessConvRate_t25['source'], 
    categories=list(source_viz['source']), 
    ordered=True
)
source_SessConvRate_t25.sort_values('source', inplace=True)

f, ax=plt.subplots(figsize=(15,7))
#sns.set(style="whitegrid")
barplot_2 = sns.barplot(x='source', y='sessionConversionRate', data=source_SessConvRate_t25)
barplot_2.axes.set_title("Session Conversion Rate in % per Source",fontsize=20)
barplot_2.set_xlabel("Source",fontsize=15)
barplot_2.set_ylabel("Session Conversion Rate in %",fontsize=15)
barplot_2.tick_params(labelsize=12.5)
sns.despine(offset=5, trim=True)
for item in barplot_2.get_xticklabels():
    item.set_rotation(90)
var='medium'
medium_viz = train_df.groupby(var).size().reset_index(name='num_sessions')

f, ax=plt.subplots(figsize=(15,7))
#sns.set(style="whitegrid")
barplot_4 = sns.barplot(x='medium', y='num_sessions', data=medium_viz)
barplot_4.axes.set_title("Number of Sessions per Medium",fontsize=20)
barplot_4.set_xlabel("Medium",fontsize=15)
barplot_4.set_ylabel("Number of Sessions",fontsize=15)
barplot_4.tick_params(labelsize=12.5)
sns.despine(offset=5, trim=True)
for item in barplot_4.get_xticklabels():
    item.set_rotation(45)
var='medium'
medium_viz = train_df.groupby(var).mean()['transactionRevenue'].reset_index(name='TransRevenueMean')

f, ax=plt.subplots(figsize=(15,7))
#sns.set(style="whitegrid")
barplot_2 = sns.barplot(x='medium', y='TransRevenueMean', data=medium_viz)
barplot_2.axes.set_title("Mean of Transaction Revenue per Medium",fontsize=20)
barplot_2.set_xlabel("Medium",fontsize=15)
barplot_2.set_ylabel("Mean Transaction Revenue",fontsize=15)
barplot_2.tick_params(labelsize=12.5)
sns.despine(offset=5, trim=True)
for item in barplot_2.get_xticklabels():
    item.set_rotation(45)
medium_SessConvRate = (round(((train_df[train_df['transactionRevenue'] > 0].groupby(['medium']).size())/train_df.groupby('medium').size())*100,2)).reset_index(name="sessionConversionRate")

f, ax=plt.subplots(figsize=(15,7))
#sns.set(style="whitegrid")
barplot_2 = sns.barplot(x='medium', y='sessionConversionRate', data=medium_SessConvRate)
barplot_2.axes.set_title("Session Conversion Rate in % per Medium",fontsize=20)
barplot_2.set_xlabel("Medium",fontsize=15)
barplot_2.set_ylabel("Session Conversion Rate in %",fontsize=15)
barplot_2.tick_params(labelsize=12.5)
sns.despine(offset=5, trim=True)
for item in barplot_2.get_xticklabels():
    item.set_rotation(45)