########### Library Imports #######
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns ## Data plotting
print(os.listdir("../input"))
train_df = train_df = pd.read_csv("../input/train.csv")
train_df.head()

# Any results you write to the current directory are saved as output.
train_df['date'] = pd.to_datetime(train_df['date'],format='%Y-%m-%d')
start_date = train_df['date'].min()
end_date = train_df['date'].max()
number_of_stores = len(train_df['store'].unique().tolist())
number_of_items = len(train_df['item'].unique().tolist())
print("start_date",start_date,"\n end_date",end_date,"\n number_of_stores",number_of_stores,"\n number_of_items",number_of_items)
train_df['year'] = train_df['date'].dt.year
total_sales_per_store_per_year = train_df.groupby(['store','year'])['sales'].sum().reset_index()

sns.catplot(x= 'store',y='sales',data=total_sales_per_store_per_year,hue='year',aspect= 1.5,kind='bar')
mask = (((train_df['store'] == 1) | (train_df['store'] == 7)) & (train_df['year'] == 2017))
train_df  = train_df[mask]
train_df['date1'] = train_df['date'].dt.strftime("%b")
train_df_weekly = train_df.groupby(['store','date1'],as_index=False)['sales'].sum()
sort_order = train_df['date1'].unique().tolist()
g = sns.catplot(x= 'date1',y='sales',data=train_df_weekly,hue='store',aspect= 3,kind='bar',height = 5,order=sort_order)
g.set_xticklabels(rotation=70)
