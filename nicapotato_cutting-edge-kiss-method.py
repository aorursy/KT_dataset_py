# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Viz
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Import data
sales = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/shops.csv')
items = pd.read_csv('../input/items.csv')
cats = pd.read_csv('../input/item_categories.csv')
val = pd.read_csv('../input/test.csv')

# Rearrange the raw data to be monthly sales by item-shop
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df["item_cnt_day"].clip(0.,20.,inplace=True)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

# Merge data from monthly sales to specific item-shops in test data
df = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

# Centralize results for Analysis
main = pd.DataFrame()
# 1.16777 LB
previous_month = pd.DataFrame(df.iloc[:,-1].values,columns=['item_cnt_month'])
main["previous_month"] = df.iloc[:,-1].values
previous_month.to_csv('previous_month.csv',index_label='ID')
previous_month.head()
# 1.07641 LB
pred = df.iloc[:,-3:].mean(axis=1)
main["threemonthmean"] = pred
threemonthmean = pd.DataFrame(pred.values,columns=['item_cnt_month'])
threemonthmean.to_csv('threemonthmean.csv',index_label='ID')
threemonthmean.head()
# 1.07641 LB
pred = df.iloc[:,-3:].mean(axis=1)
main["twomonthmean"] = pred
twomonthmean = pd.DataFrame(pred.values,columns=['item_cnt_month'])
twomonthmean.to_csv('twomonthmean.csv',index_label='ID')
twomonthmean.head()
# 1.07775 LB
pred = df.iloc[:,-5:].mean(axis=1)
main["fivemonthmean"] = pred
fivemonthmean = pd.DataFrame(pred.values,columns=['item_cnt_month'])
fivemonthmean.to_csv('fivemonthmean.csv',index_label='ID')
fivemonthmean.head()
df.head()
# 1.16435 LB
pred = df.iloc[:,4:].mean(axis=1)
main["fullmean"] = pred
fullmean = pd.DataFrame(pred.values,columns=['item_cnt_month'])
fullmean.to_csv('fullmean.csv',index_label='ID')
fullmean.head()
# 1.10767 LB
pred = df.iloc[:,-2:].median(axis=1)
main["median"] = pred
median = pd.DataFrame(pred.values,columns=['item_cnt_month'])
median.to_csv('median.csv',index_label='ID')
median.head()
# Blending
essemble_df = main[["threemonthmean","twomonthmean","fivemonthmean","median"]]
# 1.06578 LB
pred =essemble_df.mean(axis=1)
main["mean_essemble"] = pred
mean_essemble = pd.DataFrame(pred.values,columns=['item_cnt_month'])
mean_essemble.to_csv('mean_essemble.csv',index_label='ID')
mean_essemble.head()
main_melt = pd.melt(main)
sns.boxplot(data = main_melt, x="value",y="variable")
plt.title("Distribution of Submissions")
plt.xlabel("Predict Items Sold")
plt.xlim(-0.5, 2)
plt.show()
