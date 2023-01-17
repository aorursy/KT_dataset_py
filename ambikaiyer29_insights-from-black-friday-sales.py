import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
pd.set_option("display.max_rows",1001)
pd.set_option("display.max_columns",300)
bfd = pd.read_csv("../input/BlackFriday.csv")
bfd.head()
grp = bfd.groupby(['User_ID','Product_ID']).count()['Purchase'].reset_index()
grp[grp.Purchase>1]
grp = bfd.groupby(['Gender']).nunique()['User_ID']
grp.apply(lambda x : 100*x/grp.sum()).reset_index()
grp_city = bfd.groupby(['Gender','City_Category']).nunique()['User_ID'].reset_index()
pivot_city = grp_city.pivot(index='City_Category',columns='Gender',values='User_ID')
colors = ["#006D2C", "#31A354"]
pivot_city.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Gender by City_category')
grp_city = bfd.groupby(['Gender','City_Category']).sum()['Purchase'].reset_index()
pivot_city = grp_city.pivot(index='City_Category',columns='Gender',values='Purchase')
pivot_city.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Purchase value by Gender and City_category')
grp_age = bfd.groupby(['Gender','Age']).nunique()['User_ID'].reset_index()
pivot_age = grp_age.pivot(index='Age',columns='Gender',values='User_ID')
pivot_age.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Gender by Age')
grp_occ = bfd.groupby(['Gender','Occupation']).nunique()['User_ID'].reset_index().sort_values(by='User_ID',ascending=False)
pivot_occ = grp_occ.pivot(index='Occupation',columns='Gender',values='User_ID')
pivot_occ.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Gender by Occupation')
grp_occ = bfd.groupby(['Gender','Occupation']).sum()['Purchase'].reset_index()
pivot_occ = grp_occ.pivot(index='Occupation',columns='Gender',values='Purchase')
pivot_occ.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Purchase amount by Gender and Occupation')
age_prd = bfd.groupby(['Age','Gender','Product_ID']).count()['User_ID'].reset_index().sort_values(by=['Age','Gender','User_ID'],ascending=False).reset_index(drop=True)
top_20 = age_prd.groupby(['Age','Gender'])['Product_ID','User_ID'].apply(lambda x : x.head(20))
top_20 = top_20.reset_index()
top_20.dtypes
#top_5.head()
pivot_top_20 = top_20.pivot_table(index='Product_ID',columns=['Gender','Age'],values='User_ID',fill_value=0)
pivot_top_20
