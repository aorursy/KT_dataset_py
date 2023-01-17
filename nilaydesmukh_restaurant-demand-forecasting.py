import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_row',None)

sns.set_style('darkgrid') 
df = pd.read_excel('../input/restaurant-dataxlsx/Data.xlsx')

df.head()
# Dropping unnecessary columns

df = df.drop(['StoreCode','DTS','Month','Date','Year','TicketCode'],axis = 1)

df.head()
# Cheking data type of all columns

df.info()
# Replacing null by '0' as there is no peoples when food orderd online.

df['PartySize'] = df['PartySize'].replace(['na'],0)

df.head()
# Normalizing text 

df['MenuCateogry'] = df['MenuCateogry'].str.capitalize()

df['MenuItem'] = df['MenuItem'].str.capitalize()

df.head()
# Dividing data based on Weekday

weekday = df[df['Day Type']=='Weekday']

print(weekday.shape)

weekday.head()
# Dividing data based on Weekend

weekend = df[df['Day Type']=='Weekend']

print(weekend.shape)

weekend.head()
# Monday - WeekDay

monday = weekday.copy()

monday['Day'] = weekday['Day'].str.replace('Tuesday','Monday') 



# Tuseday - WeekDay

tuesday = weekday.copy() 



# Wednesday - WeekEnd

wednesday = weekend.copy()



# Thusday - WeekDay

thursday = weekday.copy()                                         

thursday['Day'] = thursday['Day'].str.replace('Tuesday','Thursday')



# Friday - WeekDay

friday = weekday.copy()                                          

friday['Day'] = friday['Day'].str.replace('Tuesday','Friday')



# Saturday - WeekEnd

saturday = weekend.copy()                                        

saturday['Day'] = saturday['Day'].str.replace('Wednesday','Saturday')



# Sunday - WeekEnd

sunday = weekend.copy()                                          

sunday['Day'] = sunday['Day'].str.replace('Wednesday','Sunday')
# Creating data for one week 

week = []

week = pd.concat([tuesday,wednesday,thursday,friday,saturday,sunday,monday,],axis = 0)

print(week.shape)

week.head()
# Creating data for 6 Months with help of above data

months = week.copy()

x = 0

while x < 25:

    months = pd.concat([months,week],axis = 0)

    x = x+1

months.reset_index(drop=True, inplace=True)

print(months.shape)

months.head(10)
months.info()
# Creating dates for dataframe.

o = pd.date_range(start='1/1/2019', periods=(len(months)/100), freq='D')

date = []

for i in o:

    for j in range(100):

        date.append(i)

date = pd.DataFrame(date,columns = ['Date'])

date.head()
# Concating Dates and Months Dataframe

final = pd.concat([date,months],axis = 1)

final.head()
# Changing Columns Postions for better understanding

final = final[['Date', 'Shift', 'Day Type', 'Day', 'PartySize', 'MenuCateogry','MenuItem', 'ItemPrice', 'ItemQty']]

final = final.iloc[:18100,:]

final.head()
final.info()
df  = final

df.head()
# Extracting Two Columns for visualization purpose.

product_1 = df[['MenuItem','ItemQty']]



# Combining two rows 'MenuItem' and 'ItemQty' for Analysis and multiplying based on ItemQty

product = product_1.loc[product_1.index.repeat(product_1.ItemQty)].reset_index(drop=True)

product = product[['MenuItem']]

product.head()
#pip install WordCloud  # Kindly install WordCloud package for furthur process 

# Joinining all the reviews into single paragraph  

speaker_rev_string1 = " ".join(product['MenuItem'])

from wordcloud import WordCloud

wordcloud_sp = WordCloud(width=6000,height=1800).generate(speaker_rev_string1)

plt.axis("off")

plt.tight_layout(pad=0)

plt.imshow(wordcloud_sp)
p = pd.DataFrame(product_1.groupby(['MenuItem']).sum())

p = p.reset_index()

p.sort_values(by=['ItemQty'], inplace=True,ascending=False)

plt.figure(figsize=(35,8))

chart = sns.barplot(x="MenuItem", y="ItemQty", data=p)

plt.xticks(rotation=90)
df = df[['Date','Shift','MenuItem','ItemQty']]

df.head()
# Preparing data to use to make Cross Table

new = df.loc[df.index.repeat(df.ItemQty)]

new = new[['Date','Shift','MenuItem']]

new.head(10)
# Shifting Table

table = pd.DataFrame(pd.crosstab(new.Date,[new.Shift,new.MenuItem]))

table.head()
# Normalizing Table Names.

table.columns = table.columns.map('{0[1]}-{0[0]}'.format) 

print(table.shape)

table.head()


plt.figure(figsize=(5,60))

plt.rcParams["figure.figsize"] = [35, 8]

table.plot(legend = False)

# We Can see that data follows seasonality
Train = table[:int(0.85*(len(table)))]

Test = table[int(0.85*(len(table))):]

print(Train.shape,Test.shape)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 

from statsmodels.tsa.holtwinters import Holt 

from statsmodels.tsa.holtwinters import ExponentialSmoothing 

import warnings

warnings.filterwarnings('ignore')
p = []

for i in table.columns:

    hwe_model_add_add = ExponentialSmoothing(Train[i],seasonal="add",trend="add",seasonal_periods=7).fit()

    pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])

    rmse_hwe_add_add = np.sqrt(np.mean((Test[i]-pred_hwe_add_add)**2))

    p.append(round(rmse_hwe_add_add,3))

p = pd.DataFrame(p, columns = ['Winter_Exponential_Smoothing_RMSE'])
q = []

for j in table.columns:

    hw_model = Holt(Train[j]).fit()

    pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])

    rmse_hw = np.sqrt(np.mean((Test[j]-pred_hw)**2))

    q.append(round(rmse_hw,3)) 

p['Holt method Model_RMSE']= pd.DataFrame(q, columns = ['Holt method Model_RMSE'])
r = []

for o in table.columns:

    ses_model = SimpleExpSmoothing(Train[o]).fit()

    pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])

    rmse_ses = np.sqrt(np.mean((Test[o]-pred_ses)**2))

    r.append(round(rmse_ses,3)) # 0.49

p['Simple Exponential Mode_RMSE']= pd.DataFrame(r, columns = ['Simple Exponential Mode_RMSE'])
p.head()
p.sum()
def Daily_menu_forcasting(table,start_date,end_date):

    da = pd.date_range(start = start_date, end = end_date , freq='D')

    for_pred = pd.DataFrame(da,columns = ['Date'] )

    for_pred = for_pred.set_index('Date')

    for i in table.columns:

        hwe_model_add_add = ExponentialSmoothing(table[i],seasonal="add",trend="add",seasonal_periods=7).fit()

        pred_hwe_add_add = hwe_model_add_add.predict(start = for_pred.index[0],end = for_pred.index[-1])

        for_pred[i]=((round(pred_hwe_add_add)).astype(int))

    final_pred =  for_pred

    p = pd.DataFrame(final_pred.stack())

    p = p.reset_index()

    p[['MenuItem','Shift']] = p.level_1.str.split("-",expand=True,)

    p = p.rename(columns={0: "ItemQty"})

    p = p[['Date','Shift','MenuItem',"ItemQty"]]

    p = p[p['ItemQty'] != 0]

    # Makind Dataframe with dinner and lunch columns

    new = p.loc[p.index.repeat(p.ItemQty)]

    f = pd.DataFrame(pd.crosstab([new.Date,new.MenuItem],[new.Shift]))

    f = f.reset_index()



    # Shorting Data Frame on the basis top item

    f['Total orders of Day'] = f.Dinner + f.Lunch

    f = f.sort_values(['Date', 'Total orders of Day'], ascending=[True, False]).reset_index(drop= True)

    f

    Daily_req_FiNal_Ans = f.copy()

    return Daily_req_FiNal_Ans
def Daily_top_menu_forcasting(table,start_date,end_date,N=5):

    da = pd.date_range(start = start_date, end = end_date , freq='D')

    for_pred = pd.DataFrame(da,columns = ['Date'] )

    for_pred = for_pred.set_index('Date')

    for i in table.columns:

        hwe_model_add_add = ExponentialSmoothing(table[i],seasonal="add",trend="add",seasonal_periods=7).fit()

        pred_hwe_add_add = hwe_model_add_add.predict(start = for_pred.index[0],end = for_pred.index[-1])

        for_pred[i]=((round(pred_hwe_add_add)).astype(int))

    final_pred =  for_pred

    p = pd.DataFrame(final_pred.stack())

    p = p.reset_index()

    p[['MenuItem','Shift']] = p.level_1.str.split("-",expand=True,)

    p = p.rename(columns={0: "ItemQty"})

    p = p[['Date','Shift','MenuItem',"ItemQty"]]

    p = p[p['ItemQty'] != 0]

    # Makind Dataframe with dinner and lunch columns

    new = p.loc[p.index.repeat(p.ItemQty)]

    f = pd.DataFrame(pd.crosstab([new.Date,new.MenuItem],[new.Shift]))

    f = f.reset_index()



    # Shorting Data Frame on the basis top item

    f['Total orders of Day'] = f.Dinner + f.Lunch

    f = f.sort_values(['Date', 'Total orders of Day'], ascending=[True, False]).reset_index(drop= True)

    f

    # Finding Topr product for days.

    name =((f['Date'].astype(str)).unique()).tolist()

    t = pd.DataFrame(columns = f.columns)

    for i in name:

        v = pd.DataFrame((f[f['Date']==i]).head(N))

        t = pd.concat([t,v],axis = 0)

    Daily_top_FiNal_Ans = t.reset_index(drop = True)

    return(Daily_top_FiNal_Ans)
all_menu = Daily_menu_forcasting(table,'7/1/2019','7/7/2019')

all_menu.head(10) # top manu day wise
# Here N = 8

top_8_menu = Daily_top_menu_forcasting(table,'7/1/2019','7/7/2019',8)

top_8_menu.head(10)