import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml
basket = pd.read_csv('../input/transactions-from-a-bakery/BreadBasket_DMS.csv')
basket.head(10)
basket.shape
basket.isnull().any()
basket.dtypes
basket['Item'].unique()
basket['Date'].value_counts().iloc[:10]
sns.countplot(x = 'Date', data = basket, order = basket['Date'].value_counts().iloc[:10].index)
plt.xticks(rotation=45)
basket['Time'].value_counts()
# plt.figure(figsize=(30,20))
sns.countplot(x = 'Time', data = basket, order = basket['Time'].value_counts().iloc[:25].index)
plt.xticks(rotation=90)
basket['Item'].value_counts().iloc[:10]
plt.xticks(rotation=45)
sns.countplot(x = 'Item', data = basket, order = basket['Item'].value_counts().iloc[:10].index)
#TEA OR COFFEE?

TC = basket.groupby("Date")["Item"].apply(lambda x: x[x=="Tea"].count()).rename("Teas sold").to_frame()
TC["Coffees sold"] = basket.groupby("Date")["Item"].apply(lambda x: x[x=="Coffee"].count())

TC.plot(figsize=(12,5))
plt.ylim([0,80])
plt.grid(True)
plt.legend()

print("Maxiumum numbers of teas sold: " + str(TC["Teas sold"].max()))
print("Maxiumum numbers of coffees sold: " + str(TC["Coffees sold"].max()))
month_year=basket.copy()  
month_year['Date']=pd.to_datetime(month_year['Date'])
month_year['Month'],month_year['Year']=month_year['Date'].dt.month,month_year['Date'].dt.year

grp_month_year=month_year.groupby(['Month','Year'])['Transaction'].count().reset_index()
grp_month_year['Period'] = grp_month_year.Month.astype(str).str.cat(grp_month_year.Year.astype(str), sep=' / ')

#plot graph for each month of the year 2016,2017
fig,axis=plt.subplots(figsize=(5,4))
axis=sns.barplot(data=grp_month_year,x='Period',y='Transaction',color = ("#FC4E07"))
axis.set_xlabel('Month & Year')
axis.set_ylabel('Nombre of Transarctions')
axis.set_xticklabels(grp_month_year['Period'], rotation=60)
# number of transactions per day
basket["Date"] = pd.to_datetime(basket["Date"])
basket["Weekday"] = basket["Date"].dt.weekday_name
basket['Weekday'].value_counts()
# plot of items sold per day
products_days = basket["Date"].value_counts().sort_index().rename("Products sold")

mean = round(products_days.mean(),0)
maximum_day = products_days.index[products_days==products_days.max()][0]

ax = products_days.plot(figsize=(12,5), x_compat=True)

text1 = '$\mu=$' + str(mean)[:-2]
# x1 = mdates.date2num(pd.Timestamp("2016/10/31"))
# x2 = mdates.date2num(maximum_day)
# ax.text(x1-5, mean+5, text1, fontsize=15)
# ax.text(x2+1, 300, str(maximum_day)[:-8], fontsize=10)
plt.grid(True)
plt.title("Number of products sold every day")
plt.axhline(mean, c="k", linestyle='--')
plt.tight_layout()
# items sold per weekday
fig, ax = plt.subplots()
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
products_weekdays = basket["Weekday"].value_counts().rename("Products sold").reindex(days)
products_weekdays.plot(kind='bar', figsize=(8,7), ax=ax, color=("#00b3b0"))
plt.xticks(rotation=0)
for i, v in enumerate(products_weekdays.values):
    ax.text(i, v+80, str(v), color="#000d1a", fontweight='bold', fontsize=14, ha='center')

#part of the day groupby
part_of_day=basket.copy()
part_of_day['timestamp'] = part_of_day.Date.astype(str).str.cat(part_of_day.Time.astype(str), sep=' ')
part_of_day['timestamp']=pd.to_datetime(part_of_day['timestamp'])

part_of_day['hour'] = part_of_day['timestamp'].dt.round('H').dt.hour
part_of_day.drop(['Time','Date'],axis=1,inplace=True)
#Coffee hours
cof_hours=part_of_day[(part_of_day['Item']== 'Coffee')]
cof_hours=cof_hours.groupby('hour')['Item'].count()
cof_hours=cof_hours.reset_index()
cof_hours
fig,ax=plt.subplots(figsize=(8,5))
ax=sns.barplot(data=cof_hours,x='hour',y='Item')
ax.set_xlabel('Hours Of The Day')
ax.set_ylabel('Nombre of Times Coffee is Sold')
#Bread hours
bread_hours=part_of_day[part_of_day['Item']=='Bread']
bread_hours=bread_hours.groupby('hour')['Item'].count()
bread_hours=bread_hours.reset_index()

fig,ax=plt.subplots(figsize=(8,5))
ax=sns.barplot(data=bread_hours,x='hour',y='Item')
ax.set_xlabel('Hours Of The Day')
ax.set_ylabel('Nombre of Times Bread is Sold')
#Cake or Pastry hours
cake_hours=part_of_day.loc[(part_of_day['Item']=='Cake') | (part_of_day['Item']=='Pastry')]
cake_hours=cake_hours.groupby('hour')['Item'].count()
cake_hours=cake_hours.reset_index()

fig,ax=plt.subplots(figsize=(8,5))
ax=sns.barplot(data=cake_hours,x='hour',y='Item')
ax.set_xlabel('Hours Of The Day')
ax.set_ylabel('Nombre of Times Cake and pastry is Sold')
#Cake hours
cake_hours=part_of_day.loc[(part_of_day['Item']=='Cake')]
cake_hours=cake_hours.groupby('hour')['Item'].count()
cake_hours=cake_hours.reset_index()

fig,ax=plt.subplots(figsize=(8,5))
ax=sns.barplot(data=cake_hours,x='hour',y='Item')
ax.set_xlabel('Hours Of The Day')
ax.set_ylabel('Nombre of Times Cake is Sold')
#Tea hours
tea_hours=part_of_day.loc[(part_of_day['Item']=='Tea')]
tea_hours=tea_hours.groupby('hour')['Item'].count()
tea_hours=tea_hours.reset_index()

fig,ax=plt.subplots(figsize=(8,5))
ax=sns.barplot(data=tea_hours,x='hour',y='Item')
ax.set_xlabel('Hours Of The Day')
ax.set_ylabel('Nombre of Times Tea is Sold')
df = basket.groupby(['Transaction','Item']).size().reset_index(name='count')
basket0 = (df.groupby(['Transaction', 'Item'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Transaction'))
#The encoding function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket0.applymap(encode_units)
basket_sets.head()
basket_sets.corr() 
# correlation of Churn with other variables
basket_sets.corr()['Coffee'].sort_values(ascending = False).plot(kind='bar', figsize=(20,5), color=("#4951ad"))
basket_sets.corr()['Coffee'].sort_values(ascending = False)
basket_sets.corr()['Tea'].sort_values(ascending = False).plot(kind='bar', figsize=(20,5), color=("#7fc94d"))
basket_sets.corr()['Tea'].sort_values(ascending = False)
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift")
rules.sort_values('confidence', ascending = False, inplace = True)
print(rules.shape)
rules.head(5)
# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='BuPu');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();
# Set the metric "lift" with a minimum threshold = 1.2

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules2 = association_rules(frequent_itemsets, metric="lift",  min_threshold=1.2)
rules2.sort_values('confidence', ascending = False, inplace = True)
print(rules2.shape)
rules2
# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(10, 6))
plt.scatter(rules2['support'], rules2['confidence'], c=rules2['lift'], alpha=0.9, cmap='BuPu');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
print(rules.shape)
rules.head(10)
rules[rules['antecedent_len'] >=2 ]
# find rules with some conditions :
rules[(rules['antecedent_len'] >= 2) &
      (rules['confidence'] >= 0.1)& 
      (rules['lift'] >= 1.2) ]