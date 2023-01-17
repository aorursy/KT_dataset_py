# import the libraries we will use

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# read data

df = pd.read_csv('../input/brazilian-real-bank-dataset/MiBolsillo.csv',encoding = 'unicode_escape',sep=';')
# we take a look at the data

df.head(5)
# translate variable names into English



df.columns = ['id','branch_number','city','state','age','gender','total_credit_card_limit','current_available_limit' ,'date','amount','category_expense','purchase_city','purchase_country']
df.date = pd.to_datetime(df.date,dayfirst=True)

df.head()
# I create a dataframe as customers with old and new customer names



customers = pd.DataFrame(df.id.unique())

customers.columns = ['old_customer_name']

new_list = list(range(1, len(customers)+1))

customers['new_customer_name'] = new_list

customers
# I am changing id to understand customers more easily



df["id"].replace(customers['old_customer_name'].values,customers['new_customer_name'].values, inplace=True)
# we have information about our dataset

df.info()
# we need to float the type of amount variable

for i in range(0,len(df.amount)):

    df.amount[i] = df.amount[i].replace('.','').replace(',','.')



#I export values containing '-' in the amount variable as df_amount_nan

df_amount_nan = df[df.amount == ' -   ']



# There are '-' values in the amount column. I am deleting theese rows because we can't convert it to the float type with these values

df = df[df.amount != ' -   ']



# Now the amount variable is ready to convert to float type.

df.amount = df.amount.astype(float)
# I am converting the date variable to date_time format

df.date = pd.to_datetime(df.date,dayfirst=True)
# We convert the gender variable to dummy_variable

dms = pd.get_dummies(df['gender'])

df = pd.concat([df,dms],axis=1)

df.drop(['gender', 'M'], axis=1,inplace=True)

df.rename(columns={'F': 'Female'}, inplace=True)

df
# I want to look at the total expense in each category



categ = df.category_expense.value_counts().sort_values(ascending=False)





plt.figure(figsize=(15,10))

sns.barplot(x=categ.index,y=categ.values)

plt.xlabel('Cataegories')

plt.ylabel('Count')

plt.title("Cataegories Count")

plt.xticks(rotation= 45);


# I look at the number of transactions each client has with a credit card



freq = df.groupby('id')[['amount']].count().sort_values('amount',ascending=False)

freq.rename(columns={'amount': 'Frequency'}, inplace=True)

freq
# I look at the total spending of each customer

total = df.groupby('id')[['amount']].sum().sort_values('amount',ascending=False)

total.rename(columns={'amount': 'Total_spending'}, inplace=True)

total
# I want to look at the total spending of each customers



plt.figure(figsize=(15,10))

sns.barplot(x=total.index,y=total['Total_spending'])

plt.xlabel('Customers')

plt.ylabel('Total Spending')

plt.title("Total Spending vs Customers")

plt.xticks(rotation= 45);
# I add the special days(Carnival,Good Friday,Christmas,Corpus Christi,New Year's Day,Black Friday,Halloween) of Brazil to the dataset



df['carnival'] = '0'

df["carnival"].replace(df[(df.date == '2020-02-24') | (df.date == '2020-02-25') | (df.date == '2019-03-04') | (df.date == '2019-03-05')]['carnival'],'1', inplace=True)



df['good_friday'] = '0'

df["good_friday"].replace(df[(df.date == '2019-04-19') | (df.date == '2020-04-10')]['good_friday'],'1', inplace=True)





df['christmas'] = '0'

df["christmas"].replace(df[(df.date == '2019-12-25') | (df.date == '2020-12-25')]['christmas'],'1', inplace=True)





df['corpus_christi'] = '0' 

df["corpus_christi"].replace(df[(df.date == '2019-06-20') | (df.date == '2020-06-11')]['corpus_christi'],'1', inplace=True)





df['new_year'] = '0'

df["new_year"].replace(df[(df.date == '2019-01-01') | (df.date == '2020-01-01')]['new_year'],'1', inplace=True)





df['black_friday'] = '0'

df["black_friday"].replace(df[(df.date == '2019-11-29') | (df.date == '2020-11-27')]['black_friday'],'1', inplace=True)





df['halloween'] = '0'

df["halloween"].replace(df[(df.date == '2019-10-31') | (df.date == '2020-10-31')]['halloween'],'1', inplace=True)



# I want to see the total transaction on special days as a barplot



special_days = ['carnival','good_friday','christmas','corpus_christi','new_year','black_friday','halloween']

counts = []



for i in special_days:

    counts.append(df[i].value_counts()[1])

special_days_dict = dict( zip( special_days, counts))





plt.figure(figsize=(15,10))

sns.barplot(x=list(special_days_dict.keys()),y=list(special_days_dict.values()))

plt.xlabel('Special Days')

plt.ylabel('Total Transaction')

plt.title('Total Transaction of Special Days')

plt.xticks(rotation= 45);
# I create data sets covid and pre covid



# pre covid

pre_covid = df[(df.date > '2020-01-01') & (df.date < '2020-03-18')]



#covid

covid = df[(df.date >= '2020-03-18')]
covid.head()
# Frequency of use



covid_freq = covid.groupby('id')[['age']].count().sort_values('age',ascending=False)

covid_freq.columns = ['frequency']

print('average number of transactions frequency: ',covid_freq.frequency.mean())

print('max number of transactions frequency: ',covid_freq.frequency.max())

print('min number of transactions frequency: ',covid_freq.frequency.min())
plt.figure(figsize=(15,10))

sns.barplot(x=covid_freq.index,y=covid_freq['frequency'])

plt.xlabel('Customers')

plt.ylabel('Frequency of use')

plt.title("The frequency of transactions made by customers during the covid")

plt.xticks(rotation= 45);
# The frequency of using credit card for each customer during covid

covid_freq
covid['category_expense'].value_counts()
# Since there is no specific rule, I create the essential and non-essential list myself.



essential_list = ['FARMACIAS','VAREJO','HOSP E CLINICA','SUPERMERCADOS','POSTO DE GAS','TRANS FINANC']



non_essential_list = ['SERVI\x82O','M.O.T.O.','ARTIGOS ELETRO','LOJA DE DEPART','VESTUARIO','SEM RAMO','MAT CONSTRUCAO','RESTAURANTE','CIA AEREAS','MOVEIS E DECOR','JOALHERIA','AGENCIA DE TUR','HOTEIS','AUTO PE AS','INEXISTENTE','']
# The transaction amount from the essential list during the COVID



covid[covid.category_expense.isin(essential_list)]['category_expense'].value_counts()
# The transaction amount from the non-essential list during the COVID



covid[covid.category_expense.isin(non_essential_list)]['category_expense'].value_counts()
# % of essential

print('Total spending during corid: ',covid['amount'].sum(),'Brazillian R')

print('Total spending in essential category during covid',covid[covid.category_expense.isin(essential_list)]['amount'].sum(),'Brazillian R')

print('Essential : %',covid[covid.category_expense.isin(essential_list)]['amount'].sum() * 100 / covid['amount'].sum())

essential_covid = covid[covid.category_expense.isin(essential_list)]['amount'].sum() * 100 / covid['amount'].sum()
# % of non essential

print('Total spending during corid: ',covid['amount'].sum(),'Brazillian R')

print('Total spending in non essential category during covid',covid[covid.category_expense.isin(non_essential_list)]['amount'].sum(),'Brazillian R')

print('Non - Essential : %',covid[covid.category_expense.isin(non_essential_list)]['amount'].sum() * 100 / covid['amount'].sum())

non_essential_covid = covid[covid.category_expense.isin(non_essential_list)]['amount'].sum() * 100 / covid['amount'].sum()
# Top 3 essential expenses
covid[covid.category_expense.isin(essential_list)]['category_expense'].value_counts()[:3]
# Top 3 non - essential expenses
covid[covid.category_expense.isin(non_essential_list)]['category_expense'].value_counts()[:3]
# The lowest paid spending and category of each customers during covid



for i in range(1,30):

    print(covid[covid['id'] == i].sort_values('amount')[:3][['category_expense','amount']].set_index(covid[covid['id'] == i]['id'][:3]))
# The lowest paid spending and category of each customers during covid



for i in range(1,30):

    print(covid[covid['id'] == i].sort_values('amount',ascending=False)[:3][['category_expense','amount']].set_index(covid[covid['id'] == i]['id'][:3]))
pre_covid.head()
# Frequency of use



pre_covid_freq = pre_covid.groupby('id')[['age']].count().sort_values('age',ascending=False)

pre_covid_freq.columns = ['frequency']

print('average number of transactions frequency: ',pre_covid_freq.frequency.mean())

print('max number of transactions frequency: ',pre_covid_freq.frequency.max())

print('min number of transactions frequency: ',pre_covid_freq.frequency.min())
plt.figure(figsize=(15,10))

sns.barplot(x=pre_covid_freq.index,y=pre_covid_freq['frequency'])

plt.xlabel('Customers')

plt.ylabel('Frequency of use')

plt.title("The frequency of transactions made by customers before the covid")

plt.xticks(rotation= 45);
# The frequency of using credit card for each customer before covid

pre_covid_freq
# Since there is no specific rule, I create the essential and non-essential list myself.



essential_list = ['FARMACIAS','VAREJO','HOSP E CLINICA','SUPERMERCADOS','POSTO DE GAS','TRANS FINANC']



non_essential_list = ['SERVI\x82O','M.O.T.O.','ARTIGOS ELETRO','LOJA DE DEPART','VESTUARIO','SEM RAMO','MAT CONSTRUCAO','RESTAURANTE','CIA AEREAS','MOVEIS E DECOR','JOALHERIA','AGENCIA DE TUR','HOTEIS','AUTO PE AS','INEXISTENTE','']
# The transaction amount from the essential list before the COVID



pre_covid[pre_covid.category_expense.isin(essential_list)]['category_expense'].value_counts()
# The transaction amount from the non-essential list before the COVID 



pre_covid[pre_covid.category_expense.isin(non_essential_list)]['category_expense'].value_counts()
# % of essential

print('Total spending before covid: ',pre_covid['amount'].sum(),'Brazillian R')

print('Total spending in essential category before covid',pre_covid[pre_covid.category_expense.isin(essential_list)]['amount'].sum(),'Brazillian R')

print('Essential : %',pre_covid[pre_covid.category_expense.isin(essential_list)]['amount'].sum() * 100 / pre_covid['amount'].sum())

essential_pre_covid = pre_covid[pre_covid.category_expense.isin(essential_list)]['amount'].sum() * 100 / pre_covid['amount'].sum()
# % of non essential

print('Total spending before corid: ',pre_covid['amount'].sum(),'Brazillian R')

print('Total spending in non essential category before covid',pre_covid[pre_covid.category_expense.isin(non_essential_list)]['amount'].sum(),'Brazillian R')

print('Non - Essential : %',pre_covid[pre_covid.category_expense.isin(non_essential_list)]['amount'].sum() * 100 / pre_covid['amount'].sum())

non_essential_pre_covid = pre_covid[pre_covid.category_expense.isin(non_essential_list)]['amount'].sum() * 100 / pre_covid['amount'].sum()
# Top 3 essential expenses
pre_covid[pre_covid.category_expense.isin(essential_list)]['category_expense'].value_counts()[:3]
# Top 3 non - essential expenses
pre_covid[pre_covid.category_expense.isin(non_essential_list)]['category_expense'].value_counts()[:3]
# The lowest paid spending and category of each customers before covid



for i in range(1,30):

    print(pre_covid[pre_covid['id'] == i].sort_values('amount')[:3][['category_expense','amount']].set_index(pre_covid[pre_covid['id'] == i]['id'][:3]))
# The lowest paid spending and category of each customers before covid



for i in range(1,30):

    print(pre_covid[pre_covid['id'] == i].sort_values('amount',ascending=False)[:3][['category_expense','amount']].set_index(pre_covid[pre_covid['id'] == i]['id'][:3]))
freq = pd.concat([covid_freq.sort_index(),pre_covid_freq.sort_index()],axis=1)

freq.columns = ['covid_freq','pre_covid_freq']
# The frequency of using credit card for each customer before covid vs during covid





fig, ax = plt.subplots(2,2,sharey=True)





ax[0,0].plot(pre_covid_freq.sort_index().index,pre_covid_freq.sort_index().values,color='g',marker='o')

ax[0,0].set_title('Pre Covid')





ax[0,1].plot(covid_freq.sort_index().index,covid_freq.sort_index().values,color='b',marker='o')

ax[0,1].set_title('Covid')



plt.show();
# The frequency of using credit card for each customer before covid vs during covid



import plotly.graph_objs as go

#import chart_studio.plotly as py





fig = go.Figure()

fig.add_trace(go.Box(y=freq.covid_freq, name='The frequency of using credit card for each customer during covid',

                marker_color = 'indianred'))

fig.add_trace(go.Box(y=freq.pre_covid_freq, name = 'The frequency of using credit card for each customer before covid',

                marker_color = 'lightseagreen'))



fig.show()
from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go





# Creating trace1

trace1 = go.Scatter(

                    x = freq.index,

                    y = freq.covid_freq,

                    mode = "lines",

                    name = "Covid Freq",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= freq.covid_freq)

# Creating trace2

trace2 = go.Scatter(

                    x = freq.index,

                    y = freq.pre_covid_freq,

                    mode = "lines+markers",

                    name = "Pre Covidreq",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= freq.pre_covid_freq)

data = [trace1, trace2]

layout = dict(title = 'The frequency of using credit card for each customer before covid vs during covid',

              xaxis= dict(title= 'Customers',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
df.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.2f',ax=ax);