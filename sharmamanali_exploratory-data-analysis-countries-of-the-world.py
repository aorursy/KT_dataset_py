from IPython.display import HTML

from IPython.display import Image

Image(url= "https://upload.wikimedia.org/wikipedia/commons/b/b4/2002_six-color_world_political_map.png")
from IPython.core.display import HTML

HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
# importing libraries 

%matplotlib inline 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import re

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# importing the dataset

df=pd.read_csv('../input/countries of the world.csv', decimal = ',')
#first 5 rows of the data set to see what sort of data is there

df.head()
#statistcal analysis of given data set

df.describe()
# Checking for null values

print('Dataset has null values?')

df.isnull().values.any()
#Finding missing values in the data set 

total = df.isnull().sum()[df.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(df)*100,2))

pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])

#Sorting the values of GDP for different countries in descending order

top_gdp_countries = df.sort_values('GDP ($ per capita)',ascending=False)

#Visual Representation of the graph using seaborn for first 33 values

fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Country', y='GDP ($ per capita)', data=top_gdp_countries.head(33), palette='Set1')

ax.set_title('Top 33 Countries vs GDP per capita')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

plt.xticks(rotation=90)

plt.show()

#Visual Representation of the graph using seaborn for last 33 values 

fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Country', y='GDP ($ per capita)', data=top_gdp_countries.tail(33), palette='Set1')

ax.set_title('Last 33 Countries vs GDP per capita')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

plt.xticks(rotation=90)

plt.show()



#
df.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()
#Missing values being filled in columns

for col in df.columns.values:

    if df[col].isnull().sum() == 0:

        continue

    if col == 'Climate':

        guess_values = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())

    else:

        guess_values = df.groupby('Region')[col].median()

    for region in df['Region'].unique():

        df[col].loc[(df[col].isnull())&(df['Region']==region)] = guess_values[region]
print('Are there anymore null values?')
df.isnull().values.any()
#check if we filled all missing values

print(df.isnull().sum())
#correlation

df.corr()
#Visual representation in form of heatmap for correlated data

plt.figure(figsize=(16,12))

ax=plt.axes()

sns.heatmap(data=df.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

ax.set_title('Heatmap showing correlated values for the Dataset')

plt.show()

# choose attributes which shows relation

x = df[['GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Service','Infant mortality (per 1000 births)','Birthrate','Agriculture']]
# show corr of the same

plt.figure(figsize=(10,5))

ax=plt.axes()

sns.heatmap(x.corr(), annot=True,ax=ax)

ax.set_title('Heatmap showing correlated values for the Dataset')

plt.show()

#scatter plot to show correlation between GDP and other attributes

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(23,20))

plt.subplots_adjust(hspace=0.4)



corr_to_gdp = pd.Series()

for col in df.columns.values[2:]:

    if ((col!='GDP ($ per capita)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

        corr_to_gdp[col] = df['GDP ($ per capita)'].corr(df[col])

abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)

corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]





for i in range(3):

    for j in range(3):

        sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=df,

                   ax=axes[i,j], fit_reg=False, marker='.')

        title = 'correlation='+str(corr_to_gdp[i*3+j])

        axes[i,j].set_title(title)

axes[1,2].set_xlim(0,102)

fig.suptitle('Scatterplot between GDP per capita and factors', fontsize='30')

plt.show()
x = df[['GDP ($ per capita)','Phones (per 1000)','Service','Region']]



g=sns.pairplot(x, hue="Region", diag_kind='hist')

g.fig.suptitle('Pairplot showing GDP per capita, Services and Phones per(1000)',y=1.05)







fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(23,20))

plt.subplots_adjust(hspace=0.4)



z = pd.Series()

for col in df.columns.values[2:]:

    if ((col!='Deathrate')&(col!='Net migration')&(col!='Industry')&(col!='Agriculture')&(col!='Birthrate')&(col!='Area (sq. mi.)')&(col!='Population')&(col!='Other (%)')&(col!='Crops (%)')&(col!='Arable (%)')&(col!='Infant mortality (per 1000 births)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

      

        colums=np.array(df[col])

        z[col]=colums

#p=z.loc[z.index]

#print (z)



for i in range(2):

    for j in range(2):

        

        #x=z.index.values[i*3+j]

        #sns.barplot(z.index[i*3+j],z.values[i*3+j])

        #x=z.index.values[i*3+j]

        

        y=z.index[i*2+j]

        x=z[i*2+j]

        print(y)

        sns.distplot(x,axlabel=y,ax=axes[i,j])



fig.suptitle('Univariate Distribution of Positively Correlated Factors', fontsize='25')

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

plt.subplots_adjust(hspace=0.1)



z = pd.Series()

for col in df.columns.values[2:]:

     if ((col!='Service')&(col!='Deathrate')&(col!='Net migration')&(col!='Industry')&(col!='Literacy (%)')&(col!='GDP ($ per capita)')&(col!='Area (sq. mi.)')&(col!='Population')&(col!='Other (%)')&(col!='Crops (%)')&(col!='Arable (%)')&(col!='Phones (per 1000)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

            

        colums=np.array(df[col])

        z[col]=colums

p=z

#print (p)



for i in range(1):

    for j in range(3):

        y=z.index[j]

        x=z[j]

        #print(x)

        #print(y)

        #print(z[j].size)

        sns.distplot(x,ax=axes[j],axlabel=y)



fig.suptitle('Univariate Distribution of Negatively Correlated Factors', fontsize='20')

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(23,20))

plt.subplots_adjust(hspace=0.4)



z = pd.Series()

for col in df.columns.values[2:]:

     if ((col!='Deathrate')&(col!='Net migration')&(col!='Industry')&(col!='GDP ($ per capita)')&(col!='Area (sq. mi.)')&(col!='Population')&(col!='Other (%)')&(col!='Crops (%)')&(col!='Arable (%)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

        colums=np.array(df[col])

        z[col]=colums



for i in range(2):

    for j in range(3):

        

        x=z.index[i*3+j]

        y=z[i*3+j]

        sns.boxplot(z[i*3+j],ax=axes[i,j])

        title = str(z.index[i*3+j])

        axes[i,j].set_title(title)

        axes[0,0].set_xlim(0,175)



fig.suptitle('Boxplot Distribution for Correlated Attributes', fontsize='30')

      

plt.show()

df.head(5)
LE = LabelEncoder()

df['Regional_label'] = LE.fit_transform(df['Region'])

df1 = df[['Region','Regional_label']]

df1.head(5)
train, test = train_test_split(df, test_size=0.3, shuffle=True)

training_features = ['Population', 'Area (sq. mi.)',

       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',

       'Net migration', 'Infant mortality (per 1000 births)',

       'Literacy (%)', 'Phones (per 1000)',

       'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',

       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Regional_label',

       'Service']

target = 'GDP ($ per capita)'

train_X = train[training_features]

train_Y = train[target]

test_X = test[training_features]

test_Y = test[target]

model = LinearRegression()

model.fit(train_X, train_Y)

train_pred_Y = model.predict(train_X)

test_pred_Y = model.predict(test_X)

train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)

test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)



rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))

msle_train = mean_squared_log_error(train_pred_Y, train_Y)

rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))

msle_test = mean_squared_log_error(test_pred_Y, test_Y)



#q=model.score(rmse_test,rmse_train)



print('rmse_train: %.2f '% (rmse_train),'msle_train: %.2f ' %(msle_train))

print('rmse_test: %.2f ' %(rmse_test),'msle_test:%.2f ' %(msle_test))
train, test = train_test_split(df, test_size=0.3, shuffle=True)

training_features = ['Phones (per 1000)']

target = 'GDP ($ per capita)'

train_X = train[training_features]

train_Y = train[target]

test_X = test[training_features]

test_Y = test[target]

model = LinearRegression()

model.fit(train_X, train_Y)

train_pred_Y = model.predict(train_X)

test_pred_Y = model.predict(test_X)

train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)

test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)



rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))

msle_train = mean_squared_log_error(train_pred_Y, train_Y)

rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))

msle_test = mean_squared_log_error(test_pred_Y, test_Y)



print('rmse_train:%.2f '%(rmse_train),'msle_train:%.2f '%(msle_train))

print('rmse_test:%.2f '% (rmse_test),'msle_test:%.2f '%(msle_test))



plt.scatter(test_X, test_Y, color = 'red')

plt.plot(train_X, train_pred_Y, color = 'blue')

plt.xlabel('Phones per 1000')

plt.ylabel('GDP per capita')

plt.title('Linear Regression between Phones per 1000 and GDP per capita')

plt.show()

df['Total_GDP ($)'] = df['GDP ($ per capita)'] * df['Population']

top_gdp_countries = df.sort_values('Total_GDP ($)',ascending=False).head(10)

other = pd.DataFrame({'Country':['Other'], 'Total_GDP ($)':[df['Total_GDP ($)'].sum() - top_gdp_countries['Total_GDP ($)'].sum()]})

gdps = pd.concat([top_gdp_countries[['Country','Total_GDP ($)']],other],ignore_index=True)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7),gridspec_kw = {'width_ratios':[2,1]})

sns.barplot(x='Country',y='Total_GDP ($)',data=gdps,ax=axes[0],palette='Set2')

axes[0].set_xlabel('Country',labelpad=30,fontsize=16)

axes[0].set_ylabel('Total_GDP',labelpad=30,fontsize=16)



colors = sns.color_palette("Set2", gdps.shape[0]).as_hex()

axes[1].pie(gdps['Total_GDP ($)'], labels=gdps['Country'],colors=colors,autopct='%1.1f%%',shadow=True)

axes[1].axis('equal')

plt.show()
Rank_total_gdp = df[['Country','Total_GDP ($)']].sort_values('Total_GDP ($)', ascending=False).reset_index()

Rank_gdp = df[['Country','GDP ($ per capita)']].sort_values('GDP ($ per capita)', ascending=False).reset_index()

Rank_total_gdp= pd.Series(Rank_total_gdp.index.values+1, index=Rank_total_gdp.Country)

Rank_gdp = pd.Series(Rank_gdp.index.values+1, index=Rank_gdp.Country)

Rank_change = (Rank_gdp-Rank_total_gdp).sort_values(ascending=False)

print('rank of total GDP - rank of GDP per capita:')

Rank_change.loc[top_gdp_countries.Country]
plt.figure(figsize=(16,12))

ax=plt.axes()

y=df[df.columns[2:]].apply(lambda x: x.corr(df['Total_GDP ($)']))

print(y)

sns.heatmap(data=df.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

ax.set_title('Heatmap showing correlated values for the Dataset with respect to total ')

plt.show()