from IPython.display import HTML

from IPython.display import Image

Image(url= "https://www.worldatlas.com/r/w728-h425-c728x425/upload/0f/59/b2/untitled-design-275.jpg")
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

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import r2_score
# read csv file

df=pd.read_csv('../input/countries of the world.csv', decimal = ',')
# View first 5 rows(default) to see the general distribution of data

df.head()
# run basic statistical analysis on the given data to find any abnormal values

df.describe()
print("Are there Null Values in the dataset? ")

df.isnull().values.any()
# finding the missing or null values in the data

total = df.isnull().sum()[df.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(df)*100,2))

pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
# sorting and plotting Countries based on GDP

top_gdp_countries = df.sort_values('GDP ($ per capita)',ascending=False)

fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Country', y='GDP ($ per capita)', data=top_gdp_countries.head(33), palette='Set1')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

ax.set_title('GDP of the top 33 countries sorted in a descending order.')

plt.xticks(rotation=90)

plt.show()
fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Country', y='GDP ($ per capita)', data=top_gdp_countries.tail(33), palette='Set1')

ax.set_xlabel(ax.get_xlabel(), labelpad=15)

ax.set_ylabel(ax.get_ylabel(), labelpad=30)

ax.xaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontsize(16)

ax.set_title('GDP of the bottom 33 countries sorted in a descending order.')

plt.xticks(rotation=90)

plt.show()
df.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()
for col in df.columns.values:

    if df[col].isnull().sum() == 0:

        continue

    if col == 'Climate':

        guess_values = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())

    else:

        guess_values = df.groupby('Region')[col].median()

    for region in df['Region'].unique():

        df[col].loc[(df[col].isnull())&(df['Region']==region)] = guess_values[region]
print("Are there Null Values in the dataset? ")

df.isnull().values.any()
print(df.isnull().sum())
df.corr()
plt.figure(figsize=(16,12))

ax=plt.axes()

sns.heatmap(data=df.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

ax.set_title('Heatmap of all the Correlated values')

plt.show()
# choose attributes which shows relation

x = df[['GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Service','Infant mortality (per 1000 births)','Birthrate','Agriculture']]
# show corr of the same

plt.figure(figsize=(9,7))

ax=plt.axes()

sns.heatmap(x.corr(), annot=True, cmap='coolwarm',ax=ax)

ax.set_title('Heatmap of all the highly Correlated values')

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25,16))

plt.subplots_adjust(hspace=0.4)



corr_to_gdp = pd.Series()

for col in df.columns.values[2:]:

    if ((col!='GDP ($ per capita)')&(col!='Climate')&(col!='Coastline (coast/area ratio)')&(col!='Pop. Density (per sq. mi.)')):

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

fig.suptitle('Scatter Plot GDP against the factors',fontsize=30)

plt.show()
x = df[['GDP ($ per capita)','Phones (per 1000)','Service', 'Region']]

pp=sns.pairplot(x, hue="Region", diag_kind="hist", aspect=1.55, markers="o")

pp.fig.suptitle('Scatter Plot of GDP, Phones per Thousand and Service',y=1.05)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(23,20))

plt.subplots_adjust(hspace=0.4)



z = pd.Series()

for col in df.columns.values[2:]:

    if ((col!='Deathrate')&(col!='Net migration')&(col!='Industry')&(col!='Agriculture')&(col!='Birthrate')&(col!='Area (sq. mi.)')&(col!='Population')&(col!='Other (%)')&(col!='Crops (%)')&(col!='Arable (%)')&(col!='Infant mortality (per 1000 births)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

    # if ((col=='GDP ($ per capita)')&(col=='Literacy (%)')&(col=='Service')&(col=='Phones (per 1000)')):

        colums=np.array(df[col])

        z[col]=colums

for i in range(2):

    for j in range(2):

        

        x=z[i*2+j]

        y=z.index[i*2+j]

        sns.distplot(x,axlabel=y,ax=axes[i,j])

        

fig.suptitle('Distplot of Positively Correlated Factors with GDP per Capita',fontsize=30)        

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(23,20))

plt.subplots_adjust(hspace=0.2)



z = pd.Series()

for col in df.columns.values[2:]:

     if ((col!='Service')&(col!='Deathrate')&(col!='Net migration')&(col!='Industry')&(col!='Literacy (%)')&(col!='Area (sq. mi.)')&(col!='Population')&(col!='Other (%)')&(col!='Crops (%)')&(col!='Arable (%)')&(col!='Phones (per 1000)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

            

        colums=np.array(df[col])

        z[col]=colums

        

for i in range(2):

    for j in range(2):

        

        x=z[i*2+j]

        y=z.index[i*2+j]

        sns.distplot(x,axlabel=y,ax=axes[i,j])

      

fig.suptitle('Distplot of Negatively Correlated Factors with GDP per capita',fontsize=30)                

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(23,20))

plt.subplots_adjust(hspace=0.2)



z = pd.Series()

for col in df.columns.values[2:]:

     if ((col!='Deathrate')&(col!='Net migration')&(col!='Industry')&(col!='GDP ($ per capita)')&(col!='Area (sq. mi.)')&(col!='Population')&(col!='Other (%)')&(col!='Crops (%)')&(col!='Arable (%)')&(col!='Climate')&(col!='Coastline (coast/area ratio)') &(col!='Pop. Density (per sq. mi.)')):

            

        colums=np.array(df[col])

        z[col]=colums



for i in range(2):

    for j in range(3):

        x=z[i*3+j]

        y=z.index[i*3+j]

        

        sns.boxplot(x,ax=axes[i,j])

        title=str(y)

        axes[i,j].set_title(title)

        

fig.suptitle('Boxplot of Correlated Factors with GDP per Capita',fontsize=30)                      

plt.show()
df.head()
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

       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Regional_label','Service']

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



print('Root Mean Squared Error for Training Data is:', '%.2f' %rmse_train,'\t\tMean Squared Log Error for Training Data is:', '%.2f' %msle_train)

print('Root Mean Squared Error for Test Data is:', '%.2f' %rmse_test,'\t\tMean Squared Log Error for Test Data is:', '%.2f' %msle_test)

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



print('Root Mean Squared Error for Training Data is:', '%.2f' %rmse_train,'\t\tMean Squared Log Error for Training Data is:', '%.2f' %msle_train)

print('Root Mean Squared Error for Test Data is:', '%.2f' %rmse_test,'\t\tMean Squared Log Error for Test Data is:', '%.2f' %msle_test)



plt.scatter(test_X, test_Y,  color='red')

plt.plot(test_X, test_pred_Y, color='blue', linewidth=1)

plt.title('Linear Regression of Phones per Thousand with GDP')

plt.xlabel('Phones per Thousand')

plt.ylabel('GDP per Capita')

plt.xticks()

plt.yticks()



plt.show()
df['Total_GDP ($)'] = df['GDP ($ per capita)'] * df['Population']

#plt.figure(figsize=(16,6))

top_gdp_countries = df.sort_values('Total_GDP ($)',ascending=False).head(10)

other = pd.DataFrame({'Country':['Other'], 'Total_GDP ($)':[df['Total_GDP ($)'].sum() - top_gdp_countries['Total_GDP ($)'].sum()]})

gdps = pd.concat([top_gdp_countries[['Country','Total_GDP ($)']],other],ignore_index=True)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7), gridspec_kw = {'width_ratios':[2,1]})

sns.barplot(x='Country', y='Total_GDP ($)', data=gdps, ax=axes[0], palette='Set3')

axes[0].set_xlabel('Country', labelpad=30, fontsize=16)

axes[0].set_ylabel('Total_GDP', labelpad=30, fontsize=16)



colors = sns.color_palette("Set3", gdps.shape[0]).as_hex()

axes[1].pie(gdps['Total_GDP ($)'], labels=gdps['Country'], colors=colors, autopct='%1.1f%%', shadow=True)

axes[1].axis('equal')

plt.show()
Rank1 = df[['Country','Total_GDP ($)']].sort_values('Total_GDP ($)', ascending=False).reset_index()

Rank2 = df[['Country','GDP ($ per capita)']].sort_values('GDP ($ per capita)', ascending=False).reset_index()

Rank1 = pd.Series(Rank1.index.values+1, index=Rank1.Country)

Rank2 = pd.Series(Rank2.index.values+1, index=Rank2.Country)

Rank_change = (Rank2-Rank1).sort_values(ascending=False)

print('rank of total GDP - rank of GDP per capita:')

Rank_change.loc[top_gdp_countries.Country]
df.corr()
plt.figure(figsize=(16,12))

ax=plt.axes()

sns.heatmap(data=df.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

ax.set_title('Heatmap of all the Correlated values')

plt.show()
# choose attributes which shows relation

x = df[['Total_GDP ($)','Population','Area (sq. mi.)','GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Service','Infant mortality (per 1000 births)','Birthrate','Agriculture']]
# show corr of the same

plt.figure(figsize=(9,7))

ax=plt.axes()

sns.heatmap(x.corr(), annot=True, cmap='coolwarm',ax=ax)

ax.set_title('Heatmap of all the highly Correlated values')

plt.show()