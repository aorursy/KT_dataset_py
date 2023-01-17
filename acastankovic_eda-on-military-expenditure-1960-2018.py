# import this libraryies just for practice, i dont need all of them

import numpy as np # used for handling numbers

import pandas as pd # used for handling the dataset

import matplotlib.pyplot as plt

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

%matplotlib inline

import seaborn as sns #data visualization library

from sklearn.model_selection import train_test_split # used for splitting training and testing data

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.multioutput import MultiOutputClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA

from sklearn.metrics import confusion_matrix

from sklearn.impute import SimpleImputer # used for handling missing data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data

from sklearn.preprocessing import StandardScaler # used for feature scaling

#@title Import modules

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

df = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv')

df.head()
# Proveravamo koliko procentualno imamo tipova

df['Type'].value_counts(normalize = True) * 100
# Kreiramo novu kolonu Total kao zbir potrosnja kroz godine za svaku drzavu

df1 = df.assign(Total = df.sum(axis=1))

df1.head(5)
# Sa nulom zamenjujemo NaN vrednosti

df1.fillna(0, inplace=True)

df1.head(5)
df1.describe()
# Delimo vrednosti sa 10 na 9 kako bi dobili vrednosti u bilionima USD (preglednije je)

columns=[str(i) for i in list((range(1960,2019)))]

columns=columns+["Total"]

for i in columns:

    df1[i]=df1[i]/1.e+9

df1=np.round(df1, decimals=2)

df1.head()
# Sortiranje po totalnoj potrosnji za tip Country

df1.sort_values(by=['Type','Total'],ascending=[False,False],inplace=True)

df1=df1[df1['Type'].str.contains("Country")]

# Prvih 20 zemalja po ukupnoj potrosnji uz izbacivanje nepotrebnih atributa

df2 = df1[:20]

df3 = df2.drop(['Indicator Name', 'Code', 'Type'], axis=1)

new = df3.reset_index(drop=True)

new.head(20)
# Vizualizacija

plt.figure(figsize=(12,8))

sns.barplot(x = 'Total', y = 'Name', data = df3)

plt.title('Total Millitary Spending from 1960 to 2018')

plt.xlabel('Total in bilions USD')

plt.ylabel('Countries')

plt.grid()
fig = px.pie(df2, values='Total', names='Name', title='Total military spendings in percentage from 1960 to 2018 ')

fig.show()
fig = px.pie(df2, values='1960', names='Name', title='Military spendings in percentage in 1960')

fig.show()
fig = px.scatter_geo(df2, locations="Code",hover_name="Name")

fig.update_layout(title="First 20 most powerful country")

fig.show()
fig = px.scatter_geo(df2, locations = 'Code',hover_name="Name",size = '2018')

fig.show()
df4 = df3.drop(['Total'], axis=1)


Top20 = df4.set_index('Name')

Top20.index = Top20.index.rename('Year')

Top20 = Top20.T

Top20.head()
plt.figure(figsize=(20,10))

plt.plot(Top20.index, Top20.values)

plt.ylabel('Spendings through year (in bilion USD)')

plt.title('Top 20 Countries  in Military Expenditure ')

plt.xticks(rotation=45)

plt.legend(Top20.columns)

plt.grid(True)

plt.show()
# Procenat rasta potrosnje po godinama za prvih 20 zemalja

PercUSA = (Top20['United States'].iloc[-1] - Top20['United States'].iloc[0])*100/Top20['United States'].iloc[0]

PercChina = (Top20['China'].iloc[-1] - Top20['China'].iloc[29])*100/Top20['China'].iloc[29]

PercRUS = (Top20['Russian Federation'].iloc[-1] - Top20['Russian Federation'].iloc[33])*100/Top20['Russian Federation'].iloc[33]

PercISR = (Top20['Israel'].iloc[-1] - Top20['Israel'].iloc[0])*100/Top20['Israel'].iloc[0]

PercITA = (Top20['Italy'].iloc[-1] - Top20['Italy'].iloc[0])*100/Top20['Italy'].iloc[0]

PercJPN = (Top20['Japan'].iloc[-1] - Top20['Japan'].iloc[0])*100/Top20['Japan'].iloc[0]

PercNET = (Top20['Netherlands'].iloc[-1] - Top20['Netherlands'].iloc[0])*100/Top20['Netherlands'].iloc[0]

PercPOL = (Top20['Poland'].iloc[-1] - Top20['Poland'].iloc[0])*100/Top20['Poland'].iloc[0]

PercSAU = (Top20['Saudi Arabia'].iloc[-1] - Top20['Saudi Arabia'].iloc[0])*100/Top20['Saudi Arabia'].iloc[0]

PercKOR = (Top20['South Korea'].iloc[-1] - Top20['South Korea'].iloc[0])*100/Top20['South Korea'].iloc[0]

PercSPA = (Top20['Spain'].iloc[-1] - Top20['Spain'].iloc[0])*100/Top20['Spain'].iloc[0]

PercTUR = (Top20['Turkey'].iloc[-1] - Top20['Turkey'].iloc[0])*100/Top20['Turkey'].iloc[0]

PercUK = (Top20['United Kingdom'].iloc[-1] - Top20['United Kingdom'].iloc[0])*100/Top20['United Kingdom'].iloc[0]

PercAUS = (Top20['Australia'].iloc[-1] - Top20['Australia'].iloc[0])*100/Top20['Australia'].iloc[0]

PercBRA = (Top20['Brazil'].iloc[-1] - Top20['Brazil'].iloc[0])*100/Top20['Brazil'].iloc[0]

PercCAN = (Top20['Canada'].iloc[-1] - Top20['Canada'].iloc[0])*100/Top20['Canada'].iloc[0]

PercFRA = (Top20['France'].iloc[-1] - Top20['France'].iloc[0])*100/Top20['France'].iloc[0]

PercGER = (Top20['Germany'].iloc[-1] - Top20['Germany'].iloc[0])*100/Top20['Germany'].iloc[0]

PercIND = (Top20['India'].iloc[-1] - Top20['India'].iloc[0])*100/Top20['India'].iloc[0]

PercIRA = (Top20['Iran'].iloc[-1] - Top20['Iran'].iloc[0])*100/Top20['Iran'].iloc[0]
data = [['United States', PercUSA], ['China', PercChina], ['France', PercFRA], ['United Kingdom', PercUK], ['Germany', PercGER], ['Japan', PercJPN], ['Saudi Arabia', PercSAU], ['Russian Federation', PercRUS], ['India', PercIND], ['Italy', PercITA], ['South Korea', PercKOR], ['Brazil', PercBRA], ['Canada', PercCAN], ['Spain', PercSPA], ['Australia', PercAUS], ['Iran', PercIRA], ['Israel', PercISR], ['Turkey', PercTUR], ['Poland', PercPOL], ['Netherlands', PercNET]]
percdf= pd.DataFrame(data, columns=['Country', 'Percentage growth'])

percdf.head(20)
plt.figure(figsize=(15,8))

sns.barplot(x = 'Country', y = 'Percentage growth', data = percdf)

plt.xticks(rotation=45)
model = percdf.join(new['Total'])

model
plt.figure(figsize=(15,8))

sns.barplot(x = 'Total', y = 'Percentage growth', hue='Country',data = model)

plt.xticks(rotation = 90)
