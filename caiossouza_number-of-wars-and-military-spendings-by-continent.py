import pandas as pd

import numpy as np
data = pd.read_csv("../input/military-expenditure-of-countries-19602019/Military Expenditure.csv", low_memory=False)
data.head(n=2)
data["Indicator Name"].nunique()
data.drop(['Indicator Name'],axis=1,inplace=True)
data.fillna(0,inplace=True)

data["Total USD"]=data.iloc[:,2:].sum(axis=1)
columns=[str(i) for i in list((range(1960,2019)))]

columns=columns+["Total USD"]

for i in columns:

    data[i]=data[i]/1.e+9

data=np.round(data, decimals=2)

data.head()
print("Number of rows before:",len(data))

data.drop(data.loc[data["Total USD"]<1].index,inplace=True)

print("Number of rows after:",len(data))
data.groupby(['Type',"Name"])['Total USD'].sum()
data.sort_index(by=['Type','Total USD'],ascending=[False,False],inplace=True)

data=data[data['Type'].str.contains("Country")]

data
contcodes = pd.read_csv(r"../input/continents-codes-and-number-of-wars/country-and-continent-codes-list-csv_csv.csv", sep=';',usecols=["Continent_Name","Three_Letter_Country_Code"])

contcodes=contcodes.rename(columns={"Three_Letter_Country_Code": "Code"})

contcodes.head()
contcodes.groupby('Code').agg("count").sort_values(["Continent_Name"], ascending = False).head(n=10)
contcodes.drop_duplicates(subset='Code', keep="last",inplace=True)
data=pd.merge(data, contcodes , how='left')

data.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.DataFrame(data, columns = ["Name", 'Total USD'])

top = df.iloc[:10,]

x=top["Name"]

y=top["Total USD"]
sns.set_color_codes("pastel")

ax = sns.barplot(y=x,  x=y, data=df)

ax.set_xlabel('Total USD')

ax.axes.xaxis.label.set_text("Total Spendings (billions)")

ax.axes.yaxis.label.set_text("Countries")

plt.title('Total Millitary Spending from 1968 to 2018')

for p in ax.patches:

    width = p.get_width()

    plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:1.2f}'.format(width),

             ha='left', va='center')
time=data.copy()

time=time.drop(["Type","Code","Continent_Name","Total USD"],axis=1)

time=time.T

time=time.loc[:,:9]

new_header = time.iloc[0]

time = time[1:] 

time.columns = new_header 

time.head()
plt.figure()

time.plot(linestyle='-', marker='*',legend=True)

plt.title('Timeline of Millitary Spending of the top 10 countries (billions)')
df2 = pd.DataFrame(data, columns = ["Name", "2018"])

df2= df2.sort_values(['2018'],ascending=False).reset_index()

top18 = df2.iloc[:10,]

a=top18["Name"]

b=top18["2018"]
sns.set_color_codes("pastel")

ax1 = sns.barplot(y=a,  x=b, data=top18)

ax1.set_xlabel('2018')

ax1.axes.xaxis.label.set_text("Total Spendings (billions)")

ax1.axes.yaxis.label.set_text("Countries")

plt.title('Total Millitary Spending in 2018')

for p in ax1.patches:

    width = p.get_width()

    plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:1.2f}'.format(width),

             ha='left', va='center')
df4 = pd.DataFrame(data, columns = ["Continent_Name", "Name",'Total USD'])

df4.groupby(["Continent_Name"]).apply(lambda x: x.sort_values(["Total USD"], ascending = False).head(5)).reset_index(drop=True)
warscontinent = pd.read_csv(r"../input/continents-codes-and-number-of-wars/wars.csv", low_memory=False,sep=';')

warscontinent=pd.DataFrame(warscontinent, columns = ["Continent_Name", 'Year'])

warscontinent.head()
warscontinent=warscontinent.groupby(["Continent_Name"]).count()

warscontinent= warscontinent.sort_values(['Year'],ascending=False).reset_index()

warscontinent
height=warscontinent['Year']

bars=warscontinent['Continent_Name']

y_pos = np.arange(len(bars))

plt.bar(y_pos, height)

plt.xticks(y_pos, bars, rotation=60)

plt.title ('Number of Wars by Continent 1968-2018')

plt.show()
USDcontinent = pd.DataFrame(data, columns = ["Continent_Name", 'Total USD'])

USDcontinent=USDcontinent.groupby(["Continent_Name"]).sum()

USDcontinent= USDcontinent.sort_values(['Total USD'],ascending=False).reset_index()

USDcontinent
height=USDcontinent['Total USD']

bars=USDcontinent['Continent_Name']

y_pos = np.arange(len(bars))

plt.bar(y_pos, height)

plt.xticks(y_pos, bars, rotation=60)

plt.title ('Spendings in Military by Continent (1968-2018)')

plt.show()
final=pd.merge(USDcontinent, warscontinent , how='left')

final.rename(columns={'Year': 'Number of Wars','Total USD':'Total Amount Spent (USD billions)'},inplace=True)

final.set_index('Continent_Name')
x=final['Continent_Name']

y1=final['Total Amount Spent (USD billions)']

y2=final['Number of Wars']





ax = final.plot(secondary_y="Number of Wars", kind="bar")

ax.set_xlabel('Continents')

ax.set_ylabel('Total Amount Spent')

ax.right_ax.set_ylabel('Number of Wars', color='red')

ax.right_ax.tick_params(axis='y', labelcolor='red')



ax.set_xticklabels(x,rotation=25)

plt.title ('Number of Wars & Military Spendings by Continent (1968-2018)')



plt.show()
#Adding a world map picture form google

ruh_m = plt.imread('../input/continents-codes-and-number-of-wars/world.kpeg.jpg')



#Defining the area of the picture

BBox = ((0, 20, 0, 10)) 

print(BBox)



final['lat']=[3,14,10.5,5.0,10,17.5] #X

final['long']=[8,8,8,3.0,5,2.2]  #Y

final
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(final.lat, final.long,color="orange",marker="p",s=(final["Number of Wars"]*1.5))

ax.scatter(final.lat+0.7, final.long,color="blue",alpha=0.5,s=(final["Total Amount Spent (USD billions)"]/100))

ax.set_title('Number of Wars (orange) and Military Spendings (blue) by continent (1968-2018)')

ax.set_xlim(BBox[0],BBox[1])

ax.set_ylim(BBox[2],BBox[3])

ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')