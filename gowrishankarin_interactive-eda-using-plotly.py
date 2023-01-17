# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

DATA_DIR = "../input"





import os

print(os.listdir(DATA_DIR))



# Any results you write to the current directory are saved as output.
import json

from pandas.io.json import json_normalize
%%time

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

train_df = pd.read_csv(DATA_DIR + "/train_v2.csv", dtype={"fullVisitorId": "str"}, 

    converters={column: json.loads for column in JSON_COLUMNS}, nrows=500000)
import json

from pandas.io.json import json_normalize

def json_to_dataframe(data, column_name):

    json_as_df = json_normalize(data[column_name])

    json_as_df.columns = [f"{column_name}.{subcolumn}" for subcolumn in json_as_df.columns]

    data = data.drop(column_name, axis=1).merge(json_as_df, right_index=True, left_index=True)

    return data
%%time

train_df = json_to_dataframe(train_df, "device")

train_df = json_to_dataframe(train_df, "geoNetwork")

train_df = json_to_dataframe(train_df, "totals")

train_df = json_to_dataframe(train_df, "trafficSource")

train_df.shape
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate("count").reset_index()
dtype_df.head(55)
train_df.head(2)
missing_values_df = train_df.isnull().sum(axis=0).reset_index()

missing_values_df.columns = ["Column_Name", "Missing_Count"]

missing_values_df = missing_values_df.loc[missing_values_df["Missing_Count"] > 0]

print(missing_values_df.shape)

missing_values_df
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999

pd.options.display.float_format = '{:20,.2f}'.format
continent_series = train_df["geoNetwork.continent"].value_counts()

labels = (np.array(continent_series.index))

sizes = (np.array((continent_series / continent_series.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(title="Continent Distribution")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="continent")
country_series = train_df["geoNetwork.country"].value_counts().head(25)

country_count = country_series.shape[0]

print("Total No. Of Countries: ", country_count)

country_series = country_series.head(25)



trace = go.Bar(

    x=country_series.index,

    y=country_series.values,

    marker=dict(

        color=country_series.values,

        showscale=True

    ),

)

layout = go.Layout(title="Countrywise Observation Count")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="country")
americas_df = train_df[train_df["geoNetwork.continent"] == "Americas"]

print("# of Observations in Americas: ", americas_df.shape[0])

americas_series = americas_df["geoNetwork.country"].value_counts()

trace = go.Bar(

    x=americas_series.index,

    y=americas_series.values,

    marker=dict(

        color=americas_series.values,

        colorscale="Viridis",

        showscale=True

    ),

)

layout = go.Layout(title="Americas Observation Count")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="americas")
not_usa_df = train_df[train_df["geoNetwork.country"] != "United States"]

print("# of Observations from Rest of the World: ", not_usa_df.shape[0])

not_usa_series = not_usa_df["geoNetwork.country"].value_counts().head(50)

trace = go.Bar(

    x=not_usa_series.index,

    y=not_usa_series.values,

    marker=dict(

        color=not_usa_series.values,

        colorscale="Viridis",

        showscale=True

    ),

)

layout = go.Layout(title="Not USA Observation Count")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="not_usa")
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')

revenue_by_country = train_df.groupby("geoNetwork.country")["totals.transactionRevenue"].sum().reset_index()

print("Total Transaction Revenue: %8.2f" % revenue_by_country["totals.transactionRevenue"].sum())

#revenue_by_country = np.sort(revenue_by_country.transactionRevenue.values)

revenue_by_country.sort_values(by ="totals.transactionRevenue", ascending=False, inplace=True)

revenue_by_country = revenue_by_country[revenue_by_country["totals.transactionRevenue"] > 0]



active_countries_count = revenue_by_country.shape[0]

print("% of Countries contributing: ", (active_countries_count/country_count)*100)

revenue_by_country.head(25)
LOCDATA="""COUNTRY,GDP (BILLIONS),CODE

Afghanistan,21.71,AFG

Albania,13.40,ALB

Algeria,227.80,DZA

American Samoa,0.75,ASM

Andorra,4.80,AND

Angola,131.40,AGO

Anguilla,0.18,AIA

Antigua and Barbuda,1.24,ATG

Argentina,536.20,ARG

Armenia,10.88,ARM

Aruba,2.52,ABW

Australia,1483.00,AUS

Austria,436.10,AUT

Azerbaijan,77.91,AZE

"Bahamas, The",8.65,BHM

Bahrain,34.05,BHR

Bangladesh,186.60,BGD

Barbados,4.28,BRB

Belarus,75.25,BLR

Belgium,527.80,BEL

Belize,1.67,BLZ

Benin,9.24,BEN

Bermuda,5.20,BMU

Bhutan,2.09,BTN

Bolivia,34.08,BOL

Bosnia and Herzegovina,19.55,BIH

Botswana,16.30,BWA

Brazil,2244.00,BRA

British Virgin Islands,1.10,VGB

Brunei,17.43,BRN

Bulgaria,55.08,BGR

Burkina Faso,13.38,BFA

Burma,65.29,MMR

Burundi,3.04,BDI

Cabo Verde,1.98,CPV

Cambodia,16.90,KHM

Cameroon,32.16,CMR

Canada,1794.00,CAN

Cayman Islands,2.25,CYM

Central African Republic,1.73,CAF

Chad,15.84,TCD

Chile,264.10,CHL

"People 's Republic of China",10360.00,CHN

Colombia,400.10,COL

Comoros,0.72,COM

"Congo, Democratic Republic of the",32.67,COD

"Congo, Republic of the",14.11,COG

Cook Islands,0.18,COK

Costa Rica,50.46,CRI

Cote d'Ivoire,33.96,CIV

Croatia,57.18,HRV

Cuba,77.15,CUB

Curacao,5.60,CUW

Cyprus,21.34,CYP

Czech Republic,205.60,CZE

Denmark,347.20,DNK

Djibouti,1.58,DJI

Dominica,0.51,DMA

Dominican Republic,64.05,DOM

Ecuador,100.50,ECU

Egypt,284.90,EGY

El Salvador,25.14,SLV

Equatorial Guinea,15.40,GNQ

Eritrea,3.87,ERI

Estonia,26.36,EST

Ethiopia,49.86,ETH

Falkland Islands (Islas Malvinas),0.16,FLK

Faroe Islands,2.32,FRO

Fiji,4.17,FJI

Finland,276.30,FIN

France,2902.00,FRA

French Polynesia,7.15,PYF

Gabon,20.68,GAB

"Gambia, The",0.92,GMB

Georgia,16.13,GEO

Germany,3820.00,DEU

Ghana,35.48,GHA

Gibraltar,1.85,GIB

Greece,246.40,GRC

Greenland,2.16,GRL

Grenada,0.84,GRD

Guam,4.60,GUM

Guatemala,58.30,GTM

Guernsey,2.74,GGY

Guinea-Bissau,1.04,GNB

Guinea,6.77,GIN

Guyana,3.14,GUY

Haiti,8.92,HTI

Honduras,19.37,HND

Hong Kong,292.70,HKG

Hungary,129.70,HUN

Iceland,16.20,ISL

India,2048.00,IND

Indonesia,856.10,IDN

Iran,402.70,IRN

Iraq,232.20,IRQ

Ireland,245.80,IRL

Isle of Man,4.08,IMN

Israel,305.00,ISR

Italy,2129.00,ITA

Jamaica,13.92,JAM

Japan,4770.00,JPN

Jersey,5.77,JEY

Jordan,36.55,JOR

Kazakhstan,225.60,KAZ

Kenya,62.72,KEN

Kiribati,0.16,KIR

"Korea, North",28.00,PRK

"Korea, South",1410.00,KOR

Kosovo,5.99,KSV

Kuwait,179.30,KWT

Kyrgyzstan,7.65,KGZ

Laos,11.71,LAO

Latvia,32.82,LVA

Lebanon,47.50,LBN

Lesotho,2.46,LSO

Liberia,2.07,LBR

Libya,49.34,LBY

Liechtenstein,5.11,LIE

Lithuania,48.72,LTU

Luxembourg,63.93,LUX

Macau,51.68,MAC

Macedonia,10.92,MKD

Madagascar,11.19,MDG

Malawi,4.41,MWI

Malaysia,336.90,MYS

Maldives,2.41,MDV

Mali,12.04,MLI

Malta,10.57,MLT

Marshall Islands,0.18,MHL

Mauritania,4.29,MRT

Mauritius,12.72,MUS

Mexico,1296.00,MEX

"Micronesia, Federated States of",0.34,FSM

Moldova,7.74,MDA

Monaco,6.06,MCO

Mongolia,11.73,MNG

Montenegro,4.66,MNE

Morocco,112.60,MAR

Mozambique,16.59,MOZ

Namibia,13.11,NAM

Nepal,19.64,NPL

Netherlands,880.40,NLD

New Caledonia,11.10,NCL

New Zealand,201.00,NZL

Nicaragua,11.85,NIC

Nigeria,594.30,NGA

Niger,8.29,NER

Niue,0.01,NIU

Northern Mariana Islands,1.23,MNP

Norway,511.60,NOR

Oman,80.54,OMN

Pakistan,237.50,PAK

Palau,0.65,PLW

Panama,44.69,PAN

Papua New Guinea,16.10,PNG

Paraguay,31.30,PRY

Peru,208.20,PER

Philippines,284.60,PHL

Poland,552.20,POL

Portugal,228.20,PRT

Puerto Rico,93.52,PRI

Qatar,212.00,QAT

Romania,199.00,ROU

Russia,2057.00,RUS

Rwanda,8.00,RWA

Saint Kitts and Nevis,0.81,KNA

Saint Lucia,1.35,LCA

Saint Martin,0.56,MAF

Saint Pierre and Miquelon,0.22,SPM

Saint Vincent and the Grenadines,0.75,VCT

Samoa,0.83,WSM

San Marino,1.86,SMR

Sao Tome and Principe,0.36,STP

Saudi Arabia,777.90,SAU

Senegal,15.88,SEN

Serbia,42.65,SRB

Seychelles,1.47,SYC

Sierra Leone,5.41,SLE

Singapore,307.90,SGP

Sint Maarten,304.10,SXM

Slovakia,99.75,SVK

Slovenia,49.93,SVN

Solomon Islands,1.16,SLB

Somalia,2.37,SOM

South Africa,341.20,ZAF

South Sudan,11.89,SSD

Spain,1400.00,ESP

Sri Lanka,71.57,LKA

Sudan,70.03,SDN

Suriname,5.27,SUR

Swaziland,3.84,SWZ

Sweden,559.10,SWE

Switzerland,679.00,CHE

Syria,64.70,SYR

Taiwan,529.50,TWN

Tajikistan,9.16,TJK

Tanzania,36.62,TZA

Thailand,373.80,THA

Timor-Leste,4.51,TLS

Togo,4.84,TGO

Tonga,0.49,TON

Trinidad and Tobago,29.63,TTO

Tunisia,49.12,TUN

Turkey,813.30,TUR

Turkmenistan,43.50,TKM

Tuvalu,0.04,TUV

Uganda,26.09,UGA

Ukraine,134.90,UKR

United Arab Emirates,416.40,ARE

United Kingdom,2848.00,GBR

United States,17420.00,USA

Uruguay,55.60,URY

Uzbekistan,63.08,UZB

Vanuatu,0.82,VUT

Venezuela,209.20,VEN

Vietnam,187.80,VNM

Virgin Islands,5.08,VGB

West Bank,6.64,WBG

Yemen,45.45,YEM

Zambia,25.61,ZMB

Zimbabwe,13.74,ZWE

    """

with open("location.csv", "w") as ofile:

    ofile.write(LOCDATA)
location_df = pd.read_csv("location.csv")

revenue_by_country_new = pd.merge(

    revenue_by_country[["geoNetwork.country", "totals.transactionRevenue"]], location_df, 

    left_on="geoNetwork.country", right_on="COUNTRY")



revenue_by_country_new.head(25)
revenue_by_country_new = revenue_by_country_new[revenue_by_country_new["geoNetwork.country"] != "United States"]



data = [dict(

    type="choropleth",

    locations=revenue_by_country_new["CODE"],

    z=revenue_by_country_new["totals.transactionRevenue"],

    text=revenue_by_country_new["geoNetwork.country"],

    colorscale="Viridis",

    autocolorscale=False,

    reversescale=True,

    marker=dict(

        line=dict(

            color="rgb(180, 180, 180)",

            width=0.5

        )

    ),

    colorbar=dict(

        autotick=False,

        title="Transaction Revenue"

    )

)]

layout = dict(

    title="Countrywise Transaction Revenue",

    geo=dict(

        showframe=False,

        showcoastlines=True,

        projection=dict(type="Mercator")

    )

)

fig = dict(data=data, layout=layout)

# py.iplot(fig, validate=False, filename="d3-world-map")
train_active_df = train_df[train_df["totals.transactionRevenue"] > 0]

country_series = train_active_df["device.browser"].value_counts()

country_count = country_series.shape[0]

print("Total No. Of Countries: ", country_count)

country_series = country_series.head(25)

trace = go.Bar(

    x=country_series.index,

    y=country_series.values,

    marker=dict(

        color=country_series.values,

        colorscale="Viridis",

        showscale=True

    ),

)

layout = go.Layout(title="Countrywise Observation Count")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="country")
device_category = train_active_df["device.deviceCategory"].value_counts()

labels = (np.array(device_category.index))

sizes = (np.array((device_category / device_category.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(title="Device Category Distribution")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="device_category")
is_mobile = train_active_df["device.isMobile"].value_counts()

labels = (np.array(is_mobile.index))

sizes = (np.array((is_mobile / is_mobile.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(title="Mobile or Not")

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="device_category")
visitor_transactions = train_active_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

visitor_transactions.head(25)
trace = go.Scatter(

    x=visitor_transactions.index,

    y=np.sort(visitor_transactions["totals.transactionRevenue"].values.astype(float)),

    mode='markers',

    marker=dict(

        sizemode="diameter",

        sizeref=2,

        size=2,

        color=visitor_transactions["totals.transactionRevenue"].values.astype(float),

        colorscale="Viridis",

        showscale=True

    )

    

)



layout = go.Layout(

    title='Transaction Revenue Distribution by Visitor'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="transact_rev_by_vistor")


trace = go.Scatter(

    x=visitor_transactions.index,

    y=np.sort(np.log1p(visitor_transactions["totals.transactionRevenue"].values.astype(float))),

    mode='markers',

    marker=dict(

        color=visitor_transactions["totals.transactionRevenue"].values.astype(float),

        colorscale="Viridis",

        showscale=True

    )

    

)



layout = go.Layout(

    title='Transaction Revenue Distribution by Visitor'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="transact_rev_by_vistor")