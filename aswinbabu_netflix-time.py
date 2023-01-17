# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/netflix-shows/netflix_titles.csv")
df.describe()
print(df.dtypes)
#df=df.head()
pd.set_option("display.max_rows",30)
df["date_added"]=df.date_added.str.split(",").str[1]
print(df["date_added"])
df.dropna(subset=["date_added","release_year"],axis=0,inplace=True)
print("dataset is ")
print(df)
df["date_added"]=df["date_added"].astype(int)
#dfNT.reset_index(inplace=True,drop=True)
#print(dfNT)


import matplotlib.pyplot as plt
x=df["release_year"]
y=df["date_added"]

plt.scatter(x,y)
plt.title("time between theatre and netflix release")
plt.xlabel("Release Year")
plt.ylabel("Year added")
df.head()
df_1=df.groupby(["date_added"],as_index=False)["show_id"].count()
#print(df_1)
df_1.pivot_table(index="date_added",values=["show_id"])
df_1=df_1.rename(columns={"date_added":"year_added","show_id":"Number_shows/movies_added"},inplace=False)
print("\nYear wise- number of contents added by netflix\n")
print(df_1)
x=df_1["year_added"]
y=df_1["Number_shows/movies_added"]
plt.scatter(x,y)
plt.title("Increase in number of contents added by netflix each year")
plt.xlabel("---year---",color="Black")
plt.ylabel("--number of contents added--")
plt.show()
print(df["listed_in"].head(2))
dft=df["type"].value_counts()
print("Types of contents in Netflix\n",dft)
a=df["listed_in"].tolist()
b=max(a,key=len)
print("longest string in the list is :",b)

df["listed_in1"]=df.listed_in.str.split(",").str[0]
df["listed_in2"]=df.listed_in.str.split(",").str[1]
df["listed_in3"]=df.listed_in.str.split(",").str[2]

#print(df.head())
print("\nCategories of movies added")
dfa=df["listed_in1"].value_counts()
print(dfa)
print("\nCategories of movies added 2")
dfb=df["listed_in2"].value_counts()
print(dfb)
print("\nCategories of movies added 3")
dfc=df["listed_in3"].value_counts()
print(dfc)
dfb=dfb.reset_index(inplace=False,drop=False)
dfa=dfa.reset_index(inplace=False,drop=False)
dfc=dfc.reset_index(inplace=False,drop=False)
#print("\n1st category\n",dfa,"\n2nd category\n",dfb,"\n3rd category\n",dfc)
dfa["index"]=dfa["index"].str.strip()
dfb["index"]=dfb["index"].str.strip()
dfc["index"]=dfc["index"].str.strip()


#combining the 2(all 3) dataframe using outer join
mergedDf=pd.merge(left=dfa,right=dfb,how="outer",left_on="index",right_on="index")
mergedDf=pd.merge(left=mergedDf,right=dfc,how="outer",left_on="index",right_on="index")
mergedDf["listed_in1"].replace(np.nan,0,inplace=True)
mergedDf["listed_in2"].replace(np.nan,0,inplace=True)
mergedDf["listed_in3"].replace(np.nan,0,inplace=True)

mergedDf["total"]=mergedDf["listed_in1"]+mergedDf["listed_in2"]+mergedDf["listed_in3"]
pd.set_option("display.max_rows",None)
print("different categories and number of releases are :")
mergedDf.sort_values("total",inplace=True,ascending=False)
print(mergedDf)



#visulaization using bar graph
import matplotlib.pyplot as plt
x=mergedDf["index"]
y=mergedDf["total"]
plt.bar(x,y)
plt.xticks(mergedDf["index"],fontsize=5,rotation=90)
plt.xlabel("categories")
plt.ylabel("number of movies/shows")
plt.show()
print(df["country"].isnull().sum())
df2=df[["country","show_id"]]
df2.dropna(subset=["country"],axis=0,inplace=True)
#df2=df2.drop(columns=["show_id"])
df2=pd.DataFrame(df2["country"].str.split(",").tolist(),index=df2.show_id).stack()
print(df2.index)
df2=pd.DataFrame(df2.reset_index(drop=False,inplace=False))
df2=df2.drop(columns=["level_1"])
df2.rename(columns={0:"countries"},inplace=True)
print(df2["countries"].isnull().sum())
#df2=pd.DataFrame(df.country.str.split(",").tolist())
#df1=pd.DataFrame(df1.str.split(",").tolist())
#df1=pd.DataFrame(df1).stack()
df2.head(10)

df2["countries"]=df2.countries.str.strip()
dfv=df2["countries"].value_counts()
dfv=dfv.reset_index(drop=False,inplace=False)
dfv.rename(columns={"index":"countries","countries":"releases"},inplace=True)
print(dfv["countries"].isnull().sum())
dfv["countries"].replace("",np.nan,inplace=True)
dfv

country_codes = {'afghanistan': 'AFG',
 'albania': 'ALB',
 'algeria': 'DZA',
 'american samoa': 'ASM',
 'andorra': 'AND',
 'angola': 'AGO',
 'anguilla': 'AIA',
 'antigua and barbuda': 'ATG',
 'argentina': 'ARG',
 'armenia': 'ARM',
 'aruba': 'ABW',
 'australia': 'AUS',
 'austria': 'AUT',
 'azerbaijan': 'AZE',
 'bahamas': 'BHM',
 'bahrain': 'BHR',
 'bangladesh': 'BGD',
 'barbados': 'BRB',
 'belarus': 'BLR',
 'belgium': 'BEL',
 'belize': 'BLZ',
 'benin': 'BEN',
 'bermuda': 'BMU',
 'bhutan': 'BTN',
 'bolivia': 'BOL',
 'bosnia and herzegovina': 'BIH',
 'botswana': 'BWA',
 'brazil': 'BRA',
 'british virgin islands': 'VGB',
 'brunei': 'BRN',
 'bulgaria': 'BGR',
 'burkina faso': 'BFA',
 'burma': 'MMR',
 'burundi': 'BDI',
 'cabo verde': 'CPV',
 'cambodia': 'KHM',
 'cameroon': 'CMR',
 'canada': 'CAN',
 'cayman islands': 'CYM',
 'central african republic': 'CAF',
 'chad': 'TCD',
 'chile': 'CHL',
 'china': 'CHN',
 'colombia': 'COL',
 'comoros': 'COM',
 'congo democratic': 'COD',
 'Congo republic': 'COG',
 'cook islands': 'COK',
 'costa rica': 'CRI',
 "cote d'ivoire": 'CIV',
 'croatia': 'HRV',
 'cuba': 'CUB',
 'curacao': 'CUW',
 'cyprus': 'CYP',
 'czech republic': 'CZE',
 'denmark': 'DNK',
 'djibouti': 'DJI',
 'dominica': 'DMA',
 'dominican republic': 'DOM',
 'ecuador': 'ECU',
 'egypt': 'EGY',
 'el salvador': 'SLV',
 'equatorial guinea': 'GNQ',
 'eritrea': 'ERI',
 'estonia': 'EST',
 'ethiopia': 'ETH',
 'falkland islands': 'FLK',
 'faroe islands': 'FRO',
 'fiji': 'FJI',
 'finland': 'FIN',
 'france': 'FRA',
 'french polynesia': 'PYF',
 'gabon': 'GAB',
 'gambia, the': 'GMB',
 'georgia': 'GEO',
 'germany': 'DEU',
 'ghana': 'GHA',
 'gibraltar': 'GIB',
 'greece': 'GRC',
 'greenland': 'GRL',
 'grenada': 'GRD',
 'guam': 'GUM',
 'guatemala': 'GTM',
 'guernsey': 'GGY',
 'guinea-bissau': 'GNB',
 'guinea': 'GIN',
 'guyana': 'GUY',
 'haiti': 'HTI',
 'honduras': 'HND',
 'hong kong': 'HKG',
 'hungary': 'HUN',
 'iceland': 'ISL',
 'india': 'IND',
 'indonesia': 'IDN',
 'iran': 'IRN',
 'iraq': 'IRQ',
 'ireland': 'IRL',
 'isle of man': 'IMN',
 'israel': 'ISR',
 'italy': 'ITA',
 'jamaica': 'JAM',
 'japan': 'JPN',
 'jersey': 'JEY',
 'jordan': 'JOR',
 'kazakhstan': 'KAZ',
 'kenya': 'KEN',
 'kiribati': 'KIR',
 'north korea': 'PRK',
 'south korea': 'KOR',
 'kosovo': 'KSV',
 'kuwait': 'KWT',
 'kyrgyzstan': 'KGZ',
 'laos': 'LAO',
 'latvia': 'LVA',
 'lebanon': 'LBN',
 'lesotho': 'LSO',
 'liberia': 'LBR',
 'libya': 'LBY',
 'liechtenstein': 'LIE',
 'lithuania': 'LTU',
 'luxembourg': 'LUX',
 'macau': 'MAC',
 'macedonia': 'MKD',
 'madagascar': 'MDG',
 'malawi': 'MWI',
 'malaysia': 'MYS',
 'maldives': 'MDV',
 'mali': 'MLI',
 'malta': 'MLT',
 'marshall islands': 'MHL',
 'mauritania': 'MRT',
 'mauritius': 'MUS',
 'mexico': 'MEX',
 'micronesia': 'FSM',
 'moldova': 'MDA',
 'monaco': 'MCO',
 'mongolia': 'MNG',
 'montenegro': 'MNE',
 'morocco': 'MAR',
 'mozambique': 'MOZ',
 'namibia': 'NAM',
 'nepal': 'NPL',
 'netherlands': 'NLD',
 'new caledonia': 'NCL',
 'new zealand': 'NZL',
 'nicaragua': 'NIC',
 'nigeria': 'NGA',
 'niger': 'NER',
 'niue': 'NIU',
 'northern mariana islands': 'MNP',
 'norway': 'NOR',
 'oman': 'OMN',
 'pakistan': 'PAK',
 'palau': 'PLW',
 'panama': 'PAN',
 'papua new guinea': 'PNG',
 'paraguay': 'PRY',
 'peru': 'PER',
 'philippines': 'PHL',
 'poland': 'POL',
 'portugal': 'PRT',
 'puerto rico': 'PRI',
 'qatar': 'QAT',
 'romania': 'ROU',
 'russia': 'RUS',
 'rwanda': 'RWA',
 'saint kitts and nevis': 'KNA',
 'saint lucia': 'LCA',
 'saint martin': 'MAF',
 'saint pierre and miquelon': 'SPM',
 'saint vincent and the grenadines': 'VCT',
 'samoa': 'WSM',
 'san marino': 'SMR',
 'sao tome and principe': 'STP',
 'saudi arabia': 'SAU',
 'senegal': 'SEN',
 'serbia': 'SRB',
 'seychelles': 'SYC',
 'sierra leone': 'SLE',
 'singapore': 'SGP',
 'sint maarten': 'SXM',
 'slovakia': 'SVK',
 'slovenia': 'SVN',
 'solomon islands': 'SLB',
 'somalia': 'SOM',
 'south africa': 'ZAF',
 'south sudan': 'SSD',
 'spain': 'ESP',
 'sri lanka': 'LKA',
 'sudan': 'SDN',
 'suriname': 'SUR',
 'swaziland': 'SWZ',
 'sweden': 'SWE',
 'switzerland': 'CHE',
 'syria': 'SYR',
 'taiwan': 'TWN',
 'tajikistan': 'TJK',
 'tanzania': 'TZA',
 'thailand': 'THA',
 'timor-leste': 'TLS',
 'togo': 'TGO',
 'tonga': 'TON',
 'trinidad and tobago': 'TTO',
 'tunisia': 'TUN',
 'turkey': 'TUR',
 'turkmenistan': 'TKM',
 'tuvalu': 'TUV',
 'uganda': 'UGA',
 'ukraine': 'UKR',
 'united arab emirates': 'ARE',
 'united kingdom': 'GBR',
 'united states': 'USA',
 'uruguay': 'URY',
 'uzbekistan': 'UZB',
 'vanuatu': 'VUT',
 'venezuela': 'VEN',
 'vietnam': 'VNM',
 'virgin islands': 'VGB',
 'west bank': 'WBG',
 'yemen': 'YEM',
 'zambia': 'ZMB',
 'zimbabwe': 'ZWE'}

#converting Dicionary into a dataframe
dfcoun=pd.DataFrame(country_codes.items(),columns=["countries","country_code"])


#joining the dfcoun and dfv
dfv["countries"]=dfv.countries.str.lower()
dfcoun["countries"]=dfcoun.countries.str.lower()
mergedDf=pd.merge(left=dfv,right=dfcoun,how="left",left_on="countries",right_on="countries")
mergedDf.head()
import geopandas as gpd
loc="../input/shapefiles/ne_110m_admin_0_countries.shp"
gdf=gpd.read_file(loc)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.rename(columns={'ADMIN':"country", 'ADM0_A3':"country_code"},inplace=True)
gdf.head()
mergedGdf=pd.merge(left=mergedDf,right=gdf,how="left",left_on="country_code",right_on="country_code")
print(mergedGdf.dtypes)
mergedGdf.dropna(subset=["countries","releases","country_code","country","geometry"],axis=0,inplace=True)
print(mergedGdf.count())
mergedGdf

import json
merged_json=json.loads(mergedGdf.to_json(default_handler=str))
json_data=json.dumps(merged_json)
from bokeh.io import output_notebook,show, output_file
from bokeh.plotting import figure,output_file,show
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
from bokeh.models import Slider, HoverTool
from bokeh.sampledata.sample_geojson import geojson ###
from bokeh.io import curdoc, output_notebook
#from bokeh.models import Slider, HoverTool
from bokeh.layouts import widgetbox, row, column
#geojson is used for plotting using xs and ys cordinates
geosource=GeoJSONDataSource(geojson=json_data)

palette=brewer['YlGnBu'][8] #defining the color for visuals
palette=palette[::-1] #reversing the color order so the most obese is blue

#linear color mapper is useful in mapping the values linearrly to colors
color_mapper=LinearColorMapper(palette=palette,low=0,high=2600)
hover = HoverTool(tooltips = [('Country/region','@country'),('Releases', '@releases')])

#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
border_line_color=None,location = (0,0), orientation = 'horizontal')
#Create figure object.
p = figure(title = 'country wise release numbers', plot_height = 600 , plot_width = 950, toolbar_location = None, tools=[hover])
p.xgrid.grid_line_color = None 
p.ygrid.grid_line_color = None 
#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource,fill_color = {'field' : 'releases', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
#Specify figure layout.
p.add_layout(color_bar, 'below')
#Display figure inline in Jupyter Notebook.
output_notebook()
#Display figure.
show(p)