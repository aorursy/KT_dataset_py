# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")
datayedek=data.copy()
data.shape
data.country.unique()
data.info()
data.describe().T
data.country.value_counts().plot.barh()
data.sex.value_counts().plot.barh()
data.age.value_counts().plot.barh()
data["suicides/100k pop"].value_counts().plot.barh()
data["country-year"].value_counts().plot.barh()
data.generation.value_counts().plot.barh()
data.isnull().sum()

data["HDI for year"].fillna(data["HDI for year"].mean(),inplace=True)
data.isnull().sum()


data.year=data[data.year<2016].year
sns.lineplot(x="year",y="suicides/100k pop",data=data);


data.year=data[data.year<2016].year
sns.lineplot(x="year",y="suicides/100k pop",hue="sex",data=data);

f=plt.figure(figsize=[7,5])
axes=f.add_axes([0.1,0.1,0.9,0.9])
axes.plot(data["gdp_per_capita ($)"],data.population)
axes.set_xlabel("GDP Per Capita")
axes.set_xlabel("Population")
axes.set_title("Population-gdp_per_capita Plot")

data.head()
#sns.lineplot(x="gdp_per_capita ($)",y="suicides/100k pop",data=data);
#sns.barplot(x="gdp_per_capita ($)",y="suicides/100k pop",data=data);  
sns.catplot(x="gdp_per_capita ($)",y="suicides/100k pop",kind="point",data=data);

sns.catplot(x="gdp_per_capita ($)",y="suicides/100k pop",kind="point",data=data);
sns.barplot(x="generation",y=data.sex.index,hue="sex",data=data);

sns.boxplot(x="age",y="suicides/100k pop",data=data);
sns.barplot(data.age,data["suicides/100k pop"],data=data);
sns.boxplot(x="generation",y="suicides/100k pop",data=data);
sns.barplot(data.generation,data["suicides/100k pop"],data=data);
europe = ["Albania","Andorra","Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria",
          "Croatia","Czechia","Denmark","Estonia","Faroe Islands","Finland","France","Germany",
          "Gibraltar","Greece","Guernsey","Hungary","Iceland","Ireland","Isle of Man","Italy",
          "Jersey","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg",
          "Former Yugoslav Republic of Macedonia","Malta","Moldova","Monaco","Czech Republic",
          "Montenegro","Netherlands","Norway","Poland","Portugal","Romania","Russian Federation","San Marino","Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom","Vatican City"]
africa = ["South Africa",'Morocco', 'Tunisia', 'Africa', 'ZA', 'Kenya',"Algeria","Angola","Benin",
          "Botswana","Burkina Faso","Burundi","Cameroon","Central African Republic","Chad","Comoros","Democratic Republic of Congo",
"Republic of Congo","Côte d'Ivoire","Djibouti","Egypt","Equatorial Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau",
"Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Mayotte","Morocco","Mozambique","Namibia","Niger",
"Nigeria","Réunion","Rwanda","Saint Helena","Sao Tome and Principe","Cabo Verde","Seychelles","Sierra Leona","Somalia","Sudan","Swaziland","Tanzania","Togo","Tunisia","Uganda","Western Sahara","Zambia","Zimbabwe"]

antarctica= ["South Orkney Islands","South Shetland Islands","South Georgia","South Sandwich Islands","Australia","Bouvet Island","Heard Island and McDonald Island","Scott Island and the Balleny Islands"]

asia= ["Afghanistan","Armenia","Azerbaijan","Bahrain","Bangladesh","Bhutan","British Indian Ocean Territory","Brunei","Cambodia","People's Republic of China",
"Christmas Island","Cocos","Cyprus","Georgia","Hong Kong","India","Indonesia","Iran","Iraq","Israel","Japan","Jordan","Kazakhstan",
"Democratic People's Republic of Korea","Republic of Korea","Kuwait","Kyrgyzstan","Laos","Lebanon","Macau","Malaysia","Maldives","Mongolia","Myanmar","Nepal","Oman","Pakistan","Palestinian Territories","Philippines","Qatar","Saudi Arabia","Singapore",
"Sri Lanka","Syria","Taiwan","Tajikistan","Thailand","Timor-Leste","Turkey","Turkmenistan","United Arab Emirates","Uzbekistan","Vietnam","Yemen"]
southamerica= ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Falkland Islands","French Guiana","Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"]

northamerica= ["Anguilla","Antigua and Barbuda","Aruba",
               "Bahamas","Barbados","Belize","Bermuda","British Virgin Islands",
               "Canada","Cayman Islands","Clipperton Island","Costa Rica","Cuba",
               "Dominica","Dominican Republic","El Salvador","Greenland","Grenada",
               "Guadeloupe","Guatemala","Haiti","Honduras","Jamaica","Martinique",
               "Mexico","Montserrat","Navassa Island","Netherlands Antilles",
               "Nicaragua","Panama","Puerto Rico","Saint Barthélemy","Saint Kitts and Nevis",
               "Saint Lucia","Saint Martin","Saint Pierre and Miquelon","Saint Vincent and Grenadines","Trinidad and Tobago","Turks and Caicos Islands","United States","United States Virgin Islands"]

Oceania= ["American Samoa","Australia","Baker Island","Cooks Islands","Fiji","French Polynesia","Guam (Hagåtña)","Howland Island","Jarvis Island","Johnston Atoll","Kingman Reef","Kiribati","Marshall Islands","Micronesia","Midway Atoll","Nauru","New Caledonia","New Zealand","Niue","Norfolk Island","Northern Mariana Islands","Palau","Palmyra Atoll,""Papua New Guinea","Pitcairn Islands","Samoa","Solomon Islands","Tokelau","Tonga","Tuvalu","Vanuatu","Wake Island","Wallis and Futuna"]
def GetConti(counry):
    if counry in asia:
        return "Asia"
    elif counry in europe:
        return "Europe"
    elif counry in africa:
        return "Africa"
    elif counry in Oceania:
        return "Oceania"
    elif counry in northamerica:
        return "North America"    
    elif counry in southamerica:
        return "South America"
    elif counry in antarctica:
        return "Antarctica"
    else:
        return "other"


data['Continent'] = data['country'].apply(lambda x: GetConti(x))

sns.barplot(data.Continent,data["suicides/100k pop"],data=data);

