import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
suicides = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

suicides.tail()
suicides.info()
suicides.isnull().any()
suicides.isnull().sum()
scc = suicides.copy()#for more manipulation we take a copy of dataset.
scc.drop(['HDI for year'],axis = 1,inplace = True) ## Due to we do not need HDI for year column we drop it anyway.
plt.figure(figsize = (14,5)) # Shape of the figure/graphic
best_20_countries = scc.sort_values(by= 'suicides_no',ascending = True) #Making the data best suitable our graphic
sns.boxplot(x='country', y = 'suicides_no',data = best_20_countries); #Visualization
plt.xticks(rotation = 90);# rotate the country names for clearly see in the table
last_20_countries = scc.sort_values(by= 'suicides_no',ascending = True) #Making the data best suitable our graphic
sns.boxplot(x='country', y = 'suicides_no',data = last_20_countries[-500:]); #Visualization
plt.xticks(rotation = 90);# rotate the country names for clearly see in the table
plt.figure(figsize = (14,5))
sns.barplot(x='year',y ='suicides_no',data = scc );#Suicide rates from 1985 to 2016. hue = "sex"
plt.xticks(rotation=90);

#For making much clear of the axis and table names
plt.xlabel('Year 1985-2016')
plt.ylabel('Sucides No')
plt.title('Total Sucides From 1985-2016');

#scc.groupby("year").sum().suicides_no
sns.barplot(x='year',y ='suicides/100k pop',data = scc );#Suicide rates per 100k population from 1985 to 2016.
plt.xticks(rotation=90);
#For making much clear of the axis and table names
plt.xlabel('Year 1985-2016')
plt.ylabel('Sucides per 100k pop')
plt.title('Sucides per 100k Population From 1985-2016');
plt.figure(figsize = (12,6))

sns.pointplot(x='year',y =scc['suicides_no']/20,data = scc,color = 'lime',alpha = 0.1);
sns.pointplot(x='year',y ='suicides/100k pop',data = scc,color = 'blue',alpha = 0.1);
plt.xticks(rotation=90);
#For making much clear of the axis and table names
plt.xlabel('Year 1985-2016')
plt.ylabel('Sucides num-Suicides per 100k pop')
plt.title('Sucides per 100k Population From 1985-2016');


plt.text(35,0,'Sucides number',color='lime',fontsize = 12,style = 'italic')
plt.text(35,0.8,'Suicides 100kpop',color='blue',fontsize = 12,style = 'italic');
plt.grid()

g = sns.jointplot(scc['year'],scc.suicides_no, kind="kde", size=7,ratio = 3,color = 'red')
# plt.savefig('graph.png')
# plt.show()
scc.groupby('generation').sum()
labels = scc.groupby('generation').sum().index
colors = ['grey','blue','red','yellow','green','brown']#["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
explode = [0.05,0.05,0.05,0.05,0.05,0.05]
sizes = scc.groupby('generation').sum()['suicides/100k pop']
# sizes = scc.groupby('generation').sum()['suicides_no']

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')#colors=colors,
plt.title('Suicides 100k pop - Generation',color = 'blue',fontsize = 15);
scc.corr()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(scc.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)#
plt.show()
current_palette = sns.color_palette("husl", 8)
sns.palplot(current_palette)
ASIA = ['Afghanistan',
'Bangladesh',
'Bhutan',
'Brunei',
'Burma',
'Cambodia',
'China',
'East Timor',
'Hong Kong',
'India',
'Indonesia',
'Iran',
'Japan',
'Republic of Korea',
'Laos',
'Macau',
'Malaysia',
'Maldives',
'Mongolia',
'Nepal',
'Pakistan',
'Philippines',
'Singapore',
'Sri Lanka',
'Taiwan',
'Thailand',
'Vietnam']

C_W_OF_IND_STATES=['Armenia',
'Azerbaijan',
'Belarus',
'Georgia',
'Kazakhstan',
'Kyrgyzstan',
'Moldova',
'Russian Federation',
'Tajikistan',
'Turkmenistan',
'Ukraine',
'Uzbekistan']
EASTERN_EUROPE=['Albania','Bosnia and Herzegovina','Bulgaria','Croatia','Czech Republic','Hungary','Macedonia','Poland','Romania']
EASTERN_EUROPE+=['Serbia','Slovakia','Slovenia']
LATIN_AMER_CARIB=['Anguilla',
'Antigua and Barbuda',
'Argentina',
'Aruba',
'Bahamas',
'Barbados',
'Belize',
'Bolivia',
'Brazil',
'British Virgin Is.',
'Cayman Islands',
'Chile',
'Colombia',
'Costa Rica',
'Cuba',
'Dominica',
'Dominican Republic',
'Ecuador',
'El Salvador',
'French Guiana',
'Grenada',
'Guadeloupe',
'Guatemala',
'Guyana',
'Haiti',
'Honduras',
'Jamaica',
'Martinique',
'Mexico',
'Montserrat',
'Netherlands Antilles',
'Nicaragua',
'Panama',
'Paraguay',
'Peru',
'Puerto Rico',
'Saint Kitts and Nevis',
'Saint Lucia',
'Saint Vincent and Grenadines',
'Suriname',
'Trinidad and Tobago',
'Turks and Caicos Is',
'Uruguay',
'Venezuela',
'Virgin Islands']

NEAR_EAST=['Bahrain',
'Cyprus',
'Gaza Strip',
'Iraq',
'Israel',
'Jordan',
'Kuwait',
'Lebanon',
'Oman',
'Qatar',
'Saudi Arabia',
'Syria',
'Turkey',
'United Arab Emirates',
'West Bank',
'Yemen']

NORTHERN_AFRICA=['Algeria',
'Egypt',
'Libya',
'Morocco',
'Tunisia',
'Western Sahara']
NORTHERN_AMERICA=['Bermuda',
'Canada',
'Greenland',
'St Pierre and Miquelon',
'United States']

OCEANIA=['American Samoa',
'Australia',
'Cook Islands',
'Fiji',
'French Polynesia',
'Guam',
'Kiribati',
'Marshall Islands',
'Micronesia, Fed. St.',
'Nauru',
'New Caledonia',
'New Zealand',
'N. Mariana Islands',
'Palau',
'Papua New Guinea',
'Samoa',
'Solomon Islands',
'Tonga',
'Tuvalu',
'Vanuatu',
'Wallis and Futuna']

SUB_SAHARAN_AFRICA=['Angola',
'Benin',
'Botswana',
'Burkina Faso',
'Burundi',
'Cameroon',
'Cape Verde',
'Central African Rep.',
'Chad',
'Comoros',
'Congo, Dem. Rep.',
'Congo, Repub. of the',
'Cote dIvoire',
'Djibouti',
'Equatorial Guinea',
'Eritrea',
'Ethiopia',
'Gabon',
'Gambia, The',
'Ghana',
'Guinea',
'Guinea-Bissau',
'Kenya',
'Lesotho',
'Liberia',
'Madagascar',
'Malawi',
'Mali',
'Mauritania',
'Mauritius',
'Mayotte',
'Mozambique',
'Namibia',
'Niger',
'Nigeria',
'Reunion',
'Rwanda',
'Saint Helena',
'Sao Tome & Principe',
'Senegal',
'Seychelles',
'Sierra Leone',
'Somalia',
'South Africa',
'Sudan',
'Swaziland',
'Tanzania',
'Togo',
'Uganda',
'Zambia',
'Zimbabwe']
WESTERN_EUROPE=['Andorra',
'Austria',
'Belgium',
'Denmark',
'Faroe Islands',
'Finland',
'France',
'Germany',
'Gibraltar',
'Greece',
'Guernsey',
'Iceland',
'Ireland',
'Isle of Man',
'Italy',
'Jersey',
'Liechtenstein',
'Luxembourg',
'Malta',
'Monaco',
'Netherlands',
'Norway',
'Portugal',
'San Marino',
'Spain',
'Sweden',
'Switzerland',
'United Kingdom']
def GetConti(counry):
    if counry in ASIA:
        return "ASIA"
    elif counry in C_W_OF_IND_STATES:
        return "C_W_OF_IND_STATES"
    elif counry in EASTERN_EUROPE:
        return "EASTERN_EUROPE"
    elif counry in LATIN_AMER_CARIB:
        return "LATIN_AMER_CARIB"
    elif counry in NEAR_EAST:
        return "NEAR_EAST"
    elif counry in NORTHERN_AFRICA:
        return "NORTHERN_AFRICA"
    elif counry in NORTHERN_AMERICA:
        return "NORTHERN_AMERICA"
    elif counry in OCEANIA:
        return "OCEANIA"
    elif counry in SUB_SAHARAN_AFRICA:
        return "SUB_SAHARAN_AFRICA"
    elif counry in WESTERN_EUROPE:
        return "WESTERN_EUROPE"
    else:
        return "other"
country=scc["country"]
country=pd.DataFrame(country)
# list(country["country"])
df1 = pd.DataFrame({"Country": list(country["country"])})
df1['Continent'] = df1['Country'].apply(lambda x: GetConti(x))
scc["continent"]=df1["Continent"]
scc[scc["continent"]=="other"]["country"]
continent_list=list(scc['continent'].unique())
suicides_100k_pop = []
for i in continent_list:
    x = scc[scc['continent']==i]
    rate = sum(x['suicides/100k pop'])/len(x)
    suicides_100k_pop.append(rate)
data1 = pd.DataFrame({'Continent_list': continent_list,'suicides/100k pop':suicides_100k_pop})

plt.figure(figsize = (15,15))
plt.subplot(2,2,1)
sns.barplot(x=scc.groupby('continent')['suicides/100k pop'].mean().index,y=scc.groupby('continent')['suicides/100k pop'].mean().values)
plt.title("Global Suicides(per 100K) by Continent")
plt.ylabel("Suicide per 100K")
plt.xlabel("Continents")
plt.xticks(rotation=90)

plt.subplot(2,2,2)
labels =data1.Continent_list
colors = ['grey','blue','red','yellow','green',"orange", "darkblue","purple","maroon","gold"]
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = data1['suicides/100k pop']
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Global Suicides(per 100K) rate of Continents',color = 'blue',fontsize = 15)
plt.show()
scc.info()
scc.groupby('sex')['suicides/100k pop'].sum()
sns.lineplot(x="year", y='suicides/100k pop',data = scc);
sns.lineplot(x='year',y='suicides/100k pop' ,hue = 'sex',data = scc); #hue = "generation",
plt.figure(figsize=(8,6))
sns.scatterplot(x='year', y='suicides_no', data=scc, hue='sex');
plt.figure(figsize=(12,18))
sns.scatterplot(x='population',y = 'country',hue = 'gdp_per_capita ($)',data = scc);
scc.head()
sns.pairplot(scc,kind='reg');#hue="sex"
sns.jointplot(x='suicides/100k pop',y = 'gdp_per_capita ($)',data = scc,kind = 'reg');
plt.figure(figsize=(10,8))
sns.scatterplot(x='suicides/100k pop',y='gdp_per_capita ($)', hue ="age",data=scc);
plt.figure(figsize=(10,8))
sns.scatterplot(x='suicides/100k pop',y='gdp_per_capita ($)',hue= "generation", data=scc);#gedp and Sucides rate dispersion over generations.

gen = scc.groupby('generation').sum()
gen
sns.barplot(x="age",y = 'suicides_no',data=scc);
plt.xticks(rotation=90);
plt.figure(figsize=(10,7))
sns.barplot(x="sex",y = 'suicides_no',hue = 'age',data=scc);
sns.barplot(x="generation",y = 'suicides_no',data=scc);
plt.xticks(rotation=90);
plt.figure(figsize=(10,7))
sns.barplot(x="sex",y = 'suicides_no',hue = 'generation',data=scc);
scc.groupby('generation').sum()
plt.figure(figsize=(12,14))
sns.catplot(x='generation', y="suicides/100k pop", kind="boxen",data=scc);
plt.xticks(rotation=90);
plt.figure(figsize=(12,14))
sns.catplot(x='generation', y="suicides/100k pop", kind="box",data=scc);
plt.xticks(rotation=90);
# scc.groupby('sex').sum()
# scc.sex.value_counts()
sns.countplot(x=scc['generation'], data=scc); #split = True,hue = "sex", kind="count",
plt.xticks(rotation=90);
# plt.figure(figsize=(12,14))
sns.catplot(x='sex', y="suicides_no", kind="violin", data=scc); #split = True,hue = "sex",
plt.xticks(rotation=90);