import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns 



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/countries of the world.csv')

#data reading
#first of all, make smaller the names of the colon and changing name be easy to write code

data.columns=[each.lower() for each in data]

new_columns_name = ['country', 'region', 'population', 'area', 'pop_density','coastline', 

                    'net_migration', 'infant_mortality', 'GDP','literacy', 'phone_using',

                    'arable', 'crops',  'other', 'climate', 'birthrate', 

                    'deathrate', 'agriculture', 'industry','service']



data.columns = new_columns_name
data.info()
# We can show there are NaN values in the data. It's problem because requires columns to be numeric.

data.dropna(how="any",inplace=True)



#NaN values destroyed :)
def convert_currency(val):

    new_val = val.replace(',','.')

    return float(new_val)

data["infant_mortality"] = data["infant_mortality"].apply(convert_currency)

data["net_migration"] = data["net_migration"].apply(convert_currency)

data["agriculture"] = data["agriculture"].apply(convert_currency)

data["birthrate"] = data["birthrate"].apply(convert_currency)

data["deathrate"] = data["deathrate"].apply(convert_currency)

data["pop_density"] = data["pop_density"].apply(convert_currency)

data["literacy"] = data["literacy"].apply(convert_currency)

data["phone_using"] = data["phone_using"].apply(convert_currency)

data["industry"] = data["industry"].apply(convert_currency)

data["coastline"] = data["coastline"].apply(convert_currency)

data["arable"] = data["arable"].apply(convert_currency)

data["climate"] = data["climate"].apply(convert_currency)

data["crops"] = data["crops"].apply(convert_currency)

data["service"] = data["service"].apply(convert_currency)



#               ________"Object" convert to "Integer"_________
data.info()
# we can show the correlation of data columns relative to each other.

data.corr()

plt.figure(figsize=(16,12))

sns.heatmap(data=data.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm')

plt.show()

# If the value "1", the colon is related.
data.region
data["infant_mortality"] = data["infant_mortality"].astype(str)

data["infant_mortality"] = data["infant_mortality"].apply(convert_currency)

data1=data

area_list = list(data1['region'].unique())

mortality_ratio = []

for i in area_list:

    x = data1[data1['region']==i]

    mortality_rate = sum(x["infant_mortality"])/len(x)

    mortality_ratio.append(mortality_rate)

data1 = pd.DataFrame({'area_list': area_list,'area_mortality_ratio':mortality_ratio})

new_index = (data1['area_mortality_ratio'].sort_values(ascending=False)).index.values

sorted_data = data1.reindex(new_index)
plt.figure(figsize=(15,9))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_mortality_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Regions')

plt.ylabel('Mortality Birth Rate')

plt.title('Mortality Birth Rate Given States')

plt.show()
data_mortality_country_high = data[(data['infant_mortality'] > 80)]

print(data_mortality_country_high)

len(data_mortality_country_high)
data_mortality_country_low= data[(data['infant_mortality'] < 5)]

print(data_mortality_country_low)

len(data_mortality_country_low)
print(data.loc[:10,"country":"infant_mortality"])
plt.subplot(2,1,1)

plt.scatter(data_mortality_country_low.pop_density, data_mortality_country_low.GDP, color="blue", label="mortality low")

plt.ylabel("GDP")

plt.legend()

plt.subplot(2,1,2)

plt.scatter(data_mortality_country_high.pop_density, data_mortality_country_high.GDP, color="red", label="mortality high")

plt.ylabel("GDP")

plt.xlabel("Population Density")

plt.legend()

plt.show()
plt.subplot(2,1,1)

plt.scatter(data_mortality_country_low.pop_density, data_mortality_country_low.literacy, color="green", label="mortality low")

plt.ylabel("Literacy")

plt.legend()

plt.subplot(2,1,2)

plt.scatter(data_mortality_country_high.pop_density, data_mortality_country_high.literacy, color="black", label="mortality high")

plt.ylabel("Literacy")

plt.xlabel("Population Density")

plt.legend()

plt.show()
plt.subplot(2,1,1)

plt.scatter(data_mortality_country_low.GDP, data_mortality_country_low.population, color="purple", label="mortality low")

plt.ylabel("GDP")

plt.legend()

plt.subplot(2,1,2)

plt.scatter(data_mortality_country_high.GDP, data_mortality_country_high.population, color="black", label="mortality high")

plt.ylabel("GDP")

plt.xlabel("Population")

plt.legend()

plt.show()
data["literacy"].plot(kind="hist",color="magenta",bins=30,grid=True,alpha=0.6,label="Birthrate",figsize=(20,8))

plt.legend()

plt.xlabel("Birthrate",color="brown",size=12)

plt.ylabel("Frequency",color="brown",size=12)

plt.title("Birthrate Distribution")

plt.show()
# _______Data Melting and uisng Seaborn lib.



datamelt1 = data.melt(value_vars="literacy",value_name="literacy",id_vars="country")

datamelt1["literacy"].head()

plt.figure(figsize=(20,35))

sns.set()

sns.barplot(x=datamelt1["literacy"],y=datamelt1["country"].unique())

plt.xlabel("Deathrate")

plt.ylabel("Country")

plt.title("Deathrate-Country")

plt.tight_layout()

plt.show()



sns.jointplot(x = 'literacy', y ='GDP', data = data, kind = 'hex',size=15, joint_kws={'gridsize':35})

    

#Literacy increased as GDP increased or vice versa.
#_____Filtering ____



MCC = data[data.population > 100000000]

MCC

# MCC : Most Crowded Country
#sns.lmplot(x="birthrate", y="deathrate", hue="GDP", data=data_mortality_country_high,  palette="Set1")

#print(data_mortality_country_high.country)





plt.scatter(data_mortality_country_high.birthrate, data_mortality_country_high.deathrate, s=120)

plt.xlabel("Birthrate")

plt.ylabel("Deathrate")

plt.show()

print(data_mortality_country_high.country)

# Death and birth rate in countries with high GDP.
## Death and birth rate in countries with low GDP.



plt.scatter(data_mortality_country_low.birthrate, data_mortality_country_low.deathrate, s=120)

plt.xlabel("Birthrate")

plt.ylabel("Deathrate")

plt.show()

print(data_mortality_country_low.country)
sns.set(style="whitegrid")              

g = sns.jointplot("arable", "climate", data=data, kind="reg",xlim=(0, 60), ylim=(0, 12), color="Green")

plt.show()



# I think so;



#1: Spring

#2: Summer 

#3: Autumn

#4: Winter

#we can see peak in climate=2 values.
data[data.climate==2].arable.hist(bins=20, color='blue')

data[data.climate==3].crops.hist(bins=20, color='purple')

plt.xlabel('arable')

plt.ylabel('crops')

plt.show()
x= data.agriculture

y= data.GDP

sns.set(style="white", color_codes=True)

grid = sns.JointGrid(x, y)

grid.plot_joint(plt.scatter, color="purple")

grid.plot_marginals(sns.rugplot, height=1, color="g")
x= data.industry

y= data.GDP

sns.set(style="white", color_codes=True)

grid = sns.JointGrid(x, y)

grid.plot_joint(plt.scatter, color="black")

grid.plot_marginals(sns.rugplot, height=1, color="g")
x1= data.industry

y1= data.agriculture

fig = plt.figure()

sns.set_style("ticks")

g = sns.JointGrid(x1, y1, xlim=[0, max(x1)], ylim=[0, max(y1)])

g.plot_marginals(sns.distplot, color=".5")

g.plot_joint(plt.hexbin, bins='log', gridsize=40)

plt.show()
#filtering

culture_low_country = data[(data['GDP']<7000) & (data['literacy']<50)].min()

culture_low_country
culture_high_country = data[(data['GDP']>20000) & (data['literacy']>95)].max()

culture_high_country
print(":)")