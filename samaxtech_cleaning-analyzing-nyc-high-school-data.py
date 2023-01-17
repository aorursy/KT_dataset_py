%autosave 2

import pandas as pd
import numpy
import re
import warnings
warnings.filterwarnings('ignore') 
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

data = {}

for f in data_files:
    d = pd.read_csv("../input/nyc-data/nyc_highschool_data/schools/{0}".format(f))
    data[f.replace(".csv", "")] = d
all_survey = pd.read_csv("../input/nyc-data/nyc_highschool_data/schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
d75_survey = pd.read_csv("../input/nyc-data/nyc_highschool_data/schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey = pd.concat([all_survey, d75_survey], axis=0)

survey["DBN"] = survey["dbn"]

survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_11", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]
survey = survey.loc[:,survey_fields]
data["survey"] = survey
data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation
    
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")
class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

class_size = class_size.groupby("DBN").agg(numpy.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size

data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]
cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    data["ap_2010"][col] = pd.to_numeric(data["ap_2010"][col], errors="coerce")
combined = data["sat_results"]

combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

to_merge = ["class_size", "demographics", "survey", "hs_directory"]

for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)
def get_first_two_chars(dbn):
    return dbn[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)
school_dist = combined["school_dist"]
correlations = combined.corr()
correlations = correlations["sat_score"]
print(correlations)
# Remove DBN since it's a unique identifier, not a useful numerical value for correlation.
survey_fields.remove("DBN")
survey_fields
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
combined.corr().loc['sat_score', survey_fields].plot.bar()
plt.ylabel('sat_score')

combined.plot.scatter(x='sat_score', y='saf_s_11') 
import numpy as np
from mpl_toolkits.basemap import Basemap

grouped = combined.groupby('school_dist')
av_school_dist = grouped.agg(np.mean)

m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='i'
    )

longitudes = av_school_dist['lon'].tolist()
latitudes = av_school_dist['lat'].tolist()

fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Safety Score (Students) by District')
m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=av_school_dist['saf_s_11'], cmap='summer')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('SAT Scores by District')
m.drawmapboundary(fill_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)
m.scatter(longitudes, latitudes, s=50, zorder=2, latlon=True, c=av_school_dist['sat_score'], cmap='summer')

plt.show()


races = ['white_per', 'asian_per', 'black_per', 'hispanic_per']

fig = plt.figure()
combined.corr().loc['sat_score', races].plot.bar()
plt.ylabel('sat_score')
combined.plot.scatter(x='hispanic_per', y='sat_score')
high_hispanic = combined[combined['hispanic_per']>95]
print(high_hispanic['SCHOOL NAME'])
low_hispanic = combined[combined['hispanic_per']<10]
high_SAT_hisp = low_hispanic[low_hispanic['sat_score']>1800]

print(high_SAT_hisp['SCHOOL NAME'])
gender = ['male_per', 'female_per']

fig = plt.figure()
combined.corr().loc['sat_score', gender].plot.bar()
plt.ylabel('sat_score')
combined.plot.scatter(x='female_per', y='sat_score')
high_female = combined[combined['female_per']>60]
high_SAT_female = high_female[high_female['sat_score']>1700]

print(high_SAT_female['SCHOOL NAME'])
combined['ap_per'] = combined['AP Test Takers '] / combined['total_enrollment']
combined.plot.scatter(x='ap_per', y='sat_score')
print("Pearson coefficient (r):", combined.corr().loc['sat_score', 'ap_per'])
combined.plot.scatter(x='AVERAGE CLASS SIZE', y='sat_score')
print("Pearson coefficient (r):", combined.corr().loc['AVERAGE CLASS SIZE', 'ap_per'])
properties = pd.read_csv('../input/nyc-data/nyc_highschool_data/nyc_properties/nyc-rolling-sales.csv')
data['properties'.replace('.csv','')] = properties

print(properties.columns)
properties.head()
properties.describe()
boroughs = {
            '1': 'Manhattan',
            '2': 'The Bronx',
            '3': 'Brooklyn',
            '4': 'Queens',
            '5': 'Staten Island'
            }

def number2borough(number):
    return boroughs[str(number)]

properties['BOROUGH'] = properties['BOROUGH'].apply(number2borough)
properties['SALE PRICE'].value_counts()
properties['SALE PRICE'] = pd.to_numeric(properties['SALE PRICE'], errors='coerce')
properties.drop(properties.index[properties[properties['SALE PRICE']<=10].index], inplace=True)

properties['SALE PRICE'].value_counts()
properties = properties.groupby('BOROUGH').agg(numpy.mean)
properties.reset_index(inplace=True)

properties.plot.bar(x='BOROUGH', y='SALE PRICE', rot=30, legend=False)
plt.ylabel('SALE PRICE ($)')

sorted_SAT = combined.sort_values('sat_score', ascending=False)

sorted_SAT['SCHOOL NAME'].reset_index()
