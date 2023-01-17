import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import time
%matplotlib inline

def get_indicator(ind_code, ind_text):
    BASE_URL = 'https://ghoapi.azureedge.net/api/'
    DATE_2000S = '?$filter=date(TimeDimensionBegin) ge 2000-01-01'    
    service_url = BASE_URL + ind_code + DATE_2000S
    response = requests.get(service_url)

# make sure we got a valid response
    if(response.ok):
        data_j = response.json()
# SpatialDim = country_code, TimeDim = year, Numeric_Value
        data = pd.DataFrame(data_j["value"]).rename(
            columns = {'NumericValue':ind_text, 'SpatialDim':'country_code', 'TimeDim':'year'})
        data = data[(data.SpatialDimType != 'REGION') & (data.SpatialDimType != 'WORLDBANKINCOMEGROUP')]

        print("Data for \"{}\" loaded, set {} rows {} columns".format(ind_text, data.shape[0], data.shape[1]))
        return data
    else:
        print("Response was not OK", response)
        return None

def remove_duplicates(data_set):
    dup_set = data_set.duplicated(subset=["country_code", "year"], keep='last')
    return data_set[~dup_set]

def test_dump(data_set):
    print(data_set.shape)
    print(data_set.info())
    print(data_set[data_set['country_code'] == 'BEL'])

    col_names = list(data_set.columns.values)
    for name in col_names:
        print(name,data_set[name].nunique())

# Import Drive API and authenticate.
from google.colab import drive
# Mount your Drive to the Colab VM.
drive.mount('/gdrive')

print (time.asctime( time.localtime(time.time()) ))
# Create basic table, with countries and years

service_url0 = 'https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues/'
response0 = requests.get(service_url0)

# make sure we got a valid response
print(response0)
if (response0.ok):
    # get the full data from the response
    data0j = response0.json()
    print(data0j.keys())
else:
    print("Response was not OK")

data0a = pd.DataFrame(data0j["value"])
data0a = data0a[data0a['Title'] != 'SPATIAL_SYNONYM']

remove_list = ['PRI', 'KNA', 'DMA', 'PSE', 'AND', 'SMR', 'MCO', 'LIE', 'COK', 
               'TUV', 'PLW', 'TKL', 'MHL', 'NIU', 'NRU', 'ME1', 'SDF']

#['Puerto Rico' 'Saint Kitts and Nevis' 'Dominica'
# 'occupied Palestinian territory, including east Jerusalem' 'Andorra'
# 'San Marino' 'Monaco' 'Liechtenstein' 'Cook Islands' 'Tuvalu' 'Palau'
# 'Tokelau' 'Marshall Islands' 'Niue' 'Nauru'
# 'The former state union Serbia and Montenegro' 'Sudan (former)']

data0a = data0a[~data0a['Code'].isin(remove_list)]

num_years = 17 # from 2000-2016
country_year_list =[] 
for index, rows in data0a.iterrows(): 
    for year in range(2000, (2000+num_years), 1):
        sub_list = [rows.Title, rows.Code, rows.ParentTitle, year] 
        country_year_list.append(sub_list) 
data0 = pd.DataFrame(country_year_list, columns=['country','country_code','region','year'])

print(data0.shape)
print(data0.info())
#  WHOSIS_000001
#      Life expectancy at birth (years)
#  choose:  Dim1 = BTSX  (ignore mle and fmle)

ind_code = 'WHOSIS_000001'
ind_text = 'life_expect'
data_raw = get_indicator(ind_code, ind_text)

data01 = data_raw[data_raw['Dim1'] == 'BTSX'][['country_code', 'year', ind_text]]
data01 = remove_duplicates(data01)
test_dump(data01)

data0 = data0.merge(data01, how='left')
#  WHOSIS_000015
#      Life expectancy at age 60 (years)
#  choose:  Dim1 = BTSX  (ignore mle and fmle)

ind_code = 'WHOSIS_000015'
ind_text = 'life_exp60'
data_raw = get_indicator(ind_code, ind_text)

data01a = data_raw[data_raw['Dim1'] == 'BTSX'][['country_code', 'year', ind_text]]
data01a = remove_duplicates(data01a)
test_dump(data01a)

data0 = data0.merge(data01a, how='left')
#  WHOSIS_000004
#      Adult mortality rate (probability of dying between 15 and 60 years per 1000 population)
#  choose:  Dim1 = BTSX  (ignore mle and fmle)

ind_code = 'WHOSIS_000004'
ind_text = 'adult_mortality'
data_raw = get_indicator(ind_code, ind_text)

data02 = data_raw[data_raw['Dim1'] == 'BTSX'][['country_code', 'year', ind_text]]
data02 = remove_duplicates(data02)
test_dump(data02)

data0 = data0.merge(data02, how='left')
#  LIFE_0000000029
#      nMx - age-specific death rate between ages x and x+n
#  A few countries have a gender-averaged rate, but most don't. Need to use the average.
#  choose:  Dim1 = BTSX  (calculate average of mle and fmle)
#           Dim2 = AGELT1, AGE1-4

ind_code = 'LIFE_0000000029'
ind_text = 'mortality'
data_raw = get_indicator(ind_code, ind_text)

data_temp = data_raw[data_raw['Dim2'] == 'AGELT1'][['country_code', 'year', ind_text, 'Dim1']]
data_temp = data_temp.pivot_table(index=["country_code", "year"], columns=["Dim1"], values=ind_text)
data_temp.reset_index(inplace=True)
data_temp['BTSX'] = 0.5 * (data_temp['MLE'] + data_temp['FMLE'])
data_temp = data_temp.melt(id_vars=['country_code', 'year'], var_name='Dim1', value_name='infant_mort')

data03 = data_temp[data_temp['Dim1'] == 'BTSX'][['country_code', 'year', 'infant_mort']]
data03 = remove_duplicates(data03)
test_dump(data03)

data_temp = data_raw[data_raw['Dim2'] == 'AGE1-4'][['country_code', 'year', ind_text, 'Dim1']]
data_temp = data_temp.pivot_table(index=["country_code", "year"], columns=["Dim1"], values=ind_text)
data_temp.reset_index(inplace=True)
data_temp['BTSX'] = 0.5 * (data_temp['MLE'] + data_temp['FMLE'])
data_temp = data_temp.melt(id_vars=['country_code', 'year'], var_name='Dim1', value_name='age1-4mort')

data04 = data_temp[data_temp['Dim1'] == 'BTSX'][['country_code', 'year', 'age1-4mort']]
data04 = remove_duplicates(data04)
data04 = data04.rename(columns = {'age1-4 mortality':'age1-4_mort'})
test_dump(data04)

data0 = data0.merge(data03, how='left')
data0 = data0.merge(data04, how='left')
#  SA_0000001400
#      Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
#  choose:  Dim1 = SA_TOTAL  (ignore beer, wine, spirits, other)
ind_code = 'SA_0000001400'
ind_text = 'alcohol'
data_raw = get_indicator(ind_code, ind_text)

data05 = data_raw[data_raw['Dim1'] == 'SA_TOTAL'][['country_code', 'year', ind_text]]
data05 = remove_duplicates(data05)
test_dump(data05)

data0 = data0.merge(data05, how='left')
#  NCD_BMI_MEAN
#      Mean BMI (kg/m^2) (age-standardized estimate)
#  choose:  Dim1 = BTSX  (same number as mle and fmle)
#           Dim2 is always "YEARS18-PLUS"
# For the Sudan (former) entry, there are two sets of data, entered January 2017 and September 2017                                               
ind_code = 'NCD_BMI_MEAN'
ind_text = 'bmi'
data_raw = get_indicator(ind_code, ind_text)

# remove duplicated entries for Sudan (former)
dup_list = data_raw.duplicated(subset=["country_code", "year", 'Dim1'], keep='last')
data_raw = data_raw[~dup_list]

data06 = data_raw[data_raw['Dim1'] == 'BTSX'][['country_code', 'year', ind_text]]
data06 = remove_duplicates(data06)
test_dump(data06)

data0 = data0.merge(data06, how='left')
#  NCD_BMI_MINUS2C
#      Prevalence of thinness among children and adolescents,
#      BMI < -2 standard deviations below the median (crude estimate) (%)
#  choose:  Dim1 = BTSX  (same number as mle and fmle)
#           Dim2 can be YEARS05-09, YEARS10-19, and YEARS05-19

ind_code = 'NCD_BMI_MINUS2C'
ind_text = 'age5-19thinness'
data_raw = get_indicator(ind_code, ind_text)

data07 = data_raw[(data_raw['Dim1'] == 'BTSX') & (data_raw['Dim2'] == 'YEARS05-19')][['country_code', 'year', ind_text]]
data07 = remove_duplicates(data07)
test_dump(data07)

data0 = data0.merge(data07, how='left')
#  NCD_BMI_PLUS2C
#      Prevalence of thinness among children and adolescents,
#      BMI > +2 standard deviations above the median (crude estimate) (%)
#  choose:  Dim1 = BTSX  (same number as mle and fmle)
#           Dim2 can be YEARS05-09, YEARS10-19, and YEARS05-19

ind_code = 'NCD_BMI_PLUS2C'
ind_text = 'age5-19obesity'
data_raw = get_indicator(ind_code, ind_text)

data08 = data_raw[(data_raw['Dim1'] == 'BTSX') & (data_raw['Dim2'] == 'YEARS05-19')][['country_code', 'year', ind_text]]
data08 = remove_duplicates(data08)
test_dump(data08)

data0 = data0.merge(data08, how='left')
#  WHS4_117
#      Hepatitis B (HepB3) immunization coverage among 1-year-olds (%)
#  choose:  no "DIM1" settings; 19 years and 186 countries

ind_code = 'WHS4_117'
ind_text = 'hepatitis'
data_raw = get_indicator(ind_code, ind_text)

data09 = data_raw[['country_code', 'year', ind_text]]
data09 = remove_duplicates(data09)
test_dump(data09)

data0 = data0.merge(data09, how='left')
#  WHS8_110
#      Measles-containing-vaccine first-dose (MCV1) immunization coverage among 1-year-olds (%)
#  choose:  no "DIM1" settings; 19 years and 194 countries
#      mslv, vmsl looked unusable - many missing entries, sporadic years, multiple data sources, etc.

ind_code = 'WHS8_110'
ind_text = 'measles'
data_raw = get_indicator(ind_code, ind_text)

data10 = data_raw[['country_code', 'year', ind_text]]
data10 = remove_duplicates(data10)
test_dump(data10)

data0 = data0.merge(data10, how='left')
#  WHS4_544
#      Polio (Pol3) immunization coverage among 1-year-olds (%)
#  choose:  no "DIM1" settings; 19 years and 194 countries

ind_code = 'WHS4_544'
ind_text = 'polio'
data_raw = get_indicator(ind_code, ind_text)

data11 = data_raw[['country_code', 'year', ind_text]]
data11 = remove_duplicates(data11)
test_dump(data11)

data0 = data0.merge(data11, how='left')
#  WHS4_100
#      Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
#  choose:  no "DIM1" settings; 19 years and 194 countries (missing 19 non-null values)

ind_code = 'WHS4_100'
ind_text = 'diphtheria'
data_raw = get_indicator(ind_code, ind_text)

data12 = data_raw[['country_code', 'year', ind_text]]
data12 = remove_duplicates(data12)
test_dump(data12)

data0 = data0.merge(data12, how='left')
#  WSH_WATER_BASIC
#      Population using at least basic drinking-water services (%)
#  choose:  Dim1 = TOTL  (ignore RUR and URB)

ind_code = 'WSH_WATER_BASIC'
ind_text = 'basic_water'
data_raw = get_indicator(ind_code, ind_text)

data13 = data_raw[data_raw['Dim1'] == 'TOTL'][['country_code', 'year', ind_text]]
data13 = remove_duplicates(data13)
test_dump(data13)

data0 = data0.merge(data13, how='left')
#  HWF_0001
#      Medical doctors (per 10,000)
#  choose:  193 country codes, but about half the year entries are missing

ind_code = 'HWF_0001'
ind_text = 'doctors'
data_raw = get_indicator(ind_code, ind_text)

data14 = data_raw[['country_code', 'year', ind_text]]
data14 = remove_duplicates(data14)
test_dump(data14)

data0 = data0.merge(data14, how='left')
#  DEVICES00
#      Total density per 100 000 population: Hospitals

ind_code = 'DEVICES00'
ind_text = 'hospitals'
data_raw = get_indicator(ind_code, ind_text)

data14a = data_raw[['country_code', 'year', ind_text]]
data14a = remove_duplicates(data14a)
test_dump(data14a)

data0 = data0.merge(data14a, how='left')
#  WHS9_93
#      Gross national income per capita (PPP int. $)
#  choose:  only up to 2013

ind_code = 'WHS9_93'
ind_text = 'gni_capita'
data_raw = get_indicator(ind_code, ind_text)

data15 = data_raw[['country_code', 'year', ind_text]]
data15 = remove_duplicates(data15)
test_dump(data15)

data0 = data0.merge(data15, how='left')
#  GHED_GGHE-DGDP_SHA2011
#      Domestic general government health expenditure (GGHE-D) as percentage of gross domestic product (GDP) (%)
#  choose:  191 country codes, 18 years

ind_code = 'GHED_GGHE-DGDP_SHA2011'
ind_text = 'gghe-d'
data_raw = get_indicator(ind_code, ind_text)

data16 = data_raw[['country_code', 'year', ind_text]]
data16 = remove_duplicates(data16)
test_dump(data16)

data0 = data0.merge(data16, how='left')
#  GHED_CHEGDP_SHA2011
#      Current health expenditure (CHE) as percentage of gross domestic product (GDP) (%)
#  choose:  191 country codes, 18 years

ind_code = 'GHED_CHEGDP_SHA2011'
ind_text = 'che_gdp'
data_raw = get_indicator(ind_code, ind_text)

data17 = data_raw[['country_code', 'year', ind_text]]
data17 = remove_duplicates(data17)
test_dump(data17)

data0 = data0.merge(data17, how='left')
# save the GHO data to a local CSV file

#data0.to_csv('gho_data.csv',index=False)

data0.to_csv('/gdrive/My Drive/Thinkful/capstone2_who_life_exp/gho_data.csv', index=False)

print ('GHO data finished at: ',time.asctime( time.localtime(time.time()) ))
# if the GHO file has already been made, read it in
#gho_data = pd.read_csv('gho_data.csv', skipinitialspace=True)

# read in the CSV files for the UNESCO features
un1_data = pd.read_csv('unesco_population.csv', skipinitialspace=True).rename(
    columns = {'LOCATION':'country_code', 'TIME':'year'})
un2_data = pd.read_csv('unesco_gni.csv', skipinitialspace=True).rename(
    columns = {'LOCATION':'country_code', 'TIME':'year'})
un3_data = pd.read_csv('unesco_educ.csv', skipinitialspace=True).rename(
    columns = {'LOCATION':'country_code', 'TIME':'year'})
indicators = [('une_pop', 'Total population '),
              ('une_infant', 'Mortality rate, infant (per 1,000 live births)'),
              ('une_life', 'Life expectancy at birth, total (years)'),
              ('une_hiv', 'Prevalence of HIV, total (% of population ages 15-49)')]

for ind_name, ind_text in indicators:
    print("Trying",ind_name,"indicator:",ind_text, len(un1_data[un1_data['Indicator'] == ind_text]))

    temp_df = un1_data[un1_data['Indicator'] == ind_text][['country_code', 'year', 'Value']].rename(
                  columns = {'Value':ind_name})
    gho_data = gho_data.merge(temp_df, how='left')

# The database is missing population information from Japan and Lebanon.

japan_pop = [127524, 127714, 127893, 128058, 128204, 128326, 128423, 128494, 128539, 128555, 128542, 128499, 128424, 128314, 128169, 127985, 127763]
gho_data.loc[(gho_data['country_code'] == 'JPN'), 'une_pop'] = japan_pop

lebanon_pop = [3843, 3991, 4182, 4388, 4569, 4699, 4760, 4767, 4765, 4813, 4953, 5202, 5538, 5913, 6261, 6533, 6714]
gho_data.loc[(gho_data['country_code'] == 'LBN'), 'une_pop'] = lebanon_pop
indicators = [('une_gni', 'GNI per capita, PPP (current international $)'),
              ('une_poverty', 'Poverty headcount ratio at $1.90 a day (PPP) (% of population)')]

for ind_name, ind_text in indicators:
    print("Trying",ind_name,"indicator:",ind_text, len(un2_data[un2_data['Indicator'] == ind_text]))

    temp_df = un2_data[un2_data['Indicator'] == ind_text][['country_code', 'year', 'Value']].rename(
                  columns = {'Value':ind_name})
    gho_data = gho_data.merge(temp_df, how='left')
indicators = [('une_edu_spend', 'Government expenditure on education as a percentage of GDP (%)'),
              ('une_literacy', 'Adult literacy rate, population 15+ years, both sexes (%)'),
              ('une_school', 'Mean years of schooling (ISCED 1 or higher), population 25+ years, both sexes')]

for ind_name, ind_text in indicators:
    print("Trying",ind_name,"indicator:",ind_text, len(un3_data[un3_data['Indicator'] == ind_text]))

    temp_df = un3_data[un3_data['Indicator'] == ind_text][['country_code', 'year', 'Value']].rename(
                  columns = {'Value':ind_name})
    gho_data = gho_data.merge(temp_df, how='left')
print(gho_data.info())

gho_data.to_csv('who_life_exp.csv',index=False)

print ('Notebook finished at: ',time.asctime( time.localtime(time.time()) ))