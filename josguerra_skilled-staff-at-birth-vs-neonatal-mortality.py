import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

import re

from IPython.display import display



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# read file

data = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')

data.shape
# verify if there are any nan or null  values in the dataset

print(data.isnull().values.any())

# how many countries are in the data set. check if the country codes are the same. Check number of indicators

countries = data['CountryName'].unique().tolist()

print("number of contries: ", len(countries))



country_codes = data['CountryCode'].unique().tolist()

print("number of contriy codes: ", len(country_codes))



indicators = data['IndicatorName'].unique().tolist()

print("number of indicators: ", len(indicators))

# Create list of unique indicators, indicator codes

# Idea taken from https://www.kaggle.com/kmravikumar/choosing-topics-to-explore

Indicator_array =  data[['IndicatorName','IndicatorCode']].drop_duplicates().values
modified_indicators = []

unique_indicator_codes = []

for ele in Indicator_array:

    indicator = ele[0]

    indicator_code = ele[1].strip()

    if indicator_code not in unique_indicator_codes:

        # delete , ( ) from the IndicatorNames

        new_indicator = re.sub('[,()]',"",indicator).lower()

        # replace - with "to" and make all words into lower case

        new_indicator = re.sub('-'," to ",new_indicator).lower()

        modified_indicators.append([new_indicator,indicator_code])

        unique_indicator_codes.append(indicator_code)



Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])

Indicators = Indicators.drop_duplicates()

print(Indicators.shape)
key_word_dict = {}

key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']

key_word_dict['Food'] = ['food','grain','nutrition','calories']

key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']

key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']

key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']

key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']

key_word_dict['Education'] = ['education','literacy']

key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']

key_word_dict['Rural'] = ['rural','village']

key_word_dict['Urban'] = ['urban','city']
feature = 'Health'

for indicator_ele in Indicators.values:

    for ele in key_word_dict[feature]:

        word_list = indicator_ele[0].split()

        if ele in word_list or ele+'s' in word_list:

            print(indicator_ele)

            break
# Main indicators to compare contries

# skill staff and mortality rate respectively

chosen_indicators = ['SH.STA.BRTC.ZS', 'SH.DYN.NMRT' ]



# Subset of data with the required features alone

df_subset = data[data['IndicatorCode'].isin(chosen_indicators)]



# Chose only Portugal and Japan for Analysis

df_Portugal = df_subset[data['CountryName']=="Portugal"]

df_Japan = df_subset[data['CountryName']=="Japan"]



df_Portugal.head(5)
# PLotting function for comparing development indicators

def plot_indicator(indicator):

    ds_Portugal = df_Portugal[['IndicatorName','Year','Value']][df_Portugal['IndicatorCode']==indicator]

    try:

        title = ds_Portugal['IndicatorName'].iloc[0]

    except:

        title = "None"



    xPortugal = ds_Portugal['Year'].values

    yPortugal = ds_Portugal['Value'].values

    ds_Japan = df_Japan[['IndicatorName','Year','Value']][df_Japan['IndicatorCode']==indicator]

    xJapan = ds_Japan['Year'].values

    yJapan = ds_Japan['Value'].values

    

    plt.figure(figsize=(14,4))

    

    plt.subplot(121)

    plt.plot(xPortugal,yPortugal,label='Portugal')

    plt.plot(xJapan,yJapan,label='Japan')

    plt.title(title)

    plt.legend(loc=2)
plot_indicator(chosen_indicators[0])
plot_indicator(chosen_indicators[1])
# Portugal case

# Understand the size of both vectors (years) to see where to make the cross correlation



skilledStaffPortugal = df_Portugal[['Year','Value']][df_Portugal['IndicatorCode']=='SH.STA.BRTC.ZS']

skilledStaffPortugal.head()



neonatalPortugal = df_Portugal[['Year','Value']][df_Portugal['IndicatorCode']=='SH.DYN.NMRT']



display(skilledStaffPortugal.head(50))

display(neonatalPortugal.head(50))



display(np.correlate(skilledStaffPortugal.Value, neonatalPortugal.Value, mode='valid'))

display(np.correlate(skilledStaffPortugal.Value-np.mean(skilledStaffPortugal.Value), neonatalPortugal.Value-np.mean(neonatalPortugal.Value), mode='valid')/(np.std(skilledStaffPortugal.Value)*np.std(neonatalPortugal.Value)))
# for Portugal case



# the birth rate only has 4 years and we need to adapts that to the skilled staff

# the vector with skilled staff will only have data corresponding to the same years as the motality rate neonatal

# if the year does not coincide, we add a year, wich occurs for the year 1989 (skilled staff only has for 1990)



neonatalPortugal_final = pd.DataFrame()



for ind in skilledStaffPortugal.index:

    if skilledStaffPortugal['Year'][ind] in neonatalPortugal['Year'].values:

        staff_final = neonatalPortugal[['Year','Value']][neonatalPortugal['Year']==skilledStaffPortugal['Year'][ind]]

        print(skilledStaffPortugal['Year'][ind])

        neonatalPortugal_final = neonatalPortugal_final.append(staff_final)

    else:

        staff_final = neonatalPortugal[['Year','Value']][neonatalPortugal['Year']==skilledStaffPortugal['Year'][ind]+1]

        print(skilledStaffPortugal['Year'][ind]+1)

        neonatalPortugal_final = neonatalPortugal_final.append(staff_final)

    

    

# sample_birthrate = df_Portugal[['Year','Value']][df_Portugal['IndicatorCode']=='SH.DYN.NMRT']
display(skilledStaffPortugal['Value'])

display(neonatalPortugal_final['Value'])
# Correlate both series

display(np.correlate(skilledStaffPortugal['Value'], neonatalPortugal_final['Value'], mode='full'))



# the respective normalization

display(np.correlate(skilledStaffPortugal.Value-np.mean(skilledStaffPortugal.Value), neonatalPortugal_final.Value-np.mean(neonatalPortugal_final.Value), mode='full')/(np.std(skilledStaffPortugal.Value)*np.std(neonatalPortugal_final.Value)))
# Using a graphical approach



x= skilledStaffPortugal['Value']

y= neonatalPortugal_final['Value']



fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)



ax1.xcorr(x, y, usevlines=True, maxlags=3, normed=True, lw=2)

ax1.grid(True)

ax1.title.set_text('Cross-correlation between skilled staff and neonatal mortality')



ax2.acorr(x, usevlines=True, normed=True, maxlags=3, lw=2)

ax2.grid(True)

ax2.title.set_text('Skilled Staff Autocorrelation')



ax3.acorr(y, usevlines=True, normed=True, maxlags=3, lw=2)

ax3.grid(True)

ax3.title.set_text('Neonatal Mortality Autocorrelation')



fig.suptitle('Fig. 1 - Portugal Case', fontsize=16)

fig.tight_layout()

fig.subplots_adjust(top=0.85)

plt.show()
# Using the same graphical approach but also normalizing the data



x= skilledStaffPortugal['Value']

y= neonatalPortugal_final['Value']



# https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python

a = (x - np.mean(x)) / (np.std(x) * len(x))

b = (y - np.mean(y)) / (np.std(y))



fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)



ax1.xcorr(a, b, usevlines=True, maxlags=3, normed=True, lw=2)

ax1.grid(True)

ax1.title.set_text('Normalized Cross-correlation between skilled staff and neonatal mortality')



ax2.acorr(a, usevlines=True, normed=True, maxlags=3, lw=2)

ax2.grid(True)

ax2.title.set_text('Skilled Staff Normalized Autocorrelation')



ax3.acorr(b/len(b), usevlines=True, normed=True, maxlags=3, lw=2)

ax3.grid(True)

ax3.title.set_text('Neonatal Mortality Normalized Autocorrelation')



fig.suptitle('Fig.2 - Portugal Case (Normalized)', fontsize=16)

fig.tight_layout()

fig.subplots_adjust(top=0.85)

plt.show()
# Japan case

# Understand the size of both vectors (years) to see where to make the cross correlation

# 



skilledStaffJapan = df_Japan[['Year','Value']][df_Japan['IndicatorCode']=='SH.STA.BRTC.ZS']



neonatalJapan = df_Japan[['Year','Value']][df_Japan['IndicatorCode']=='SH.DYN.NMRT']





display(skilledStaffJapan.head(50))

display(neonatalPortugal.head(50))
# for Japan case



# the birth rate only has 3 years and we need to adapts that to the skilled staff

# the vector with skilled staff will only have data corresponding to the same years as the motality rate neonatal



neonatalJapan_final = pd.DataFrame()



for ind in skilledStaffJapan.index:

    if skilledStaffJapan['Year'][ind] in neonatalJapan['Year'].values:

        staff_final = neonatalJapan[['Year','Value']][neonatalJapan['Year']==skilledStaffJapan['Year'][ind]]

        print(skilledStaffJapan['Year'][ind])

        neonatalJapan_final = neonatalJapan_final.append(staff_final)

    else:

        staff_final = neonatalJapan[['Year','Value']][neonatalJapan['Year']==skilledStaffJapan['Year'][ind]+1]

        print(skilledStaffJapan['Year'][ind]+1)

        neonatalJapan_final = neonatalJapan_final.append(staff_final)

    

# sample_birthrate = df_Portugal[['Year','Value']][df_Portugal['IndicatorCode']=='SH.DYN.NMRT']
display(skilledStaffJapan['Value'])

display(neonatalJapan_final['Value'])
# Using a graphical approach



x= skilledStaffJapan['Value']

y= neonatalJapan_final['Value']



fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)



ax1.xcorr(x, y, usevlines=True, maxlags=2, normed=True, lw=2)

ax1.grid(True)

ax1.title.set_text('Cross-correlation between skilled staff and neonatal mortality')



ax2.acorr(x, usevlines=True, normed=True, maxlags=2, lw=2)

ax2.grid(True)

ax2.title.set_text('Skilled Staff Autocorrelation')



ax3.acorr(y, usevlines=True, normed=True, maxlags=2, lw=2)

ax3.grid(True)

ax3.title.set_text('Neonatal Mortality Autocorrelation')



fig.suptitle('Fig.3 - Japan Case', fontsize=16)

fig.tight_layout()

fig.subplots_adjust(top=0.85)

plt.show()
# Using the same graphical approach but also normalizing the data



x= skilledStaffJapan['Value']

y= neonatalJapan_final['Value']



# https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python

a = (x - np.mean(x)) / (np.std(x) * len(x))

b = (y - np.mean(y)) / (np.std(y))



fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)



ax1.xcorr(a, b, usevlines=True, maxlags=2, normed=True, lw=2)

ax1.grid(True)

ax1.title.set_text('Normalized Cross-correlation between skilled staff and neonatal mortality')



ax2.acorr(a, usevlines=True, normed=True, maxlags=2, lw=2)

ax2.grid(True)

ax2.title.set_text('Skilled Staff Normalized Autocorrelation')



ax3.acorr(b/len(b), usevlines=True, normed=True, maxlags=2, lw=2)

ax3.grid(True)

ax3.title.set_text('Neonatal Mortality Normalized Autocorrelation')



fig.suptitle('Fig.4 - Japan Case (Normalized)', fontsize=16)

fig.tight_layout()

fig.subplots_adjust(top=0.85)

plt.show()