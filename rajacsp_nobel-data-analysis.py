# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set()
nobel_data = pd.read_csv("../input/archive.csv",parse_dates=True)
nobel_data.head()
nobel_data = nobel_data.dropna()
headers = nobel_data.dtypes.index  
    
columnNames = list(nobel_data.head(0))
columnNames
# Get the Modern country name

def get_country_new_name(row, col_name):
    d = str(row[col_name])   
    
    if "(" not in d:
        return d
    
    d = d[d.index("(")+1:]
    d = d[:d.index(")")] 
    
    #then Germany, now France
    if "then Germany" in d:
        d = "France" # need to verify
    
    return d
nobel_data['Birth Country New Name'] = nobel_data.apply (lambda row: get_country_new_name(row, 'Birth Country'),axis=1)
nobel_data['Organization Country New Name'] = nobel_data.apply (lambda row: get_country_new_name(row, 'Organization Country'),axis=1)
nobel_data['Death Country New Name'] = nobel_data.apply (lambda row: get_country_new_name(row, 'Death Country'),axis=1)
nobel_data['Birth Country New Name'].head()
# Check the year min and max
print(nobel_data['Year'].min())
print(nobel_data['Year'].max())
'''
    Get the Decade 
    
    Sample: 
    1998 --> 1990
'''
def get_decade(row, col_name):
    
    d = int(row[col_name])   
    
    if(d > 1900 and d < 1910):
        return "1900+"    
    elif(d >= 1910 and d < 1920):
        return "1910+"
    elif(d >= 1920 and d < 1930):
        return "1920+"
    elif(d >= 1930 and d < 1940):
        return "1930+"
    elif(d >= 1940 and d < 1950):
        return "1940+"
    elif(d >= 1950 and d < 1960):
        return "1950+"
    elif(d >= 1960 and d < 1970):
        return "1960+"
    elif(d >= 1970 and d < 1980):
        return "1970+"
    elif(d >= 1980 and d < 1990):
        return "1980+"
    elif(d >= 1990 and d < 2000):
        return "1990+"
    elif(d >= 2000 and d < 2010):
        return "2000+"
    elif(d >= 2010 and d < 2020):
        return "2010+"
    
    return 'Unknown'
nobel_data['Decade'] = nobel_data.apply (lambda row: get_decade(row, 'Year'),axis=1)
nobel_data['Decade'].head()
# Analyze Countries

print(nobel_data.groupby(['Birth Country New Name']).size().sort_values(ascending=False))
print(nobel_data['Death Country New Name'].value_counts())
type(nobel_data['Birth Country New Name'])
nobel_data['Birth Country New Name'].value_counts()
plt.figure(figsize=(10,12))
sns.countplot(y="Birth Country New Name", data=nobel_data,
              order=nobel_data['Birth Country New Name'].value_counts().index,
              palette="rocket"
              )
plt.show()
nobel_data['Sex'].value_counts().plot(kind='pie', autopct='%.2f')
plt.show()
nobel_data_female = nobel_data[nobel_data['Sex'] == 'Female']
nobel_data_female.head()
print(nobel_data_female['Birth Country New Name'].value_counts())
nobel_data_female['Birth Country New Name'].value_counts().plot(kind='pie', autopct='%.2f')
plt.show()
print(nobel_data_female.groupby(['Organization Country New Name']).size().sort_values(ascending=False))
nobel_data_female['Organization Country New Name'].value_counts().plot(kind='pie', autopct='%.2f')
plt.show()
nobel_data_female['Year'].value_counts()
nobel_data_female['Year'].value_counts().plot(kind='pie', autopct='%.2f')
plt.show()
nobel_data_female['Decade'].value_counts()
nobel_data_female.groupby(['Decade']).size().plot(kind='bar')
plt.show()
nobel_data_chemistry = nobel_data[nobel_data['Category'] == 'Chemistry']
nobel_data_chemistry.head()
nobel_data_chemistry.groupby(['Birth Country New Name']).size().sort_values(ascending=False)
nobel_data_chemistry.groupby(['Decade']).size().sort_values(ascending=False)
nobel_data_chemistry.groupby(['Decade']).size().plot(kind='barh')
plt.show()
nobel_data.groupby(['Category']).size().plot(kind='barh')
plt.show()
# Find unique countries (birth country)

nobel_data['Birth Country New Name'].unique()
def get_continent(row, col_name):
    
    country = str(row[col_name]).lower()
    
    if (
        country in "Burundi, Comoros, Djibouti, Eritrea, Ethiopia, Kenya, Madagascar, Malawi, Mauritius, Mayotte, Mozambique, Réunion, Rwanda, Seychelles, Somalia, South Sudan, Uganda".lower()
        or country in "United Republic of Tanzania, Zambia, Zimbabwe, Angola, Cameroon, Central African Republic, Chad, Congo, Democratic Republic of the Congo, Equatorial Guinea, Gabon".lower()
        or country in "Sao Tome and Principe, Algeria, Egypt, Libya, Morocco, Sudan, Tunisia, Western Sahara, Botswana, Lesotho, Namibia, South Africa, Swaziland, Benin, Burkina Faso, Cabo Verde".lower()
        or country in "Cote d'Ivoire, Gambia, Ghana, Guinea,Guinea-Bissau Liberia, Mali,Mauritania Niger, Nigeria,Saint Helena Senegal, Sierra Leone,Togo, Cape Verde".lower()
        or country in "Congo, Dem Rep".lower()
    ):
        return "africa"
    
    if (
        country in "Anguilla, Antigua and Barbuda, Aruba, Bahamas, Barbados, Bonaire, Sint Eustatius and Saba, British Virgin Islands, Cayman Islands, Cuba,".lower()
        or country in "Curaçao, Dominica, Dominican Republic, Grenada, Guadeloupe, Haiti, Jamaica, Martinique, Montserrat, Puerto Rico, Saint-Barthélemy, Saint Kitts and Nevis,".lower()
        or country in "Saint Lucia, Saint Martin (French part), Saint Vincent and the Grenadines, Sint Maarten (Dutch part), Trinidad and Tobago, Turks and Caicos Islands,".lower()
        or country in "United States Virgin Islands, Belize, Costa Rica, El Salvador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Bermuda, Canada, Greenland,".lower()
        or country in "Saint Pierre and Miquelon, United States of America".lower()
    ):
        return "north america"   
    
  
    if (
        country in "Argentina, Bolivia (Plurinational State of), Brazil, Chile, Colombia, Ecuador, Falkland Islands (Malvinas), French Guiana, Guyana,".lower()
        or country in "Paraguay, Peru, Suriname, Uruguay, Venezuela (Bolivarian Republic of)".lower()
    ):
        return "south america"
    
    if (
        country in "Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan,China, China, Hong Kong Special Administrative Region, China, Macao Special Administrative Region,".lower()
        or country in "Democratic People's Republic of Korea, Japan, Mongolia, Republic of Korea, Afghanistan, Bangladesh, Bhutan, India, Iran (Islamic Republic of), Maldives,".lower()
        or country in "Nepal, Pakistan, Sri Lanka, Brunei Darussalam, Cambodia, Indonesia, Lao People's Democratic Republic, Malaysia, Myanmar, Philippines, Singapore, Thailand,".lower()
        or country in "Timor-Leste, Viet Nam, Armenia, Azerbaijan, Bahrain, Cyprus, Georgia, Iraq, Israel, Jordan, Kuwait, Lebanon, Oman, Qatar, Saudi Arabia, State of Palestine,".lower()
        or country in "Syrian Arab Republic, Turkey, United Arab Emirates, Yemen, ".lower()
        or country in "East Timor (Timor-Leste), Korea, North, Korea, South".lower()
        or country in "Laos, Burma, Palestine, Occupied Territories, Taiwan, Vietnam".lower()        
    ):
        return "asia"
    
    if (
        country in "Belarus, Bulgaria, Czechia, Hungary, Poland, Republic of Moldova, Romania, Russian Federation, Slovakia, Ukraine, Åland Islands, Channel Islands,".lower()
        or country in "Denmark, Estonia,  Faeroe Islands, Finland, Guernsey, Iceland, Ireland, Isle of Man, Jersey, Latvia, Lithuania, Norway, Sark, Svalbard and Jan Mayen Islands,".lower()
        or country in "Sweden, United Kingdom of Great Britain and Northern Ireland, Albania, Andorra, Bosnia and Herzegovina, Croatia, Gibraltar,Greece, Holy See, Italy, Malta,".lower()
        or country in "Montenegro, Portugal, San Marino, Serbia, Slovenia, Spain, The former Yugoslav Republic of Macedonia, Austria, Belgium, France, Germany,".lower()
        or country in "Liechtenstein, Luxembourg, Monaco, Netherlands, Switzerland, Netherlands Antilles*, Czech Republic".lower()
    ):
        return "europe"
    
    if (
        country in "Australia, New Zealand, Norfolk Island, Fiji, New Caledonia, Papua New Guinea,Solomon Islands, Vanuatu, Guam, Kiribati, Marshall Islands, Micronesia (Federated States of),".lower()
        or country in "Nauru, Northern Mariana Islands, Palau, American Samoa, Cook Islands, French Polynesia, Niue, Pitcairn, Samoa,Tokelau,".lower()
        or country in "Tonga, Tuvalu, Wallis and Futuna Islands".lower()
    ):
        return "oceania"
     
    return "unknown"

nobel_data['Birth Continent'] = nobel_data.apply(lambda row: get_continent(row, 'Birth Country New Name'),axis=1)
nobel_data['Birth Continent'].head()
nobel_data['Birth Continent'].value_counts().plot(kind='pie', autopct='%.2f')
plt.show()
