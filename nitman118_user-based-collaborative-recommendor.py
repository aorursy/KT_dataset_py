# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/advertising.csv')

df.head()
#One hot encoding

df = pd.get_dummies(df,columns=['Male','Country'])
df.head()
#since logistic regression does not accept string or datetime values

del df['Ad Topic Line']

del df['Timestamp']

del df['City']

df.head()
df.shape
X= df.iloc[:,[x for x in range(0,len(df.columns)) if x !=4]].values # .values returns a numpy representation
y=df.iloc[:,4].values
LogReg = LogisticRegression()

LogReg.fit(X,y)
cols = df.columns.tolist()

print(f'[')

for item in cols:

    print('\t0, '+'#'+item)

print(']')

print(df.shape)

      
#variables:

[

	0, #Daily Time Spent on Site

	0, #Age

	0, #Area Income

	0, #Daily Internet Usage

	0, #Male_0

	0, #Male_1

	0, #Country_Afghanistan

	0, #Country_Albania

	0, #Country_Algeria

	0, #Country_American Samoa

	0, #Country_Andorra

	0, #Country_Angola

	0, #Country_Anguilla

	0, #Country_Antarctica (the territory South of 60 deg S)

	0, #Country_Antigua and Barbuda

	0, #Country_Argentina

	0, #Country_Armenia

	0, #Country_Aruba

	0, #Country_Australia

	0, #Country_Austria

	0, #Country_Azerbaijan

	0, #Country_Bahamas

	0, #Country_Bahrain

	0, #Country_Bangladesh

	0, #Country_Barbados

	0, #Country_Belarus

	0, #Country_Belgium

	0, #Country_Belize

	0, #Country_Benin

	0, #Country_Bermuda

	0, #Country_Bhutan

	0, #Country_Bolivia

	0, #Country_Bosnia and Herzegovina

	0, #Country_Bouvet Island (Bouvetoya)

	0, #Country_Brazil

	0, #Country_British Indian Ocean Territory (Chagos Archipelago)

	0, #Country_British Virgin Islands

	0, #Country_Brunei Darussalam

	0, #Country_Bulgaria

	0, #Country_Burkina Faso

	0, #Country_Burundi

	0, #Country_Cambodia

	0, #Country_Cameroon

	0, #Country_Canada

	0, #Country_Cape Verde

	0, #Country_Cayman Islands

	0, #Country_Central African Republic

	0, #Country_Chad

	0, #Country_Chile

	0, #Country_China

	0, #Country_Christmas Island

	0, #Country_Colombia

	0, #Country_Comoros

	0, #Country_Congo

	0, #Country_Cook Islands

	0, #Country_Costa Rica

	0, #Country_Cote d'Ivoire

	0, #Country_Croatia

	0, #Country_Cuba

	0, #Country_Cyprus

	0, #Country_Czech Republic

	0, #Country_Denmark

	0, #Country_Djibouti

	0, #Country_Dominica

	0, #Country_Dominican Republic

	0, #Country_Ecuador

	0, #Country_Egypt

	0, #Country_El Salvador

	0, #Country_Equatorial Guinea

	0, #Country_Eritrea

	0, #Country_Estonia

	0, #Country_Ethiopia

	0, #Country_Falkland Islands (Malvinas)

	0, #Country_Faroe Islands

	0, #Country_Fiji

	0, #Country_Finland

	0, #Country_France

	0, #Country_French Guiana

	0, #Country_French Polynesia

	0, #Country_French Southern Territories

	0, #Country_Gabon

	0, #Country_Gambia

	0, #Country_Georgia

	0, #Country_Germany

	0, #Country_Ghana

	0, #Country_Gibraltar

	0, #Country_Greece

	0, #Country_Greenland

	0, #Country_Grenada

	0, #Country_Guadeloupe

	0, #Country_Guam

	0, #Country_Guatemala

	0, #Country_Guernsey

	0, #Country_Guinea

	0, #Country_Guinea-Bissau

	0, #Country_Guyana

	0, #Country_Haiti

	0, #Country_Heard Island and McDonald Islands

	0, #Country_Holy See (Vatican City State)

	0, #Country_Honduras

	0, #Country_Hong Kong

	0, #Country_Hungary

	0, #Country_Iceland

	0, #Country_India

	0, #Country_Indonesia

	0, #Country_Iran

	0, #Country_Ireland

	0, #Country_Isle of Man

	0, #Country_Israel

	0, #Country_Italy

	0, #Country_Jamaica

	0, #Country_Japan

	0, #Country_Jersey

	0, #Country_Jordan

	0, #Country_Kazakhstan

	0, #Country_Kenya

	0, #Country_Kiribati

	0, #Country_Korea

	0, #Country_Kuwait

	0, #Country_Kyrgyz Republic

	0, #Country_Lao People's Democratic Republic

	0, #Country_Latvia

	0, #Country_Lebanon

	0, #Country_Lesotho

	0, #Country_Liberia

	0, #Country_Libyan Arab Jamahiriya

	0, #Country_Liechtenstein

	0, #Country_Lithuania

	0, #Country_Luxembourg

	0, #Country_Macao

	0, #Country_Macedonia

	0, #Country_Madagascar

	0, #Country_Malawi

	0, #Country_Malaysia

	0, #Country_Maldives

	0, #Country_Mali

	0, #Country_Malta

	0, #Country_Marshall Islands

	0, #Country_Martinique

	0, #Country_Mauritania

	0, #Country_Mauritius

	0, #Country_Mayotte

	0, #Country_Mexico

	0, #Country_Micronesia

	0, #Country_Moldova

	0, #Country_Monaco

	0, #Country_Mongolia

	0, #Country_Montenegro

	0, #Country_Montserrat

	0, #Country_Morocco

	0, #Country_Mozambique

	0, #Country_Myanmar

	0, #Country_Namibia

	0, #Country_Nauru

	0, #Country_Nepal

	0, #Country_Netherlands

	0, #Country_Netherlands Antilles

	0, #Country_New Caledonia

	0, #Country_New Zealand

	0, #Country_Nicaragua

	0, #Country_Niger

	0, #Country_Niue

	0, #Country_Norfolk Island

	0, #Country_Northern Mariana Islands

	0, #Country_Norway

	0, #Country_Pakistan

	0, #Country_Palau

	0, #Country_Palestinian Territory

	0, #Country_Panama

	0, #Country_Papua New Guinea

	0, #Country_Paraguay

	0, #Country_Peru

	0, #Country_Philippines

	0, #Country_Pitcairn Islands

	0, #Country_Poland

	0, #Country_Portugal

	0, #Country_Puerto Rico

	0, #Country_Qatar

	0, #Country_Reunion

	0, #Country_Romania

	0, #Country_Russian Federation

	0, #Country_Rwanda

	0, #Country_Saint Barthelemy

	0, #Country_Saint Helena

	0, #Country_Saint Kitts and Nevis

	0, #Country_Saint Lucia

	0, #Country_Saint Martin

	0, #Country_Saint Pierre and Miquelon

	0, #Country_Saint Vincent and the Grenadines

	0, #Country_Samoa

	0, #Country_San Marino

	0, #Country_Sao Tome and Principe

	0, #Country_Saudi Arabia

	0, #Country_Senegal

	0, #Country_Serbia

	0, #Country_Seychelles

	0, #Country_Sierra Leone

	0, #Country_Singapore

	0, #Country_Slovakia (Slovak Republic)

	0, #Country_Slovenia

	0, #Country_Somalia

	0, #Country_South Africa

	0, #Country_South Georgia and the South Sandwich Islands

	0, #Country_Spain

	0, #Country_Sri Lanka

	0, #Country_Sudan

	0, #Country_Suriname

	0, #Country_Svalbard & Jan Mayen Islands

	0, #Country_Swaziland

	0, #Country_Sweden

	0, #Country_Switzerland

	0, #Country_Syrian Arab Republic

	0, #Country_Taiwan

	0, #Country_Tajikistan

	0, #Country_Tanzania

	0, #Country_Thailand

	0, #Country_Timor-Leste

	0, #Country_Togo

	0, #Country_Tokelau

	0, #Country_Tonga

	0, #Country_Trinidad and Tobago

	0, #Country_Tunisia

	0, #Country_Turkey

	0, #Country_Turkmenistan

	0, #Country_Turks and Caicos Islands

	0, #Country_Tuvalu

	0, #Country_Uganda

	0, #Country_Ukraine

	0, #Country_United Arab Emirates

	0, #Country_United Kingdom

	0, #Country_United States Minor Outlying Islands

	0, #Country_United States Virgin Islands

	0, #Country_United States of America

	0, #Country_Uruguay

	0, #Country_Uzbekistan

	0, #Country_Vanuatu

	0, #Country_Venezuela

	0, #Country_Vietnam

	0, #Country_Wallis and Futuna

	0, #Country_Western Sahara

	0, #Country_Yemen

	0, #Country_Zambia

	0, #Country_Zimbabwe

]
new_user =[

	66.00, #Daily Time Spent on Site

	48, #Age

	24593.33, #Area Income

	131.76, #Daily Internet Usage

	0, #Male_0

	1, #Male_1

	0, #Country_Afghanistan

	1, #Country_Albania

	0, #Country_Algeria

	0, #Country_American Samoa

	0, #Country_Andorra

	0, #Country_Angola

	0, #Country_Anguilla

	0, #Country_Antarctica (the territory South of 60 deg S)

	0, #Country_Antigua and Barbuda

	0, #Country_Argentina

	0, #Country_Armenia

	0, #Country_Aruba

	0, #Country_Australia

	0, #Country_Austria

	0, #Country_Azerbaijan

	0, #Country_Bahamas

	0, #Country_Bahrain

	0, #Country_Bangladesh

	0, #Country_Barbados

	0, #Country_Belarus

	0, #Country_Belgium

	0, #Country_Belize

	0, #Country_Benin

	0, #Country_Bermuda

	0, #Country_Bhutan

	0, #Country_Bolivia

	0, #Country_Bosnia and Herzegovina

	0, #Country_Bouvet Island (Bouvetoya)

	0, #Country_Brazil

	0, #Country_British Indian Ocean Territory (Chagos Archipelago)

	0, #Country_British Virgin Islands

	0, #Country_Brunei Darussalam

	0, #Country_Bulgaria

	0, #Country_Burkina Faso

	0, #Country_Burundi

	0, #Country_Cambodia

	0, #Country_Cameroon

	0, #Country_Canada

	0, #Country_Cape Verde

	0, #Country_Cayman Islands

	0, #Country_Central African Republic

	0, #Country_Chad

	0, #Country_Chile

	0, #Country_China

	0, #Country_Christmas Island

	0, #Country_Colombia

	0, #Country_Comoros

	0, #Country_Congo

	0, #Country_Cook Islands

	0, #Country_Costa Rica

	0, #Country_Cote d'Ivoire

	0, #Country_Croatia

	0, #Country_Cuba

	0, #Country_Cyprus

	0, #Country_Czech Republic

	0, #Country_Denmark

	0, #Country_Djibouti

	0, #Country_Dominica

	0, #Country_Dominican Republic

	0, #Country_Ecuador

	0, #Country_Egypt

	0, #Country_El Salvador

	0, #Country_Equatorial Guinea

	0, #Country_Eritrea

	0, #Country_Estonia

	0, #Country_Ethiopia

	0, #Country_Falkland Islands (Malvinas)

	0, #Country_Faroe Islands

	0, #Country_Fiji

	0, #Country_Finland

	0, #Country_France

	0, #Country_French Guiana

	0, #Country_French Polynesia

	0, #Country_French Southern Territories

	0, #Country_Gabon

	0, #Country_Gambia

	0, #Country_Georgia

	0, #Country_Germany

	0, #Country_Ghana

	0, #Country_Gibraltar

	0, #Country_Greece

	0, #Country_Greenland

	0, #Country_Grenada

	0, #Country_Guadeloupe

	0, #Country_Guam

	0, #Country_Guatemala

	0, #Country_Guernsey

	0, #Country_Guinea

	0, #Country_Guinea-Bissau

	0, #Country_Guyana

	0, #Country_Haiti

	0, #Country_Heard Island and McDonald Islands

	0, #Country_Holy See (Vatican City State)

	0, #Country_Honduras

	0, #Country_Hong Kong

	0, #Country_Hungary

	0, #Country_Iceland

	0, #Country_India

	0, #Country_Indonesia

	0, #Country_Iran

	0, #Country_Ireland

	0, #Country_Isle of Man

	0, #Country_Israel

	0, #Country_Italy

	0, #Country_Jamaica

	0, #Country_Japan

	0, #Country_Jersey

	0, #Country_Jordan

	0, #Country_Kazakhstan

	0, #Country_Kenya

	0, #Country_Kiribati

	0, #Country_Korea

	0, #Country_Kuwait

	0, #Country_Kyrgyz Republic

	0, #Country_Lao People's Democratic Republic

	0, #Country_Latvia

	0, #Country_Lebanon

	0, #Country_Lesotho

	0, #Country_Liberia

	0, #Country_Libyan Arab Jamahiriya

	0, #Country_Liechtenstein

	0, #Country_Lithuania

	0, #Country_Luxembourg

	0, #Country_Macao

	0, #Country_Macedonia

	0, #Country_Madagascar

	0, #Country_Malawi

	0, #Country_Malaysia

	0, #Country_Maldives

	0, #Country_Mali

	0, #Country_Malta

	0, #Country_Marshall Islands

	0, #Country_Martinique

	0, #Country_Mauritania

	0, #Country_Mauritius

	0, #Country_Mayotte

	0, #Country_Mexico

	0, #Country_Micronesia

	0, #Country_Moldova

	0, #Country_Monaco

	0, #Country_Mongolia

	0, #Country_Montenegro

	0, #Country_Montserrat

	0, #Country_Morocco

	0, #Country_Mozambique

	0, #Country_Myanmar

	0, #Country_Namibia

	0, #Country_Nauru

	0, #Country_Nepal

	0, #Country_Netherlands

	0, #Country_Netherlands Antilles

	0, #Country_New Caledonia

	0, #Country_New Zealand

	0, #Country_Nicaragua

	0, #Country_Niger

	0, #Country_Niue

	0, #Country_Norfolk Island

	0, #Country_Northern Mariana Islands

	0, #Country_Norway

	0, #Country_Pakistan

	0, #Country_Palau

	0, #Country_Palestinian Territory

	0, #Country_Panama

	0, #Country_Papua New Guinea

	0, #Country_Paraguay

	0, #Country_Peru

	0, #Country_Philippines

	0, #Country_Pitcairn Islands

	0, #Country_Poland

	0, #Country_Portugal

	0, #Country_Puerto Rico

	0, #Country_Qatar

	0, #Country_Reunion

	0, #Country_Romania

	0, #Country_Russian Federation

	0, #Country_Rwanda

	0, #Country_Saint Barthelemy

	0, #Country_Saint Helena

	0, #Country_Saint Kitts and Nevis

	0, #Country_Saint Lucia

	0, #Country_Saint Martin

	0, #Country_Saint Pierre and Miquelon

	0, #Country_Saint Vincent and the Grenadines

	0, #Country_Samoa

	0, #Country_San Marino

	0, #Country_Sao Tome and Principe

	0, #Country_Saudi Arabia

	0, #Country_Senegal

	0, #Country_Serbia

	0, #Country_Seychelles

	0, #Country_Sierra Leone

	0, #Country_Singapore

	0, #Country_Slovakia (Slovak Republic)

	0, #Country_Slovenia

	0, #Country_Somalia

	0, #Country_South Africa

	0, #Country_South Georgia and the South Sandwich Islands

	0, #Country_Spain

	0, #Country_Sri Lanka

	0, #Country_Sudan

	0, #Country_Suriname

	0, #Country_Svalbard & Jan Mayen Islands

	0, #Country_Swaziland

	0, #Country_Sweden

	0, #Country_Switzerland

	0, #Country_Syrian Arab Republic

	0, #Country_Taiwan

	0, #Country_Tajikistan

	0, #Country_Tanzania

	0, #Country_Thailand

	0, #Country_Timor-Leste

	0, #Country_Togo

	0, #Country_Tokelau

	0, #Country_Tonga

	0, #Country_Trinidad and Tobago

	0, #Country_Tunisia

	0, #Country_Turkey

	0, #Country_Turkmenistan

	0, #Country_Turks and Caicos Islands

	0, #Country_Tuvalu

	0, #Country_Uganda

	0, #Country_Ukraine

	0, #Country_United Arab Emirates

	0, #Country_United Kingdom

	0, #Country_United States Minor Outlying Islands

	0, #Country_United States Virgin Islands

	0, #Country_United States of America

	0, #Country_Uruguay

	0, #Country_Uzbekistan

	0, #Country_Vanuatu

	0, #Country_Venezuela

	0, #Country_Vietnam

	0, #Country_Wallis and Futuna

	0, #Country_Western Sahara

	0, #Country_Yemen

	0, #Country_Zambia

	0, #Country_Zimbabwe

]
y_pred = LogReg.predict([new_user])

y_pred