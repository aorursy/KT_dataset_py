import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
dunya = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
df = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
df["Date"] = pd.to_datetime(df["Date"])

df_temp=pd.read_csv('/kaggle/input/covid19-global-weather-data/temperature_dataframe.csv')
df_temp["date"] = pd.to_datetime(df_temp["date"])
df_pop=pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
dunya.head().T
df.head().T
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as ply
ply.init_notebook_mode(connected=True)
import plotly.express as px
df.sort_values(by=["Confirmed"], ascending=False, inplace=True)
fig = px.pie(
    df.head(50),
    values = "Confirmed",
    names = "Country/Region",
    title = "En Yüksek Vaka Sayısına Sahip 5 Ülke"
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.show()
df1=pd.Series(dunya['Country/Region'],name="Country")
df2=pd.Series(dunya['Date'],name="Date")
df3=pd.Series(dunya['Confirmed'],name="Confirmed")
df4=pd.Series(dunya['Deaths'],name="Deaths")
df5=pd.Series(dunya['Recovered'],name="Recovered")
df_world=pd.concat([df1, df2,df3, df4,df5], axis=1)
plt.figure()
df_world.boxplot(column=['Confirmed','Deaths','Recovered'])

fig,axs=plt.subplots(2,2) 
axs[0, 0].boxplot(df_world['Confirmed'])
axs[0, 0].set_title('Hasta Sayısı')

axs[0, 1].boxplot(df_world['Recovered'])
axs[0, 1].set_title('İyileşen Hasta Sayısı')

axs[1, 0].boxplot(df_world['Deaths'])
axs[1, 0].set_title('Hayatını Kaybeden Hasta Sayısı')
import matplotlib.pyplot as plt
%matplotlib inline
# Matplotlib ile basit bir dagilim grafigi
ax = plt.axes()

ax.scatter(df.Deaths, df.Recovered)

# Eksenleri isimlendirme
ax.set(xlabel='Ölen Kişi',
       ylabel='İyileşen Hasta',
       title='Ölen Kişi vs İyileşen Hasta');
plt.axes().set(xlabel='Hasta Sayısı',
       ylabel='Confirmed',
       title='Dünya Genelinde Hastalar');
# Histogram
# bins = number of bar in figure
df.Confirmed.plot(kind = 'hist',bins = 20,figsize = (10,5))

plt.show()
df_pop.rename(columns={'Country (or dependency)': 'country',
                             'Population (2020)' : 'population',
                             'Density (P/Km²)' : 'density',
                             'Fert. Rate' : 'fertility',
                             'Med. Age' : "age",
                             'Urban Pop %' : 'urban_percentage'}, inplace=True)
df.rename(columns={'Country/Region': 'country'}, inplace=True)
df_temp.rename(columns={'date': 'Date'}, inplace=True)
df_temp['country'] = df_temp['country'].replace('USA', 'US')
df_pop['country'] = df_pop['country'].replace('United States', 'US')
df['country'] = df['country'].replace('Mainland China', 'China')
df_pop = df_pop[["country", "population", "density", "fertility", "age", "urban_percentage"]]
df = df.merge(df_pop, on=['country'], how='left')
df_temp.drop_duplicates(subset =["Date",'country'], 
                     keep = 'first', inplace = True)
df = df.merge(df_temp, on=['Date','country'], how='left')
tarih=df['Date'].max()
guncel=df[df['Date']==tarih]
olum=guncel['Deaths'].sum()
iyilesme=guncel['Recovered'].sum()
vaka=guncel['Confirmed'].sum()
turkiye=guncel[guncel['country']=='Turkey']
turkiye_vaka=turkiye['Confirmed'].sum()
turkiye_olum=turkiye['Deaths'].sum()
turkiyeOlum_orani=(turkiye_olum/turkiye_vaka)*100
turkiye_iyilesme=turkiye['Recovered'].sum()
print ('Bilgilerin Son Güncellenme Tarihi: {}'.format(tarih))
print ('Türkiye Vaka: {:,.0f}'.format(turkiye_vaka))
print ('Türkiye Ölüm: {:,.0f}'.format(turkiye_olum))
print ('Türkiye İyileşme: {:,.0f}'.format(turkiye_iyilesme))
print ('Türkiye Ölüm Oranı: {:,.1f}%'.format(turkiyeOlum_orani))
print ('Toplam Ölüm: {:,.0f}'.format(olum))
print ('Toplam İyileşme: {:,.0f}'.format(iyilesme))
print ('Toplam Vaka: {:,.0f}'.format(vaka))

who_region = {}

# African Region 
africa = "Algeria, Angola, Cabo Verde, Eswatini, Sao Tome and Principe, Benin, South Sudan, Western Sahara, Congo (Brazzaville), Congo (Kinshasa), Cote d'Ivoire, Botswana, Burkina Faso, Burundi, Cameroon, Cape Verde, Central African Republic, Chad, Comoros, Ivory Coast, Democratic Republic of the Congo, Equatorial Guinea, Eritrea, Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mauritius, Mozambique, Namibia, Niger, Nigeria, Republic of the Congo, Rwanda, São Tomé and Príncipe, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, Swaziland, Togo, Uganda, Tanzania, Zambia, Zimbabwe"
africa = [i.strip() for i in africa.split(',')]
for i in africa:
    who_region[i] = 'Afrıka'
    

america = 'Antigua and Barbuda, Argentina, Bahamas, Barbados, Belize, Bolivia, Brazil, Canada, Chile, Colombia, Costa Rica, Cuba, Dominica, Dominican Republic, Ecuador, El Salvador, Grenada, Guatemala, Guyana, Haiti, Honduras, Jamaica, Mexico, Nicaragua, Panama, Paraguay, Peru, Saint Kitts and Nevis, Saint Lucia, Saint Vincent and the Grenadines, Suriname, Trinidad and Tobago, United States, US, Uruguay, Venezuela'
america = [i.strip() for i in america.split(',')]
for i in america:
    who_region[i] = 'Amerika'


asia = 'Bangladesh, Bhutan, North Korea, India, Indonesia, Maldives, Myanmar, Burma, Nepal, Sri Lanka, Thailand, Timor-Leste'
asia = [i.strip() for i in asia.split(',')]
for i in asia:
    who_region[i] = 'Asya'


euro = 'Albania, Andorra, Greenland, Kosovo, Holy See, Liechtenstein, Armenia, Czechia, Austria, Azerbaijan, Belarus, Belgium, Bosnia and Herzegovina, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Georgia, Germany, Greece, Hungary, Iceland, Ireland, Israel, Italy, Kazakhstan, Kyrgyzstan, Latvia, Lithuania, Luxembourg, Malta, Monaco, Montenegro, Netherlands, North Macedonia, Norway, Poland, Portugal, Moldova, Romania, Russia, San Marino, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Tajikistan, Turkey, Turkmenistan, Ukraine, United Kingdom, Uzbekistan'
euro = [i.strip() for i in euro.split(',')]
for i in euro:
    who_region[i] = 'Avrupa'


emro = 'Afghanistan, Bahrain, Djibouti, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Pakistan, Palestine, West Bank and Gaza, Qatar, Saudi Arabia, Somalia, Sudan, Syria, Tunisia, United Arab Emirates, Yemen'
emro = [i.strip() for i in emro.split(',')]
for i in emro:
    who_region[i] = 'Orta Dogu'


wpro = 'Australia, Brunei, Cambodia, China, Cook Islands, Fiji, Japan, Kiribati, Laos, Malaysia, Marshall Islands, Micronesia, Mongolia, Nauru, New Zealand, Niue, Palau, Papua New Guinea, Philippines, South Korea, Samoa, Singapore, Solomon Islands, Taiwan, Taiwan*, Tonga, Tuvalu, Vanuatu, Vietnam'
wpro = [i.strip() for i in wpro.split(',')]
for i in wpro:
    who_region[i] = 'Guney Asya'


other='nan,NAN'
other = [i.strip() for i in other.split(',')]
for i in other:
    who_region[i]='Dıger'
df_world['Region'] = df_world['Country'].map(who_region)
df_world[df_world['Region'].isna()]['Region'].unique()
df_world['Region'].unique()
df_1=df_world
df_2=pd.Series(dunya['Long'],name="Long")
df_3=pd.Series(dunya['Lat'],name="Lat")
df_location=pd.concat([df_1,df_2,df_3], axis=1)
import plotly.express as px
fig = px.choropleth(df_location, locations="Country", locationmode='country names', color=np.log(df_location["Confirmed"]), 
                    hover_name="Country", animation_frame=df_location["Date"],
                    title='Zaman İçerisindeki Değişim', color_continuous_scale=px.colors.sequential.Purp)
fig.update(layout_coloraxis_showscale=False)
fig.show()
import plotly.express as px
fig = px.bar(df_world.sort_values("Confirmed"),
            x='Region', y="Confirmed",
            hover_name="Region",
            hover_data=["Recovered","Deaths","Confirmed"],
            title='COVID-19: Test Sonucu Pozitif Olan Hasta Sayısı Kıtalara Göre',
)
fig.update_xaxes(title_text="Region")
fig.update_yaxes(title_text="Positif Test Sayısı(%)")
fig.show()
fig = px.bar(df_world.sort_values("Recovered"),
            x='Region', y="Recovered",
            hover_name="Region",
            hover_data=["Confirmed","Deaths","Recovered"],
            title='COVID-19: İyileşen Hasta Sayısı Kıtalara Göre',
)
fig.update_xaxes(title_text="Region")
fig.update_yaxes(title_text="İyileşen Hasta Sayısı")
fig.show()
fig = px.bar(df_world.sort_values("Deaths"),
            x='Region', y="Deaths",
            hover_name="Region",
            hover_data=["Confirmed","Recovered","Deaths"],
            title='COVID-19: Hayatını Kaybeden Hasta Sayısı Kıtalara Göre ',
)
fig.update_xaxes(title_text="Region")
fig.update_yaxes(title_text="Hayatını Kaybeden Hasta Sayısı")
fig.show()
df['Active']=df['Confirmed']-df['Deaths']-df['Recovered']
temp = df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')


fig = px.area(temp, x="Date", y="Count", color='Case',
             title='Yayılma Hızı', color_discrete_sequence = ['#21bf73', '#ff2e63', '#fe9801'])
fig.show()