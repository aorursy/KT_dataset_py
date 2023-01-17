



import numpy as np 

import pandas as pd 







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_time           = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')

df_peroid         = df_time[df_time["date"] >= '2020-02-25'].copy()

df_peroid["rate"] = df_peroid["confirmed"]/df_peroid["test"] * 100 

import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))

barplot = sns.barplot(x=df_peroid['date'], y=df_peroid['rate'], palette="rocket")

plt.xticks(rotation=90)



plt.show()


df_non_acc = df_time.copy()

r, d = df_time.shape

for i in range(1, r):              # skip the first row

    for j in range(2,d) :          # skip the first two columns

        df_non_acc.iloc[i,j] = df_non_acc.iloc[i,j] - df_time.iloc[(i-1),j]





file_name='/kaggle/working/Time_daily.csv'

df_non_acc.to_csv(file_name, sep=',', encoding='utf-8')



df_peroid      = df_non_acc[df_non_acc["date"] >= '2020-02-25'].copy()

df_peroid["rate"] = df_peroid["confirmed"]/df_peroid["test"] * 100 



plt.figure(figsize=(10,5))

barplot = sns.barplot(x=df_peroid['date'], y=df_peroid['rate'], palette="rocket")

plt.xticks(rotation=90)



plt.show()
df_age        = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')

df_daily_age  = df_age.copy()



r, d = df_daily_age.shape

for i in range(9, r):        

    for j in range(3,d) :   

        df_daily_age.iloc[i,j] = df_daily_age.iloc[i,j] - df_age.iloc[(i-9),j]

    





file_name2='/kaggle/working/TimeAge_daily.csv'

df_daily_age.to_csv(file_name2, sep=',', encoding='utf-8')





sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="age", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_age)

plt.title('Accuminated Total Confirmed Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)





sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="age", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_daily_age[df_daily_age["date"] > '2020-03-02'])

plt.title('Daily Confirmed Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)





sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="age", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_age)

plt.title('Accuminated Total Deceased Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)





sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="age", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_daily_age[df_daily_age["date"] > '2020-03-02'])

plt.title('Daily Deceased Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)
df_gender = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')

df_daily_gender  = df_gender.copy()



r, d = df_daily_gender.shape

for i in range(2, r):              

    for j in range(3,d) :          

        df_daily_gender.iloc[i,j]   = df_daily_gender.iloc[i,j] - df_gender.iloc[(i-2),j]





file_name1='/kaggle/working/TimeGender_daily.csv'

df_daily_gender.to_csv(file_name1, sep=',', encoding='utf-8')





sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="sex", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted", height=6, data=df_gender)

plt.title('Accuminated Total Confirmed Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)





sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="sex", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted", height=6, data=df_daily_gender[df_daily_gender["date"] > '2020-03-02'])

plt.title('Daily Confirmed Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)





sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="sex", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_gender[df_gender["date"] > '2020-03-02'])

plt.title('Accuminated Total Deceased Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)





sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="sex", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_daily_gender[df_daily_gender["date"] > '2020-03-02'])

plt.title('Daily Deceased Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)
def wordcloud_column(dataframe):

    from wordcloud import WordCloud, STOPWORDS 

 

    comment_words = ' '

    stopwords = set(STOPWORDS) 

  

    

    for k in range(len(dataframe)):

       

        val = str(dataframe.iloc[k,0]) 

        

        tokens = val.split()

    

        

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 

          

        for words in tokens: 

            comment_words = comment_words + words + ' '

  

    

    wordcloud = WordCloud(width=400, height=200,background_color ='white', max_font_size=60).generate(comment_words)

    plt.figure()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()





df_patient = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')

df_reason  = df_patient[['infection_case']]

df_reason  = df_reason[(df_reason['infection_case'].notna())]

wordcloud_column(df_reason)





df_case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

df_case = df_case[['infection_case']]

df_case = df_case[(df_case['infection_case'].notna())]

wordcloud_column(df_case)


legend_html = '''

        <div style="position: fixed; bottom: 300px; left: 50px; width: 160px; height: 110px; 

                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;"

                    >&nbsp; <b>Legend</b> <br>

                    &nbsp; Confirmed < 100 &nbsp&nbsp&nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#ff9900"></i><br>

                    &nbsp; Confirmed < 1000 &nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#cc33ff"></i><br>

                    &nbsp; Confirmed < 3000 &nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#ff0000"></i><br>

                    &nbsp; Confirmed >= 3000

                        <i class="fa fa-circle" style="font-size:14px;color:#660000"></i>

        </div>

        ''' 



def color(total):

    

    col_100  = "#ff9900"

    col_1000 = "#cc33ff"

    col_3000 = "#ff0000"

    over     = "#660000"

    if (total < 100):   

            rad = total/10

            color = col_100

    elif (total < 1000): 

            rad = min(total/10, 20)

            color = col_1000

    elif (total < 3000): 

            rad = min(total/10, 30)

            color = col_3000

    else: 

            rad = 35

            color = over

    return rad, color







import folium

from   folium import plugins



df_province = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')

df_region   = pd.read_csv('/kaggle/input/region-new/Region_New.csv')

df_current = df_province[df_province['date']==df_province['date'].max()]

df_row = df_current.join(df_region.set_index('city')[['latitude','longitude']], on='province')





map0 = folium.Map(location=[35.7982008,125.6296572], control_scale=True, zoom_start=7)

folium.TileLayer('openstreetmap').add_to(map0)

folium.TileLayer('CartoDB positron',name='Positron').add_to(map0)

folium.TileLayer('CartoDB dark_matter',name='Dark Matter').add_to(map0)

folium.TileLayer('Stamen Terrain',name='Terrain').add_to(map0)

folium.TileLayer('Stamen Toner',name='Toner').add_to(map0)



folium.LayerControl().add_to(map0)



plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(map0) 

map0.get_root().html.add_child(folium.Element(legend_html))



for index, row in df_row.iterrows():

    date      = row['date']

    confirmed = row["confirmed"]

    deceased  = row["deceased"]

    released  = row['released']

    province  = row["province"]

    lat   = row["latitude"]

    long  = row["longitude"]

    

    

    popup_text = "<b>Date:</b> {}<br><b>Province: </b>{}<br><b>Confirmed:</b> {}<br><b>Deceased: </b>{}"

    popup_text = popup_text.format(date, province, confirmed, deceased)          

    

   

    rad, col = color(confirmed)

    folium.CircleMarker(location=(lat,long), radius = rad, color=col, popup=popup_text, 

                        opacity= 4.0, fill=True).add_to(map0)



map0.save('SKConfirmed_Mar20.html')

display(map0)