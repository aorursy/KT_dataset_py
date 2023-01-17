import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import plotly.offline as py

import seaborn as sns

py.offline.init_notebook_mode()
national_names = pd.read_csv("../input/NationalNames.csv", index_col='Id')

state_names = pd.read_csv("../input/StateNames.csv",index_col='Id')

jewish_names = np.asarray(['Aharon','Abba','Avraham','Adam','Akiva','Alon', 'Alter','Amos','Amram','Ariel','Aryeh','Asher','Avi','Avigdor','Avner','Azriel','Barak','Baruch','Betzalel','Benyamin','Ben-Tzion','Berel','Boaz','Calev','Carmi', 'Chagai','Chaim','Chanan','Chananya','Chananel','Chanoch','Chizkiyahu','Dan','David','Doron','Dov','Ephraim','Ehud','Eitan','Elchanan','Eldad','Elazar','Eliezer','Eli','Elimelech','Elisha','Eliyahu','Elyakim','Emanuel','Ezra','Fishel','Fivel','Gad','Gamliel','Gavriel','Gedaliah','Gershom','Gershon', 'Gidon','Gil','Hirsh','Hillel','Ilan','Issur','Itamar','Kalman','Kalonymos', 'Leib','Levi','Malkiel','Manoach','Matitiyahu','Medad','Meir','Menachem','Menashe',  'Mendel','Meshulam','Micha','Michael','Mordechai','Moshe','Nachshon','Nachman','Nachum','Naftali','Natan','Nechemia','Netanel',

'Nissan','Nissim','Noam','Noach','Oren','Ovadia','Paltiel','Peretz','Pesach', 'Pesachya','Pinchas','Rachamim','Rafael','Reuven','Selig','Seth','Shabtai','Shalom','Shaul','Shay','Shimshon','Shimon','Shlomo','Shmuel','Shmariyahu','Shneur','Shraga','Simcha','Tamir','Tanchum','Tuvia','Tzion','Tzvi','Tzadok','Tzemach','Tzephania','Tzuriel','Uri','Uriel','Uziel','Velvel','Yair','Yaakov','Yechezkel','Yechiel','Yedidya','Yehoshua','Yehuda','Yigal','Yerachmiel','Yirmiyahu','Yitzhak','Yisrael','Yissachar','Yishayahu','Yochanan','Yoel','Yom','Tov','Yosef','Yona','Yonatan','Yoram','Yuval','Zalman','Zechariah','Zev','Zerach','Zevulun','Adina','Ahuva','Aliza','Anat','Ariella','Atara','Avigail','Avishag','Avital','Aviva','Ayala','Ayelet','Bat','Sheva','Batya','Bat','Tziyon','Bayla','Bina','Bracha','Bruriah','Carmel ','Chana','Chava','Chagit','Chaviva',

'Chaya','Dafna','Dalia','Dalit','Daniella','Devorah','Dinah','Efrat','Eliana','Elisheva','Emunah','Esther','Faige','Freida','Fruma','Gavriella','Geula','Gila','Golda','Hadassah','Hadar','Hinda','Hodaya','Idit','Ilana','Irit','Keila','Keren','Kinneret','Leah','Leeba','Levana','Levona','Lila','Liora','Machla','Malka','Maya','Mayan','Mazal','Meira','Meirav','Menucha','Michal','Milka','Miriam','Moriah','Naama','Naomi','Netanya','Nava','Nechama','Noa','Nurit','Ora',

'Orli','Orna','Osnat','Penina','Rachel','Raizel','Rina','Rivka','Ruth','Sarah','Sarai','Serach','Sharon','Shayna','Shifra','Shira','Shoshana','Shlomit','Shulamit','Sigal,','','Sigalit,','','Sigalia','Simcha','Tal','Talia','Tamar','Techiya','Tehilla','Tikva','Tirtzah','Tova','Tzipporah','Tziona','Tzivia','Tzofiya','Tzviya','Uriella','Vered',

'Yael','Yaffa','Yakova','Yardena','Yehudit','Yiskah','Yocheved','Zahava','Zissel'], dtype=object)
jewish_nationals = national_names[national_names.Name.isin(jewish_names)]

jewish_by_name = jewish_nationals.groupby('Name',as_index=False).Count.sum()

extremes = jewish_by_name[jewish_by_name['Count'] > 150000]

no_extremes = jewish_by_name[jewish_by_name['Count'] < 150000]

small = jewish_by_name[jewish_by_name['Count'] < 20000]

popular_jews = jewish_nationals[jewish_nationals.Name.isin(extremes['Name'])]

jews = jewish_nationals[jewish_nationals.Name.isin(no_extremes['Name'])]

jews_small = jewish_nationals[jewish_nationals.Name.isin(small['Name'])]

jewish_nationals.tail()
# Let's build wordcloud 

def visualize(wordcloud):

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.figure()

    

# Exploring Jews names. Has anyone knows how to visualize a name proportionally to it's weight(count)?

wordcloud_all = WordCloud(background_color="white").generate(" ".join(pd.unique(jews['Name'])))

visualize(wordcloud_all)
sns.set(color_codes=True)

popular_jews.pivot_table('Count',index='Year',aggfunc=sum).plot()

jews.pivot_table('Count',index=['Year'],aggfunc=sum).plot()

plt.title("All Jewish names trend",fontsize=16)
jews.pivot_table('Count',index=['Name'],aggfunc=sum).plot()

plt.title("Jewish names distribution",fontsize=16)
jews_small.pivot_table('Count',index=['Name'],aggfunc=sum).plot()

plt.title("Least popular Jewish names distribution",fontsize=16)
o_names = jews['Name'].unique().tolist()

p_names = popular_jews['Name'].unique().tolist()

pjews = pd.DataFrame()

ojews = []

for name in p_names:

    pjews[name] = jewish_nationals.loc[jewish_nationals['Name'] == name].groupby('Year').Count.sum()

for name in o_names:

    ojews.append(pd.DataFrame(jewish_nationals.loc[jewish_nationals['Name'] == name]))

pjews.tail()
for df in ojews:

    ax = df.pivot_table('Count',index='Year').plot(figsize=(10,6))

plt.title("All names trend", fontsize=16)

ax = pjews.plot(figsize=(10,6),marker='o')

plt.title('Popular names trend', fontsize=16)
# Explore state distribution, let's find communities of jews

j_states = state_names[state_names.Name.isin(jewish_names)]

j_states_no_extreme = j_states[j_states.Name.isin(jews_small['Name'])]

j_states = j_states.groupby(as_index=False, by="State").Count.sum()

j_states_no_extreme = j_states_no_extreme.groupby(as_index=False, by="State").Count.sum()

j_states_no_extreme['CumulativePercentage'] = 100 * j_states_no_extreme.Count.cumsum()/j_states_no_extreme.Count.sum()

j_states['CumulativePercentage'] = 100 * j_states.Count.cumsum()/j_states.Count.sum()
import plotly.offline as py

py.offline.init_notebook_mode()

data1 = [ dict(

        type = 'choropleth',

        locations = j_states['State'],

        z = j_states['Count'],

        locationmode = 'USA-states',

        text = j_states['State'],

        colorscale = [[0,"rgb(253, 141, 143)"],[0.33,"rgb(156, 12, 12)"],[1,"rgb(100, 12, 12)"]],

        autocolorscale = False,

        marker = dict(

            line = dict(color = 'rgb(58,100,69)', width = 0.6)),

            colorbar = dict(autotick = True, tickprefix = '', title = '# of Names')

            )

       ]



data2 = [ dict(

        type = 'choropleth',

        locations = j_states_no_extreme['State'],

        z = j_states_no_extreme['Count'],

        locationmode = 'USA-states',

        text = j_states_no_extreme['State'],

        colorscale = [[0,"rgb(253, 141, 143)"],[0.33,"rgb(156, 12, 12)"],[1,"rgb(100, 12, 12)"]],

        autocolorscale = False,

        marker = dict(

            line = dict(color = 'rgb(58,100,69)', width = 0.6)),

            colorbar = dict(autotick = True, tickprefix = '', title = '# of Names')

            )

       ]



layout = dict(

    title = 'Total number of Jewish names by state',

    geo = dict(

        scope='usa',

        projection=dict( type='albers usa' ),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

    )



fig1 = dict(data=data1, layout=layout)

fig2 = dict(data=data2, layout=layout)

py.iplot(fig1, validate=False, filename='USmap')

py.iplot(fig2, validate=False, filename='USmap')