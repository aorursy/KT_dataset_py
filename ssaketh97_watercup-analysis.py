# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Talukas = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv');

Marking = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/MarkingSystem.csv');

Winners = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv');

Donations = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/VillagesSupportedByDonationsWaterCup2019.csv');

print('Talukas dataset');

print('----------------');

print(Talukas.head());

print('  .\n  .\n  .\n');

print('Marking dataset');

print('----------------');

print(Marking.head());

print('  .\n  .\n  .\n');
print('State level Winners dataset');

print('----------------');

print(Winners.head());

print('  .\n  .\n  .\n');

print('Donations in 2019 dataset');

print('----------------');

print(Donations.head());

print('  .\n  .\n  .\n');
plt.figure(figsize=(14,10))

plt.subplot(224)

districtwise = Talukas.drop(columns=Talukas.columns[[0,2]]).drop_duplicates()

plot=sb.countplot(x=districtwise['Region'],hue=districtwise['Year'],data=districtwise,palette=sb.color_palette('dark'));

sb.despine(left=True,bottom=True);

plot.set_yticks([]);

plot.set_ylabel('count of Talukas');

plot.set_title('count of Talukas in every region participated in watercup yearwise');

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.4,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.subplot(222)

districtwise = Talukas.drop(columns=Talukas.columns[[0,3]]).drop_duplicates()

plot=sb.countplot(x=districtwise['Region'],hue=districtwise['Year'],data=districtwise,palette=sb.color_palette('dark'));

sb.despine(left=True,bottom=True);

plot.set_yticks([]);

plot.set_ylabel('');

plot.set_title('count of Districts in every region participated in watercup yearwise');

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.1,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.subplot(221)

districtwise = Talukas.drop(columns=Talukas.columns[[0,1,3]]).drop_duplicates()

plot=sb.countplot(x=districtwise['Year'],data=districtwise,palette=sb.color_palette('dark'));

sb.despine(left=True,bottom=True);

plot.set_yticks([]);

plot.set_ylabel('count of Districts');

plot.set_title('count of Districts participated in watercup yearwise');

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 0.5,

            '{:1.0f}'.format(height),

            ha="center") ;

plt.subplot(223)

districtwise = Talukas.drop(columns=Talukas.columns[[0,1,2]])

plot=sb.countplot(x=districtwise['Year'],data=districtwise,palette=sb.color_palette('dark'));

sb.despine(left=True,bottom=True);

plot.set_yticks([]);

plot.set_ylabel('count of Talukas');

plot.set_title('count of Talukas participated in watercup yearwise');

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x()+p.get_width()/2.,

            height + 1,

            '{:1.0f}'.format(height),

            ha="center") ;
total_districts=36

total_talukas = 358



plt.figure(figsize=(16,8));

plt.subplot(121)

districtwise = Talukas.drop(columns=Talukas.columns[[0,1,3]]).drop_duplicates()

districtwise=districtwise['Year'].value_counts()

new_data = pd.DataFrame({'Year':districtwise.index,'count':districtwise.values})

new_data['percentparticipationdistrict'] = (new_data['count']/36)*100;



plot=sb.lineplot(x="Year",y="percentparticipationdistrict",data=new_data,marker='o',color='darkblue');

sb.despine(left=True,bottom=True);

for x,y in zip(new_data['Year'],new_data['percentparticipationdistrict']):

    plot.text(x-0.15,y+1,'{:1.2f}%'.format(y));

plot.text(2018,30,'**Total=36 districts**');

plt.xticks([2016,2017,2018,2019],['2016','2017','2018','2019']);

plt.yticks([]);

plt.ylabel('Percent of total districts particapted');

plt.title('Percentage of districts from total which participated in challenge yearwise');



plt.subplot(122);

districtwise = districtwise = Talukas.drop(columns=Talukas.columns[[0,1,2]])

districtwise=districtwise['Year'].value_counts()

new_data = pd.DataFrame({'Year':districtwise.index,'count':districtwise.values})

new_data['percentparticipationdistrict'] = (new_data['count']/358)*100;



plot=sb.lineplot(x="Year",y="percentparticipationdistrict",data=new_data,marker='o',color='darkblue');

sb.despine(left=True,bottom=True);

for x,y in zip(new_data['Year'],new_data['percentparticipationdistrict']):

    plot.text(x-0.15,y+0.5,'{:1.2f}%'.format(y));

plot.text(2018,8,'**Total=358 talukas**');

plt.xticks([2016,2017,2018,2019],['2016','2017','2018','2019']);

plt.yticks([]);

plt.ylabel('Percent of total talukas particapted');

plt.title('Percentage of talukas from total which participated in challenge yearwise');

plt.tight_layout();
plt.figure(figsize=(18,12))

plt.subplot(234)

Winner = Winners['District'].value_counts()

Winner=pd.DataFrame({'District':Winner.index,'count':Winner.values}).sort_values(by=['count'])

plot=plt.barh(y='District',width='count',data=Winner,color='cadetblue');

sb.despine(left=True,bottom=True);

for i, v in enumerate(Winner['count']):

    plt.text(v + 0.1,i, str(round(v,1)));

plot[9].set_color("coral");

plt.xticks([]);

plt.title('Count of How many times a district has won');



plt.subplot(235)

Winner = Winners['Taluka'].value_counts()

Winner=pd.DataFrame({'Taluka':Winner.index,'count':Winner.values})

plot=plt.barh(y='Taluka',width='count',data=Winner,color='cadetblue');

sb.despine(left=True,bottom=True);

for i, v in enumerate(Winner['count']):

    plt.text(v + 0.1,i, str(round(v,1)));

plot[0].set_color("coral");

plt.xticks([]);

plt.title('Count of How many times a Taluka has won');



plt.subplot(236)

plot=sb.countplot(y='Village',data=Winners,color='cadetblue');

sb.despine(left=True,bottom=True);

plot.set_xticks([]);

plot.set_title('count of How many times each village has won');

for p in plot.patches:

    width = p.get_width()

    plot.text(width + 0.03,

              p.get_y()+p.get_height()/2.,

            '{:1.0f}'.format(width),

            ha="center") ;

plt.subplot(211)  

colors = ['lightskyblue','lightskyblue','steelblue','dodgerblue','deepskyblue','deepskyblue','lightskyblue','lightskyblue','lightskyblue','deepskyblue','lightskyblue'];

explode =  (0, 0, 0.1,0.1,0,0,0,0,0,0,0)

plt.pie(Marking['Maximum Marks'],labels=Marking['Component'],autopct='%1.0f%%',explode =explode,colors=colors);

plt.title('Marking System Weightage for 100 Marks');

plt.tight_layout();
plt.figure(figsize=(10,10))



data = Donations['District'].value_counts()

colors = sb.cubehelix_palette(24,reverse=True)

#explode =  (0, 0, 0.1,0.1,0,0,0,0,0,0,0)

plt.pie(data.values,labels=data.index,autopct='%1.2f%%',radius=1,wedgeprops=dict(width=0.5, edgecolor='w'),colors=colors);

plt.title('How much percent of villages present in donations dataset for every district participated');
plt.figure(figsize=(20,10))

text="";

for i in range(0,Donations.shape[0]):

    text = text +" " +Donations.iloc[i]['Taluka']

mask=np.array(Image.open("/kaggle/input/for-watercup-analysis/682420-maharashtra-state-outline.jpg"))

# Create and generate a word cloud image:

wordcloud = WordCloud(max_font_size=50, max_words=75, background_color="white",mask=mask,contour_color='firebrick').generate(text);



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear');

plt.title('word cloud of talkuas which received donations');

plt.axis('off');

plt.show();