# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
median_house_hold_in_come = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
'''
Her bir eyaletin fakirlik oranını gösteren grafik (seaborn barplot)
'''
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0, inplace = True) #  datada bulunan - değerleri 0 a eşitledik
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float) # str olan sayısal verileri float cinsine çevirdik
area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique()) # unique metodu ile eyalet isimlerini liste içine aldık (unique = dataset içindeki elemanlardar tekrar etmeyenleri alır)
area_poverty_ratio = [] # for döngüsünden gelenleri dizi içine alıyoruz.
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"]==i]
    area_poverty_rate = sum(x.poverty_rate/len(x))
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list' : area_list, 'area_poverty_ratio' : area_poverty_ratio})  # oluşturduğumuz listeleri bir data frame içine atıyoruz
new_index = (data['area_poverty_ratio'].sort_values(ascending = False).index.values) # area_poverty_ratio listemizi sıralı hale indexledik
sorted_data = data.reindex(new_index) # sıralı listeyi yeni bir datanın içine attık

#visualization

plt.figure(figsize = (15,10))  # matplotlib ile bir figür oluşturduk
sns.barplot(x = sorted_data['area_list'], y = sorted_data['area_poverty_ratio'])  # Seaborn barplot oluşumu
plt.xticks(rotation = 45)  # Şehir isimleri 45 derece eğik
plt.xlabel('States') # x ekseni ismi
plt.ylabel('Poverty Rate') # y ekseni ismi
plt.title('Poverty Rate Given States') # grafik ismi
plt.show()

'''
İsim veya soyisim olarak en çok öldürülen 15 İnsan
'''
separate = kill.name[kill.name != 'TK TK'].str.split()  # Datada bozuk veri olan TK TK isimlerini ayırdık
a,b = zip(*separate)
name_list = a+b
name_count = Counter(name_list)  # İsimlerin kaç tane olduğunu aldık
most_common_names = name_count.most_common(15) # en çok olan 15 tanesini aldık
x,y = zip(*most_common_names) 
x,y = list(x), list(y)

plt.figure(figsize = (15,10))
sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(len(x)))  # palette grafik renklendirme
plt.xlabel("Name or Surname")
plt.ylabel("Frequency")
plt.title("Name or Surname of Killeo People")
plt.show()
'''
Eyaletlerde 25 yaşından büyük insanların lise mezunu olma oranı
'''
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'], 0.0, inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
area_list = list(percent_over_25_completed_highSchool["Geographic Area"].unique())
area_highschool = []

for i in area_list:
    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool["Geographic Area"] == i]
    area_highschool_rate = sum(x.percent_completed_hs) / len(x)
    area_highschool.append(area_highschool_rate)
data = pd.DataFrame({'area_list' : area_list, 'area_highschool_ratio' : area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values
sorted_data2 = data.reindex(new_index)

plt.figure(figsize = (15,10))
sns.barplot(x = sorted_data2['area_list'], y = sorted_data2['area_highschool_ratio'])
plt.xticks(rotation = 90)
plt.xlabel("States")
plt.ylabel("High School Graduate Rate")
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
plt.show()
'''
Eyaletlerdeki etnik kimlik oranı
'''
share_race_city.replace(['-'], 0.0, inplace = True)
share_race_city.replace(['(X)'], 0.0, inplace = True)
share_race_city.loc[:, ['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']] = share_race_city.loc[:, ['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']].astype(float)
area_list = list(share_race_city['Geographic area'].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []
for i in area_list:
    x = share_race_city[share_race_city['Geographic area'] == i]
    share_white.append(sum(x.share_white) / len(x))
    share_black.append(sum(x.share_black) / len(x))
    share_native_american.append(sum(x.share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))

f, ax = plt.subplots(figsize = (9,15))                #Alpha grafiğin saydamlığını ayarlar
sns.barplot(x = share_white, y = area_list, color = 'green', alpha = 0.5, label = 'White')
sns.barplot(x = share_black, y = area_list, color = 'blue', alpha = 0.7, label = 'African American')
sns.barplot(x = share_native_american, y = area_list, color = 'cyan', alpha = 0.6, label = 'Native American')
sns.barplot(x = share_asian, y = area_list, color = 'yellow', alpha = 0.6, label = 'Asian')
sns.barplot(x = share_hispanic, y = area_list, color = 'red', alpha = 0.6, label = 'Hispanic')

ax.legend(loc = 'lower right', frameon = True)   # label lerin yer tutması için 
ax.set(xlabel = 'Percentage of Races', ylabel = 'States', title = 'Percentage of States Population According to Races')

'''
Eyaletlerdeki liseden mezun olma ve fakirlik oranı karşılaştırması  (Point Plot)
'''
# İlk iki satırda değerleri max değere bölerek normalization yaptık

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio'] / max(sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio'] / max(sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data, sorted_data2['area_highschool_ratio']], axis = 1)
data.sort_values('area_poverty_ratio', inplace = True)

f, ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x = 'area_list', y = 'area_poverty_ratio', data = data, color='lime', alpha = 0.8)
sns.pointplot(x = 'area_list', y = 'area_highschool_ratio', data = data, color='red', alpha = 0.8)
plt.text(40,0.6, 'high school graduate ratio', color = 'red', fontsize = 17, style = 'italic')
plt.text(40,0.55, 'poverty ratio', color = 'lime', fontsize = 17, style = 'italic')
plt.xlabel('States', fontsize = 15, color = 'blue')
plt.ylabel('Values', fontsize = 15, color = 'blue')
plt.title('High School Graduate ve Poverty Ratio', fontsize = 20, color = 'blue')
plt.grid()


'''
Bir üstteki grafiği (JointPlot) ile tekrar yapalım

Pearsonr = 1 e ne kadar yakınsa o kadar doğru -1 e ne kadar yakınsa o kadar ters orantı vardır.
Eğer değer sıfır 0 ise oran yoktur.
kde = kernel density estimation
'''

g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind='kde', size = 7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot('area_poverty_ratio', 'area_highschool_ratio', data = data, size = 5, ratio = 3, color = 'r')
'''
Öldürülen insanların ırklarının oranı      (Pie Plot)
'''
kill.race.dropna(inplace = True)  # race kısmı boş olan verileri çıkarttık
labels = kill.race.value_counts().index    # race kısmında hangi veriden kaç tane oluğunu bulup indexlerini aldık ('w','a','b') gibi
colors = ['grey', 'blue', 'red', 'yellow', 'green', 'brown']
explode = [0,0,0,0,0,0]
sizes = kill.race.value_counts().values   # kill.race.value_counts() kısmındaki valueleri aldık

# visual

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed People According to Races', color = 'blue', fontsize = 15)
plt.show()
'''
Eyaletlerdeki liseden mezun olma ve fakirlik oranı karşılaştırması (LM Plot)
lmplot
Show the results of a linear regression within each dataset
'''

sns.lmplot(x = "area_poverty_ratio", y = "area_highschool_ratio", data=data)
plt.show()
'''
Eyaletlerdeki liseden mezun olma ve fakirlik oranı karşılaştırması (KDE Plot)
'''

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=5)
plt.show()
'''
Eyaletlerdeki liseden mezun olma ve fakirlik oranı karşılaştırması (Violin Plot)
Show each data both violins and points
cubehelix kullandık ve palette ile özelleştirdik
'''

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner='points')
plt.show()
'''
Correlation Map
Eyaletlerdeki liseden mezun olma ve fakirlik oranı karşılaştırması
'''

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
'''
manner of death (Ölüm şekli) shot: ateş edilerek, Tasered: şok tabancası ile    (Box Plot)
gender: cinsiyet
age: yaş
'''
sns.boxplot(x = 'gender', y = 'age', hue='manner_of_death', data=kill, palette='PRGn')
plt.show()
'''
Yukarıdaki problemi (Swarm Plot) ile çizelim
'''

sns.swarmplot(x = 'gender', y = 'age', hue='manner_of_death', data=kill)
plt.show()
'''
Eyaletlerdeki liseden mezun olma ve fakirlik oranı karşılaştırması (Pair Plot)
'''
sns.pairplot(data)
plt.show()

# manner of death
# gender

sns.countplot(kill.gender)
sns.countplot(kill.manner_of_death)
plt.title('Manner of Death', color = 'blue', fontsize = 15)
plt.show()
# kill weapon (Öldürülen insanların kullandığı silahlar)

armed = kill.armed.value_counts()
plt.figure(figsize = (10,7))
sns.barplot(x = armed[:7].index, y = armed[:7].values)  # ilk 7 veriyi kullanmak istiyoruz [:7]
plt.ylabel("Number of weapon")
plt.xlabel("Weapon Type")
plt.title("Kill Weapon", color = 'blue', fontsize = 15)
plt.show()
# Ölen insanların yaşı
above25 = ['above25' if i >= 25 else 'below25' for i in kill.age]
df = pd.DataFrame({'age': above25})
sns.countplot(x = df.age)
plt.ylabel('Number of Killed People')
plt.title('Age of Killed People', color = 'blue', fontsize = 15)
plt.show()