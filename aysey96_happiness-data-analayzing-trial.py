# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/world-happiness"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#2015-2019 Yılları Türkiye Verisinin İncelenmesi
data_2015=pd.read_csv('../input/world-happiness/2015.csv')

data_2016=pd.read_csv('../input/world-happiness/2016.csv')

data_2017=pd.read_csv('../input/world-happiness/2017.csv')

data_2018=pd.read_csv('../input/world-happiness/2018.csv')

data_2019=pd.read_csv('../input/world-happiness/2019.csv')





data_2015=data_2015.drop(['Region','Happiness Rank','Standard Error','Dystopia Residual'],axis=1)



data_2015.info()
data_2016=data_2016.drop(['Region','Happiness Rank','Lower Confidence Interval','Upper Confidence Interval','Dystopia Residual'],axis=1)

data_2016.info()

data_2017=data_2017.drop(['Happiness.Rank','Whisker.high','Dystopia.Residual','Whisker.low'],axis=1)

data_2017.info()
#Rename columns



data_2017.rename(columns={'Happiness.Score':'Happiness Score',

                          'Economy..GDP.per.Capita.':'Economy (GDP per Capita)',

                          'Health..Life.Expectancy.':'Health (Life Expectancy)',

                         'Trust..Government.Corruption.':'Trust (Government Corruption)'}, 

                 inplace=True)

data_2017.info()
data_2018=data_2018.drop(['Overall rank'],axis=1)


data_2018.rename(columns={'Country or region':'Country',

                          'Score':'Happiness Score',

                          'GDP per capita':'Economy (GDP per Capita)',

                         'Healthy life expectancy':'Health (Life Expectancy)',

                         'Freedom to make life choices': 'Freedom',

                         'Perceptions of corruption':'Trust (Government Corruption)',

                        'Social support':'Family' }, 

                 inplace=True)
data_2018.info()
data_2019=data_2019.drop(['Overall rank'],axis=1)


data_2019.rename(columns={'Country or region':'Country',

                          'Score':'Happiness Score',

                          'GDP per capita':'Economy (GDP per Capita)',

                         'Healthy life expectancy':'Health (Life Expectancy)',

                         'Freedom to make life choices': 'Freedom',

                         'Perceptions of corruption':'Trust (Government Corruption)',

                         'Social support':'Family'}, 

                 inplace=True)
data_2019.info()
data_2015['Year']='2015'

data_2016['Year']='2016'

data_2017['Year']='2017'

data_2018['Year']='2018'

data_2019['Year']='2019'

#data_2015.info()

#concating

data=pd.concat([data_2015,data_2016,data_2017,data_2018,data_2019],axis=0,sort = False)

#data_Tr=data[data['Country']=='Turkey']

data_Tr=data[data['Country']=='Turkey']

data_Tr
import matplotlib.pyplot as plt

import seaborn as sns

#Mutluluk için bileşenlere baktığımızda en yüksek skora sahip 2017 yılında GDP değeri en yüksekkken en düşük skora sahip 2015 yılında GDP ciddi seviyede düşüş yaşamıştır. Aradaki değerlerde doğrusal bir ilşki çıkmasa da   mutluluk puanı en yüksek ve en düşük yıllardaki  GDP değerinin skorla paralel gitmesi önemli diyebiliriz.

#Family bileşeni de n yüksek puana sahip 2017 ve 2018 yıllaında yüksek seviyededir. En az değee sahip 2015 yılında ise düşüktür.

#Freedom ve Trust(Yolsuzluk varlığına inanma) alanı içinde benzer bir tablo çıkarken Health, Genorisity için   doğrusallıktan uzak bir tablo çıkmaktadır.

sns.pairplot(data=data_Tr, y_vars="Happiness Score", x_vars=["Economy (GDP per Capita)","Family","Health (Life Expectancy)","Freedom","Generosity","Trust (Government Corruption)"],hue="Year",palette="muted")
#GDP

fig, ax = plt.subplots(ncols=1,figsize=(10, 8))

sns.scatterplot(x="Happiness Score", y="Economy (GDP per Capita)", hue="Year", data=data_Tr,palette="muted",ax=ax)
#Herbir bileşenin yıllara göre incelenmesi

sns.pairplot(data=data_Tr, y_vars="Year", x_vars=["Economy (GDP per Capita)","Family","Health (Life Expectancy)","Freedom","Generosity","Trust (Government Corruption)"],palette="muted")
data_2019=pd.read_csv('../input/world-happiness/2019.csv')

data_2019.head()
data_2019.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data_2019.columns]
import matplotlib.pyplot as plt

import seaborn as sns
#Türkiye 2019 

print(data_2019[data_2019['Country']=='Turkey'])
data_2019.columns
#2019'da mutluluk puanı en yüksek ülke Finlandiya çıkmıştır.

data_2019.sort_values(by='Score',ascending=False)
#Mutluluk puanı ve diğer sütunların ilişkilerine baktığımızda GDP, Social Support, Healthy sütunları ön plana çıkmaktadır.

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data_2019.corr(),annot=True,linewidth=.5,fmt='.1f')

plt.show()
X=data_2019.drop('Score',axis=1)

X.columns
#Regresyon doğrusyla da inceledğimizde GDP, Scocial Support ve Healty alanları ile Score arasında doğrusal bir ilişki görürüz. Regressyon doğrusunun eğimi yüksektir.

g = sns.pairplot(data_2019, y_vars=["Score"], x_vars=[ "GDP", "Social_support", "Healthy",

       "Freedom", "Generosity", "Perceptions"],kind="reg")

plt.show()