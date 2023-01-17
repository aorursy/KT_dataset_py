# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
happiness2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv")

happiness2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")

happiness2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")

happiness2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")

happiness2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
type(happiness2015), type(happiness2016), type(happiness2017), type(happiness2018),type(happiness2019)
len(happiness2015.columns), len(happiness2016.columns),len(happiness2017.columns),len(happiness2018.columns),len(happiness2019.columns)
happiness2015.columns
happiness2016.columns
happiness2017.columns
happiness2018.columns
happiness2019.columns
happiness2015cpp=happiness2015.copy(); happiness2016cpp=happiness2016.copy(); happiness2017cpp=happiness2017.copy(); happiness2018cpp=happiness2018.copy(); happiness2019cpp=happiness2019.copy() 
happiness2015cpp.drop(['Happiness Rank',

       'Standard Error', 'Economy (GDP per Capita)', 'Family','Dystopia Residual'],axis=1, inplace=True)
happiness2016cpp.drop(['Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval','Economy (GDP per Capita)', 'Family','Dystopia Residual'],axis=1, inplace=True)
happiness2017cpp.drop(['Happiness.Rank', 'Whisker.high',

       'Whisker.low', 'Economy..GDP.per.Capita.', 'Family','Dystopia.Residual'],axis=1, inplace=True)
happiness2018cpp.drop(['Overall rank', 'GDP per capita',

       'Social support',],axis=1, inplace=True)
happiness2019cpp.drop(['Overall rank', 'GDP per capita',

       'Social support',],axis=1, inplace=True)
len(happiness2015cpp.columns), len(happiness2016cpp.columns),len(happiness2017cpp.columns),len(happiness2018cpp.columns),len(happiness2019cpp.columns)
happiness2015cpp.columns
happiness2015cpp.rename(columns={'Health (Life Expectancy)':"Health_2015",

                               'Trust (Government Corruption)':"Corruption_2015",

                                 'Region':'Region_2015',

                                 'Happiness Score':'Happiness_2015',

                                 'Freedom':'Freedom_2015',

                                 'Trust (Government Corruption)':'Corruption_2015',

                                 'Generosity':'Generosity_2015'

                    }, inplace=True) 
happiness2015cpp.columns
happiness2016cpp.columns
happiness2016cpp.rename(columns={'Health (Life Expectancy)':"Health_2016",

                               'Trust (Government Corruption)':"Corruption_2015",

                                 'Region':'Region_2016',

                                 'Happiness Score':'Happiness_2016',

                                 'Freedom':'Freedom_2016',

                                 'Trust (Government Corruption)':'Corruption_2016',

                                 'Generosity':'Generosity_2016'

                    }, inplace=True) 
happiness2016cpp.columns
happiness2017cpp.columns
happiness2017cpp.rename(columns={'Health..Life.Expectancy.':"Health_2017",

                               'Trust..Government.Corruption.':"Corruption_2017",

                                 'Region':'Region_2017',

                                 'Happiness.Score':'Happiness_2017',

                                 'Freedom':'Freedom_2017',

                                 'Generosity':'Generosity_2017'

                    }, inplace=True) 
happiness2018cpp.columns
happiness2018cpp.rename(columns={'Healthy life expectancy':"Health_2018",

                               'Perceptions of corruption':"Corruption_2018",

                                 'Region':'Region_2018',

                                 'Score':'Happiness_2018',

                                 'Freedom to make life choices':'Freedom_2018',

                                 'Generosity':'Generosity_2018'

                    }, inplace=True) 
happiness2019cpp.columns
happiness2019cpp.rename(columns={'Healthy life expectancy':"Health_2019",

                               'Perceptions of corruption':"Corruption_2019",

                                 'Score':'Happiness_2019',

                                 'Freedom to make life choices':'Freedom_2019',

                                 'Generosity':'Generosity_2019'

                    }, inplace=True)
happiness2015cpp.columns
happiness2016cpp.columns
happiness2017cpp.columns
happiness2018cpp.columns
happiness2018cpp.rename(columns={'Country or region':"Country"

                             

                    }, inplace=True)
happiness2018cpp.columns
happiness2019cpp.rename(columns={'Country or region':"Country"

                             

                    }, inplace=True)
happiness2019cpp.columns
happinessnew=pd.merge(happiness2015cpp,happiness2016cpp, on="Country", how="outer")
happinessnew.tail(3)
happinessnew["Country"].value_counts()
number=happinessnew.Country.unique()

a=0

for each in number:

    a=a+1

print(a)
happinessnew1=pd.merge(happiness2017cpp,happiness2018cpp, on="Country", how="outer")
happinessnew2=pd.merge(happinessnew1,happiness2019cpp, on="Country", how="outer")
happinessnew.columns
happinessnew2.columns
happinessnewend=pd.merge(happinessnew,happinessnew2, on="Country", how="outer")
happinessnewend.info()
happinessnewend["Country"].value_counts()
number1=happinessnewend.Country.unique()

a=0

for each in number1:

    a=a+1

print(a)
happinessnewend.isnull().sum()
happinessnewend.Region_2015.unique()
happinessnewend[happinessnewend["Region_2015"].isnull()]
happinessnewend.iloc[-12:-6]["Region_2015"]=["Latin America and Caribbean","Latin America and Caribbean","Sub-Saharan Africa","Sub-Saharan Africa","Sub-Saharan Africa","Sub-Saharan Africa"]
happinessnewend.tail(13)
happinessnewend[happinessnewend["Region_2015"].isnull()]
happinessnewend.iloc[-6:]["Region_2015"]=["East Asia","East Asia","South America","Western Asia","Southeast Europe","Western Africa"]
happinessnewend.tail(13)

happinessnewend.drop("Region_2016", axis=1, inplace=True)
happinessnewend[happinessnewend["Region_2015"].isnull()]
happinessnewend.isnull().sum()
for each in happinessnewend.columns[2:]:

    happinessnewend[each].fillna((happinessnewend[each].mean()), inplace=True)
happinessnewend.isnull().sum()

happinessnewend.info()
happinessnewend.describe()
happinessnewend.rename(columns={'Region_2015':'Region'

                             

                    }, inplace=True)

happinessnewend.Region.unique()
len(happinessnewend.Region.unique())
happinessnewend.columns
happinessnewend[['Happiness_2015', 'Health_2015', 'Freedom_2015',

       'Corruption_2015', 'Generosity_2015']].corr()
crr_2015 = happinessnewend[['Happiness_2015', 'Health_2015', 'Freedom_2015',

       'Corruption_2015', 'Generosity_2015']].corr()

crr_2015 = crr_2015.iloc[:,0]

crr_2015 = sorted(crr_2015)

crr_2015value = crr_2015[-2]

crr_2015value
happinessnewend[['Happiness_2016', 'Health_2016',

       'Freedom_2016', 'Corruption_2016', 'Generosity_2016']].corr()
crr_2016 = happinessnewend[['Happiness_2016', 'Health_2016',

       'Freedom_2016', 'Corruption_2016', 'Generosity_2016']].corr()

crr_2016 = crr_2016.iloc[:,0]

crr_2016 = sorted(crr_2016)

crr_2016value = crr_2016[-2]

crr_2016value
happinessnewend[['Happiness_2017',

       'Health_2017', 'Freedom_2017', 'Generosity_2017', 'Corruption_2017']].corr()
crr_2017 = happinessnewend[['Happiness_2017',

       'Health_2017', 'Freedom_2017', 'Generosity_2017', 'Corruption_2017']].corr()

crr_2017 = crr_2017.iloc[:,0]

crr_2017 = sorted(crr_2017)

crr_2017value = crr_2017[-2]

crr_2017value
happinessnewend[['Happiness_2018', 'Health_2018', 'Freedom_2018', 'Generosity_2018',

       'Corruption_2018']].corr()
crr_2018 = happinessnewend[['Happiness_2018', 'Health_2018', 'Freedom_2018', 'Generosity_2018',

       'Corruption_2018']].corr()

crr_2018 = crr_2018.iloc[:,0]

crr_2018 = sorted(crr_2018)

crr_2018value = crr_2018[-2]

crr_2018value
happinessnewend[['Happiness_2019', 'Health_2019', 'Freedom_2019',

       'Generosity_2019', 'Corruption_2019']].corr()
crr_2019 = happinessnewend[['Happiness_2019', 'Health_2019', 'Freedom_2019',

       'Generosity_2019', 'Corruption_2019']].corr()

crr_2019 = crr_2019.iloc[:,0]

crr_2019 = sorted(crr_2019)

crr_2019value = crr_2019[-2]

crr_2019value
import matplotlib.pyplot as plt



year = ['2015', '2016', '2017', '2018', '2019']

corelation = [crr_2015value, crr_2016value, crr_2017value, crr_2018value, crr_2019value]

plt.plot(year, corelation, color='orange')

plt.xlabel('Years', color = 'b', size = 13)

plt.ylabel('Corelation', color = 'b', size = 13)

plt.title('Health - Happiness Corelation', color = 'r', size = 16)

plt.show()
happinessnewend.groupby(['Region']).Country.count()
happyavr = happinessnewend[['Country', 'Region', 'Happiness_2015', 'Happiness_2016', 'Happiness_2017', 'Happiness_2018', 'Happiness_2019']]

happyavr['Average'] = happyavr.mean(axis=1)

happyavr = happyavr.sort_values(by = 'Average', ascending = False)

happyavr.head(3)
happyavr.tail(3)
corruptionavr = happinessnewend[['Country', 'Region', 'Corruption_2015', 'Corruption_2016', 'Corruption_2017', 'Corruption_2018', 'Corruption_2019']]

corruptionavr['Average'] = corruptionavr.mean(axis=1)

corruptionavr = corruptionavr.sort_values(by = 'Average', ascending = False)

corruptionavr.head(3)
corruptionavr.tail(3)
freedomavr = happinessnewend[['Country', 'Region', 'Freedom_2015', 'Freedom_2016', 'Freedom_2017', 'Freedom_2018', 'Freedom_2019']]

freedomavr['Average'] = freedomavr.mean(axis=1)

freedomavr = freedomavr.sort_values(by = 'Average', ascending = False)

freedomavr
freedom_mean = freedomavr.groupby(['Region']).Average.mean()

freedom_mean = freedom_mean.sort_values() 

print("Least free region :", freedom_mean.index[0], "\nMost free region:", freedom_mean.index[-1])
healthavr = happinessnewend[['Country', 'Region', 'Health_2015', 'Health_2016', 'Health_2017', 'Health_2018', 'Health_2019']]

healthavr['Average'] = healthavr.mean(axis=1)

healthavr = healthavr.sort_values(by = 'Average', ascending = False)

healthavr
health_mean = healthavr.groupby(['Region']).Average.mean()

health_mean = health_mean.sort_values() 

print("Least healthy region :", health_mean.index[0])
happiness_mean = happyavr.groupby(['Region']).Average.mean()

happiness_mean = happiness_mean.sort_values()

happiness_mean
freedom_mean = freedomavr.groupby(['Region']).Average.mean()

freedom_mean = freedom_mean.sort_values()

freedom_mean