# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# coding: utf-8



# In[ ]:





###########################################

# NAME: MUSA T. GANIYU                    #

#                                         #

# COURSE: DATA 602 PROJECT                #

#                                         #

# DATE: 12/08/2016                        #

#                                         #

###########################################





print("Project Title: The effect of Population on Crimes in USA")

print("Abstract: The data for the analysis was obtained from the Uniform Crime Reporting of the FBI website for the year 2012 on population less than 250,000. The dataset was divided into 4 equal strata and regression analysis was done on each group to verify the effect of population on crimes. A larger population seems to affect the crime rate, while small population has little or no effect on the crimes. ")

print('*****************************************************************************')

# You may comment the line below if youre not using Jupyter.



get_ipython().magic(u'matplotlib inline')

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)

print('Kindly install of these modules before continuing Pandas, matplotlib, mpl_toolkits.basemap, numpy, statsmodels.api, lmxl, request, os,re, collections, wordcloud' )

print('*****************************************************************************')

import pandas as pd

from pandas import *

pandas_file = ("../input/LocalCrimeOneYearofData2012.csv")

files = pd.read_csv(pandas_file) 

datafile = pd.DataFrame(files)

print('The head or a glimpes of the datafile.')

print(datafile.head())
print('*****************************************************************************')



print('Below is the sampled of 10 percent of the whole data. ')

datafile.set_index(['Agency', 'lat', 'long'])

sample10 = datafile.sample(frac=0.1, replace=False)

sample10[[0, 13, 14]]


print('*********************************WORD CLOUD********************************************')



print("Below, we are going to use some of the sample data to make a map on globe,by using their respective city names, longtitudes and latitudes ")

import matplotlib

matplotlib.style.use('ggplot')

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np
map = Basemap(projection='ortho', lat_0 = 20, lon_0 = -120,

              resolution = 'h', area_thresh = 5000.)

map.bluemarble()

map.drawmapboundary()

map.drawmeridians(np.arange(0, 360, 30))

map.drawparallels(np.arange(-90, 90, 30))
cities = ["Roseline", "Fremont", "Pasadena", "South Bend", "Garland", "Knoxville", "Mongomery", "Norwalk", "Pembroke", "Escondido", "Bellevue", "Broken Arrow", "Aurora", "Stamford", "Lafeyette", "Lowell", "Cape", "Glendale", "Santa Clara", "Durham", "Billings"]

lats= [38.7521, 37.5483, 34.1478, 41.6764, 32.9126, 35.9606, 32.3668, 41.1177, 26.0078,

33.1192, 47.6104, 36.0609, 41.7606, 41.0534, 30.2241, 42.6334, 26.5629, 33.5387, 37.3541, 35.9940, 45.7833]

lons = [-121.2880, -121.9886, -118.1445, -86.2520, -96.6389, -83.9207, -86.3000, -73.4082, -80.2963, 

-117.0864, -122.2007, -95.7975, -88.3201, -73.5387, -92.0198, -71.3162, -81.9495, -112.1860, 

-121.9552, -78.8986, -108.5007]





x,y = map(lons,lats)

# plot filled circles at the locations of the cities.

map.plot(x,y,'bo')

# plot the names of those sample of 10 percent of all the cities.

for name,xpt,ypt in zip(cities,x,y):

    plt.text(xpt+50000,ypt+50000,name)

plt.show()



print('*****************************SETTING UP HYPOTHESIS.****************************************')


print('NULL HYPOTHESIS : Population has effect on Crimes')



print('               Against                     ')





print('ALTERNATIVE HYPOTHESIS : Population has no effect on Crimes')







print('REJECTION:')

print('Reject Null Hypothesis if tabulated value(alpha <= 0.05) is less than calculated value, otherwise do not reject.')

print('**********************************************************************')



datafile.State.value_counts() # Print the values of states



datafile[['Population', 'State', 'Violent_crime_total', 'Property_crime_total']].max() # The Max Total crimes per state



datafile[['Population', 'State', 'Violent_crime_total', 'Property_crime_total']].min() # The Min Total crimes per state



datafile[['Population', 'Violent_crime_total', 'Property_crime_total']].cumsum().head() # The Columns sum



crimedata = datafile.copy() 
print("We will be distributing the data into four stratas, where analysis would be done on each stratum. The effects of population on crimes would be measured on each of them. ")





sample1 = crimedata.sample(frac=0.25, replace=False)

sample2 = crimedata.sample(frac=0.25, replace=False)

sample3 = crimedata.sample(frac=0.25, replace=False)

sample4 = crimedata.sample(frac=0.25, replace=False)
%matplotlib inline

import matplotlib

matplotlib.style.use('ggplot')
df3 = pd.DataFrame(files, columns=('Murder_and_Manslaughter ', 'Forcible_rape', 'Robbery', 'Aggravated_assault',

                                  'Burglary', 'Larceny_theft', 'Motor_vehicle_theft'))



color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

df3.plot.bar(stacked=True);



df3.plot.hist(stacked=True, bins=20)



df3.plot.box(color=color, sym='r+')

df3.plot.area()
print('Below is scatter plot showing interaction between Population and Crime rates. Also interaction between Violent Crime and Property Crimes')



df4 = pd.DataFrame(files, columns=('Murder_and_Manslaughter ', 'Forcible_rape', 'Robbery', 'Aggravated_assault',

                                  'Burglary', 'Larceny_theft', 'Population', 'Motor_vehicle_theft',  'Violent_crime_total', 'Property_crime_total' ))

df4.plot.scatter(x='Violent_crime_total', y='Property_crime_total');

df4.plot.hexbin(x='Violent_crime_total', y='Property_crime_total', C='Population', reduce_C_function=np.max, gridsize=25)



import pandas as pd

import statsmodels.api as sm



import statsmodels.formula.api as smf



smresults_unsmoothed = smf.ols('Population ~ Property_crime_total', crimedata).fit()



crimedata['ols_preds_unsmoothed'] = smresults_unsmoothed.predict()



print(smresults_unsmoothed.summary)
sampledata = crimedata[['Population', 'Violent_crime_total', 'Forcible_rape', 'Robbery', 'Aggravated_assault', 'Property_crime_total', 'Burglary', 'Larceny_theft',

          'Motor_vehicle_theft']]



print(sampledata.head())

print(sampledata.describe())




print('*****************************************************************************')



print(' We are going to extract text from the Uniform Crime Reporting of the Federal Bureau of Investigation website of the data description')

from lxml import html

import requests as r

from os import path

from wordcloud import WordCloud

import matplotlib.pyplot as plt



"""

page = r.get('https://www.ucrdatatool.gov/offenses.cfm')



tree = html.fromstring(page.content)



ucr = tree.xpath('//td[@valign="top"]/text()') #Cleaning of the extracted text from the website



# Clean the text cont'd



str(ucr).strip('[]')

a = ''.join([item.rstrip('\r\n\t\r\n') for item in ucr])

print(a) #the text





print('Some paragraphs of the extracted text would be used in our wordcloud below')

"""
from wordcloud import WordCloud



# Read the whole text.

# Generate a word cloud image



text = 'The UCR Program collects statistics on the number of offenses known to law enforcement.  In the traditional Summary Reporting System (SRS), there are eight crimes, or Part I offenses, (murder and nonnegligent homicide, forcible rape, robbery, aggravated assault, burglary, motor vehicle theft, larceny-theft, and arson) to be reported to the UCR Program.  These offenses were chosen because they are serious crimes, they occur with regularity in all areas of the country, and they are likely to be reported to police.  The Part I offenses are defined as:Murder and nonnegligent manslaughter: the willful (nonnegligent) killing of one human being by another. Deaths caused by negligence, attempts to kill, assaults to kill, suicides, and accidental deaths are excluded.  The program classifies justifiable homicides separately and limits the definition to:  (1) the killing of a felon by a law enforcement officer in the line of duty; or (2) the killing of a felon, during the commission of a felony, by a private citizen. b.) Manslaughter by negligence: the killing of another person through gross negligence. Deaths of persons due to their own negligence, accidental deaths not resulting from gross negligence, and traffic fatalities are not included in the category Manslaughter by Negligence.The carnal knowledge of a female forcibly and against her will.  Rapes by force and attempts or assaults to rape, regardless of the age of the victim, are included.  Statutory offenses (no force used—victim under age of consent) are excluded.The taking or attempting to take anything of value from the care, custody, or control of a person or persons by force or threat of force or violence and/or by putting the victim in fear.An unlawful attack by one person upon another for the purpose of inflicting severe or aggravated bodily injury.  This type of assault usually is accompanied by the use of a weapon or by means likely to produce death or great bodily harm.  Simple assaults are excluded.The unlawful entry of a structure to commit a felony or a theft.  Attempted forcible entry is included.The unlawful taking, carrying, leading, or riding away of property from the possession or constructive possession of another.  Examples are thefts of bicycles, motor vehicle parts and accessories, shoplifting, pocketpicking, or the stealing of any property or article that is not taken by force and violence or by fraud.  Attempted larcenies are included.  Embezzlement, confidence games, forgery, check fraud, etc., are excluded.The theft or attempted theft of a motor vehicle.  A motor vehicle is self-propelled and runs on land surface and not on rails.  Motorboats, construction equipment, airplanes, and farming equipment are specifically excluded from this category.Any willful or malicious burning or attempt to burn, with or without intent to defraud, a dwelling house, public building, motor vehicle or aircraft, personal property of another, etc. Arson statistics are not included in this table-building tool.'





wordcloud = WordCloud().generate(text)



# Display the generated image:

# the matplotlib way:

import matplotlib.pyplot as plt





# take relative word frequencies into account, lower max_font_size

wordcloud = WordCloud(background_color="white",max_words=len(text),max_font_size=40, relative_scaling=.5).generate(text)

plt.figure()

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
smallpop = sample4[sample4['Population'] <= 120000]



bigpop = sample4[sample4['Population']>120000]



import statsmodels.formula.api as smf
print('***************Regression Analysis on the stratum (Big Population)********************')

lm = smf.ols(formula='Population ~ Violent_crime_total + Property_crime_total', data=bigpop).fit()

print(lm.summary())
print('From the above table, it reveals that P-value(0.442) is greater than 0.05 for Violent Crimes, but not with Property Crimes(0.002)')
print('******Regression Analysis on the stratum (Small Population)*******************')

lm2 = smf.ols(formula='Population ~ Violent_crime_total + Property_crime_total', data=smallpop).fit()

print(lm2.summary())
print('From the above table, it reveals that P-value greater for both Violent and Property crime for small dataset')
print('***************Regression Analysis on the whole population*****************************')

lm3 = smf.ols(formula='Population ~ Robbery + Murder_and_Manslaughter + Forcible_rape + Aggravated_assault + Burglary + Larceny_theft + Motor_vehicle_theft', data=crimedata).fit()

print(lm3.summary())
print('------From the above table, it reveals that P-values are statisticall insignificant for all variables , except Larceny for the whole dataset------')
print('********Regression Analysis on the strata (first Sampled Population)*******************')

lm4 = smf.ols(formula='Population ~ Violent_crime_total + Property_crime_total', data=sample1).fit()

print(lm4.summary())
print('From the above table, it reveals that P-value(0.898) is greater than 0.05 for Violent Crimes, but not with Property Crimes(0.000)')
print("****************DECISION AND CONCLUSION:**********************************************")

print("From the above analysis, since P-values (Calculated Values) are greater than the alpha at 0.05, we therefore fail to reject Null Hypothesis, and conclude that the Population has effect on the crime rate. The higher the population, the greater the Violent crime and vice-versa. ")
print("Disclaimer: The analysis may not have covered all the factors (variables) that may contribute to Crime in a given population. Therefore, factors like Education, Poverty Rate etc may also contribute to the crime rate in a given population.")