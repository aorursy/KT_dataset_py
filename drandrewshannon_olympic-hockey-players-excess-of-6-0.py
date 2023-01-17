import json

import math

import matplotlib.mlab as mlab

import matplotlib.patches as mpatches

import matplotlib.pyplot as pyplot



import numpy # linear algebra

import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)

import pprint

import re

import requests

from scipy.stats import norm

import seaborn

from sklearn.mixture import GaussianMixture

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using

#"Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




for YEAR in [1924, 1928, 1932, 1936, 1948, 

             1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018] :



# The category naming is consistent

# The years there were olympics isn't, so you gotta type that list out by hand.  Sorry.

    print("Loading year ", YEAR) #Just so I know it ain't frozen

    category_url = "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Ice_hockey_players_at_the_" + str(YEAR) + "_Winter_Olympics&cmlimit=10000&format=json"

    category_response = requests.get(category_url).text

    category_data = json.loads(category_response)



    names = []



    for index in range(len(category_data['query']['categorymembers'])):

    #    print(len(data['query']['categorymembers']))    

    #    print(data['query']['categorymembers'][index]['title'])

    #    These are just leftovers from when I first loaded a page and read through the data

    #    I left it here so you can follow how I did it; of course, I actually did print(data), then print(data['query'])

    #    and so forth until I got to where i needed, only opening a single page

        names.append(category_data['query']['categorymembers'][index]['title'])



    #print(players)



    value_holder = 0 #It holds value



    protoframe = []  #I'm going to assemble it as a list, for simplicity



    for name in names :



        #print(name, ' ', end='')

    

        url = "https://en.wikipedia.org/w/api.php?action=query&titles=" + name + "&prop=revisions&rvprop=content&format=json"



        response = requests.get(url).text



        data = json.loads(response)



        page_num = next(iter(data['query']['pages'].keys())) # Okay, looks odd, Wikipedia has a page number, I think for querying 

                                                             # multi pages at once.  But I think one by one is easier, so the only

                                                             # key is the one I want

                                                             # Maybe partly too because I wanted to get the script working on a 

                                                             # single page, before trying to load a whole category's worth



        page = data['query']['pages'][page_num]['revisions'][0]['*'] #As above, found iteratively



        height_inch = 0

        height_foot = 0

        

        #Right, so because Wikipedia is full of hand entered data by many people, it's pretty irregular

        #Still, the infoboxen are semi-standaradised, so that helps immensely

        #Anyways, height_foot and height_inch is the most common standard, so I'm using them

        #But, can be metric, and you can have blank metric and filled imperial or vice versa

        #Anyways, do the obvious things : use try and have the code tell you every time it fails, inspect the failure by hand

    

        #In general, using "try" on a float() tells you you've indeed found a number

        #You gotta be careful not to set the height_foot or height_inch until you're confident the try is going to succeed

        if len(re.split('height_ft\s*=\s*', page)) > 1 :

            try :

                height_temp = re.split('height_ft\s*=\s*', page)[1]

                height_temp = float(re.split('\n|\s+|<', height_temp)[0])

                height_foot = height_temp

                if len(re.split('height_in\s*=\s*', page)[1]) > 1 :

                    height_temp = re.split('height_in\s*=\s*', page)[1]

                    height_temp = float(re.split('\n|\s+|<', height_temp)[0])

                    height_inch = height_temp

                else :

                    height_inch = 0

            except :

                value_holder += 1

            

        if len(re.split('height_cm\s*=\s*', page)) > 1 :

            try :

                height_cm = re.split('height_cm\s*=\s*', page)[1]

                height_cm = float(re.split('\n|\s+|<', height_cm)[0])

                height_inch = height_cm/2.54 #We're gonna use this, but we've passed the danger zone of trying to convert to float

                height_foot = 0

    #            print("Alpha", height_inch)

            except :

                value_holder += 1

    #            print("Bravo", height_inch, height_foot)

            

        if len(re.split('height_m\s+=\s+', page)) > 1 :

            try :

                height_temp = re.split('height_m\s+=\s+', page)[1]

                height_temp = float(re.split('\n|\s+|<', height_temp)[0])/0.0254

                height_inch = height_temp

                height_foot = 0

            except :

                value_holder += 1

                

        #Found a few of these generic ones, they seem to typically be 6 ft 3 in or 1.91 m formats

        #Do check that other formats aren't used

                

        if len(re.split('height\s+=\s+', page)) > 1 :

            try : 

                height_temp = re.split('height\s+=\s+', page)[1]

                if len(re.split('\s+ft\s+', height_temp)) > 1 :

                    height_temp = float(re.split('\s+ft\s+', height_temp)[0].split('\n')[0])

                    height_foot = height_temp

                    height_inch = re.split('\s+ft\s+', page)[1]

                    if len(re.split('\s+in\s+', height_inch[:20])) > 1 : #Limiting it to the next few characters reduces the risk of a false positive

                        height_inch = float(re.split('\s+ft\s+', re.split('\s+in\s+', page)[0])[1])

                    else :

                        height_inch = 0

                else :

                    height_inch = 0

                    height_foot = 0        

            except : 

                value_holder += 1



        weight_lb = 0

        weight_temp = 0

        weight_kg = 0

    

        if len(re.split('weight_lb\s*=\s*', page)) > 1 : 

            try :

                weight_temp = re.split('weight_lb\s*=\s*', page)[1]

                weight_temp = float(re.split('\n|s+|<', weight_temp)[0])

                weight_lb = weight_temp

            except :

                value_holder += 1

        

        if len(re.split('weight_lbs\s*=\s*', page)) > 1 : 

            try : 

                weight_temp = re.split('weight_lbs\s*=\s*', page)[1]

                weight_temp = float(re.split('\n|s+|<', weight_temp)[0])

                weight_lb = weight_temp

            except :

                value_holder += 1

            

        if len(re.split('weight_kg\s*=\s*', page)) > 1 :

            try:

                weight_kg = re.split('weight_kg\s*=\s*', page)[1]

                weight_kg= float(re.split('\n|\s+|<', weight_kg)[0])

                weight_lb = weight_kg*2.2

            except : 

                value_holder += 1

            

        #As with heights, the generic weights are lb lbs kg kgs - split on those, look for successes

        #limit yourself to the next 20 chars to avoid hitting false positives

        if len(re.split('weight\s*=\s*', page)) > 1 :

            try: 

                weight_temp = re.split('weight\s*=\s*', page)[1]

                if len(re.split('\s*lb\s*', weight_temp[:20])) > 1 :

                    weight_temp = float(re.split('\s*lb\s*', weight_temp[:20])[0])

                    weight_lb = weight_temp

                elif len(re.split('\s*kg\s*', weight_lb[:20])) > 1 :

                    weight_temp = float(re.split('\s*kg\s*', weight_temp[:20])[0])

                    weight_lb = weight_temp*2.2

            except :

                value_holder += 1

            

        birth_year = page.split(' births]]')[0] #This is just the Wikipedia categorisation scheme

                                                #But it seems to be used pretty ubiquitously

    

        try:

            birth_year = int(birth_year[-4:]) #HARDCODING THEY WERE BORN IN A YEAR WITH 4 DIGITS?!  FOR SHAME!

        except :

            birth_year = 0

    

        if len(re.split('ntl_team\s*=\s*', page)) > 1 : 

            nationality = re.split('ntl_team\s*=\s*', page)[1]

            if nationality[0] == '{':

                nationality = nationality[2:5].upper()

            else :

                nationality = nationality[:3].upper()

        else :

            nationality = 'XXX'

        #print(nationality)

        

    

        #Okay, the below is not the most elegant solution, but it's pretty messy in the boxen

        if nationality not in ['AUS', 'AUT', 'BEL', 'BLR', 'BUL', 'CAN', 'CHE', 'CHI', 'CHN', 'CSK', 

                               'CZE', 'DEU', 'DDR', 'EUA', 'EUN', 'FIN', 'FRA', 'FRG', 'GBR', 'GER', 

                               'GRE', 'HUN', 'ITA', 'JAP', 'KOR',

                               'JPN', 'KAZ', 'LAT', 'LVA', 'NED', 'NET', 'NLD', 'NOR', 

                               'POL', 'PRK', 'ROM', 'SLO', 'SFR', 'SOU', 'SOV', 'SUI', 'SWI', 'SVK', 'SVN', 

                               'SWE', 'TCH', 'RUS', 'UKR', 'UNI', 'URS', 'USA', 

                               'USR', 'USS', 'WES', 'YUG', 'XXX'] :

            #print('UNREGISTERED NATIONALITY: ', nationality, ' for ', name)

            #Don't need this anymore, but it is indeed how I found all the errant nationalities

            

            nationality = 'XXX'

        else :

            if nationality == 'BEL' :

                nationality = 'BLR'

            if nationality == 'CHE' :

                nationality = 'SUI'

            if nationality == 'CHI' :

                nationality = 'CHN'

            if nationality == 'CSK' :

                nationality = 'TCH'

            if nationality == 'EUA' :

                nationality = 'GER'

            if nationality == 'EUN' :

                nationality = 'URS'

            if nationality == 'GRE' : #Could also be a GER type, but the first example O found was Great Britain typed out by hand

                nationality = 'GBR'

            if nationality == 'JAP' :

                nationality = 'JPN'

            if nationality == 'NET' :

                nationality = 'NED'

            if nationality == 'LVA' :

                nationality = 'LAT'

            if nationality == 'NLD' :

                nationality = 'NED'

            if nationality == 'SFR' :

                nationality = 'YUG'

            if nationality == 'SOU' :

                nationality = 'KOR'

            if nationality == 'SOV' :

                nationality = 'USR'

            if nationality == 'SVN' :

                nationality = 'SLO'

            if nationality == 'SWI' :

                nationality = 'SUI'

            if nationality == 'UNI' :

                nationality = 'USA' #Could be others, but probably a safe bet

            if nationality == 'USR' :

                nationality = 'URS'

            if nationality == 'USS' :

                nationality = 'URS'

            if nationality == 'WES' :

                nationality = 'FRG'

        # Right, so this is fearsome ugly, but what are you going to do?

        # Actual fields are not well standardised in the database

        # Hashtag Actual Data or the like, eh?

        

        #print("Height: ", height_foot, "'", height_inch,'"')

        #print("Weight: ", weight_lb)

        #print("Born: ", birth_year)

    

        if len(re.split('sex\s*=\s*', page)) > 1 : 

            sex = re.split('sex\s*=\s*', page)[1]           

            sex = sex[0].upper()

        else :

            sex = 'X'

        if sex not in ['F', 'M'] : 

            sex = 'X'

            

        #In practice, there are basically no men with this field filled out

        #It looks like most women are

        #Still, if I want to use it, I'll have to figure out another way to pull it from the page

        #A quick perusal of the pages of a few guys (Lemieux, Orr), I don't see a clear candidate

        

        if len(re.split('shoots\s*=\s*', page)) > 1 :

            shoots = re.split('shoots\s*=\s*', page)[1]

            shoots = shoots[0].upper()

        else :

            shoots = 'X'

        if shoots not in ['A', 'L', 'R'] :

            shoots = 'X'



    

        try : 

            print_name = re.split('\(', name)[0] #Remove any disambiggy stuff from names

                                                 # Because a lot are titles of the form John Smith (born 1982)

                                                 # to also allow for John Smith (born 1983), but things like

                                                 # Gordon MacKenzie (the hockey guy) are also used

        except :

            value_holder += 1

            print_name = 'John Doe'

        

        # print(print_name, height_foot, height_inch, weight_lb, birth_year, nationality)

        # Originally I was running one, then a couple dozen pages, and printing everything to check by hand it was working

        # Now it's commented out, but left here for posterity

    

        if ((name[:3] != 'Ice') & (name[:8] != 'Category')) :

            protoframe.append([print_name, 12*height_foot+height_inch, weight_lb, birth_year, nationality, page_num, sex, shoots])

    

        time.sleep(0.1)

    #print(data['query']['pages']['revisions']['*'])



    #print(protoframe)



    column_names = ['Name', 'Height', 'Weight', 'Birth_Year', 'Nationality', 'ID', 'Sex', 'Shoots']



    hockey_frame = pandas.DataFrame(protoframe, columns = column_names)



    #print(hockey_frame.head())



    #hockey_frame = hockey_frame.drop(hockey_frame[hockey_frame.Weight < 1].index)

    #hockey_frame = hockey_frame.drop(hockey_frame[hockey_frame.Height < 1].index)

    #hockey_frame.hist()

 

    hockey_frame.to_pickle(str(YEAR) + "_players.dat")
all_players = pandas.DataFrame()



for YEAR in [1924, 1928, 1932, 1936, 1948, 

             1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018] :

    load_frame = pandas.read_pickle(str(YEAR) + "_players.dat")

    all_players = pandas.concat([all_players, load_frame]).drop_duplicates().reset_index(drop=True)



    





    



pairs = seaborn.pairplot(all_players, hue='Sex') #A good zeroeth order look at the data

pairs.axes[0,0].set_xlim((60,85))

pairs.axes[0,0].set_ylim((60,85))

pairs.axes[0,1].set_xlim((90,285))

pairs.axes[1,0].set_ylim((90,285))

pairs.axes[0,2].set_xlim((1880,2010))

pairs.axes[2,0].set_ylim((1880,2010))

#Note that since I've set cuts to the "plausible" values of the data, I'll need to look at outliers afterwards





print(all_players.loc[(all_players['Sex'] == 'F') & (all_players['Weight'] > 200)])
print(all_players.loc[(all_players['Birth_Year'] < 1880)]) #Seems like they're all old timers who don't have birth years

                                                           #Or the list pages, but I've purged them now

print(all_players.loc[(all_players['Birth_Year'] > 2010)]) 
print(all_players.loc[(all_players['Height'] > 85)])

print(all_players.loc[(all_players['Height'] < 60) & (all_players['Height'] > 0)])

print(all_players.loc[(all_players['Weight'] > 285)])

print(all_players.loc[(all_players['Weight'] < 90) & (all_players['Weight'] > 0)])
figure, axes = pyplot.subplots(2,3, figsize=(20,14))



axes[0,0].set_xlim(60,85)

axes[0,0].set_ylim(90,285)



axes[0,1].set_xlim(60,85)

axes[0,1].set_ylim(90,285)



axes[0,2].set_xlim(60,85)



axes[1,0].set_xlim(1880,2010)

axes[1,0].set_ylim(60,85)



axes[1,1].set_xlim(1880,2010)

axes[1,1].set_ylim(90,285)



axes[1,2].set_xlim(1880,2010)

axes[1,2].set_ylim(60,85)





seaborn.scatterplot(data = all_players, x = 'Height', y = 'Weight', hue='Sex',ax=axes[0,0])







seaborn.kdeplot(all_players[(all_players['Sex'] == 'F') 

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Height, 

                all_players[(all_players['Sex'] == 'F')

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Weight, ax=axes[0,1])

seaborn.kdeplot(all_players[(all_players['Sex'] == 'X') 

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Height, 

                all_players[(all_players['Sex'] == 'X')

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Weight, ax=axes[0,1])



#The first time I did this, I forgot to exclude the zereoes

#It did not look good

#Don't do it, eh?



seaborn.distplot(all_players[all_players['Height'] > 0].Height, bins=[i for i in range(60,81)], ax=axes[0,2])



seaborn.kdeplot(all_players[(all_players['Sex'] == 'F')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Height'] > 0)].Birth_Year, 

                all_players[(all_players['Sex'] == 'F')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Height'] > 0)].Height, ax=axes[1,0])

seaborn.kdeplot(all_players[(all_players['Sex'] == 'X')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Height'] > 0)].Birth_Year, 

                all_players[(all_players['Sex'] == 'X')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Height'] > 0)].Height, ax=axes[1,0])



seaborn.kdeplot(all_players[(all_players['Sex'] == 'F')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Weight'] > 0)].Birth_Year, 

                all_players[(all_players['Sex'] == 'F')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Weight'] > 0)].Weight, ax=axes[1,1])



seaborn.kdeplot(all_players[(all_players['Sex'] == 'X')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Weight'] > 0)].Birth_Year, 

                all_players[(all_players['Sex'] == 'X')

                           & (all_players['Birth_Year'] > 0)

                           & (all_players['Weight'] > 0)].Weight, ax=axes[1,1])





seaborn.scatterplot(data = all_players, x = 'Birth_Year', y = 'Height', hue='Sex',ax=axes[1,2])
figure, axes = pyplot.subplots()

axes.set_xlim(60,80)





North_Americans = all_players[all_players['Nationality'].isin(["CAN", "USA"])]





seaborn.distplot(North_Americans['Height'], bins=[i for i in range(60,81)], ax=axes)



#print(sorted(all_players['Nationality'].unique()))

#I just wanted them listed so I could sort by continent



Europeans = all_players[all_players['Nationality'].isin(['AUS', 'BLR', 'BUL', 'CZE', 'DDR', 'DEU', 'FIN', 'FRA', 'FRG', 'GBR', 'GER', 

                               'HUN', 'ITA', 'LAT', 'NED', 'NOR', 

                               'POL', 'ROM', 'SLO', 'SOV', 'SUI', 'SVK', 

                               'SWE', 'TCH', 'RUS', 'UKR', 'UNI', 'URS', 

                               'USR', 'WES', 'YUG'])]



seaborn.distplot(Europeans['Height'], bins=[i for i in range(60,81)], ax=axes)



Asians = all_players[all_players['Nationality'].isin(['CHN', 'JPN', 'KAZ', 'KOR', 'PRK'])]



seaborn.distplot(Asians['Height'], bins=[i for i in range(60,81)], ax=axes)
figure, axes = pyplot.subplots(figsize=(12,12))



counts, bins, bars = pyplot.hist(all_players[(all_players['Sex'] == 'X')

                                             & (all_players['Height'] > 0)].Height, bins = [i for i in range(60,85)])



bins = [i + 0.5 for i in bins[:-1]]

counts = [i for i in counts]

sigma_counts = [math.sqrt(i)+1e-10 for i in counts]

#print(type(sigma_counts), type(counts), type(bins))

#print(len(sigma_counts), len(counts), len(bins))



N_players = len(all_players[(all_players['Sex'] == 'X')

                                             & (all_players['Height'] > 0)].Height)



(mu, sigma) = norm.fit(all_players[(all_players['Sex'] == 'X')

                                             & (all_players['Height'] > 0)].Height)

g_bins = [i - 0.5 for i in bins]



print(mu, sigma)

gauss_fit = norm.pdf(g_bins, mu, sigma)

pyplot.plot(bins, N_players*gauss_fit, 'b--', linewidth=2)



pyplot.bar(bins, counts, width=1.0, color='r', yerr=sigma_counts)

pyplot.show()



print("Six foot tall is ", 72.0*2.54, " cm to ", 72.999999*2.54, " cm")



pandas.options.display.max_rows = 4000

print(type(Europeans['Height'].value_counts(ascending=True)))



#print((Europeans['Height'].value_counts(ascending=True)).sort_values(axis=0,ascending=False))

print((Europeans['Height'].value_counts(ascending=True)).sort_index())

#Whether you find the former or latter format preferable is probably a matter of taste and custom



ful

figure, axes = pyplot.subplots(1,3)

Canadians = all_players[all_players['Nationality'].isin(["CAN"])]

Canadians.Shoots.value_counts().plot.bar(ax=axes[0])



Americans = all_players[all_players['Nationality'].isin(["USA"])]

Americans.Shoots.value_counts().plot.bar(ax=axes[1])



all_players.Shoots.value_counts().plot.bar(ax=axes[2])

print(all_players.Sex.value_counts())



figure, axes = pyplot.subplots(1, 2, figsize=(20,8))



axes[0].set_xlim(60,85)

axes[1].set_xlim(90,285)



seaborn.distplot(all_players[(all_players['Sex'] == 'F') & (all_players['Height'] > 0)].Height, ax=axes[0], bins = [i for i in range(60,80)])

seaborn.distplot(all_players[(all_players['Sex'] == 'X') & (all_players['Height'] > 0)].Height, ax=axes[0], bins = [i for i in range(60,80)])



seaborn.distplot(all_players[(all_players['Sex'] == 'F') & (all_players['Weight'] > 0)].Weight, ax=axes[1], bins = [5*i for i in range(20,60)])

seaborn.distplot(all_players[(all_players['Sex'] == 'X') & (all_players['Weight'] > 0)].Weight, ax=axes[1], bins = [5*i for i in range(20,60)])



#Possibly it's not quite perfect, but those look pretty normal and non-overlapping
machine_model = GaussianMixture(n_components=2)



heights = all_players[(all_players['Height'] > 0)

                    & (all_players['Weight'] > 0)].Height.tolist()

weights = all_players[(all_players['Height'] > 0)

                    & (all_players['Weight'] > 0)].Weight.tolist()







hwarray = numpy.dstack((numpy.array(heights), numpy.array(weights)))



n_elements = len(numpy.array(heights))



hwarray = hwarray.reshape(n_elements,2)



machine_model.fit(hwarray)

# assign a cluster to each example

pops = machine_model.predict(hwarray)



clusters = numpy.unique(pops)



figs, axes = pyplot.subplots(1,3, figsize=(18,8))



axes[0].set_xlabel('Height (in)')

axes[0].set_ylabel('Weight (lb)')



axes[1].set_xlabel('Height (in)')

axes[1].set_ylabel('Weight (lb)')



for cluster in clusters:

    underlying = numpy.where(pops == cluster)

    axes[0].scatter(hwarray[underlying, 0], hwarray[underlying, 1])





plotFrame = pandas.DataFrame({'H': hwarray[:, 0], 'W': hwarray[:, 1], 'Population' : pops})



seaborn.kdeplot(data = plotFrame[plotFrame['Population'] == 1], ax=axes[1])

seaborn.kdeplot(data = plotFrame[plotFrame['Population'] == 0], ax=axes[1])



labels = []



seaborn.kdeplot(all_players[(all_players['Sex'] == 'F') 

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Height, 

                all_players[(all_players['Sex'] == 'F')

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Weight, 

                ax=axes[2])

label = mpatches.Patch(

        color='Orange',

        label='Men')

labels.append(label)

seaborn.kdeplot(all_players[(all_players['Sex'] == 'X') 

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Height, 

                all_players[(all_players['Sex'] == 'X')

                            & (all_players['Height'] > 0) 

                            & (all_players['Weight'] > 0)].Weight, 

                ax=axes[2])

label = mpatches.Patch(

        color='Blue',

        label='Women')

labels.append(label)

axes[2].legend(handles=labels, loc='upper left')



pyplot.show()


