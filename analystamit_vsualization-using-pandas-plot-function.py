# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
# Reading input file

magazine = pd.read_csv("../input/archive.csv")

print(magazine.columns)
np.shape(magazine)
# Checking if there is anybody from India received this award

magazine[magazine.Country == 'India']
# getting the number of people selected from each country

dbycount =  magazine.Country.groupby(magazine.Country).count()

print(dbycount)
# Maximum record went to US :)

dbycount.plot(kind = 'box', color = 'red')
# Same analysis for the honor in award

magazine.Honor.groupby(magazine.Honor).count().plot(kind = 'pie', figsize = (10,9), autopct = '%1.1f%%')
# Mostly they call it as 'Man of the Year'

#Now checking the title for the award

magazine.Title.groupby(magazine.Title).count().plot(kind = 'barh', color = 'red', figsize = (5, 15))

#So, most of the records went to President of United States bucket. 
# And obviously now I am expecting the mojor category will be politics

magazine.Category.groupby(magazine.Category).count().plot(kind = 'bar', color = 'red')
magazine.columns
#The maximim number an individual win this award

no_aw_p = magazine.Name.groupby(magazine.Name).count()

np.max(no_aw_p)
no_aw_p[no_aw_p == 3].index
magazine[magazine.Name == 'Franklin D. Roosevelt']
#US presidents who were awared by this award

US_President = magazine[magazine['Title'] == 'President of the United States'].Name.unique()
# US presidents that were awarded for election victory in president election

US_President_awarded = magazine[magazine.Context == 'Presidential Election'].Name.unique()
US_President_Not_Awarded = []

for president in US_President:

    if president not in US_President_awarded:

        US_President_Not_Awarded.append(president)

        
#Below presidents didn't receive the honor after winning the election

US_President_Not_Awarded
magazine['Category_new'] = magazine['Category'].map({'Economics': 0, 'Diplomacy': 1, 'Revolution': 2, 'Politics': 3, 'War': 4, 'Space': 5, 'Science': 6, 'Relegion': 7, 'Technoloy': 8, 'Environment': 9, 'Media': 10, 'Philanthropy': 11})
def grp_year(x):

    return (x-1927)//10

magazine['year_range'] = grp_year(magazine['Year'])
a = magazine['Category_new'].groupby(magazine['year_range']).apply(list)