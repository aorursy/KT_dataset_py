# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stat



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



tab = pd.read_table("../input/en.openfoodfacts.org.products.tsv", nrows = 1000)

#get the places with values only

tab = tab[pd.isnull(tab["countries"]) == False]



#find which foods that have higher than 10% sugar

high_sugar = tab["sugars_100g"] > 10



#find the top 'western countries'

print(tab["countries"].value_counts() >1)

#list them

top = pd.Series(['US', 'France', 'en:FR', 'United Kingdom', 'en:GB', 'United States'  \

             "France,Royaume-Uni", "France,United Kingdom", "Deutschland", "Canada",  \

             "France,UK", "UK,France", "Germany", "en:US", "en:AU"])



#indicate in list where true for top country

countries = []

for x in tab["countries"]:

    isTop = False

    for y in top:

        if y == x:

            isTop = True

            break

    if isTop:

        countries.append(True)

    else:

        countries.append(False)



#convert back to pandas

countries = pd.Series(countries)



##one-way test

print(stat.chisquare(high_sugar.value_counts()))

print(stat.chisquare(countries.value_counts()))



#two-way

# Hypothesis - First world countries are less likely to have food products containing more sugar?

contingencyTable = pd.crosstab(high_sugar, top_countries)

stat.chi2_contingency(contingencyTable)





# Any results you write to the current directory are saved as output.