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
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from scipy import stats

import pylab as pl

import pandas as pd

import seaborn as sns; sns.set()

food_data=pd.read_csv('../input/en.openfoodfacts.org.products.tsv', sep="\t")
food_clean=food_data[food_data.product_name.notnull()]

food_clean=food_clean[food_clean.countries_en.notnull()]

alcohol=food_clean[food_clean.alcohol_100g.notnull()]
font = {'fontname':'Arial', 'size':'14'}

title_font = {'fontname':'Arial', 'weight' : 'bold','size':'16'}

alcohol=alcohol[alcohol.alcohol_100g>0]

plt.hist(alcohol.alcohol_100g, bins=[0,2,5,10, 15,20, 30, 40,50, 60,100])

plt.xticks([0,2,5,10, 15,20, 30, 40,50, 60,100],**font)

plt.yticks(**font)

plt.xlabel('Alcohol per 100g')

plt.ylabel("Number of products")

plt.title("Histogram of alcoholic products by alcohol_100g",**title_font)

plt.show()
print (alcohol.countries_en.value_counts().head(10))
countries=['France', 'United Kingdom', "Germany", "Belgium", "United States"]

alcohol=alcohol[alcohol.countries_en.isin(countries)]
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

plt.hist([alcohol.alcohol_100g[alcohol.countries_en=="France"],

          alcohol.alcohol_100g[alcohol.countries_en=="United Kingdom"],

          alcohol.alcohol_100g[alcohol.countries_en=="Germany"],

          alcohol.alcohol_100g[alcohol.countries_en=="Belgium"],

          alcohol.alcohol_100g[alcohol.countries_en=="United States"]],

         normed=True, bins=[0,2,5,10, 15,20]

        )

plt.xticks([0,2,5,10, 15,20],**font)

plt.yticks([])

plt.xlabel('Alcohol per 100g')

plt.ylabel("Number of products")

plt.title("Histogram of 'light' alcoholic products by alcohol_100g",**title_font)

plt.subplot(1,2,2)

plt.hist([alcohol.alcohol_100g[alcohol.countries_en=="France"],

          alcohol.alcohol_100g[alcohol.countries_en=="United Kingdom"],

          alcohol.alcohol_100g[alcohol.countries_en=="Germany"],

          alcohol.alcohol_100g[alcohol.countries_en=="Belgium"],

          alcohol.alcohol_100g[alcohol.countries_en=="United States"]],

         normed=True, bins=[20,30,40, 50,60]

        )

plt.xticks([20,30,40,50,60],**font)



plt.yticks([])

plt.xlabel('Alcohol per 100g')

plt.ylabel("Number of products")

plt.legend(countries, loc="upper left",bbox_to_anchor=(1,1),prop={'size':15})

plt.title("Histogram of 'hard' alcoholic products by alcohol_100g",**title_font)

plt.subplots_adjust(wspace=0.5)



plt.show()
from collections import Counter

import re

from nltk.corpus import stopwords

from operator import itemgetter

##Stop words for the languages of the different countries

french_stop=stopwords.words('french')

eng_stop=stopwords.words('english')

ger_stop=stopwords.words('german')

# most belgian products are in french no need for dutch stopwords 

stopwords=french_stop+eng_stop+ger_stop+["year","ctes", "cl", "les","sans"] #I added some stop words afterwards



def count_words(country):

    """ This takes every product names for one country cleans them and count the number of occurences of each word"""

    names=alcohol.product_name[alcohol.countries_en==country]

    giant_string=[]

    

    for string in names:

        giant_string.append(string)

        

    giant_string=' '.join(giant_string)

    giant_string=giant_string.lower()



    giant_string=re.sub(r'\([^)]*\)', '', giant_string)#remove whats in between brackets

    regex = re.compile('[^a-zA-Zéèô ]') 

    giant_string=regex.sub('', giant_string)# remove what's not letters 

    count=Counter(giant_string.split()) 

    for key in stopwords:

        if key in count:

            del count[key]#remove stopwords 

    count= sorted(count.items(), key=itemgetter(1), reverse=True) #return sorted list of tuples (word, occurence)

    return(count)
def high_freq_words( country, number=10):

    """plot the histogram of the most used words"""

    count=count_words(country)

    labels, values = zip(*count)

    plt.figure(figsize=(20,10))

    plt.bar(range(len(values[0:number])), values[0:number], align="center")

    plt.xticks(range(len(values[0:number])), labels[0:number], rotation=70,**font)

    plt.yticks(**font)

    plt.title("{} most frequent word in {} alcohols".format(number, country),**title_font)

    plt.show()
high_freq_words("France",30)

high_freq_words("United Kingdom",15)

high_freq_words("United States",9)

high_freq_words("Germany",25)

high_freq_words("Belgium",4)
alcohol[(alcohol.countries_en=="France")&(alcohol.alcohol_100g>10)&(alcohol.alcohol_100g<14)].product_name.head(10) #Trying to find out how wines are named 
alcohol[(alcohol.countries_en=="France")&(alcohol.alcohol_100g<8)].product_name.head(10)
def extract_most_common(country, names):

    most_comm=pd.DataFrame()

    for name in names:

        Lol=alcohol[(alcohol.product_name.str.lower().str.contains(name))]

        most_comm= most_comm.append(Lol[Lol.countries_en==country])

    return most_comm

belgium=extract_most_common("Belgium", ["bière","blonde"])

france=extract_most_common("France", ["bière", "blonde", "cidre", "vin"])

UK=extract_most_common("United Kingdom", ["ale", "old"])

US=extract_most_common("United States", ["sauvignon", "ale", "ipa"])

germany=extract_most_common("Germany", ["radler", "pilsener"])



plt.figure(figsize=(20,10))

plt.hist([france.alcohol_100g,UK.alcohol_100g, germany.alcohol_100g, belgium.alcohol_100g,

           US.alcohol_100g], 

        label=["FR","UK","GER","BE","US"], normed=True, bins=[0,2,5,8,12,15,20,40])

plt.legend(["FR","UK","GER","BE","US"],prop={'size':15})

plt.xticks([0,2,5,8,12,15,20,40], **font)

plt.yticks([])

plt.ylabel("Shares of products")

plt.xlabel("Alcohol per 100g")

plt.title("Alcohol contained in most found products ",**title_font)

plt.show()
plt.figure(figsize=(20,10))

plt.hist([france.alcohol_100g, UK.alcohol_100g,germany.alcohol_100g, belgium.alcohol_100g,

           US.alcohol_100g], 

        label=["FR","UK","GER","BE","US"], normed=True, bins=[0,2,5,8,12,15,20])

plt.legend(["FR","UK","GER","BE","US"],prop={'size':15})

plt.xticks([0,2,5,8,12,15,20], **font)

plt.yticks([])

plt.ylabel("Shares of products")

plt.xlabel("Alcohol per 100g")

plt.title("Alcohol contained in most found light beverages",**title_font)

plt.show()