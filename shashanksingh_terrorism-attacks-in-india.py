# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





years = range(1970,2017)

#source https://en.wikipedia.org/wiki/List_of_Prime_Ministers_of_India

#"1993":"P. V. Narasimha Rao","1970": "Indira Gandhi", "1971": "Indira Gandhi", 

# "1973": "Indira Gandhi", "1974": "Indira Gandhi", "1978":"Morarji Desai",

prime_ministers_each_year = {

         "1972": "Indira Gandhi",

         "1975": "Indira Gandhi", 

         "1976": "Indira Gandhi", "1977": "Indira Gandhi & Morarji Desai", 

          "1979":"Morarji Desai & Charan Singh", 

         "1980":"Charan Singh & Indira Gandhi", "1981":"Indira Gandhi", 

         "1982":"Indira Gandhi", "1983":"Indira Gandhi",

         "1984":"Indira Gandhi & Rajiv Gandhi", "1985":"Rajiv Gandhi", 

         "1986":"Rajiv Gandhi", "1987":"Rajiv Gandhi", 

         "1988":"Rajiv Gandhi", "1989":"Rajiv Gandhi & V. P. Singh", 

         "1990":"V. P. Singh & Chandra Shekhar", "1991":"Chandra Shekhar & P. V. Narasimha Rao", 

         "1992":"P. V. Narasimha Rao",  

         "1994":"P. V. Narasimha Rao", "1995":"P. V. Narasimha Rao", 

         "1996":"P. V. Narasimha Rao & Atal Bihari Vajpayee & H. D. Deve Gowda", 

         "1997":"H. D. Deve Gowda & Inder Kumar Gujral", "1998":"Inder Kumar Gujral & Atal Bihari Vajpayee", 

         "1999":"Atal Bihari Vajpayee", "2000":"Atal Bihari Vajpayee", 

         "2001":"Atal Bihari Vajpayee", "2002":"Atal Bihari Vajpayee", 

         "2003":"Atal Bihari Vajpayee", "2004":"Atal Bihari Vajpayee & Manmohan Singh", 

         "2005":"Manmohan Singh", "2006":"Manmohan Singh", "2007":"Manmohan Singh", 

         "2008":"Manmohan Singh", "2009":"Manmohan Singh", "2010":"Manmohan Singh", 

         "2011":"Manmohan Singh", "2012":"Manmohan Singh", "2013":"Manmohan Singh", 

         "2014":"Manmohan Singh", "2015":"Narendra Modi"}





#print("[Test] 2014 India's Prime - minister is" ,years["2014"])



terror_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1',

                          usecols=[1, 8])

terror_data_india_by_year = terror_data[(terror_data.country_txt == 'India')]

terror_data_india_by_year_grouped = terror_data_india_by_year.groupby(terror_data_india_by_year.iyear).count()

#terror_data_india_by_year_grouped_header = terror_data_india_by_year_grouped.keys()



labels = tuple(list(prime_ministers_each_year.values()))

sizes = terror_data_india_by_year_grouped

explode=range(0,40)



print(len(labels))

print(len(sizes))



fig1, ax1 = plt.subplots()

ax1.pie(sizes,  explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.





#pie(terror_data_india_by_year_grouped)



#print(terror_data_india_by_year_grouped)



plt.show()