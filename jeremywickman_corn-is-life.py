## Let's begin by importing the tools we will need

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline



#Why are there two files here? Is there a difference in the corn txt files contained in this...

#...KERNEL?

corn_data = pd.read_csv('../input/corn2013-2017.txt', header = None)

corn_data2 = pd.read_csv('../input/corn2015-2017.txt', header = None)



print(corn_data.head())

print(corn_data2.head())

print(corn_data[104:109])
#Rename columns to be worthy of the data they contain

corn_data.columns = ['Corn_Date', 'Corn_Price'] 



#Fix the date to the correct type

corn_data['Corn_Date'] = corn_data['Corn_Date'].astype('datetime64[ns]')
#Let's graph this quick and dirty

ax = corn_data.plot(x='Corn_Date', y='Corn_Price',kind='line',figsize=(16,10), color='g', legend=False)



#Adding important commentary

ax.annotate('corn can make you rich', 

            xy=('2013-02-04', 7.8), 

            xytext=('2013-02-04', 6),

            arrowprops=dict(facecolor='black', 

                            shrink=0.05)

            )

ax.annotate('the great corn hole', 

            xy=('2014-09-04', 4.3), 

            xytext=('2013-09-04', 4.3),

            arrowprops=dict(facecolor='black', 

                            shrink=0.05)

            )

ax.annotate('corn becomes fashionable again following release of The Paradigm Shift by KoRn', 

            xy=('2014-05-01', 7), 

            xytext=('2014-09-04', 8),

            arrowprops=dict(facecolor='black', 

                            shrink=0.05),

            )



ax.annotate('farmer mortality rate increases', 

            xy=('2014-07-01', 6), 

            xytext=('2015-01-01', 6),

            arrowprops=dict(facecolor='red', 

                            shrink=0.05),

            )



ax.annotate('bigger things to worry about than corn', 

            xy=('2016-11-08', 4), 

            xytext=('2016-6-08', 6),

            arrowprops=dict(facecolor='black', 

                            shrink=0.05),

            )
