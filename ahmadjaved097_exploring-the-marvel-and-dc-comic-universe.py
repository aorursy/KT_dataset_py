import warnings                       # to hide warnings if any

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
marvel = pd.read_csv('../input/marvel-wikia-data.csv')

dc = pd.read_csv('../input/dc-wikia-data.csv')
marvel.head(3)
dc.head(3)
#Function to remove unnecessary columns



def remove_col(df, *col):                    # Here df = name of dataframe, *col = names of unnecessary columns

    df.drop([*col], axis = 1, inplace = True)
remove_col(dc,'page_id','urlslug','FIRST APPEARANCE')        # removing columns from dc's dataset
remove_col(marvel,'page_id','urlslug','FIRST APPEARANCE')    # removing column's from marvel's dataset
marvel['name'] = marvel['name'].apply(lambda x: x.split('(')[0])

dc['name'] = dc['name'].apply(lambda x: x.split('(')[0])
marvel.rename(columns = {'Year':'YEAR'}, inplace = True)
marvel.head(3)
dc.head(3)
row, col = dc.shape[0], dc.shape[1]

print("-------- DC'S DATASET --------")

print('Number of rows: ',row)

print('Number of columns: ',col)



print()



row,col = marvel.shape[0], marvel.shape[1]

print("-------- MARVEL'S DATASET --------")

print('Number of rows: ',row)

print('Number of columns: ',col)

print("-------- DC'S DATASET --------")

print('')

dc.info()
print("-------- MARVEL'S DATASET --------")

print('')

marvel.info()
count = 0

for i in dc.columns:

    count = count + 1

    print(count,'. ',i)
# function to print unique values in a given column.

# Here col = column name



def print_unique(col):

    print('Unique values in the ', col, 'Column are: ')

    print()

    count = 0

    for i in dc[col].unique():

        count = count + 1

        print(count,'. ',i)

print_unique('ID')
print_unique('ALIGN')
print_unique('EYE')
print_unique('HAIR')
print_unique('SEX')
print_unique('GSM')
print_unique('ALIVE')
# function to draw count plots for both marvel and dc comics side by side

#Here col = Name of column

#color = color palette's name

#xtic = x-axis label's rotation



sns.set_style('darkgrid')



def plot_countplot(col,hue = None, color = 'magma',xtic = 0,ylim = 13000):

    plt.figure(figsize=(12,6))

    

    plt.subplot(1,2,1)

    sns.countplot(x = col, data = dc, hue = hue, palette = color)

    plt.xticks(rotation = xtic)

    plt.ylim(0,ylim)

    plt.title('DC Comics')

    

    plt.subplot(1,2,2)

    sns.countplot(x = col, data = marvel, hue = hue, palette = color)

    plt.xticks(rotation = xtic)

    plt.ylim(0,ylim)

    plt.title('Marvel Comics')

    

    plt.tight_layout()

    
# function to print the count of different values present in each column

# df = name of dataset

# col = column name



def stats(df, col):

    for i in df[col].unique():

        print(i,': ',len(df[df[col] == i]))
plot_countplot('ID',xtic=30)
stats(dc,'ID')
stats(marvel,'ID')
plot_countplot('ALIGN',xtic=45)
stats(dc,'ALIGN')
stats(marvel,'ALIGN')
plot_countplot('SEX',xtic=40)
stats(dc,'SEX')
stats(marvel,'SEX')
plot_countplot('ALIVE')
stats(dc,'ALIVE')
stats(marvel,'ALIVE')
plot_countplot('EYE',ylim = 2000,xtic=60)
stats(dc,'EYE')
stats(marvel,'EYE')
plot_countplot('ALIVE',hue='ALIGN',ylim=5000)
alive_align = dc.groupby(['ALIVE','ALIGN']).aggregate('count')

alive_align['name']
alive_align = marvel.groupby(['ALIVE','ALIGN']).aggregate('count')

alive_align['name']
plt.figure(figsize=(20,6))

sns.lineplot(x = 'YEAR',y = 'APPEARANCES',hue = 'SEX',data= dc,markers= True,dashes=False,lw=2)

plt.ylim(0,1000)

plt.show()
plt.figure(figsize=(20,6))

sns.lineplot(x = 'YEAR',y = 'APPEARANCES',hue = 'SEX',data= marvel,markers= True,dashes=False,lw=2)

plt.ylim(0,600)

plt.show()