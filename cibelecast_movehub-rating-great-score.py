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
# Import libraries necessary for this project

import numpy as np

import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

from pandas import DataFrame

import matplotlib.pyplot as plt

import pylab as plt

import scipy as scipy

from scipy.stats.stats import pearsonr

# Pretty display for notebooks

%matplotlib inline



# Load the dataset

full_data = pd.read_csv('../input/cities.csv') 

full_data2= pd.read_csv('../input/movehubqualityoflife.csv')

full_data3= pd.read_csv('../input/movehubcostofliving.csv')



# Print the first few entries of the MoveHub data

display(full_data.head())

display(full_data2.head())

display(full_data3.head())
#Verify if there is missing values inside every column of the dataset: "cities.csv"

full_data.info()
#See countries that have NAN missing

full_data[full_data['Country'].isnull()]
#Insertion of the countries that are missing

full_data.iloc[654,1]='Ukraine'

full_data.iloc[724,1]='Russia'

full_data.iloc[1529,1]='Kosovo'



#See that now, there isn't any more NAN:

full_data.info()
#Verify if there is missing values inside every column of the dataset:'movehubcostofliving.csv'

full_data2.info()
#Verify if there is missing values inside every column of the dataset:'movehubqualityoflife.csv'

full_data3.info()
#Merge Datasets

movehubcity= pd.merge(full_data2, full_data3,how='outer')

#Sort Dataset by 'City'

movehubcity=movehubcity.sort_values(by='City')

#Modification of the values of the index

movehubcity.reset_index(drop=True)
#Insert column country to dataset.

movehubcity2= pd.merge(movehubcity, full_data,how='left',on='City')

movehubcity2
#Missing Values: 30 cities don't have countries.

movehubcity2.info()
#All the cities that don't have country

movehubcity2[movehubcity2['Country'].isnull()]
#Update wrong names of the cities

movehubcity2.iloc[227,0]='Zürich'

movehubcity2.iloc[224,0]='Washington, D.C.'

movehubcity2.iloc[201,0]='Tampa, Florida'

movehubcity2.iloc[188,0]='São Paulo'

movehubcity2.iloc[185,0]='San Francisco, California'

movehubcity2.iloc[184,0]='San Diego, California'

movehubcity2.iloc[193,13]='Malta'

movehubcity2.iloc[10,13]='United States' #dado não encontrado no wikipedia

movehubcity2.iloc[51,13]='Philippines'#dado não encontrado no wikipedia

movehubcity2.iloc[61,13]='Argentina' #dado não encontrado no wikipedia

movehubcity2.iloc[66,0]='Davao City'

movehubcity2.iloc[74,0]='Düsseldorf'

movehubcity2.iloc[79,0]='Frankfurt am Main'

movehubcity2.iloc[81,13]='Ireland' #dado não encontrado no wikipedia

movehubcity2.iloc[100,0]='İstanbul'

movehubcity2.iloc[101,0]='İzmir'

movehubcity2.iloc[122,13]='Poland' #dado não encontrado no wikipedia

movehubcity2.iloc[129,0]='Málaga'

movehubcity2.iloc[130,0]='Malmö'

movehubcity2.iloc[134,13]='Spain'

movehubcity2.iloc[136,0]='Medellín'

movehubcity2.iloc[139,0]='Miami, Florida'

movehubcity2.iloc[141,0]='Minneapolis, Minnesota'

movehubcity2.iloc[164,13]='Thailand'

movehubcity2.iloc[166,0]='Philadelphia, Pennsylvania'

movehubcity2.iloc[167,0]='Phoenix, Arizona'

movehubcity2.iloc[168,0]='Portland, Oregon'

movehubcity2.iloc[176,0]='Rio de Janeiro'

movehubcity2.iloc[178,13]='United States'

movehubcity2.iloc[183,0]='San Antonio, Texas'
movehubcity2[movehubcity2['Country'].isnull()]
movehubcity2.info()
#Do merge again to recover the names of the countries with the names of the cities already updated.

data= pd.merge(movehubcity2, full_data,how='inner',on='City')

data
#Delete column 'Country_x' and alter the name of 'Country_y'

data=data.drop('Country_x',axis=1)

data=data.rename(columns={'Country_y': 'Country'})

#See there isn't any NAN countries

data[data['Country'].isnull()]
#Here we see that are four register for the City London, and this happen for others cities, too.

#We don't know the correct country for each city, so we'll maintain one register for each different 

#country

###### Duplicated register - Remotion

data[(data['City'] == 'London')]

#This code shows all the cities that are duplicated.

data.set_index('City').index.get_duplicates()
#This code shows the amount of duplicated registers:

names=data.City.value_counts()

names[names > 1]
###### Example city London: Remove the duplicated registers:

df2=pd.DataFrame(data[(data['City'] == 'London')])

df2

df2.drop_duplicates(subset=['City','Country'])
#Like in the example above, we're gonna to delete all the duplicated registers for all the cities:

data=data.drop_duplicates(subset=['City','Country'])
#You can run this code to see there isn't any more duplicated city:

#This code will remove 28 duplicated cities.

names=data.City.value_counts()

names[names > 1]
data.info() 
#Replace '' by '_'

data.columns = data.columns.str.replace(' ','_')
data
#See the other notebook "Data Exploration"

#Drop outliers:

data=data.drop([79]) #Purchase_Power

data=data.drop([105]) #Avg_Rent

data=data.drop([190]) #Cinema

data=data.drop([213]) #Wine

data=data.drop(246) #Avg_Disposable_Income

data=data.drop(128) #Avg_Disposable_Income
#Importação 'train_test_split'

from sklearn.model_selection import train_test_split



X = data.drop(['Movehub_Rating','Country','City'], axis = 1) 

y=data['Movehub_Rating']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



print ("Dados de Treinamento e dados de teste divididos.")
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit (X_train,y_train)
y_saida=reg.predict(X_test)
#Metrics for avaliation

#Explained variance regression score function

#Best possible score is 1.0, lower values are worse.



from sklearn.metrics import explained_variance_score

explained_variance_score=explained_variance_score(y_test, y_saida)  

from sklearn.metrics import mean_squared_error

mean_squared_error=mean_squared_error(y_test, y_saida)
from sklearn.metrics import mean_absolute_error

erro_medio_absoluto=mean_absolute_error(y_test, y_saida)
print ("Determination coefficient r²: {:.3f}".format(reg.score(X_train,y_train)))

print ("Determination coefficient r² of the testing set: {:.3f} ".format(reg.score(X_test,y_test)))

print("mean_squared_error: {:.2f} ".format(mean_squared_error))

print("mean_absolute_error: {:.2f} ".format(erro_medio_absoluto))

print("Explained_variance_score: {:.3f} ".format(explained_variance_score))

print ('Linear Coefficient {:.2f}'.format(reg.intercept_))

print ('Angular Coefficient{:}'.format(reg.coef_))
#Results:

pd.DataFrame(list(zip(y_test[5:10],y_saida[5:10])),columns=['True Values','Estimated Values'])
plt.scatter(y_saida,y_test)

plt.scatter(y_saida[5:10],y_test[5:10],color='b')

plt.scatter(y_saida[5:10],y_test[5:10],color='r')

plt.xlabel('Estimated Values')

plt.ylabel('True Values')

plt.title('True Values X Estimated Values')



plt.show()