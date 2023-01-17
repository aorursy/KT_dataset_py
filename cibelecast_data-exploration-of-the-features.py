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
#Insertion of the countries that are missing

full_data.iloc[654,1]='Ukraine'

full_data.iloc[724,1]='Russia'

full_data.iloc[1529,1]='Kosovo'
#Merge Datasets

movehubcity= pd.merge(full_data2, full_data3,how='outer')

#Sort Dataset by 'City'

movehubcity=movehubcity.sort_values(by='City')

#Modification of the values of the index

movehubcity.reset_index(drop=True)
#Insert column country to dataset.

movehubcity2= pd.merge(movehubcity, full_data,how='left',on='City')

#movehubcity2
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
#Do merge again to recover the names of the countries with the names of the cities already updated.

data= pd.merge(movehubcity2, full_data,how='inner',on='City')
#Delete column 'Country_x' and alter the name of 'Country_y'

data=data.drop('Country_x',axis=1)

data=data.rename(columns={'Country_y': 'Country'})

#See there isn't any NAN countries

data[data['Country'].isnull()]
#Like in the example above, we're gonna to delete all the duplicated registers for all the cities:

data=data.drop_duplicates(subset=['City','Country'])
#Replace '' by '_'

data.columns = data.columns.str.replace(' ','_')
X = data.drop(['Movehub_Rating','Country','City'], axis = 1) 

y=data['Movehub_Rating']
#Basic Information of Statistics

#### Minimum Value movehub rating.

minimum_movehubrating = np.min(y)



#### Maximum Value movehub rating

maximum_movehubrating = np.max(y)



#### Avg movehub rating

mean_movehubrating = np.mean(y)



#### Std movehub rating

std_movehubrating = np.std(y)



#### Show the calculated statistics

print ("Minimum movehub rating: ",minimum_movehubrating)

print ("Maximum movehub rating: ",maximum_movehubrating)

print ("Avg movehub rating: ",mean_movehubrating)

print ("Std de movehub rating: ",std_movehubrating)
informacoes=['Mínimo movehub rating','Máximo movehub rating:','Média movehub rating:','Desvio Padrão de movehub rating:']

valores=[minimum_movehubrating,maximum_movehubrating,mean_movehubrating,std_movehubrating]

pd.DataFrame(list(zip(informacoes,valores)),columns=['Informações básicas de estatística','Resultado'])

#Correlation between the features and the movehub rating

ccl_Pollution=list(pearsonr(X.Pollution,y))

ccl_Purchase_Power=list(pearsonr(X.Purchase_Power,y))

ccl_Gasoline=list(pearsonr(X.Gasoline,y))

ccl_Crime=list(pearsonr(X.Crime_Rating,y))

ccl_Cinema=list(pearsonr(X.Cinema,y))

ccl_Wine=list(pearsonr(X.Wine,y))

ccl_Avg_Rent=list(pearsonr(X.Avg_Rent,y))

ccl_Avg_Income=list(pearsonr(X.Avg_Disposable_Income,y))

ccl_Health=list(pearsonr(X.Health_Care,y))

ccl_Quality=list(pearsonr(X.Quality_of_Life,y))

ccl_Cappucino=list(pearsonr(X.Cappuccino,y))
ccl_Pollution1=ccl_Pollution[0]

ccl_Purchase_Power=ccl_Purchase_Power[0]

ccl_Gasoline=ccl_Gasoline[0]

ccl_Crime=ccl_Crime[0]

ccl_Cinema=ccl_Cinema[0]

ccl_Wine=ccl_Wine[0]

ccl_Avg_Rent=ccl_Avg_Rent[0]

ccl_Avg_Income=ccl_Avg_Income[0]

ccl_Health=ccl_Health[0]

ccl_Quality=ccl_Quality[0]

ccl_Cappucino=ccl_Cappucino[0]
informacoes1=X.columns[:]

valores2=[ccl_Cappucino,ccl_Cinema,ccl_Wine,ccl_Gasoline,ccl_Avg_Rent,ccl_Avg_Income,ccl_Purchase_Power,ccl_Health,ccl_Pollution1,ccl_Quality,ccl_Crime]

conjunto=pd.DataFrame(list(zip(informacoes1,valores2)),columns=['Features','Resultado do Coeficiente de Correção Linear'])

conjunto
#Before drop the outliers:

quantidade = [1,2,3,4,5,6,7,8,9,10,11]

colors = ['g','g','g','g','y','y','y','y','y','r','r']

numbers = [0.81049364263681856,0.751426,0.713844,0.542362,0.429955,0.384803,0.310625,0.142297,0.136347,-0.167520,-0.258001]



LABELS = ["Purchase_Power","Avg_Disposable_Income","Quality_of_Life","Avg_Rent","Cappuccino","Health_Care","Cinema","Gasoline","Wine", "Pollution","Crime_Rating"]



plt.bar(quantidade, numbers,color=colors)

plt.xticks(quantidade, LABELS,rotation='vertical')#rotation='vertical'

plt.ticklabel_format

plt.ylabel('Pearson')

plt.title('Coeficiente de Correlação Linear - Pearson')

#label='Maior correlação negativa')



plt.plot(color="r",label='Maior correlação negativa')

    

plt.legend(bbox_to_anchor=(1.1, 1.1))



plt.show()
#ScatterPlot of the features X Movehub_Rating

plt.scatter(data.Purchase_Power,data.Movehub_Rating)

plt.scatter(data.Purchase_Power[79],data.Movehub_Rating[79],color='r')

plt.xlabel('Purchase_Power')

plt.ylabel('Movehub Rating')

plt.title('Relação entre Purchase Power e Movehub Rating')

plt.show()
plt.scatter(data.Avg_Disposable_Income,data.Movehub_Rating)

plt.scatter(data.Avg_Disposable_Income[246],data.Movehub_Rating[246],color='r')

plt.scatter(data.Avg_Disposable_Income[128],data.Movehub_Rating[128],color='r')

plt.xlabel('Avg_Disposable_Income')

plt.ylabel('Movehub Rating')

plt.title('Relação entre Avg_Disposable_Income e Movehub Rating')

plt.show()
plt.scatter(data.Avg_Rent,data.Movehub_Rating)

plt.scatter(data.Avg_Rent[105],data.Movehub_Rating[105],color='r')

plt.xlabel('Avg_Rent')

plt.ylabel('Movehub Rating')

plt.title('Relação entre Avg_Rent e Movehub Rating')



plt.show()
plt.scatter(data.Cinema,data.Movehub_Rating)

plt.scatter(data.Cinema[190],data.Movehub_Rating[190],color='r')

#plt.plot(0.04718706,57.7712295139,)

plt.xlabel('Cinema')

plt.ylabel('Movehub Rating')

plt.title('Relação entre Cinema e Movehub Rating')

plt.show()

plt.scatter(data.Wine,data.Movehub_Rating)

plt.xlabel('Wine')

plt.scatter(data.Wine[213],data.Movehub_Rating[213],color='r')

plt.ylabel('Movehub Rating')

plt.title('Relação entre Wine e Movehub Rating')

plt.show()
#After drop the outliers:

quantidade = [1,2,3,4,5,6,7,8,9,10,11]

colors = ['g','g','g','g','y','y','g','y','y','r','r']

numbers = [0.826469,0.765779,0.712798,0.561414,0.429955,0.384803,0.585449,0.123752,0.175471,-0.245230,-0.137547]



#quantidade = [1,2,3,4,5,6,7,8,9,10,11]

#colors = ['g','g','g','g','y','y','y','y','y','r','r']

#numbers = [0.81049364263681856,0.751426,0.713844,0.542362,0.429955,0.384803,0.310625,0.142297,0.136347,-0.167520,-0.258001]



LABELS = ["Purchase_Power","Avg_Disposable_Income","Quality_of_Life","Avg_Rent","Cappuccino","Health_Care","Cinema","Gasoline","Wine", "Pollution","Crime_Rating"]



plt.bar(quantidade, numbers,color=colors)

plt.xticks(quantidade, LABELS,rotation='vertical')

plt.ticklabel_format

plt.ylabel('Pearson')

plt.title('Coeficiente de Correlação Linear - Pearson após remoção de outliers')



plt.plot(color="r",label='Maior correlação negativa')

plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.show()