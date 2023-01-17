## Answers to the questions :

#1.Which is the size of the edu DataFrame (rows x columns)?
# It is (384 rows x 3 columns)

#2.What happens if we give a number as argument to the method head()?
#Par exemple, head(3) affiche les éléments des 2 (n-1=3-1=2) premières lignes de la structure de données (edu). 

#3.What does the method tail()return?
#Cette méthode retourne tous les éléments des 5 dernières lignes de la structure de données. 

#4.Which measures does the result show? It seems that it shows some default values, can you guess which ones?
#Cette fonction donne la somme totale des valeurs des colonnes (que celles qui contiennent des valeurs numériques), ainsi que la moyenne, l'écart-type, le min, le max et les quartiles. 

#5.What does this index return? What does the first index represent? And the second one?
#Cette fonction retourne une certaine partie du tableau qu'on a spécifié. Le 1er index est le numéro de la 1ere ligne dont on veut regarder les valeurs de temps et de géographie et le 2nd est le numéro de la dernière ligne dont on veut regarder les valeurs + 1. 

#6.What does the operation edu[’Value’] > 6.5 produce? And if we apply the indexedu[edu[’Value’] > 6.5]?Is this aSeries or aDataFrame?
# Affiche les valeurs des 3 colonnes pour les lignes dont les valeurs de value sont supérieures à 6,5, mais seulement pour les 5 dernières lignes de la structure edu qui sont concernées par cette condition car il y a la fonction tail(). indexedu[edu[’Value’] > 6.5] affiche les index pour lesquels la condition entre crochets est respectées. C'est donc une Series car il n'y a qu'une ligne avec plusieurs éléments.  

#7. What do you observe regarding the parameter ascending=False?
# Ce parametre, quand il est égale à False, signifie que les éléments sont classés par ordre décroissants. S'il est égale à True, les paramètres sont rangés par ordre croissant. 





import numpy as np # bibliothèque de linear algebra, pour utiliser les matrices
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = {
'year': [2010, 2011, 2012,
2010, 2011, 2012,
2010, 2011, 2012],
'team': ['FCBarcelona', 'FCBarcelona', 'FCBarcelona',
'RMadrid', 'RMadrid', 'RMadrid',
'ValenciaCF', 'ValenciaCF', 'ValenciaCF'],
'wins': [30, 28, 32, 29, 32, 26, 21, 17, 19],
'draws': [6, 7, 4, 5, 4, 7, 8, 10, 8],
'losses': [2, 3, 2, 4, 2, 5, 9, 11, 11]
}

#The key data structure in Pandas is the DataFrameobject. A DataFrameis a tabular data structure, with rows and columns. Rows have a specific index to access them, which can be any name or value. In Pandas, the columns are called Series, a special type of data, which consists of a list of several values, where each value has an index.
football = pd.DataFrame(data, columns = ['year', 'team', 'wins', 'draws', 'losses'])

football

#La raw data est contenue dans un fichier csv
#The way to read CSV (educ_figdp_1_Data) files in Pandas is by calling the method read_csv. Besides the name of the file, we add the key argument na_valuesto this method along with the character that represents “non available data” in the file.
edu = pd.read_csv('/kaggle/input/datalab3/files/ch02/educ_figdp_1_Data.csv',
                  na_values=':', usecols=['TIME', 'GEO', 'Value'])
edu

#To see how the data looks, we can use the method head, which shows just the first five rows
edu.head() 

#Pareil que la fonction head mais affiche les derniers élements du DataFrame.
edu.tail()

#If we just want quick statistical information on all the numeric columns in a DataFrame, we can use the function describe
edu.describe()

#to select a subset of data from a DataFrame, it is necessary to indicate this subset using square brackets ([ ]). If we want to select only one column from a DataFrame, we only need to put its name between the square brackets. The result will be a Seriesdata structure, not a DataFrame, because only one column is retrieved.
edu['GEO']

#If we want to select a subset of rows from a DataFrame, we can do so by indicating a range of rows separated by a colon (:) inside the square brackets. This instruction returns the slice of rows from the 10th to the 13th position
edu[10:14]

#If we want to select a subset of columns and rows using the labels as our references instead of the positions, we can use ilocindexing
edu.iloc[90:94][['TIME','GEO']]

#Another way to select a subset of data is by applying Boolean indexing. This indexing is known as a filter. For instance, if we want to filter those values less than or equal to 6.5, we can do it like this:
#any of the usual Boolean operators can be used for filtering: < (less than),<= (less than or equal to), > (greater than), >= (greater than or equal to), == (equal to), and ! = (not equal to).
edu[edu['Value'] > 6.5].tail()

# Pandas uses the special value NaN(not a number) to represent missing values. In Python, NaNis a special floating-point value returned by certain operations when one of their results ends in an undefined value. A subtle feature ofNaNvalues is that two NaNare never equal. Because of this, the only safe way to tell whether a value is missing in a DataFrameis by using the function isnull(). Indeed, this function can be used to filter rows with missing values:
edu[edu["Value"].isnull()].head()

#you can specify if the function should be applied to the rows for each column (setting the axis=0 keyword on the invocation of the function),or it should be applied on the columns for each row (setting the axis=1 keyword on the invocation of the function).
edu.max(axis = 0)

#the Pandas maxfunction excludes NaNvalues, thus they are interpreted as missing values, while the standard Python max function will take the mathematical interpretation of NaNand return it as the maximum:
print ('Pandas max function:', edu['Value'].max())
print ('Python max function:', max(edu['Value']))

# we can apply operations over all the values in rows, columns or a selection of both. The rule of thumb is that an operation between columns means that it is applied to each row in that column and an operation between rows means that it is applied to each column in that row. For example we can apply any binary arithmetical operation (+,-,*,/) to an entire row
s = edu["Value"]/100
s.head()

#we can apply any function to a DataFrameor Seriesjust setting its name as argument of the apply method. For example, in the following code, we apply the sqrt function from the NumPylibrary to perform the square root of each value in the column Value.
s = edu["Value"].apply(np.sqrt)
s.head()

#If we need to design a specific function to apply it, we can write an in-line function, known as a λ-function. A λ-function is a function without a name. It is only necessary to specify the parameters it receives, between the lambda keyword and the colon (:). This function returns the square of that value of each element in the column Value.
s = edu["Value"].apply(lambda d: d**2)
s.head()

#we assign the Seriesthat results from dividing the columnValueby the maximum value in the same column to a new column named ValueNorm.
edu['ValueNorm'] = edu['Value']/edu['Value'].max()
edu.tail()

#if we want to remove this column from the DataFrame, we can use the function drop.
#This removes the indicated rows if axis=0, or the indicated columns if axis=1.
#return a copy of the modified data, instead of overwriting the DataFrame. Therefore, the original DataFrameis kept. If you do not want to keep the old values, you can set the keyword inplaceto True. By default, this keyword is set toFalse, meaning that a copy of the data is returned.
edu.drop('ValueNorm', axis = 1, inplace = True)
edu.head()

#if what we want to do is to insert a new row at the bottom of the DataFrame, we can use the Pandas function append. This function receives as argument the new row, which is represented as a dictionary where the keys are the name of the columns and the values are the associated value. You must be aware to setting the ignore_indexflag in the append method to True, otherwise the index 0 is given to this new row, which will produce an error if it already exists:
edu = edu.append({"TIME": 2000, "Value": 5.00, "GEO": 'a'},
                  ignore_index = True)
edu.tail()

#if we want to remove this row, we need to use the function dropagain. Now we have to set the axis to 0, and specify the index of the row we want to remove
edu.drop(max(edu.index), axis = 0, inplace = True)
edu.tail()

eduDrop = edu[~edu["Value"].isnull()].copy()
eduDrop.head()

eduDrop = edu.dropna(how = 'any', subset = ["Value"])
eduDrop.head()

eduFilled = edu.fillna(value = {"Value": 0})
eduFilled.head()

# SORTING DATA

edu.sort_values(by = 'Value', ascending = False,
                inplace = True)
edu.head()

edu.sort_index(axis = 0, ascending = True, inplace = True)
edu.head()

group = edu[["GEO", "Value"]].groupby('GEO').mean()
group.head()

filtered_data = edu[edu["TIME"] > 2005]
pivedu = pd.pivot_table(filtered_data, values = 'Value',
                        index = ['GEO'], columns = ['TIME'])
pivedu.head()

pivedu.loc[['Spain','Portugal'], [2006,2011]]

#RANKING DATA

pivedu = pivedu.drop(['Euro area (13 countries)',
                      'Euro area (15 countries)',
                      'Euro area (17 countries)',
                      'Euro area (18 countries)',
                      'European Union (25 countries)',
                      'European Union (27 countries)',
                      'European Union (28 countries)'
                      ], axis=0)
pivedu = pivedu.rename(
    index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
pivedu = pivedu.dropna()
pivedu.rank(ascending=False, method='first').head()

totalSum = pivedu.sum(axis = 1)

totalSum.rank(ascending = False, method = 'dense').sort_values().head()


totalSum = pivedu.sum(axis = 1).sort_values(ascending = False)
totalSum.plot(kind = 'bar', style = 'b', alpha = 0.4,
              title = "Total Values for Country")

my_colors = ['b', 'r', 'g', 'y', 'm', 'c']
ax = pivedu.plot(kind='barh', stacked=True, color=my_colors, figsize=(12, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Value_Time_Country.png', dpi=300, bbox_inches='tight')




index.edu[edu['Value'] > 6.5]
pivedu = pivedu.drop(['Euro area (13 countries)',
                      'Euro area (15 countries)',
                      'Euro area (17 countries)',
                      'Euro area (18 countries)',
                      'European Union (25 countries)',
                      'European Union (27 countries)',
                      'European Union (28 countries)'
                      ], axis=0)
pivedu = pivedu.rename(
    index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
pivedu = pivedu.dropna()
pivedu.rank(ascending=False, method='first').head()