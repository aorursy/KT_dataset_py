# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re #Regular Expression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', None)  # to show all the column of the dataset

#Read both datasets

googlestore = pd.read_csv("../input/googleplaystore.csv")

googlereviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")
#The row 10472 is problematic, it doesn't have the Category camp so everything is shifted to the left

googlestore.loc[[10472]]

#Best choice for the row 10472 is the record removal

googlestore=googlestore.drop(10472)
googlestore.head()
googlereviews.head()
googlestore.Size.unique()
size_to_num = re.compile('(?P<number>\d+\.{0,1}\d*)(?P<prefix>\w*)')

#In this function we use lowercase k for kilo and uppercase M for Mega.

#We Assume no file with any other unit otherwise the unit will be ignored.

def prefix_to_mult(unit):

    if unit == 'M':

        return 1000000

    if unit == 'k':

        return 1000

    return 1



#This function can understand the format of the data given to it using regex,

#If the given data is not in the expected format it will return 'Not a Number'

#For example 'Varies with device' is transformed in NaN

def ConvertSizeToByte(Size):

    searched = size_to_num.search(Size)

    if searched is None:

        return np.nan

    else:

        prefix = searched.group('prefix')

        mult = prefix_to_mult(prefix)

        result = float(searched.group('number'))

        return int(result*mult)

    

#Here We apply it

googlestore['SizeInBytes'] = googlestore['Size'].apply(ConvertSizeToByte)
googlestore.Installs.unique()
install_to_num = re.compile('(?P<number>[\d,]+)') #the format is num,num,num,...,num



#We find using regex the number in the format 

def installToNumber(installs):

    found = install_to_num.search(installs)

    if(found):

        replacedComma = found.group('number').replace(',','') #replace the commas with nothing

        return int(replacedComma)

    else:

        return np.nan

    

#Here we apply it

googlestore['InstallNumber']=googlestore['Installs'].apply(installToNumber)
#We Transform “Varies with device” into a missing value so it's easier to manage

googlestore.replace('Varies with device', np.nan,inplace=True)

#inplace is needed so the replace is executed on the dataframe itself
print('There are {0} missing data values in Current Ver out of {1} more than 12%'

      .format(googlestore[googlestore['Current Ver'].isnull()].shape[0], googlestore.shape[0]))
googlestore['Current Ver'].value_counts()
#It will search for any string with the format num.num.num...

#It should end with a number, can also not have a dot for example 

#a new app can have version 1 and not necessarily 1.0

currentVer = re.compile('(?P<ver>[\d+\.]*\d)') 



def convertCurrentVer(version):

    #Since we checked that there are some NaN values we use an extra condition

    if pd.isnull(version):

        return np.nan

    CurVer=currentVer.search(version)

    if CurVer is not None:

        return CurVer.group('ver')

    else:

        return np.nan

    

#Here we apply it

googlestore['CurrentVersion']=googlestore['Current Ver'].apply(convertCurrentVer)
googlestore['Android Ver'].value_counts()
print('There are {0} missing data values in Android Ver out of {1} more than 12%'

      .format(googlestore[googlestore['Android Ver'].isnull()].shape[0], googlestore.shape[0]))
MinVer_toNum=re.compile('^(?P<MinVer>\d+\.\d+\.{0,1}\d*)')

MaxVer_toNum=re.compile('(?P<MaxVer>\d+\.\d+\.{0,1}\d*)$')



#To be changed if new version is released, will be used instead of 'and up'

major_Android_ver = 8.1



#We Decide not to trow away the missing version since we would we throwing away more that 12% of the data!

def MinAndroidVersion_ToNum(version):

    if pd.isnull(version):

        return np.nan

    MinVer=MinVer_toNum.search(version)

    if MinVer is not None:

        return MinVer.group('MinVer')

    else:

        return np.nan

    

def MaxAndroidVersion_ToNum(version):

    #Since there are some NaN values

    if pd.isnull(version):

        return np.nan

    MaxVer=MaxVer_toNum.search(version)

    if MaxVer is not None:

        return MaxVer.group('MaxVer')

    else:

        return major_Android_ver

    

#Here we apply it

googlestore['Min_Android_Ver']=googlestore['Android Ver'].apply(MinAndroidVersion_ToNum)

googlestore['Max_Android_Ver']=googlestore['Android Ver'].apply(MaxAndroidVersion_ToNum)
googlestore.head()
print('Total Extra Apps: ',googlestore[googlestore.duplicated('App')].shape[0])
#Convert the review column to numeric

googlestore.Reviews = googlestore.Reviews.apply(pd.to_numeric)

#Check it's type now

googlestore.Reviews.dtype
#We sort according to Reviews decendingly, keeping only the first

#This way we keep the most updated instance

googlestore=googlestore.sort_values(['App','Reviews'],ascending=False).drop_duplicates('App',keep='first')
print('Remaining Extra Apps: ',googlestore[googlestore.duplicated('App')].shape[0])
googlestore.groupby('Category').size()
googlestore.groupby('Category')['Rating'].mean()
genre = pd.DataFrame([genres.split(';') for genres in googlestore.Genres])



genre_series = pd.Series() #Empty Series

Apps2 = pd.Series() #Empty Series



for i in genre.columns:

    #just for convenience it indicates the column number of the genre dataframe created before

    genre_i = genre[i] 

    #We append in a single series all the columns of the dataframe genre

    genre_series = genre_series.append(genre_i)

    #We append as many times as genre also the application, we will need to later for the join table

    Apps2 = Apps2.append(googlestore.App)

#We drop the None values assuming all the apps to hava a genre and keep only the unique values

#So no app can have a None Genre.

genre_unique = genre_series.dropna().unique()

#We sort it because we want the column of the genre ordered so it's easier to find thema

genre_unique_sorted = np.sort(genre_unique)



print(genre_unique_sorted)
false_vector = np.repeat([False],googlestore.shape[0])

for elem in genre_unique_sorted:

    googlestore[elem]=false_vector



#Checking dtype of one of the columns

googlestore[genre_unique_sorted[0]].dtype
#We zip the only two important attributes for the exercise



#We use zip instead of iterating because pandas is slow when it comes 

#to iterate over the rows using a for cycle and the loc or iloc function

for i in zip(googlestore.index, googlestore.Genres):

    genrePerApp=i[1].split(';')

    for elem in genrePerApp:

        googlestore.at[i[0],elem] = True #We assign True only where needed and leave the rest False

        #googlestore.loc[i[0],elem] = True

        

#I changed from .loc to .at because it is faster than loc because

#'loc' select multiple columns while 'at' only a single value at a time. 
#Since we created the new columns for each genre, let's use them. 

#Could have also used a groupby on each column

average_rating_per_genre=pd.Series() #Empty series, we use series because later we need use idxmax() 

for genere in genre_unique:

    ratings=googlestore[googlestore[genere]==True]['Rating']

    average_rating_per_genre[genere]=ratings.mean(skipna=True)

    #The mean is by deafault with skipna=True, but we put it explicitly to show the need here.

    

pd.DataFrame(average_rating_per_genre,columns=['average_rating'])
#The maximum is

print("'{}' is the genre with highest ratings with {} average rating"

            .format(average_rating_per_genre.idxmax(),

            average_rating_per_genre[average_rating_per_genre.idxmax()]))
#First we need to convert from Price to numbers

money_to_float = re.compile('\D*(?P<amount>\d+\.{0,1}\d*)')



def Convert(money):

    found = money_to_float.search(money)

    if found:

        return float(found.group('amount'))

    else:

        return None





googlestore['PriceInNumber']=googlestore.Price.apply(Convert)

#Or just strip('$') and convert to numeric, but regex makes the code cooler.



googlestore['approximate_income']=googlestore.InstallNumber * googlestore.PriceInNumber
googlestore[['App','Price','InstallNumber','PriceInNumber','approximate_income']][googlestore.Price!='0'].head()

#Holy Moly Look at that app it's $399.99 and 10,000 people bought it!
googlereviews.isna().any() 
minSentiment = pd.DataFrame(googlereviews.groupby('App')['Sentiment_Polarity'].min())

maxSentiment = pd.DataFrame(googlereviews.groupby('App')['Sentiment_Polarity'].max())

MinMaxSentiment = minSentiment.merge(maxSentiment, on='App',suffixes=['_min','_max'])



#### We show only the ones without NaN, because the only one useful

MinMaxSentiment[MinMaxSentiment.notnull().any(axis=1)]