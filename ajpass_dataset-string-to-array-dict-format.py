# Load data, show first 5 rows

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/top-ranked-mangas-myanimelist-mal/top500mangaMAL.csv')

df[['English Title', 'Published Dates', 'Genres', 'Author']].head()
# Dataframe information -> Dtype (data type) 

df[['English Title','Published Dates', 'Genres', 'Author']].info()
firstRowGenres = df['Genres'][0]

firstRowGenres
import ast # Abstract Syntax Tree
firstRow = ast.literal_eval(firstRowGenres)

firstRow[0]
secondRowDates = df['Published Dates'][1]

secondRowDates
datesDict  = ast.literal_eval(secondRowDates)

datesDict['from']
datesDict['to']
dates =df['Published Dates']

dates
# Array for all the dictionaries and how to use them

allDates = []

for date in dates:

    allDates.append(ast.literal_eval(date))



allDates[:5]
allDates[0]['from']
vagabondAuthors = df['Author'][5]

firstRowAuthors = ast.literal_eval(vagabondAuthors)

firstRowAuthors[0]
firstRowAuthors[1]
firstRowAuthors[1].strip()