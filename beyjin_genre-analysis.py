import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
## import necessary Files ## 
Genre = pd.read_csv("../input/Movie_Genres.csv")
Movies = pd.read_csv("../input/Movie_Movies.csv")
# reduce the Movie Dataframe to only the necessary Columns which are necessary
# This will help to get a better overview
Movies.head(n = 2)
Movies.info()
#drop Columns which are not necessary
Movies.drop(['Unnamed: 0', 'Awards', 'BoxOffice', 'DVD', 'Poster', 'Plot', 'Response', 'Website', 'Production'], axis = 1, inplace = True)
Movies.info()
Movies.drop(['Rated', 'Metascore'], axis = 1, inplace = True)
Analysis = pd.merge(Movies, Genre, on = 'imdbID')
Analysis.info()
Analysis.drop('Unnamed: 0', axis = 1, inplace = True)
#Get a general Overview of the Genre Information
Analysis['Genre'].nunique()
Analysis['Genre'].unique()
## We can cleary see that the Strings are not trimmed. 
# Therefore If we do further Analysis - we might receive Duplicates e.g. ' History' != 'History'

Analysis['Genre'] = Analysis['Genre'].apply(lambda  x : x.strip())
Analysis_TimeAnalysis = Analysis
Analysis_TimeAnalysis = Analysis_TimeAnalysis[Analysis_TimeAnalysis['imdbRating'].notnull()]
Analysis_TimeAnalysis = Analysis_TimeAnalysis[Analysis_TimeAnalysis['Year'].notnull()].reset_index()
# how many rows do we still have?
len(Analysis_TimeAnalysis.index)
# check if each Year entry is an int
# This function will help deliver all different kinds of occurences which do not fit to our Format. Because I don´t like losing data. 
# Cleaning Cleaning :-)
VariableTypes = {}
for year in Analysis_TimeAnalysis['Year']:
    if type(year) != int:
        invalidType = type(year).__name__
        if invalidType in VariableTypes.keys():
            if year not in VariableTypes[invalidType]:
                VariableTypes[invalidType].append(year)
        else:
            VariableTypes[invalidType] = []
            VariableTypes[invalidType].append(year)
VariableTypes.keys()
print(VariableTypes['float'][0:10])
print(VariableTypes['float'][-10:-1])
print(len(VariableTypes['float']))
print(VariableTypes['str'][0:10])
print(VariableTypes['str'][-10:-1])
print(len(VariableTypes['str']))
#clean float numbers 
#split the real Year and get the acutal information
Analysis_TimeAnalysis['Year'] = Analysis_TimeAnalysis['Year'].apply(lambda x : int(str(x).split(".")[0]) if type(x) == float else x)
Analysis_TimeAnalysis['Year'] = Analysis_TimeAnalysis['Year'].apply(lambda x : x.split("–")[0] if type(x) is str else x)
Analysis_TimeAnalysis['Year'] = Analysis_TimeAnalysis['Year'].apply(lambda x : int(x) if type(x) != int else x)
# check if everything is clean now
for row in Analysis_TimeAnalysis['Year']:
    if type(row) is not int:
        print(row)
        break
VariableTypes = {}
for rating in Analysis_TimeAnalysis['imdbRating']:
    if type(rating) != float:
        invalidType = type(rating).__name__
        if invalidType in VariableTypes.keys():
            if rating not in VariableTypes[invalidType]:
                VariableTypes[invalidType].append(rating)
        else:
            VariableTypes[invalidType] = []
            VariableTypes[invalidType].append(rating)
## check if there any further additional information which have to be cleaned
VariableTypes.keys()
#Get a view of the current dataframe so that I do not have to scroll up
Analysis_TimeAnalysis.head()
Analysis_TimeAnalysis['Genre'].nunique()
AnalysisGrouped = Analysis_TimeAnalysis.groupby('Genre')
GenreTopTen = AnalysisGrouped['imdbID'].count().reset_index()
GenreTopTen.sort_values('imdbID', ascending = False, inplace = True)
Genre_Top_Ten = GenreTopTen.head(n = 10)
Genre_Top_Ten.head(10)
# Drop each Row which matches to the filtered Genre
# Create Helper Variable
# via apply function I do check if the Genre is in the array of Genre which we will not look any further
Filter = []
for genre in Genre_Top_Ten['Genre']:
    Filter.append(genre)

Analysis_TimeAnalysis['Keep'] = Analysis_TimeAnalysis['Genre'].apply(lambda genre : True if genre in Filter else False)
# we will create a new Dataframe so that we our general dataframe in our backs for potential different analysis - e.g. Genre Combination 
Analysis_TimeAnalysis_Final = Analysis_TimeAnalysis[Analysis_TimeAnalysis['Keep'] == True ]
Analysis_TimeAnalysis_Final.info()
Analysis_Year_Genre = Analysis_TimeAnalysis_Final.groupby(['Year', 'Genre'])
Year = Analysis_TimeAnalysis_Final.groupby('Year')
Year['imdbID'].count()
Analysis_TimeAnalysis_Final_YearFilter = Analysis_TimeAnalysis_Final[(Analysis_TimeAnalysis_Final['Year'] >= 1990) & 
                                                                     (Analysis_TimeAnalysis_Final['Year'] <= 2016)]
Analysis_Year_Genre = Analysis_TimeAnalysis_Final_YearFilter.groupby(['Year', 'Genre'])
fig, ax = plt.subplots()

fig.set_size_inches(18, 9)
sns.pointplot(x = 'Year', y = 'imdbRating', data = Analysis_Year_Genre['imdbRating'].mean().reset_index(), hue = 'Genre', ax = ax)
