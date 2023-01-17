# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.columns #Total of 20 columns
data.dtypes
#Converting data to different type
#dataNew = data.keywords.astype('category') # 'float' ...
cv = lambda x, y: x.astype(y)
cv(data.keywords, 'category')
data.shape
data.info() #homepage property have 3091 null elements
data.head(8)
data.tail(8)
data.describe()
data.plot()
plt.show()
#Filling && Dropping Nan | Null | - data
#data.property.dropna(inplace = True) #Nan will be dropped.
#data.property.fillna('Anything',inplace = True) #Anything is a value that we wish to change Nan with.
#dropna allows to count null values
data.production_companies.value_counts(dropna = False) 
data.production_countries.value_counts(dropna = False) 
data.homepage.dropna(inplace = True) #Drops null values
#ASSERTING
#assert 1 == 1 #True returns nothing
#assert 1 == 93 #False returns error
assert data['homepage'].notnull().all() #Checks if homepage has no null value 
#returns true if homepage does not include null values.
data.homepage.fillna('myValue', inplace = True)
assert data['homepage'].notnull().all()
assert data.columns[0] == 'budget'
assert data.budget.dtype == int
#assert data.budget.dtype == pd.int #Error because pd does not have attribute int
assert data.budget.dtype == np.int
data.boxplot(column = 'budget')
plt.show()
#data.boxplot(column = 'revenue', by = ['vote_average', 'vote_count'])
#data.boxplot(column = 'revenue', by = 'vote_average')
#plt.show()
data.head()
#Melting
dataHead = data.head()
dataHead = pd.melt(frame = dataHead, id_vars = 'title', value_vars = ['budget', 'revenue'])
dataHead
#Reverse melting
dataHead.pivot(index = 'title', columns = 'variable', values = 'value')
#Concatenating
#pd.concat([DATA1, DATA2], axis = (0 | 1), ignore_index = True <- sorts index from beginning)
concatRow = pd.concat([data.head(2), data.tail(2)], axis = 0, ignore_index = True)
concatRow
#concatColumn = pd.concat([data.head(2), data.tail(2)], axis = 1)
concatColumn = pd.concat([data.title.head(2), data.popularity.head(2)], axis = 1)
concatColumn
#Filters movies which cost less than 1.000.000 and earned less than 1.000.000
data = data[(data.budget > 1000000) & (data.revenue > 1000000)]
data.loc[data.budget == data.budget.max()]
data.loc[data.revenue == data.revenue.max()]
data.plot(x = 'budget', y = 'revenue', kind = 'scatter', color = 'blue', alpha = 0.9)
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.title('Budget vs Revenue')
plt.show()
data.loc[:,['budget', 'revenue']].plot()
plt.show()
data.loc[:,['budget', 'revenue']].plot(subplots = True)
plt.show()
#data.plot(kind = 'hist', y = 'revenue', bins = 999, range = (0, 250), normed = True)
#data.plot(kind = 'hist', y = 'budget', bins = 99, range = (0, 250), density = True)
#data.plot(kind = 'hist', y = 'budget', bins = 100, range = (1, 999))
#data.plot(kind = 'hist', y = 'budget', bins = 100, ylim = (0, 1000), range = (1, 1000))
columnsToKeep = ['title', 'budget', 'revenue']
dataHead = data[columnsToKeep].head()
#dataHead.plot(kind = 'hist', y = 'revenue', bins = 100, ylim = (-1, 1), range = (-1, 1))
dataHead.plot(kind = 'hist', y = 'revenue', bins = 25)
plt.legend(loc = 'upper right')
#plt.savefig('graph.png')
plt.show()
dataHead.plot(
    kind = 'hist', 
    y = 'revenue',
    bins = 100,
    cumulative = True
)
plt.show()
firstMovie = data[data.id == 1865] # firstMovie == Pirates of the Caribbean: On Stranger Tides
secondMovie = data[data.id == 19995]  # secondMovie == Avatar
#  columnsToDrop = ['keywords', 
#                  'original_language', 
#                  'original_title', 
#                  'overview', 
#                  'production_companies',
#                  'production_countries',
#                  'release_date',
#                  'runtime',
#                  'spoken_languages',
#                  'status',
#                  'tagline'
#                 ]
# firstMovie.drop(columnsToDrop, axis = 1, inplace = True)
# secondMovie.drop(columnsToDrop, axis = 1, inplace = True)
# gives KeyError: "['keywords' 'original_title'] not found in axis"
columnsToKeep = ['title', 'popularity', 'vote_average', 'vote_count']
firstMovie = firstMovie[columnsToKeep]
secondMovie = secondMovie[columnsToKeep]
# def getDetails(movie):
#     """
#     Get popularity, vote average and vote count properties as one string
#    :param pandas.dataFrame movie: Movie to get its details from.
#    :return: str : Details as one string
#    """
#     details = np.empty(shape = (2, 4), dtype = object) # np.empty(shape = (2, 4), dtype = np.str)
#     shapeCounter = 0
#     if movie is not None:
#         for key, value in movie.items():
#             if key == 'title' or key == 'popularity' or key == 'vote_average' or key == 'vote_count':
#                 details[0, shapeCounter] = str(key[0].upper() + key[1:len(key)])
#                 details[1, shapeCounter] = str(value)
#                 shapeCounter += 1
            
#     return details
         
# details0 = getDetails(firstMovie)
# details1 = getDetails(secondMovie)
def getDetailsAsArray(movie):
    """
    Get properties as numpy array
    :param pandas.dataFrame movie: Movie to get properties from.
    :return: numpy.array: Details as numpy object array.
    """
    #details = np.empty(shape = (2, 4), dtype='U32')
    details = np.empty(shape = (2, 4), dtype = object)
    columnCounter = 0
    if movie is not None:
        for key, value in movie.items():      
            # turn value series into a str list.
            value = value.to_string().split()
            # delete first element 17
            del value[0]          
            # join the list
            value = ' '.join(value).strip()
            if key == 'vote_average' or 'vote_count':
                details[0, columnCounter] = str(key)
            else:
                details[0, columnCounter] = str(key[0].upper() + key[1:len(key)])
            details[1, columnCounter] = str(value)
            columnCounter += 1
        
    return details
details0 = getDetailsAsArray(firstMovie)
details1 = getDetailsAsArray(secondMovie)
isBigger = lambda x, y: x > y
def getPercentage(firstValue, secondValue):
    """
    Calculates a percentage of this values.
    :param float firstValue:
    :param float secondValue:
    :return float percentage
    """
    return float(secondValue) * 100 / float(firstValue)
def compare(details0, details1):
    """
    Compares popularity, vote average and vote count properties as one string
    :param np.array details0: First details
    :param np.array details1: Second details
    """
    print('Titles: ')
    print('First title: ', details0[1, 0])
    print('Second title: ', details1[1, 0])
    firstPop = details0[1, 2]
    secondPop = details1[1, 2]
    firstVoteCount = details0[1, 3]
    secondVoteCount = details1[1, 3]
    #Popularity comparison
    if isBigger(firstPop, secondPop):
        percentage = getPercentage(firstPop, secondPop)
        print('First title is', '{0:.2f}'.format(percentage), 'more popular then second title.')
    elif isBigger(secondPop, firstPop):
        percentage = getPercentage(secondPop, firstPop)
        print('Second title is', '{0:.2f}'.format(percentage), 'more popular then first title.')
    else:
        print('Titles are equally popular.')
    #Vote count comparison
    if isBigger(firstVoteCount, secondVoteCount):
        percentage = getPercentage(firstVoteCount, secondVoteCount)
        print('First title is voted', '{0:.2f}'.format(percentage), 'then second title.')
    elif isBigger(secondVoteCount, firstVoteCount):
        percentage = getPercentage(secondVoteCount, firstVoteCount)
        print('Second title is voted', '{0:.2f}'.format(percentage), 'then first title.')
    else:
        print('Titles are equally voted.')
compare(details0, details1)
#Pandas creating frames with dictionaries
# pd.DataFrame(
#     dict(
#         list(
#             zip(
#                 ['name', 'age', 'job'], 
#                 [
#                     ['Jack', 'Cindy', 'Python'], 
#                     [32, 18, 29], 
#                     ['Coal Miner', 'Secretary', 'Engineer']
#                 ]
#             )
#         )
#     )
# )
name = ['Jack', 'Cindy', 'Adam']
age = [32, 18, 29]
job = ['Coal Miner', 'Secretary', 'Engineer']
listLabels = ['name', 'age', 'job']
listColumns = [name, age, job]
zippedList = list(zip(listLabels, listColumns))
dictionary = dict(zippedList)
frame = pd.DataFrame(dictionary)
frame
#Assing new value
frame['salary'] = [300, 600, 900]
frame
#Broadcasting
frame['isWorking'] = True
frame
#Pandas Time Series
#yyyy-mm-dd hh:mm:ss)
timeList = ['1992-03-08', '1992-04-12']
dateTime = pd.to_datetime(timeList)
type(dateTime)
columnsToKeep = [
    'title', 
    'budget', 
    'revenue', 
    'popularity'
]
dataHead = data[columnsToKeep].head(9).copy()
dateList = [
    '1992-01-01', 
    '1992-01-05',
    '1992-02-01', 
    '1992-03-01',
    '1992-03-11',
    '1992-04-01',
    '1992-05-01',
    '1993-03-02',
    '1994-02-05'
]
dateTime = pd.to_datetime(dateList)
dataHead['date'] = dateTime
dataHead = dataHead.set_index('date')
dataHead
print(dataHead.loc['1992-01-01'])
print('---Slicing---')
print(dataHead.loc['1992-01-01' : '1992-04-01'])
#Resampling
# A = year M = month
dataHead.resample('A').mean()
dataHead.resample('M').mean()
dataHead.resample('M').first().interpolate('linear')
dataHead.resample('M').mean().interpolate('linear')
#excludeNan = lambda frame: [else 'aaA' var if pd.notnull(var) for var in frame]
exclude = lambda frame: [
    var if pd.notnull(var)
    else "SpaceHolder"
    for var in frame
]
dataHead.resample('M').apply(exclude)
#Not working.
#I reimported because id column is dropped at this point.
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data = data.set_index('id')
data = data.sort_values('id')
data.head() #1 : 5 and 6 : 11 missing in id
data['title'][5]
print(type(data['title']))
print(type(data[['title']]))
data.loc[1:15, 'title'] # Froö 1 to 15 get 'title'
data.loc[1:15, 'title':] #from 1 to 15 get from 'title' to end
data[(data.vote_count > 10000) & (data.vote_average > 5.0)].head()
data.title[(data.popularity > 50.0) & (data.budget == 100000000)].head()
data['profit'] = data.revenue - data.budget
#Change order of columns so the profit property will be at the beginning
columns = data.columns.tolist()
#columns[-1:] get last element columns[:-1] get everything except last element
#columns[-1:] + columns[:-1] 
columns = columns[::-1] #columns[-1:] + columns[:-1] 
data[columns].head()
#Index processing
print(data.index.name)
#data.index.name = 'index'
#data.index = range(100, 900, 1) #index range will be changed
#data.set_index('title') == data.index = data['title']
#Data index has a hierarchy
hierarchicalData = data.set_index(['status', 'runtime'])
hierarchicalData.head()
dic = {
    'title': ['aaaa', 'bbbb', 'cccc', 'dddd'],
    'status': ['R', 'R', 'C', 'C'],
    'budget': [15, 20, 15, 20],
    'revenue': [4, 8, 12, 16]
}
frame = pd.DataFrame(dic)
frame
#Pivot
frame.pivot(index = 'status', columns = 'budget', values = 'revenue')
#title: aaa status: R budget: 15, revenue: 6
#title: bbb status: R budget: 20, revenue: 9
frame = frame.set_index(['status', 'budget'])
frame
#level position of stacked index
frame.unstack(level = 0)
frame.unstack(level = 1)
frame = frame.swaplevel(0, 1)
frame
#Melting repeat
dic = {
    'title': ['aaaa', 'bbbb', 'cccc', 'dddd'],
    'status': ['R', 'R', 'C', 'C'],
    'budget': [15, 20, 15, 20],
    'revenue': [4, 8, 12, 16]
}
frame = pd.DataFrame(dic)
pd.melt(frame, id_vars = 'status', value_vars = ['title', 'revenue'])
frame.groupby('status').mean()
frame.groupby('status').budget.mean()
frame.groupby('status')['revenue', 'budget'].min()