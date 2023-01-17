import numpy as np 

import pandas as pd 

from scipy import spatial



import warnings

warnings.filterwarnings('ignore')



import re

import os

print(os.listdir("../input"))

ratings = pd.read_csv("../input/ratings.csv")

# links = pd.read_csv("../input/links.csv")

tags = pd.read_csv("../input/tags.csv")

movies = pd.read_csv("../input/movies.csv")
print(ratings.shape)

ratings.head(5)
pd.options.display.float_format = '{:f}'.format

ratings['rating'].describe()
#number of ratings

ratings['rating'].hist()
ratings['rating'].plot( kind='box',subplots=True)
userRatingsAggr = ratings.groupby(['userId']).agg({'rating': [np.size, np.mean]})

userRatingsAggr.reset_index(inplace=True)  # To reset multilevel (pivot-like) index

userRatingsAggr.head()
userRatingsAggr['rating'].describe()
userRatingsAggr['rating'].plot(kind='box', subplots=True)
movieRatingsAggr = ratings.groupby(['movieId']).agg({'rating': [np.size, np.mean]})

movieRatingsAggr.reset_index(inplace=True)

movieRatingsAggr.head()
movieRatingsAggr['rating'].describe()
movieRatingsAggr['rating'].plot(kind='box', subplots=True)
print(tags.shape)

tags.head(5)
print(movies.shape)

movies.head(5)
movies = movies.merge(movieRatingsAggr, left_on='movieId', right_on='movieId', how='left')  # ['rating']

movies.columns = ['movieId', 'title', 'genres', 'rating_count', 'rating_avg']
movies.head(5)
def getYear(title):

    result = re.search(r'\(\d{4}\)', title)

    if result:

        found = result.group(0).strip('(').strip(')')

    else: 

        found = 0

    return int(found)

    

movies['year'] = movies.apply(lambda x: getYear(x['title']), axis=1)

movies.head(10)
genresList = [

  "Action",

  "Adventure",

  "Animation",

  "Children",

  "Comedy",

  "Crime",

  "Documentary",

  "Drama",

  "Fantasy",

  "Film-Noir",

  "Horror",

  "Musical",

  "Mystery",

  "Romance",

  "Sci-Fi",

  "Thriller",

  "War",

  "Western",

  "(no genres listed)"

]



def setGenresMatrix(genres):

    movieGenresMatrix = []

    movieGenresList = genres.split('|')

    for x in genresList:

        if (x in movieGenresList):

            movieGenresMatrix.append(1)

        else:

            movieGenresMatrix.append(0) 

    return movieGenresMatrix

    

movies['genresMatrix'] = movies.apply(lambda x: np.array(list(setGenresMatrix(x['genres']))), axis=1)



movies.head(5)
movieRatingsAggr['rating'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
def setRatingGroup(numberOfRatings):

    # if (numberOfRatings is None): return 0

    if (1 <= numberOfRatings <= 10): return 1

    elif (11 <= numberOfRatings <= 30): return 2

    elif (31 <= numberOfRatings <= 100): return 3

    elif (101 <= numberOfRatings <= 300): return 4

    elif (301 <= numberOfRatings <= 1000): return 5

    elif (1001 <= numberOfRatings): return 6

    else: return 0



movies['ratingGroup'] = movies.apply(lambda x: setRatingGroup(x['rating_count']), axis=1)

movies.fillna(0, inplace=True)  # Replace NaN values to zero

movies.head(10)
stopWords = ['a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 

        'alone', 'along', 'already', 'also','although','always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 

        'another', 'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',  'at', 'back','be','became', 

        'because','become','becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 

        'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 

        'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven','else', 

        'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 

        'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 

        'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 

        'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 

        'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 

        'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 

        'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 

        'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own','part', 'per', 'perhaps', 'please', 

        'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 

        'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 

        'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 

        'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 

        'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 

        'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 

        'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 

        'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the']



tagsDict = {}



for index, x in tags.iterrows():

    wordlist = str(x['tag']).lower().split(' ')

    movieId = x['movieId']

    for y in wordlist:

        if y not in stopWords:

            if movieId in tagsDict:

                # if y not in tagsDict[movieId]:  # Switched off (we will get a non unique list)

                    tagsDict[movieId].append(y)

            else:

                tagsDict[movieId] = [y]



tags.apply(lambda x: str(x['tag']).split(' '), axis=1)

print(tagsDict[6])
titleWordsDict = {}



for index, x in movies.iterrows():

    wordlist = str(x['title']).lower().split(' ')

    movieId = x['movieId']

    for y in wordlist:

        if y not in stopWords:

            if movieId in titleWordsDict:

                    titleWordsDict[movieId].append(y)

            else:

                titleWordsDict[movieId] = [y]
# Parameter weights

genresSimilarityWeight = 0.8

tagsSimilarityWeight = 2

titleSimilarityWeight = 1

ratingAvgWeight = 0.2

ratingGroupWeight = 0.005

yearDistanceWeight = 0.1



def tagsSimilarity(basisMovieID, checkedMovieID, checkType):    

    # The higher value is the more similar (from 0 to 1) 

    if checkType == 'tag':

        dictToCheck = tagsDict

    else:

        dictToCheck = titleWordsDict

        

    counter = 0

    if basisMovieID in dictToCheck: 

        basisTags = dictToCheck[basisMovieID]

        countAllTags = len(basisTags)

        basisTagsDict = {}

        for x in basisTags:

            if x in basisTagsDict:

                basisTagsDict[x] += 1

            else:

                basisTagsDict[x] = 1   

        

        for x in basisTagsDict:

            basisTagsDict[x] = basisTagsDict[x] / countAllTags

    else: return 0

    

    if checkedMovieID in dictToCheck: 

        checkedTags = dictToCheck[checkedMovieID]

        checkedTags = set(checkedTags) # Make the list unique

        checkedTags = list(checkedTags)

        

    else: return 0

    

    for x in basisTagsDict:

        if x in checkedTags: counter += basisTagsDict[x]

    return counter

    

def checkSimilarity(movieId):

    # print("SIMILAR MOVIES TO:")

    # print (movies[movies['movieId'] == movieId][['title', 'rating_count', 'rating_avg']])

    basisGenres = np.array(list(movies[movies['movieId'] == movieId]['genresMatrix']))

    basisYear = int(movies[movies['movieId'] == movieId]['year'])

    basisRatingAvg = movies[movies['movieId'] == movieId]['rating_avg']

    basisRatingGroup = movies[movies['movieId'] == movieId]['ratingGroup']

    

    moviesWithSim = movies

    moviesWithSim['similarity'] = moviesWithSim.apply(lambda x: 

                                                      spatial.distance.cosine(x['genresMatrix'], basisGenres) * genresSimilarityWeight + 

                                                      - tagsSimilarity(movieId, x['movieId'], 'tag') * tagsSimilarityWeight +

                                                      - tagsSimilarity(movieId, x['movieId'], 'title') * titleSimilarityWeight +

                                                      abs(basisRatingAvg - x['rating_avg']) * ratingAvgWeight +

                                                      abs(basisRatingGroup - x['ratingGroup']) * ratingGroupWeight + 

                                                      abs(basisYear - x['year'])/100 * yearDistanceWeight

                                                     , axis=1)

    

    moviesWithSim = moviesWithSim.loc[(moviesWithSim.movieId != movieId)]

    return moviesWithSim[['movieId', 'title', 'genres', 'rating_count', 'rating_avg', 'similarity']].sort_values('similarity')

currentMovie = movies.loc[(movies.movieId == 3793)]

currentMovie.head(1)
# X-men

similarityResult  = checkSimilarity(3793)

similarityResult.head(5)
# Lock, Stock & Two Smoking Barrels

similarityResult  = checkSimilarity(2542)

similarityResult.head(5)
# Casino

similarityResult  = checkSimilarity(16)

similarityResult.head(5)
# Star Wars: Episode IV - A New Hope

similarityResult  = checkSimilarity(260)

similarityResult.head(5)
# Iron Man

similarityResult  = checkSimilarity(59315)

similarityResult.head(5)
# The Good, the Bad and the Ugly

similarityResult  = checkSimilarity(1201)

similarityResult.head(5)
# Mega Shark vs. Giant Octopus

similarityResult  = checkSimilarity(73829)

similarityResult.head(5)
# King Kong Escapes (Kingu Kongu no gyakushû)

similarityResult  = checkSimilarity(92518)

similarityResult.head(5)
# Armageddon

similarityResult  = checkSimilarity(1917)

similarityResult.head(5)
# Groundhog Day

similarityResult  = checkSimilarity(1265)

similarityResult.head(5)





# Cars

similarityResult  = checkSimilarity(45517)

similarityResult.head(5)