from math import sqrt

import csv

import os

import pandas as pd

import numpy as np



# Sample data to play with

critics = {

    'Lisa Rose': {

        'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5,

        'You, Me and Dupree': 2.5, 'The Night Listener': 3.0

    },

    'Gene Seymour': {

        'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0,

        'You, Me and Dupree': 3.5, 'The Night Listener': 3.0

    },

    'Michael Phillips': {

        'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5,

        'The Night Listener': 4.0

    },

    'Caludia Puig': {

        'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 4.0,

        'You, Me and Dupree': 2.5, 'The Night Listener': 4.5

    },

    'Mick LaSalle': {

        'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,

        'You, Me and Dupree': 2.0, 'The Night Listener': 3.0

    },

    'Jack Matthews': {

        'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Superman Returns': 5.0,

        'You, Me and Dupree': 3.5, 'The Night Listener': 3.0

    },

    'Toby': {

        'Snakes on a Plane': 4.5, 'Superman Returns': 4.0, 'You, Me and Dupree': 1.0

    }

}
def sim_euclidean(pref, person1, person2):

    """

    This function calculates the similarity between two data objects using Euclidean formula

    """

    si = {}

    for item in pref[person1]:

        if item in pref[person2]:

            si[item] = 1



    if len(si) == 0:

        return 0



    sum_of_squares = sum(pow(pref[person1][item] - pref[person2][item], 2) for item in si)

    return 1 / (1 + sum_of_squares)
def sim_tanimoto(pref, person1, person2):

    """

    This hunctions calculates similarity between two data objects using Tanimoto score

    """

    commonItems = {}

    for item in pref[person1]:

        if item in pref[person2]:

            commonItems[item] = 1

    

    dotProd = 0

    for item in commonItems:

        dotProd += (pref[person1][item] * pref[person2][item])

    

    person1Square = 0

    for item in pref[person1]:

        person1Square += (pref[person1][item] * pref[person1][item])

    

    person2Square = 0

    for item in pref[person2]:

        person2Square += (pref[person2][item] * pref[person2][item])

        

    score = dotProd/(person1Square+person2Square-dotProd)

    return score
def sim_pearson(pref, person1, person2):

    """

    This function calculates the similarity between two data objects using Pearson formula

    """

    si = {}

    for item in pref[person1]:

        if item in pref[person2]:

            si[item] = 1

    n = len(si)



    if n == 0:

        return 0



    sum1 = sum(pref[person1][item] for item in si)

    sum2 = sum(pref[person2][item] for item in si)



    sum1Sq = sum(pow(pref[person1][item], 2) for item in si)

    sum2Sq = sum(pow(pref[person2][item], 2) for item in si)



    pSum = sum(pref[person1][item] * pref[person2][item] for item in si)



    num = pSum - (sum1 * sum2) / n

    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))

    if den == 0:

        return 0

    return num / den
def topMatches(pref, person, n=5, similarity=sim_pearson):

    """

    This function returns the list of other users compared to the parameter person based on similarity measure

    """

    scores = [(similarity(pref, person, other), other) for other in pref if other != person]



    scores.sort()

    scores.reverse()

    return scores[0:n]
def getRecommendations(pref, person, similarity=sim_pearson):

    """

    Here we get movie recommendations for a user using user-based recommendation

    We also give the probable rating the user might give to these recommended movies

    """

    movie_total = {}

    movie_simSum = {}

    for critic in pref:

        if critic != person:

            sim_score = similarity(pref, person, critic)

            if sim_score <= 0:

                continue

            for movie in pref[critic]:

                if movie not in pref[person]:

                    movie_total.setdefault(movie, 0)

                    movie_simSum.setdefault(movie, 0)

                    movie_total[movie] += sim_score * pref[critic][movie]

                    movie_simSum[movie] += sim_score

    result = [(movie_total[movie] / movie_simSum[movie], movie) for movie in movie_total]

    result.sort()

    result.reverse()

    return result
def transformPrefs(prefs):

    """

    Here we invert the data, now for each movie we will have the names of critics and their respective ratings

    This inversion will help us determine similar movies, just like how we determined similar users

    """

    result = {}

    for person in prefs:

        for item in prefs[person]:

            result.setdefault(item, {})

            result[item][person] = prefs[person][item]



    return result
def calculateSimilarItems(prefs, n=10):

    """

    We compute here item-similarity matrix, which stores the similarity between items

    """

    result = {}

    itemPrefs = transformPrefs(prefs)



    for item in itemPrefs:

        scores = topMatches(itemPrefs, item, n, similarity=sim_euclidean)

        result[item] = scores



    return result
def getRecommendedItems(prefs, itemMatch, user):

    """

    Here we are performing item-based recommendation

    We use the item-similarity matrix to calculate the weighted sum of items similar to those already rated by the user

    The pre-computed item-similarity matrix saves time compared to user-based recommendation

    """

    userRatings = prefs[user]

    score = {}

    totalSim = {}



    for (item, rating) in userRatings.items():



        for (similarity, item2) in itemMatch[item]:



            if item2 in userRatings:

                continue



            score.setdefault(item2, 0)

            score[item2] += rating * similarity



            totalSim.setdefault(item2, 0)

            totalSim[item2] += similarity



    rankings = [(scores / totalSim[item2], item2) for item2, scores in score.items()]

    rankings.sort()

    rankings.reverse()

    return rankings
def loadMovieLens():

    movies = {}

    moviesData = pd.read_csv("../input/movie-lens-small-latest-dataset/movies.csv")

    for i in range(0, moviesData.shape[0]):

        movies[moviesData['movieId'][i]] = moviesData['title'][i]

    



    prefs = {}

    ratingsData = pd.read_csv("../input/movie-lens-small-latest-dataset/ratings.csv")

    for i in range(0, ratingsData.shape[0]):

        prefs.setdefault(ratingsData['userId'][i], {})

        prefs[ratingsData['userId'][i]][movies[ratingsData['movieId'][i]]] = float(ratingsData['rating'][i])

        

    return prefs