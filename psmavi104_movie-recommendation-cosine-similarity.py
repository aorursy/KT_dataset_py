# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from typing import List

import math, random

from collections import defaultdict, Counter



Vector = List[float]

def dot(v: Vector, w: Vector) -> float:

    """Computes v_1 * w_1 + ... + v_n * w_n"""

    assert len(v) == len(w), "vectors must be same length"



    return sum(v_i * w_i for v_i, w_i in zip(v, w))
users_interests = [

    [" Hereditary", "  The Favourite", "Blindspotting", " Spider-Man: Into the Spider-Verse", "Mission: Impossible - Fallout", " Sorry to Bother You", "Roma"],

    ["Annihilation", "Avengers: Infinity War", "Roma", "Blindspotting", " Under the Silver Lake"],

    ["Upgrade", "wrong turn", "No Mercy", "La la Land", "bahubaali 2", "BPM"],

    ["Whose Streets? ", "Upgrade", "bahubaali", "Nocturama", "demons"],

    ["manhunt", "Nocturama", "River", "Brawl in Cell Block 99 "],

    ["Upgrade", "Whose Streets? ", " Spider-Man: Into the Spider-Verse", "Hangover", " Hangover 2", "Leech"],

    ["bahubaali", "demons", "The man who knew infinity", "The Big Sick"],

    ["manhunt", "wrong turn", "abcd", "Pale"],

    ["Pale", "deep learning", "  The Favourite", "Lame"],

    [" Hereditary", " Spider-Man: Into the Spider-Verse", "I", "  The Favourite"],

    ["bahubaali", "Whose Streets? ", "bahubaali 2"],

    ["Hangover", "deep learning", "Lame", "demons"],

    ["BPM", "Whose Streets? ", "Upgrade"],

    ["Guardians of the Galaxy Vol. 2 ", "Blindspotting", " Under the Silver Lake", "Columbus", "Avengers: Infinity War"],

    ["Brawl in Cell Block 99 ", "Nocturama", "The Post "]

]
popular_interests = Counter(interest

                            for user_interests in users_interests

                            for interest in user_interests).most_common()



print("___________Popular Interests :________________")

print(popular_interests)



def most_popular_new_interests(user_interests, max_results=5):

    suggestions = [(interest, frequency)

                   for interest, frequency in popular_interests

                   if interest not in user_interests]

    return suggestions[:max_results]



print("___________Most Popular New Interests___________")

print("already like:", ["Annihilation", "Avengers: Infinity War", "Roma", "Blindspotting", " Under the Silver Lake"])

print(most_popular_new_interests(["Annihilation", "Avengers: Infinity War", "Roma", "Blindspotting", " Under the Silver Lake"]))

print()

print("___________already like:___________", ["Whose Streets? ", "Upgrade", "bahubaali", "Nocturama", "demons"])

print(most_popular_new_interests(["Whose Streets? ", "Upgrade", "bahubaali", "Nocturama", "demons"]))

print()

#

# user-based filtering

#



def cosine_similarity(v, w):

    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))



unique_interests = sorted(list({ interest

                                 for user_interests in users_interests

                                 for interest in user_interests }))



def make_user_interest_vector(user_interests):

    """given a list of interests, produce a vector whose i-th element is 1

    if unique_interests[i] is in the list, 0 otherwise"""

    return [1 if interest in user_interests else 0

            for interest in unique_interests]



user_interest_matrix = list(map(make_user_interest_vector, users_interests))



user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)

                      for interest_vector_j in user_interest_matrix]

                     for interest_vector_i in user_interest_matrix]



def most_similar_users_to(user_id):

    pairs = [(other_user_id, similarity)                      # find other

             for other_user_id, similarity in                 # users with

                enumerate(user_similarities[user_id])         # nonzero

             if user_id != other_user_id and similarity > 0]  # similarity



    return sorted(pairs,                                      # sort them

                  key=lambda pair: pair[1],                   # most similar

                  reverse=True)                               # first





def user_based_suggestions(user_id, include_current_interests=False):

    # sum up the similarities

    suggestions = defaultdict(float)

    for other_user_id, similarity in most_similar_users_to(user_id):

        for interest in users_interests[other_user_id]:

            suggestions[interest] += similarity



    # convert them to a sorted list

    suggestions = sorted(suggestions.items(),

                         key=lambda pair: pair[1],

                         reverse=True)



    # and (maybe) exclude already-interests

    if include_current_interests:

        return suggestions

    else:

        return [(suggestion, weight)

                for suggestion, weight in suggestions

                if suggestion not in users_interests[user_id]]
interest_user_matrix = [[user_interest_vector[j]

                         for user_interest_vector in user_interest_matrix]

                        for j, _ in enumerate(unique_interests)]



interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)

                          for user_vector_j in interest_user_matrix]

                         for user_vector_i in interest_user_matrix]



def most_similar_interests_to(interest_id):

    similarities = interest_similarities[interest_id]

    pairs = [(unique_interests[other_interest_id], similarity)

             for other_interest_id, similarity in enumerate(similarities)

             if interest_id != other_interest_id and similarity > 0]

    return sorted(pairs,

                  key=lambda pair: pair[1],

                  reverse=True)



def item_based_suggestions(user_id, include_current_interests=False):

    suggestions = defaultdict(float)

    user_interest_vector = user_interest_matrix[user_id]

    for interest_id, is_interested in enumerate(user_interest_vector):

        if is_interested == 1:

            similar_interests = most_similar_interests_to(interest_id)

            for interest, similarity in similar_interests:

                suggestions[interest] += similarity



    suggestions = sorted(suggestions.items(),

                         key=lambda pair: pair[1],

                         reverse=True)



    if include_current_interests:

        return suggestions

    else:

        return [(suggestion, weight)

                for suggestion, weight in suggestions

                if suggestion not in users_interests[user_id]]


    print("User based similarity")

    print("most similar to 0")

    print(most_similar_users_to(0))



    print("Suggestions for 0")

    print(user_based_suggestions(0))

    print()



    print("Item based similarity")

    print("most similar to '  The Favourite'")

    print(most_similar_interests_to(0))

    print()



    print("suggestions for user 0")

    print(item_based_suggestions(0))