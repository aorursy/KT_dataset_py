# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import math

import operator
ratings = {

'Ishaan': {

'C++': 5.00, 

'Data Structures': 4.29,

'Algorithms': 5.00, 

'Java': 2.

},

'Riya': {

'C': 4.89, 

'C++': 4.93 , 

'Java': 1.87,

'English': 1.33,

},

'Anubhav': {

'Algorithms': 5.0, 

'Data Structures': 4.89,

'C++': 4.78, 

'Java': 1.33,

'C': 4.77,

'Spanish':  1.25,

'German': 1.72

},

'Charul':{

'Machine Learning': 5.00,

'Python': 4.74,

'Artificial Intelligence': 4.5,

'C++': 1.24,

'C': 1.34,

'R': 4.56,

'Deep Learning': 4.32

},

'Manika': {

'R': 4.02, 

'Python': 5.00,

},

'Nipunika': {

'Data Structures': 4.07, 

'Algorithms': 4.29, 

'C++': 5.00, 

'Java':  4.89,

'Machine Learning': 2.54,

'R': 1.60

},

'Tanishka': {

'Big Data': 4.80, 

'Data Mining': 4.61,

'R': 4.26 

},

'Bhavya': {

'Data Mining': 5.0,

'Machine Learning': 5.0,

'Data Structures': 1.22,

'Deep Learning': 4.34

},

'Tanisi': {

'Algorithms': 4.98,

'Data Structures': 4.42,

'C++': 4.63,

'Big Data': 1.12,

'Machine Learning': 2.16

},'Utkarsh': {

'Algorithms': 5.0, 

'C': 4.84,

'Data Structures': 4.42,

'C++': 4.63,

'Big Data': 1.12,

'Machine Learning': 2.16

},

'Tanmay': {

'Machine Learning': 3.78, 

'Android': 4.96,

'C++': 1.04,

'Data Structures': 1.03

},

'Vishal': {

'Algorithms': 5.00, 

'Data Structures': 5.0, 

'R': 1.24,

'Python': 2.02

},

'Kunal': {

'Big Data': 5.0, 

'Machine Learning': 4.87,

'Algorithms': 1.14,

'Java': 4.00

},

'Raman': {

'Machine Learning': 2.98,

'Python': 3.93,

'C++': 1.37

},

'Aditi': {

'Java': 5.0, 

'Android': 5.0,

'C++': 1.07,

'C': 0.63

},

'Anant': {

'C++': 4.89, 

'Algorithms': 5.0,

'Data Structures': 4.87,

'Android': 1.32

}

}



def get_common_subjects(criticA,criticB):

    return [sub for sub in ratings[criticA] if sub in ratings[criticB]]

get_common_subjects('Ishaan','Riya')

def get_ratings(criticA,criticB):

    common_subjects = get_common_subjects(criticA,criticB)

    return [(ratings[criticA][sub], ratings[criticB][sub]) for sub in common_subjects]
get_ratings('Ishaan','Riya')

def euclidean_distance(points):

    squared_diffs = [(point[0] - point[1]) ** 2 for point in points]

    summed_squared_diffs = sum(squared_diffs)

    distance = math.sqrt(summed_squared_diffs)

    return distance
def similarity(rating):

    return 1/ (1 + euclidean_distance(rating))
def get_critic_similarity(criticA, criticB):

    ratings = get_ratings(criticA,criticB)

    return similarity(ratings)
get_critic_similarity('Ishaan','Riya')

def recommend_subjects(critic, num_suggestions):

    similarity_scores = [(get_critic_similarity(critic, other), other) for other in ratings if other != critic]

    # Get similarity Scores for all the critics

    similarity_scores.sort() 

    similarity_scores.reverse()

    similarity_scores = similarity_scores[0:num_suggestions]



    recommendations = {}

    # Dictionary to store recommendations

    for similarity, other in similarity_scores:

        reviewed = ratings[other]

        # Storing the review

        for sub in reviewed:

            if sub not in ratings[critic]:

                weight = similarity * reviewed[sub]

                # Weighing similarity with review

                if sub in recommendations:

                    sim, weights = recommendations[sub]

                    recommendations[sub] = (sim + similarity, weights + [weight])

                    # Similarity of movie along with weight

                else:

                    recommendations[sub] = (similarity, [weight])

                    

    for recommendation in recommendations:

        similarity, sub = recommendations[recommendation]

        recommendations[recommendation] = sum(sub) / similarity

        # Normalizing weights with similarity



    sorted_recommendations = sorted(recommendations.items(), key=operator.itemgetter(1), reverse=True)

    #Sorting recommendations with weight

    return sorted_recommendations
recommend_subjects('Riya',4)
