# import libraries for data exploration and basic statistical functions

import pandas as pd
import numpy as np
data = pd.read_csv('../input/HW1-data.csv')
data
# Top movies by mean score
means = data.iloc[:, 2:].mean().sort_values(ascending=False)
means
# Counts
counts = data.iloc[:, 2:].count()
counts.sort_values(ascending=False)
# Top movies by percentage of positive marks
counts_positive = data.iloc[:, 2:][data.iloc[:, 2:] >= 4].count()
counts_positive.sort_values(ascending=False)
(counts_positive / counts).sort_values(ascending=False)
# Percentage of people who watched Toy Story also watched...
associative_product = '1: Toy Story (1995)'

watched_product = data.iloc[:, 2:][data[associative_product].notnull()].count()
(watched_product / data[associative_product].count()).sort_values(ascending=False)
# Correlation between Toy Story and other movie ratings
data.iloc[:, 2:].corrwith(data[associative_product]).sort_values(ascending=False)
# Means separate by gender
gender_column_name = 'Gender (1 =F, 0=M)'
male_means = data.iloc[:, 2:][data[gender_column_name] == 0].mean()
female_means = data.iloc[:, 2:][data[gender_column_name] == 1].mean()
# Male means
male_means.sort_values(ascending=False)
#Female means
female_means.sort_values(ascending=False)
# Overall mean ratings 
male_average_mean = data.iloc[:, 2:][data[gender_column_name] == 0].sum().sum() / data.iloc[:, 2:][data[gender_column_name] == 0].count().sum()
female_average_mean = data.iloc[:, 2:][data[gender_column_name] == 1].sum().sum() / data.iloc[:, 2:][data[gender_column_name] == 1].count().sum()
print("Male avg. mean: {} Female avg. mean: {}".format(male_average_mean, female_average_mean))
# Movies that female users rate highest above male raters

(female_means - male_means).sort_values(ascending=False)
# Movies that male users rate highest above female raters

(male_means - female_means).sort_values(ascending=False)
# Positive (> 4) ratings by male

counts_positive_male = data.iloc[:, 2:][(data >= 4)][data[gender_column_name] == 0].count()
counts_positive_male.sort_values(ascending=False)
# Percentage of positive ratings by male

counts_male = data.iloc[:, 2:][data[gender_column_name] == 0].count()
percentage_positive_male = (counts_positive_male / counts_male)
percentage_positive_male.sort_values(ascending=False)
# Positive (> 4) ratings by female

counts_positive_female = data.iloc[:, 2:][(data >= 4)][data[gender_column_name] == 1].count()
counts_positive_female.sort_values(ascending=False)
# Percentage of positive ratings by female

counts_female = data.iloc[:, 2:][data[gender_column_name] == 1].count()
percentage_positive_female = (counts_positive_female / counts_female)
percentage_positive_female.sort_values(ascending=False)
# Female-male difference in the liking percentage

(percentage_positive_female - percentage_positive_male).sort_values(ascending=False)
# Male-female difference in liking percentage
(percentage_positive_male - percentage_positive_female).sort_values(ascending=False)
# Difference between the average rating overall

female_average_mean - male_average_mean
# importing raw data from excel file

raw_data = pd.read_excel("../input/cbf.xls")
raw_data
docs = raw_data.loc['doc1':'doc20', 'baseball':'family']
docs
user_ranks = raw_data.loc['doc1':'doc20', 'User 1':'User 2']
user_ranks.fillna(0, inplace=True)
user_ranks
user_profiles = np.array(docs).T @ np.array(user_ranks)
pd.DataFrame(user_profiles, docs.columns, user_ranks.columns)
user_preferences = np.matmul(np.array(docs), user_profiles)
updf = pd.DataFrame(user_preferences, docs.index, user_ranks.columns)
updf
updf.loc[:, 'User 1'].sort_values(ascending=False)
updf.loc[:, 'User 2'].sort_values(ascending=False)
normalized_docs = docs.div(docs.sum(axis=1).apply(np.sqrt), axis=0)
normalized_docs
normalized_profiles = np.matmul(np.array(normalized_docs).T, np.array(user_ranks))
pd.DataFrame(normalized_profiles, docs.columns, user_ranks.columns)
normalized_preferences = np.matmul(np.array(normalized_docs), normalized_profiles)
npdf = pd.DataFrame(normalized_preferences, docs.index, user_ranks.columns)
npdf
npdf.loc[:, 'User 1'].sort_values(ascending=False)
docs
DF = docs.sum(axis=0)
IDF = 1.0 / DF
np.array(IDF)
weighted_preferences = np.matmul(np.array(normalized_docs), np.multiply(np.array(normalized_profiles).T, np.array(IDF)).T)
pd.DataFrame(weighted_preferences, docs.index, user_ranks.columns)
# library for visualization
import seaborn
data = pd.read_excel("../input/data.xls")
data
# user correlations
correlations = pd.DataFrame(data.transpose(), data.columns, data.index).corr()
seaborn.heatmap(correlations)
# selecting 6 neighbors 

neighbours_3867 = correlations[3867].sort_values(ascending=False)[1:6]
recommendations = data.fillna(0)
recommendations.loc[neighbours_3867.index]
# calculations for top-5 movies

(recommendations.loc[neighbours_3867.index].multiply(
    neighbours_3867, axis=0).sum(axis=0) / (recommendations.iloc[:, :] != 0).multiply(
    neighbours_3867, axis=0).sum(axis=0)).sort_values(ascending=False)[:5]
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
ratings_raw_data = pd.read_excel('../input/iicf.xls')
data = pd.DataFrame(ratings_raw_data.iloc[0:20, 1:21])
data.index = ratings_raw_data.iloc[0:20,0]
data_means = data.mean(axis=0)
data
data.fillna(0, inplace=True)
similarities = pd.DataFrame(cosine_similarity(data.T, data.T), data.columns, data.columns)
seaborn.heatmap(similarities)
# Yeah, this probably can be done better with Pandas one-liners.

predictions = data.copy()
for user in data.index: #every user in index
    for movie in data.columns: #every movie for user
        mean = data_means[movie] #mean rate of this movie
        similar_movies = similarities[movie] # similar movies to this movie
        numerator = 0
        weights_sum = 0
        for sm in similar_movies.index: # for every similar movie
            weight = similar_movies[sm]
            rating = data.loc[user, sm]
            if weight > 0 and rating > 0: #which is non-negative (sim) and rated by user
                numerator += weight * (rating - mean)
                weights_sum += weight
        predictions.loc[user, movie] = mean + (numerator / weights_sum)
predictions.loc[755].sort_values(ascending=False)