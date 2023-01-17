import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
reviews.info()
median_points = reviews.points.median()

print(median_points)



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()

print(countries)



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()

print(reviews_per_country)



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
price_mean = reviews.price.mean()

print(price_mean)



centered_price = reviews.price.map(lambda p: p - price_mean)

print(centered_price)



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
df_ratio = reviews[['title', 'points', 'price']]

print(df_ratio.head())



bargain_idx = (reviews['points']/reviews['price']).idxmax()    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.idxmax.html

bargain_wine = reviews.loc[bargain_idx, 'title']



print(bargain_wine)



# Check your answer

q5.check()
#q5.hint()

#q5.solution()
tropical = reviews['description'].map(lambda x: 'tropical' in x)

fruity = reviews['description'].map(lambda x: 'fruity' in x)

print(tropical.sum())

print(fruity.sum())
tropical = reviews['description'].map(lambda x: 'tropical' in x)

fruity = reviews['description'].map(lambda x: 'fruity' in x)



descriptor_counts = pd.Series([tropical.sum(),fruity.sum()],index=['tropical','fruity'])



print(descriptor_counts)



# Check your answer

q6.check()
#q6.hint()

#q6.solution()
def ratings(row):

    if(row['country']=='Canada'):

        return 3

    elif(row['points']>=95):

        return 3

    elif(row['points']>=85 and row['points']<95):

        return 2

    else:

        return 1

star_ratings = reviews.apply(ratings,axis='columns')



# Check your answer

q7.check()
#q7.hint()

#q7.solution()