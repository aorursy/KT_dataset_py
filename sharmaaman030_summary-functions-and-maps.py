import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews['points'].median()





q1.check()
#q1.hint()

#q1.solution()
countries = reviews['country'].unique()

type(countries)



q2.check()
#q2.hint()

#q2.solution()
cp=reviews['country']

Countries_apperead=pd.Series(cp)



reviews_per_country = Countries_apperead.value_counts()

reviews_per_country



q3.check()
#q3.hint()

#q3.solution()
mean_price=reviews['price'].mean()

pr=reviews['price']

centered_price = pr-mean_price

centered_price



q4.check()
#q4.hint()

#q4.solution()
point=reviews['points']

pr=reviews['price']

bargain_win = (point/pr)







bargain_id=bargain_win.idxmax()

bargain_wine = reviews.loc[bargain_id, 'title']



















q5.check()
#q5.hint()

#q5.solution()
tropi=reviews['description'].str.contains('tropical').astype(int)

t=tropi.value_counts()[1]





fru=reviews['description'].str.contains('fruity').astype(int)

f=fru.value_counts()[1]





    

descriptor_counts =pd.Series([t,f],index=['tropical', 'fruity'])

descriptor_counts



q6.check()



#q6.hint()

#q6.solution()
def stars(row):

    if row.country == 'Canada':

        return 3

    elif row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(stars, axis=1)

star_ratings

# points=reviews['points']

# s_95=points>=95

# s_85_1=points>=85

# s_85_2=points<95

# s_85=points[s_85_1 & s_85_2]



# def star(x):

#     if s_95:

#         return 3

#     if s_85:

#         return 2

#     else:

#         return 1

    

# star_ratings=points.apply(star)

# star_ratings









































#q7.check()
#q7.hint()

#q7.solution()