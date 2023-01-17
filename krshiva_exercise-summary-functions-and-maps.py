import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



q2.check()
q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews['price']-reviews['price'].mean()



q4.check()
q4.hint()

q4.solution()
label = (reviews.points/reviews.price).idmax()

bargain_wine = reviews.loc[label, 'title'] 

q5.check()
q5.hint()

q5.solution()
l1 = 'tropical'

l2 = 'fruity'

descriptor_counts_tropival= reviews.description.isna(l1).value_counts()

descriptor_counts_fruity= reviews.description.isna(l2).value_counts()

descriptor_counts = df.DataFrame(['tropical','fruity'],{'tropical':descriptor_counts_tropival,'fruity': descriptor_counts_fruity}, index = 'count')



q6.check()
q6.hint()

q6.solution()
ratings = reviews.points

def row(ratings):

        if reviews.country =='Canada':

            star = 3

        elif ratings>=95:

            star = 3

        elif ratings >=85:

            star = 2

        else:

            star = 1

    

star_ratings = reviews.apply(row)

#q7.check()
#q7.hint()

q7.solution()