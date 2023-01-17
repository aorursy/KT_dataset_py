import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews.points.median()

# Check your answer
q1.check()
median_points
#q1.hint()
#q1.solution()
#countries = reviews.country.value_counts()
countries = reviews.country.unique()

# Check your answer
q2.check()
countries
#q2.hint()
#q2.solution()
reviews_per_country = reviews.country.value_counts()

# Check your answer
q3.check()
reviews_per_country
#q3.hint()
#q3.solution()
#centered_price = reviews.price.map(lambda p : p - reviews.price.mean())
reviews.price - reviews.price.mean()

# Check your answer
q4.check()
centered_price
#q4.hint()
#q4.solution()
ratio_point_price_max = (reviews.points/reviews.price).max()
s = reviews.title.loc[(reviews.Ratio_points_to_price == ratio_point_price_max)]
bargain_wine = s.iloc[0]
#Normaly two wines have the best bargain, it have the highest points-to-price ratio!!!
#bargain_wine1 = s.iloc[0]
#bargain_wine2 = s.iloc[1]
# Check your answer
q5.check()
bargain_wine

#q5.hint()
#q5.solution()
# count_tropical = reviews.description.str.count("tropical").sum()
# count_fruity = reviews.description.str.count("fruity").sum()
check_tropical = [1 if "tropical" in des else 0 for des in reviews.description]
check_fruity = [1 if "fruity" in des else 0 for des in reviews.description]
count_tropical = sum(check_tropical)
count_fruity = sum(check_fruity)
descriptor_counts = pd.Series([count_tropical, count_fruity], index=['tropical', 'fruity'])
# Check your answer
q6.check()
descriptor_counts
#q6.hint()
#q6.solution()
def rating(row):
    if row.country == 'Canada' or row.points >= 95:
        row.points = 3
    elif row.points >= 85 and row.points < 95:
        row.points = 2
    else:
        row.points = 1        
    return row

df = reviews.apply(rating, axis='columns')
star_ratings = df.points

# Check your answer
q7.check()
star_ratings
#q7.hint()
#q7.solution()