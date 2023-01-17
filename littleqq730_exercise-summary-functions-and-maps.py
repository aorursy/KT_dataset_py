import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews.points.median()
median_points
# Check your answer
q1.check()
median_points
#q1.hint()
#q1.solution()
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
q3.solution()
reviews.price_mean=reviews.price.mean()
centered_price = reviews.price.map(lambda p:p-reviews.price_mean)

# Check your answer
q4.check()
q4.hint()
#q4.solution()
bargin_idx = (reviews.points/reviews.price).idxmax()
bargin_wine = reviews.loc[bargin_idx,'title']
bargin_wine
bargain_wine_idx = (reviews.points/reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_wine_idx, "title"]
bargain_wine

# Check your answer
#q5.check()
dfk = pd.DataFrame({"A":[4, 5, 2, None], 
                   "B":[11, 2, None, 8],  
                   "C":[1, 8, 66, 4]}) 
dfk
#
dfk.idxmax( skipna = True)
q5.hint()
q5.solution()
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_idx

reviews.head()
descriptor_counts_fruity = reviews.description.map(lambda description : "fruity" in description).sum()
descriptor_counts_tropical = reviews.description.map(lambda description : "tropical" in description).sum()
descriptor_counts = pd.Series([descriptor_counts_tropical,descriptor_counts_fruity],index = ["tropical","fruity"])
q6.check()
descriptor_counts
descriptor_trop_counts = reviews.description.map(lambda description:"tropical" in description).sum()
descriptor_frui_counts = reviews.description.map(lambda description:"fruity" in description).sum()
descriptor_counts = pd.Series ([descriptor_trop_counts,descriptor_frui_counts],index=['tropical', 'fruity'])
# Check your answer
q6.check()
descriptor_counts
q6.hint()
q6.solution()
reviews.head()
def srars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >=95:
        return 3
    elif row.points >=85:
        return 2
    else:
        return 1
star_ratings = reviews.apply(srars,axis = 'columns')  
q7.check()

star_ratings
# Check your answer

q7.hint()
q7.solution()