import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews.points.median()

q1.check()
# Uncomment the line below to see a solution
#q1.solution()
countries = reviews.country.unique()
countries
q2.check()
q2.solution()
reviews_per_country = reviews.country.value_counts()

q3.check()
#q3.solution()
mean = reviews.price.mean()
centered_price = reviews.price - mean

q4.check()
#q4.solution()
chu  = (reviews.points / reviews.price).idxmax()  # 找到点价比最大的索引
bargain_wine = reviews.loc[chu,'title']  # 根据索引和列名找到值
# idxmax() 函数返回值醉的索引
q5.check()
q5.solution()
tropical = reviews.description.map(lambda desc:'tropical' in desc).sum()
fruity = reviews.description.map(lambda desc:'fruity' in desc ).sum()
descriptor_counts = pd.Series([tropical, fruity],index = ['tropical', 'fruity'])
print(descriptor_counts)
q6.check()
q6.solution()
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >=85:
        return 2
    else:
        return 1
    
# 第三步，组Series
star_ratings = reviews.apply(stars,axis = 'columns' )

q7.check()
q7.solution()