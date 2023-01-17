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



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews['country'].value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews['price']-reviews['price'].mean()



q4.check()
#q4.hint()

#q4.solution()


bargain_wine =reviews.loc[(reviews['points']/reviews['price']).idxmax(),'title']



q5.check()
#q5.hint()

#q5.solution()


tropCount = 0

fruitCount = 0



for desc in reviews.description:

    tropCount = tropCount + int('tropical' in desc)

    fruitCount = fruitCount + int('fruity' in desc)





descriptor_counts = pd.Series([tropCount,fruitCount],index=['tropical','fruity'])



#q6.hint()

#q6.solution()
star_list = []

for i in range(len(reviews)):

    if(reviews.loc[i,'country'] == 'Canada'):

        star_list.append(3)

    elif(reviews.loc[i,'points'] >= 95):

        star_list.append(3)

    elif(reviews.loc[i,'points'] >= 85):

        star_list.append(2)

    else:

        star_list.append(1)

        

    



star_ratings = pd.Series(star_list)

star_ratings



q7.check()
#q7.hint()

#q7.solution()