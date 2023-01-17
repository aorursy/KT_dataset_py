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
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
m=reviews.price.mean()

centered_price = reviews.price-m



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
x=(reviews.points/reviews.price).idxmax()

bargain_wine=reviews.title[x]

# Check your answer

q5.check()
q5.hint()

q5.solution()

c=0

s=0



for x in range(len(reviews.index)):

    if 'tropical' in reviews.description[x]:

        c+=1

    if 'fruity' in reviews.description[x]:

        s+=1

    

descriptor_counts =pd.Series([c,s],index=['tropical','fruity']) 





# Check your answer

q6.check()
#q6.hint()

#q6.solution()
x=[]

for i in range(len(reviews.index)):

    if reviews.points[i]>=95:

        x.append(3)

    elif reviews.points[i]<95 and reviews.points[i]>=85:

        if reviews.country[i]=='Canada':

            x.append(3)

        else :

            x.append(2)

    elif reviews.points[i]<85:

        if reviews.country[i]=='Canada':

            x.append(3)

        else:

            x.append(1)

star_ratings = pd.Series(x)



# Check your answer

q7.check()
#q7.hint()

#q7.solution()