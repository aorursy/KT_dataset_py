import pandas as pd

user = pd.read_csv("../input/peoplerecommender.csv")

user.head()



user.Store.value_counts()
userid = user.ID.copy()

user_brands = dict(user.groupby('ID').Store.apply(lambda x:list(x)))

#Print top 2 dictionary items

dict(list(user_brands.items())[0:2])
def jac_sim (x, y):

    x = set(x)

    y = set(y)

    return len(x & y) / float(len( x | y))
#Example

print(jac_sim(user_brands[80139],user_brands[80135])) # This is for user 80135

# This is for user 80139, and it should give 1 as we are calcualting similarity with itself.

print(jac_sim(user_brands[80139],user_brands[80139])) 


test_user = ['Target','Old Navy', 'Banana Republic', 'H&M']

jac_list = {}

for userid, brand in user_brands.items():

    jac_list[userid] = jac_sim(test_user,brand)

#Print top 2 dictionary items

dict(list(jac_list.items())[0:2])

    
top_users = sorted(jac_list.items(), key = lambda x: x[1], reverse = True)[:5]

top_users
#K Most similar users

recommendedbrands = set()

for user in top_users:

    for brand in user_brands[user[0]]:

        if brand not in test_user:

            recommendedbrands.add(brand)

    
recommendedbrands
def getRecommendedBrands(userid):

    userbrand = user_brands[userid]

    jac_list = {}

    for userid, brand in user_brands.items():

        jac_list[userid] = jac_sim(userbrand,brand)

    top_users = sorted(jac_list.items(), key = lambda x: x[1], reverse = True)[:5]

    recommendedbrands = set()

    for user in top_users:

        for brand in user_brands[user[0]]:

            if brand not in test_user:

                recommendedbrands.add(brand)

    return recommendedbrands

    
#Testing the method for the user id 80010

getRecommendedBrands(80010)

 
user_vs_recommenders = {}

for userid, brand in user_brands.items():

    user_vs_recommenders[userid] = getRecommendedBrands(userid)



#Print top 10 dictionary items

dict(list(user_vs_recommenders.items())[0:10])