import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import random



pd.set_option('mode.chained_assignment', None)



# Note that there are no NANs in these data; '?' is

# used when there is missing information

accepts = pd.read_csv('../input/chefmozaccepts.csv')

cuisine = pd.read_csv('../input/chefmozcuisine.csv')

hours = pd.read_csv('../input/chefmozhours4.csv')

parking = pd.read_csv('../input/chefmozparking.csv')

geo = pd.read_csv('../input/geoplaces2.csv') 

usercuisine = pd.read_csv('../input/usercuisine.csv')

payment = pd.read_csv('../input/userpayment.csv')

profile = pd.read_csv('../input/userprofile.csv')

rating = pd.read_csv('../input/rating_final.csv')
res_all = np.concatenate((accepts.placeID, cuisine.placeID, 

                          hours.placeID, parking.placeID, geo.placeID))

res_all = np.sort( np.unique(res_all) ) # All the placeID's



print("There are {} restaurants.".format(len(res_all)))
user_all = np.concatenate((usercuisine.userID, payment.userID, profile.userID))

user_all = np.sort( np.unique(user_all) ) # All the userID's



print("There are {} users.".format(len(user_all)))
rating.head(10) # There are three types of ratings
print("There are {} restaurants with ratings.".format(len(rating.placeID.unique())))

print("There are {} users who gave ratings.".format(len(rating.userID.unique())))
overall_rating = pd.DataFrame( np.zeros((len(res_all),len(user_all)))-1.0, 

                              columns=user_all, index=res_all )

food_rating = overall_rating.copy()

service_rating = overall_rating.copy() 



for r, u, o, f, s in zip(rating.placeID, rating.userID, rating.rating, rating.food_rating, 

                         rating.service_rating):

    overall_rating.loc[r,u] = o

    food_rating.loc[r,u] = f

    service_rating.loc[r,u] = s



# This tells us whether a restaurant-user pair has a rating. 0 means No and 1 means Yes.

review = pd.DataFrame( np.zeros(overall_rating.shape), columns=user_all, index=res_all)

review[overall_rating >= 0] = 1
cuisine.head(10)
# use dummy variables for different cuisine categories of the restaurants

res_cuisine = pd.get_dummies(cuisine,columns=['Rcuisine'])



# remove duplicate restaurant ID's. 

# A restaurant with multiple cuisine categories would have multiple columns equal 1

res_cuisine = res_cuisine.groupby('placeID',as_index=False).sum()
parking.head(10)
# use dummy variables for different cuisine categories of the restaurants

res_parking = pd.get_dummies(parking,columns=['parking_lot'])



# remove duplicate restaurant ID's. 

# A restaurant with multiple parking options would have multiple columns equal 1

res_parking = res_parking.groupby('placeID',as_index=False).sum()
geo.columns.values
geo.head()
# These are the ones that I think might be relevant

res_features = geo[['placeID','alcohol','smoking_area','other_services','price','dress_code',

                         'accessibility','area']]



df_res = pd.DataFrame({'placeID': res_all})

df_res = pd.merge(left=df_res, right=res_cuisine, how="left", on="placeID")

df_res = pd.merge(left=df_res, right=res_parking, how="left", on="placeID")

df_res = pd.merge(left=df_res, right=res_features, how="left", on="placeID")
# The placeID's for the 130 restaurants with ratings

res_rated = res_all[np.sum(review,axis=1) > 0] 



# tells us whether a restaurant-user pair has a rating. 0 means No and 1 means Yes.

R = review.loc[res_rated].values  # shape = (130,138)



# Now these have a shape of (130, 138) too

Y_overall = overall_rating.loc[res_rated].values

Y_food  = food_rating.loc[res_rated].values

Y_service = service_rating.loc[res_rated].values



# select the indices of "df_res" where a restaurant has ratings

index = [x in res_rated for x in df_res['placeID'].values] #np.array()



# restaurant features for the 130 restaurants with ratings

X = df_res.loc[index, :].reset_index(drop=True)



X.isnull().sum() # all the NANs are from cuisine 
# fill all NANs with 0

X = X.fillna(0) 



# drop a feature if the entire column are 0

features_to_drop = X.columns.values[np.sum(X,axis=0) == 0] 

X = X.drop(features_to_drop, axis=1)



# drop placeID

X = X.drop(['placeID'], axis=1)



# There are the restaurant features we'll explore

X.columns.values 
profile.columns.values
user_info = profile[['smoker','drink_level','transport','budget','dress_preference']] 



print(user_info.smoker.value_counts())

print('\n')

print(user_info.drink_level.value_counts())

print('\n')

print(user_info.transport.value_counts())

print('\n')

print(user_info.budget.value_counts())

print('\n')

print(user_info.dress_preference.value_counts())
user_info.transport = user_info.transport.replace({'public':'no car', 'on foot':'no car'})

user_info.dress_preference = user_info.dress_preference.replace({'elegant':'formal'})
# Calculate the mean rating for each restaurant

def GetMean(Y,R):



    Y = Y*R

    mean =  (np.sum(Y, axis=1)/np.sum((R == 1.0), axis=1)).reshape(Y.shape[0],1)

    return mean



Y_overall_mean = GetMean(Y_overall,R)

Y_food_mean = GetMean(Y_food,R)

Y_service_mean = GetMean(Y_service,R)
# This is the function I'll use to plot and print the mean ratings of different 

# groups of restaurants based on different values of a given feature

def plot_mean_rating(df,rotate=False):

    

    n = df.shape[1]

    columns = df.columns.values

    

    if n > 1:

        y_overall = [ Y_overall_mean[df[i] == 1].mean() for i in columns ]

        y_food = [ Y_food_mean[df[i] == 1].mean() for i in columns ]

        y_service = [ Y_service_mean[df[i] == 1].mean() for i in columns ] 

        y = pd.DataFrame({'overall':y_overall, 'food':y_food, 'service':y_service},

                         columns=['overall','food','service'],index=columns) 

        ticks = columns

        

    else:

        values = df[columns[0]].unique()

        values = values[values != '?']

        y_overall = [ Y_overall_mean[df[columns[0]] == i].mean() for i in values ]

        y_food = [ Y_food_mean[df[columns[0]] == i].mean() for i in values ]

        y_service = [ Y_service_mean[df[columns[0]] == i].mean() for i in values ] 

        y = pd.DataFrame({'overall':y_overall, 'food':y_food, 'service':y_service},

                         columns=['overall','food','service'],index=values) 

        ticks = values

   

    fig = plt.figure()

    plt.plot(range(y.shape[0]),y['overall'],'-o',c='k',label='overall')

    plt.plot(range(y.shape[0]),y['food'],'-o',c='r',label='food')   

    plt.plot(range(y.shape[0]),y['service'],'-o',c='b',label='service')

    plt.xticks(range(y.shape[0]),ticks,fontsize=13)

    if rotate: plt.xticks(rotation=40)

    plt.yticks(fontsize=13) 

    if n == 1: plt.xlabel(columns[0],fontsize=15)

    plt.ylabel('mean rating',fontsize=15)

    plt.legend(fontsize=15,frameon=False)

    plt.show()

    

    print(y)
# This function is similar to the one above, but this time the result is split

# into different groups of users as well

def plot_mean_rating_split(df,userinfo,rotate=False):

    

    n = df.shape[1]

    columns = df.columns.values

    

    cases = userinfo.unique()

    cases = cases[cases != '?']

    

    num = len(cases)

    y = {}

    

    if n > 1:

        

        for i in range(num):

            

            index = (userinfo == cases[i])

            R_case = np.zeros(R.shape)

            R_case[:,index] = R[:,index]



            Y_overall_case = GetMean(Y_overall,R_case)

            Y_food_case = GetMean(Y_food,R_case)

            Y_service_case = GetMean(Y_service,R_case)

        

            isnan = np.isnan(Y_overall_case).reshape(-1)

            y_overall = [Y_overall_case[(df[j] == 1) & 

                        (isnan == False)].mean() for j in columns]

            y_food = [Y_food_case[(df[j] == 1) & 

                     (isnan == False)].mean() for j in columns]

            y_service = [Y_service_case[(df[j] == 1) & 

                        (isnan == False)].mean() for j in columns]

            

            y[cases[i]] = pd.DataFrame({'overall':y_overall, 'food':y_food, 'service':

                          y_service}, columns=['overall','food','service'],index=columns)

            

        ticks = columns



     

    else:

        

        for i in range(num):

            

            values = df[columns[0]].unique()

            values = values[values != '?']

            

            index = (userinfo == cases[i])

            R_case = np.zeros(R.shape)

            R_case[:,index] = R[:,index]



            Y_overall_case = GetMean(Y_overall,R_case)

            Y_food_case = GetMean(Y_food,R_case)

            Y_service_case = GetMean(Y_service,R_case)

        

            isnan = np.isnan(Y_overall_case).reshape(-1)

            y_overall = [Y_overall_case[(df[columns[0]] == j) & 

                        (isnan == False)].mean() for j in values]

            y_food = [Y_food_case[(df[columns[0]] == j) & 

                     (isnan == False)].mean() for j in values]

            y_service = [Y_service_case[(df[columns[0]] == j) & 

                        (isnan == False)].mean() for j in values]

                        

            y[cases[i]] = pd.DataFrame({'overall':y_overall, 'food':y_food, 'service':

                          y_service}, columns=['overall','food','service'],index=values)



        ticks = values

   



    f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True, figsize=(24,6))

 

    color = ['k','r','b'] 



    for i in range(num):

        ax1.plot(range(len(ticks)),y[cases[i]]['overall'],'-o',c=color[i],label=cases[i])

        ax2.plot(range(len(ticks)),y[cases[i]]['food'],'-o',c=color[i],label=cases[i])

        ax3.plot(range(len(ticks)),y[cases[i]]['service'],'-o',c=color[i],label=cases[i])

    

    ax1.set_title('overall',fontsize=20)

    ax2.set_title('food',fontsize=20)

    ax3.set_title('service',fontsize=20)



    ax1.tick_params(labelsize=16)

    ax2.tick_params(labelsize=16)

    ax3.tick_params(labelsize=16)

    

    if rotate:

        ax1.set_xticks(range(len(ticks)))

        ax1.set_xticklabels(ticks, rotation=40)

        ax2.set_xticks(range(len(ticks)))

        ax2.set_xticklabels(ticks, rotation=40)

        ax3.set_xticks(range(len(ticks)))

        ax3.set_xticklabels(ticks, rotation=40)

    else:

        plt.xticks(range(len(ticks)),ticks)

                           

    if n == 1: 

        ax2.set_xlabel(columns[0],fontsize=20)

    

    ax1.set_ylabel('mean rating',fontsize=20)

    

    plt.legend(fontsize=20,frameon=False)

    plt.show()

    

    return y
X.price.value_counts()
user_info.budget.value_counts()
# mean rating as a function of price

plot_mean_rating(X[['price']])
# mean rating as a function of price, split into three groups of consumers

# with low, medium, and high budget

y = plot_mean_rating_split(X[['price']],user_info.budget)



print('low budget:')

print(y['low'])

print('\nmedium budget:')

print(y['medium'])

print('\nhigh budget:')

print(y['high'])
columns = ['parking_lot_none','parking_lot_public', 'parking_lot_valet parking','parking_lot_yes']

X[columns].sum()
X[columns].sum().sum()
user_info.transport.value_counts()
# mean rating as a function of parking option

plot_mean_rating(X[columns], rotate=True)
# mean rating as a function of parking option, split into car owners and

# non car owners

y = plot_mean_rating_split(X[columns],user_info.transport,rotate=True)



print('no car:')

print(y['no car'])

print('\ncar owner:')

print(y['car owner'])
X.price[X['parking_lot_valet parking']==1].value_counts()
X.price[X['parking_lot_public']==1].value_counts()
X.price[X['parking_lot_none']==1].value_counts()
X.smoking_area.value_counts()
user_info.smoker.value_counts()
# mean rating as a function of smoking area

plot_mean_rating(X[['smoking_area']])
# mean rating as a function of smoking area, split into smokers and

# non smokers

y = plot_mean_rating_split(X[['smoking_area']],user_info.smoker)



print('false:')

print(y['false'])

print('\ntrue:')

print(y['true'])
X.alcohol.value_counts()
user_info.drink_level.value_counts()
# mean rating as a function of alcohol

plot_mean_rating(X[['alcohol']])
# mean rating as a function of smoking alcohol, split into abstemious, 

# casual and social drinkers

y = plot_mean_rating_split(X[['alcohol']],user_info.drink_level)



print('abstemious:')

print(y['abstemious'])

print('\ncasual drinker:')

print(y['casual drinker'])

print('\nsocial drinker:')

print(y['social drinker'])
X.price[X.alcohol == 'No_Alcohol_Served'].value_counts()
X.price[X.alcohol == 'Wine-Beer'].value_counts()
X.price[X.alcohol == 'Full_Bar'].value_counts()
X.other_services.value_counts()
# mean rating as a function of other_services

plot_mean_rating(X[['other_services']])
X.price[X.other_services == 'variety'].value_counts()
X.price[X.other_services == 'Internet'].value_counts()
X.price[X.other_services == 'none'].value_counts()
X.dress_code.value_counts()
# mean rating as a function of dress code

plot_mean_rating(X[['dress_code']])
# mean rating as a function of smoking alcohol, split into different 

# groups of consumers with different dress preferences

y = plot_mean_rating_split(X[['dress_code']],user_info.dress_preference)



print('no preference:')

print(y['no preference'])

print('\ninformal:')

print(y['informal'])

print('\nformal:')

print(y['formal'])
X.price[X.dress_code == 'formal'].value_counts()
X.price[X.dress_code == 'informal'].value_counts()
X.price[X.dress_code == 'casual'].value_counts()
X.accessibility.value_counts()
# mean rating as a function of dress accessibility

plot_mean_rating(X[['accessibility']])
X.price[X.accessibility == 'partially'].value_counts()
X.price[X.accessibility == 'no_accessibility'].value_counts()
X.price[X.accessibility == 'completely'].value_counts()
X.area.value_counts()
# mean rating as a function of area

plot_mean_rating(X[['area']])
# the number of restaurants for each cuisine type 

X.iloc[:,:23].sum()
# Besides Mexican, Cafeteria and Fast_Food, group the other types into 

# the following 5 types



X['Rcuisine_Bar_Pub'] = np.zeros(X.shape[0])

index = ((X.Rcuisine_Bar == 1) | (X.Rcuisine_Bar_Pub_Brewery == 1))

X.loc[index,'Rcuisine_Bar_Pub'] = 1



X['Rcuisine_Asian'] = np.zeros(X.shape[0])

index = ((X.Rcuisine_Chinese == 1) | (X.Rcuisine_Japanese == 1) | (X.Rcuisine_Vietnamese == 1))

X.loc[index,'Rcuisine_Asian'] = 1



X['Rcuisine_Western'] = np.zeros(X.shape[0])

index = ((X.Rcuisine_Armenian == 1) | (X.Rcuisine_Italian == 1) | (X.Rcuisine_Mediterranean == 1) 

         | (X.Rcuisine_Pizzeria == 1) | (X.Rcuisine_Seafood == 1))

X.loc[index,'Rcuisine_Western'] = 1



X['Rcuisine_American_Burgers'] = np.zeros(X.shape[0])

index = ((X.Rcuisine_American == 1) | (X.Rcuisine_Burgers == 1))

X.loc[index,'Rcuisine_American_Burgers'] = 1



X['Rcuisine_Others'] = np.zeros(X.shape[0])

index = (((X.Rcuisine_Bakery == 1) | (X["Rcuisine_Breakfast-Brunch"] == 1) | (X["Rcuisine_Cafe-Coffee_Shop"] == 1) 

        | (X.Rcuisine_Contemporary == 1) | (X.Rcuisine_Family == 1) | (X.Rcuisine_Game == 1) 

        | (X.Rcuisine_International == 1) | (X.Rcuisine_Regional == 1)))

X.loc[index,'Rcuisine_Others'] = 1
print("Number of restaurants for each type")

print("Mexican: {}".format(int(X.Rcuisine_Mexican.sum())))

print("American/Burgers: {}".format(int(X.Rcuisine_American_Burgers.sum())))

print("Asian: {}".format(int(X.Rcuisine_Asian.sum()))) 

print("Bar/Pub: {}".format(int(X.Rcuisine_Bar_Pub.sum()))) 

print("Cafeteria: {}".format(int(X.Rcuisine_Cafeteria.sum()))) 

print("Fast Food: {}".format(int(X.Rcuisine_Fast_Food.sum()))) 

print("Others: {}".format(int(X.Rcuisine_Others.sum())))

print("Western: {}".format(int(X.Rcuisine_Western.sum())))
columns = ['Rcuisine_Mexican','Rcuisine_American_Burgers', 'Rcuisine_Asian','Rcuisine_Bar_Pub', 'Rcuisine_Cafeteria',

           'Rcuisine_Fast_Food','Rcuisine_Others','Rcuisine_Western']

plot_mean_rating(X[columns],rotate=True)
X_cuisine = X[['Rcuisine_Mexican','Rcuisine_American_Burgers','Rcuisine_Asian','Rcuisine_Bar_Pub','Rcuisine_Cafeteria',

           'Rcuisine_Fast_Food','Rcuisine_Western','Rcuisine_Others']]



X_cuisine['Y_food'] = Y_food_mean

X_cuisine['Y_overall'] = Y_overall_mean

X_cuisine['Y_service'] = Y_service_mean
print(X_cuisine.corr()['Y_overall'][:-3])

print('\n')

print(X_cuisine.corr()['Y_food'][:-3])

print('\n')

print(X_cuisine.corr()['Y_service'][:-3])

print('\n')