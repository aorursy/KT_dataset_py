import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from surprise import Dataset,Reader

from surprise.model_selection import train_test_split



from surprise import KNNWithMeans

from surprise import accuracy



from surprise import SVD

from surprise import accuracy



from sklearn.model_selection import train_test_split as tt_split

import os

os.listdir('../input/amazon-product-reviews')
input_data=pd.read_csv('../input/amazon-product-reviews/ratings_Electronics (1).csv',names=['CustomerID','ItemID','Rating','Timestamp'])

input_data.head()
input_data.info()
input_data.shape
input_data['CustomerID'].value_counts()
input_data['ItemID'].value_counts()
data=input_data.groupby('CustomerID').filter(lambda x : len(x) > 100)

data.head()
data.shape
sns.distplot(data['Rating'])
data.reset_index(inplace=True)

data.drop(columns=['index'],inplace=True)

data.head()
#Since the timestamp column is not needed, we can drop it
data.drop(columns='Timestamp',inplace=True)
#Unique items in DF

print("The no of unique items in the data is", len(data['ItemID'].unique()))



#Unique Customers in DF

print("The no of unique customers in the data is", len(data['CustomerID'].unique()))
#Active customers - Those who given more no of ratings

data['CustomerID'].value_counts().head()
reader = Reader(rating_scale=(1, 5))
surp_data=Dataset.load_from_df(data[['CustomerID','ItemID','Rating']],reader)

surp_data
trainset,testset =train_test_split(surp_data, test_size=0.3,random_state=1)
print(type(testset))

print(type(trainset))



#Raw ids are normal data. the Raw ids are mapped to inner ids

#trainset contain these inner ids
#From the above link we see that .ur represent user ratings



user_ratings=trainset.ur

type(user_ratings)
for keys in user_ratings.keys():

    print(keys)
#For first user

#These are inner ids



user_ratings[0]
# However the ids are the inner ids and not the raw ids

# raw ids can be obatined as follows



#Convert a user inner id to a raw id.

print(trainset.to_raw_uid(0))



#Convert an item inner id to a raw id.

print(trainset.to_raw_iid(0))
knn_model = KNNWithMeans(k=10,sim_options={'name':'cosine' , 'user_based':False})

knn_model.fit(trainset)



#Item Item similarity matrix has been created now
len(testset)
#Evaluation on testset



test_pred_knn=knn_model.test(testset)



# compute RMSE

accuracy.rmse(test_pred_knn)
#Lets predict for 11th one



test_pred_knn[10]
#Convert result to DF



test_pred_df = pd.DataFrame(test_pred_knn)

test_pred_df
#was_impossible=false: are only calculated



testset_new = trainset.build_anti_testset()

len(testset_new)
#Lets fetch top 5 values



testset_new[0:5]
predictions = knn_model.test(testset_new)
predictions_df = pd.DataFrame([[x.uid,x.iid,x.est] for x in predictions])
predictions_df.head()
predictions_df.columns = ["CustomerID","ItemID","Est_rating"]

predictions_df.sort_values(by = ["CustomerID","ItemID", "Est_rating"],ascending=False,inplace=True)
predictions_df.head()
#Representing top 5 Recommendations for each Customers





top_5_recos = predictions_df.groupby("CustomerID").head(5).reset_index(drop=True)

top_5_recos
#No of ratings for each Item



item_group = data.groupby(['ItemID']).agg({'Rating' : 'count'}).reset_index()

item_group.head()
#Rating's total



grouped_sum = item_group['Rating'].sum()

print(grouped_sum)



#Thus we have a rating sum of 43309 
item_group['Percentage'] = item_group['Rating'].div(grouped_sum)*100
item_group.sort_values(['Rating'], ascending = False)
train_data, test_data = train_test_split(data, test_size = 0.3, random_state = 1)
#Class for Popularity based Recommender System model

class popularity_recommender_py():

    def __init__(self):

        self.train_data = None

        self.user_id = None

        self.item_id = None

        self.popularity_recommendations = None

        

    #Create the popularity based recommender system model

    def create(self, train_data, user_id, item_id):

        self.train_data = train_data

        self.user_id = user_id

        self.item_id = item_id



        #Get a count of user_ids for each unique song as recommendation score

        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()

        train_data_grouped.rename(columns = {'CustomerID': 'score'},inplace=True)

    

        #Sort the songs based upon recommendation score

        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])

    

        #Generate a recommendation rank based upon score

        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        

        #Get the top 10 recommendations

        self.popularity_recommendations = train_data_sort.head(10)



    #Use the popularity based recommender system model to

    #make recommendations

    def recommend(self, user_id):    

        user_recommendations = self.popularity_recommendations

        

        #Add user_id column for which the recommendations are being generated

        user_recommendations['CustomerID'] = user_id

    

        #Bring user_id column to the front

        cols = user_recommendations.columns.tolist()

        cols = cols[-1:] + cols[:-1]

        user_recommendations = user_recommendations[cols]

        

        return user_recommendations
pm = popularity_recommender_py()
pm.create(train_data,'CustomerID','ItemID')
customers = data['CustomerID'].unique()

len(customers)
items = data['ItemID'].unique()

len(items)
#Find Recommendation for a particular Customer



cust_id=customers[7]

pm.recommend(cust_id)





#This represents the top 10 items recommended for the customer A3PD8JD9L4WEII
#Class for Item similarity based Recommender System model

class item_similarity_recommender_py():

    def __init__(self):

        self.train_data = None

        self.user_id = None

        self.item_id = None

        self.cooccurence_matrix = None

        self.songs_dict = None

        self.rev_songs_dict = None

        self.item_similarity_recommendations = None

        

    #Get unique items (songs) corresponding to a given user

    def get_user_items(self, user):

        user_data = self.train_data[self.train_data[self.user_id] == user]

        user_items = list(user_data[self.item_id].unique())

        

        return user_items

        

    #Get unique users for a given item (song)

    def get_item_users(self, item):

        item_data = self.train_data[self.train_data[self.item_id] == item]

        item_users = set(item_data[self.user_id].unique())

            

        return item_users

        

    #Get unique items (songs) in the training data

    def get_all_items_train_data(self):

        all_items = list(self.train_data[self.item_id].unique())

            

        return all_items

        

    #Construct cooccurence matrix

    def construct_cooccurence_matrix(self, user_songs, all_songs):

            

        ####################################

        #Get users for all songs in user_songs.

        ####################################

        user_songs_users = []        

        for i in range(0, len(user_songs)):

            user_songs_users.append(self.get_item_users(user_songs[i]))

            

        ###############################################

        #Initialize the item cooccurence matrix of size 

        #len(user_songs) X len(songs)

        ###############################################

        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

           

        #############################################################

        #Calculate similarity between user songs and all unique songs

        #in the training data

        #############################################################

        for i in range(0,len(all_songs)):

            #Calculate unique listeners (users) of song (item) i

            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]

            users_i = set(songs_i_data[self.user_id].unique())

            

            for j in range(0,len(user_songs)):       

                    

                #Get unique listeners (users) of song (item) j

                users_j = user_songs_users[j]

                    

                #Calculate intersection of listeners of songs i and j

                users_intersection = users_i.intersection(users_j)

                

                #Calculate cooccurence_matrix[i,j] as Jaccard Index

                if len(users_intersection) != 0:

                    #Calculate union of listeners of songs i and j

                    users_union = users_i.union(users_j)

                    

                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))

                else:

                    cooccurence_matrix[j,i] = 0

                    

        

        return cooccurence_matrix



    

    #Use the cooccurence matrix to make top recommendations

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):

        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        

        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.

        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])

        user_sim_scores = np.array(user_sim_scores)[0].tolist()

 

        #Sort the indices of user_sim_scores based upon their value

        #Also maintain the corresponding score

        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

    

        #Create a dataframe from the following

        columns = ['CustomerID', 'ItemID', 'score', 'rank']

        #index = np.arange(1) # array of numbers for the number of samples

        df = pd.DataFrame(columns=columns)

         

        #Fill the dataframe with top 10 item based recommendations

        rank = 1 

        for i in range(0,len(sort_index)):

            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:

                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]

                rank = rank+1

        

        #Handle the case where there are no recommendations

        if df.shape[0] == 0:

            print("The current user has no item for training the item similarity based recommendation model.")

            return -1

        else:

            return df

 

    #Create the item similarity based recommender system model

    def create(self, train_data, user_id, item_id):

        self.train_data = train_data

        self.user_id = user_id

        self.item_id = item_id



    #Use the item similarity based recommender system model to

    #make recommendations

    def recommend(self, user):

        

        ########################################

        #A. Get all unique songs for this user

        ########################################

        user_songs = self.get_user_items(user)    

            

        print("No. of unique items for the user: %d" % len(user_songs))

        

        ######################################################

        #B. Get all unique items (songs) in the training data

        ######################################################

        all_songs = self.get_all_items_train_data()

        

        print("no. of unique items in the training set: %d" % len(all_songs))

         

        ###############################################

        #C. Construct item cooccurence matrix of size 

        #len(user_songs) X len(songs)

        ###############################################

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        

        #######################################################

        #D. Use the cooccurence matrix to make recommendations

        #######################################################

        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

                

        return df_recommendations

    

    #Get similar items to given items

    def get_similar_items(self, item_list):

        

        user_songs = item_list

        

        ######################################################

        #B. Get all unique items (songs) in the training data

        ######################################################

        all_songs = self.get_all_items_train_data()

        

        print("no. of unique items in the training set: %d" % len(all_songs))

         

        ###############################################

        #C. Construct item cooccurence matrix of size 

        #len(user_songs) X len(songs)

        ###############################################

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        

        #######################################################

        #D. Use the cooccurence matrix to make recommendations

        #######################################################

        user = ""

        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

         

        return df_recommendations

item_model = item_similarity_recommender_py()
item_model.create(train_data,'CustomerID','ItemID')
#Find recommendation for User with id 5



custo_id = customers[5]
user_items = item_model.get_user_items(custo_id)
print('--------------------------------------------------------')

print("Training data items for the user userId: %s:" %custo_id)

print('--------------------------------------------------------')

for user_item in user_items:

    print(user_item)

    

print('--------------------------------------------------------')

print("Recommendation process is going on:" )

print('--------------------------------------------------------')



#Recommend items for the user using personalized model

item_model.recommend(custo_id)