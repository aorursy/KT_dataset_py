# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import pandas as pd  

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
user1 = [10,20,30]

user2 = [100,200,300]

print (np.corrcoef(x = user1,y = user2,rowvar = False))
user1 = [10,20,40]

user3 = [30,5,2]

print (np.corrcoef(x = user1,y = user3,rowvar = False))
#First of all, we have to modify columns (using this method we are deleting the first row, but we won't

#use this row anyway)

df = pd.read_csv("../input/steam-200k.csv")

df.columns = ["user_id","game","activity","hours","unkknown"]



df_users = df[["user_id","game"]][df["activity"]=="play"].groupby("user_id", axis = 0).count().reset_index()

df_users["game"][df_users["game"]<10].hist()

plt.title("How many different games are played by how many users?")

plt.show()

print ("Top users:",len(df_users["user_id"][df_users["game"]>2].unique()))
user_games_dict = df[["user_id","game"]][df["activity"]=="play"].groupby("user_id", axis = 0).count().to_dict()

#Once we got a dict from the entire dataframe, we create a new dict with only "true gamers"

true_gamers_dict = {}

for user in user_games_dict["game"]:

	if user_games_dict["game"][user]>2: #if he plays more than two games,

		true_gamers_dict[user] = user_games_dict["game"][user] #he is a "true gamer"

        

#Let's def a function which will return 1 if user is "true gamer" and 0 if not.    

def top_gamer (x):

	if x in true_gamers_dict:

		return 1

	else:

		return 0

#Now, we create a new column on the dataframe, mapping user's id in order to take only "true gamers"    

df["gamer"] = df["user_id"].map(lambda x: top_gamer(x))

#Finally, we can work with our own dataframe

df_gamers = df[df["gamer"]==1][df["activity"]=="play"]

#And print the number of different users we have, just to be sure

print ("Top users:",len(df_gamers["user_id"].unique()))
df_recom = df_gamers[["user_id","game","hours"]] #taking only neccessary columns



vectors = {}                                     #this dict object will contain the vectors

                                                                

for index in df_recom.index:                     #we map the dataset in order to get our own dict

    row = df_recom.loc[index, :]

    user_id = row["user_id"]

    game = row["game"]

    hours = row["hours"]

    if user_id not in vectors:

        vectors[user_id] = {}

    else:

        pass

    vectors[user_id][game] = hours

    

user_example =  103804924       #Taking an user_id

#NOTE: I chose this user because he have played a large list of games.

print (vectors[103804924])   #Let's print the vector of the user that we will use to show recoms
user_example =  103804924



def corr_users (vectors, random_id):  #params are: dict containing vectors and the user we will recom games

	best = []  #list saving tuples (corr,{possible recoms: hours played})

	for user in vectors:              #for every user in the dict



		possible_recom = []           #possible games to recom

		matched_games = []			  #games played in common



		vector_1 = vectors[random_id] #our user's vector as dict

		vector_2 = vectors[user]      #another user's vector as dict



		given_vector = []			  #user's vector of hours played for the matched games

		matched_vector = []			  #another user's vector of hours played for the matched games



		if user != random_id:	      #for the rest of the users that are not the given user

									  #we are matching up the games in common



			for game in vector_2:	  #for each game played by an strange user



				if game in vector_1:  #if our user plays it too

					matched_games.append(game) #we append it as matched game



				else:				  #if not

					possible_recom.append(game) #it's a possible recommendation



			for game in matched_games:  #Once we have matched_games



				given_vector.insert(0,vector_1[game])    #we construct the vectors with the number

				matched_vector.insert(0,vector_2[game])  #of hours that both users played the matched

                                                         #games



		if len(matched_games)>4:      #if we have enough games matched (we can play with this number

                                      #in order to get better results)

                                      #we calculate similarity

			corr = np.corrcoef(x = given_vector,y = matched_vector,rowvar = True)[0][1] 



			dic = {}                          #we need a dict for possible recoms

			for game in possible_recom:

				dic[game]= vector_2[game] 

                

			best.append((corr,dic))#and we append the tuple with neccesary data

            

            #print ("Matched vectors:")

			#print (matched_vector,given_vector)              #we can print every pair of matched vectors 

			#print ("Correlation: "+str(corr*100)[:5]+"%")      #and their correlation level





		else:						   #if we have not enough games matched, just try with the next user

			pass

        

	#When we finally got best matches

	print ("You were matched up with this number of gamers:",len(best))



	if len(best)==0: #If there are no matches

		print ("Warning: No matches")



	else:            #If there are matches:

		print ("Coincidence levels are: ")              #we can print every correlation levels we found

		for i in best:

			print (str(i[0]))

		

		best_positive = sorted(best,key=lambda x: x[0], reverse = True)[0]	#The most  correlated tuple

		second_positive = sorted(best,key=lambda x: x[0], reverse = True)[1] #The second most correlated tuple

		second_recom = max(second_positive[1])      #We recommend the most played game of the second most similar user

		first_recom = max(best_positive[1])      #We recommend the most played game of the most similar user 

		recoms = (first_recom,second_recom)



		print ("We recommend : ")

		print ("-"+recoms[0]+"  Coincidence: "+str(best_positive[0]*100)[:5]+"%")

		print ("-"+recoms[1]+"  Coincidence: "+str(abs(second_positive[0]*100))[:5]+"%")



#Finally, we can call the function and show the results

correlated_users = corr_users(vectors,user_example)