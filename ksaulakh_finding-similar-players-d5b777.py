import math

import sqlite3 as lite

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def get_data(sql_command):

	con = lite.connect("../input/database.sqlite")

	data = pd.read_sql(sql_command,con)

	con.close()

	return data



#We will try to find similar players according to a large list of stats

sqlcommand = "SELECT  A1.player_name NAME, height, weight, overall_rating, potential, preferred_foot, attacking_work_rate, defensive_work_rate, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, marking, standing_tackle, sliding_tackle, gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes FROM Player A1 INNER JOIN Player_Attributes A2 ON A1.player_api_id = A2.player_api_id WHERE A2.date LIKE '2015%'"

#Note that we are taking only rows updated in 2015

data = get_data(sqlcommand)

print ("Total rows 2015:", str(len(data)))

print ("Unique players 2015:", str(len(data["NAME"].unique())))



#I'm dropping rows which have 1 or more null values.

data = data.dropna()

unique = data["NAME"].unique() 

print ("Total rows after dropna:", str(len(data)))

print ("Unique players after dropna", str(len(data["NAME"].unique())))
unique = data["NAME"].unique()

dic = {}

for name in unique:

	row = data[data["NAME"]==name].iloc[[0]]

	name = row.iloc[0]["NAME"]

	height = row.iloc[0]["height"]

	weight = row.iloc[0]["weight"]

	potential= row.iloc[0]["potential"]

	crossing= row.iloc[0]["crossing"]

	finishing= row.iloc[0]["finishing"]

	heading_accuracy= row.iloc[0]["heading_accuracy"]

	short_passing= row.iloc[0]["short_passing"]

	volleys= row.iloc[0]["volleys"]

	dribbling= row.iloc[0]["dribbling"]

	curve= row.iloc[0]["curve"]

	free_kick_accuracy= row.iloc[0]["free_kick_accuracy"]

	long_passing= row.iloc[0]["long_passing"]

	ball_control= row.iloc[0]["ball_control"]

	acceleration= row.iloc[0]["acceleration"]

	sprint_speed= row.iloc[0]["sprint_speed"]

	agility= row.iloc[0]["agility"]

	reactions= row.iloc[0]["reactions"]

	balance= row.iloc[0]["balance"]

	shot_power= row.iloc[0]["shot_power"]

	jumping= row.iloc[0]["jumping"]

	stamina= row.iloc[0]["stamina"]

	strength= row.iloc[0]["strength"]

	long_shots= row.iloc[0]["long_shots"]

	aggression= row.iloc[0]["aggression"]

	interceptions= row.iloc[0]["interceptions"]

	positioning= row.iloc[0]["positioning"]

	vision= row.iloc[0]["vision"]

	penalties=row.iloc[0]["penalties"]

	marking= row.iloc[0]["marking"]

	standing_tackle= row.iloc[0]["standing_tackle"]

	sliding_tackle= row.iloc[0]["sliding_tackle"]

	gk_diving=row.iloc[0]["gk_diving"]

	gk_handling= row.iloc[0]["gk_handling"]

	gk_kicking= row.iloc[0]["gk_kicking"]

	gk_positioning= row.iloc[0]["gk_positioning"]

	gk_reflexes = row.iloc[0]["gk_reflexes"]

	vector = (height, weight,

              potential, crossing, finishing, heading_accuracy, short_passing, volleys,

              dribbling, curve, free_kick_accuracy, long_passing, ball_control,

              acceleration, sprint_speed, agility, reactions, balance, shot_power,

              jumping, stamina, strength, long_shots, aggression, interceptions,

              positioning, vision, penalties, marking, standing_tackle, sliding_tackle,

              gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes)



	dic[name] = vector

    

print ("Final number of players:", len(dic))
#Method that returns the squared sum's root of a given vector

def squared_sum (vector):

	suma_cuadrados = sum(x**2 for x in vector)

	return math.sqrt(suma_cuadrados)



#Method that returns the sum of the product (element by element) of two given vectors

def sum_product (vector1, vector2):

	return sum(vector1[i]*vector2[i] for i in range(len(vector1)))



#Method that returns an array containing tree tuples (similar_player_name, cosine_distance_value)

def cos_dista (asked_player):

	vector_player1 = list(dic[asked_player])

	sqr_sum_player1 = squared_sum(vector_player1)

	similars = {}

	for player in dic:

		if player!= asked_player:

			vector_player2 = list(dic[player])

			sqr_sum_player2 = squared_sum(vector_player2)

			product = sum_product(vector_player1,vector_player2)

			cosine_distance = product/(sqr_sum_player1*sqr_sum_player2)

			if cosine_distance>0.8:

				similars[player]=cosine_distance

			else:

				pass

		else:

			pass

	similars_names = sorted(similars, key=similars.get, reverse=True)

	result = []

	cont = 0

	for similar in similars_names:

		if cont >4:

			break

		cosine_dist = similars[similar]

		result.append((similar,cosine_dist))

		cont +=1



	return result



similars_cd = cos_dista("Cristiano Ronaldo")

print ("======================COSINE DISTANCE========================")

print ("First: ", similars_cd[0][0], ". Cosine Distance: ",similars_cd[0][1])

print ("Second: ", similars_cd[1][0], ". Cosine Distance: ",similars_cd[1][1])

print ("Third: ", similars_cd[2][0], ". Cosine Distance: ",similars_cd[2][1])

print ("Fourth: ", similars_cd[3][0], ". Distance: ",similars_cd[3][1])

print ("Fifth: ", similars_cd[4][0], ". Distance: ",similars_cd[4][1])
#Method that returns the manhattan distance of two given vectors

def manhattan_distance(vector1,vector2):

        return sum(abs(vector1[i]-vector2[i]) for i in range(len(vector1)))

                

##Method that returns an array containing tree tuples (similar_player_name, manhattan_distance_value)

def man_dist (asked_player):

        vector_player1 = list(dic[asked_player])

        similars = {}

        for player in dic:

                if player!= asked_player:

                        vector_player2 = list(dic[player])

                        distance = manhattan_distance(vector_player1,vector_player2)

                        similars[player]=distance

                else:

                        pass

        similars_names = sorted(similars, key=similars.get, reverse=False)

        result = []

        cont = 0

        for similar in similars_names:

                if cont >4:

                        break

                man_dist = similars[similar]

                result.append((similar,man_dist))

                cont +=1 

        return result

    

similars_md = man_dist("Cristiano Ronaldo")

print ("======================MANHATTAN DISTANCE========================")

print ("First: ", similars_md[0][0], ". Distance: ",similars_md[0][1])

print ("Second: ", similars_md[1][0], ". Distance: ",similars_md[1][1])

print ("Third: ", similars_md[2][0], ". Distance: ",similars_md[2][1])

print ("Fourth: ", similars_md[3][0], ". Distance: ",similars_md[3][1])

print ("Fifth: ", similars_md[4][0], ". Distance: ",similars_md[4][1])
#Method that returns the euclidean distance of two given vectors

def euclidean_distance (vector1,vector2):

        return math.sqrt(sum((vector1[i]-vector2[i])**2 for i in range(len(vector1))))



#Method that returns an array containing tree tuples (similar_player_name, euclidean_distance_value)

def euc_dist(asked_player):

        vector_player1 = list(dic[asked_player])

        similars = {}

        for player in dic:

                if player!= asked_player:

                        vector_player2 = list(dic[player])

                        distance = euclidean_distance(vector_player1,vector_player2)

                        similars[player]=distance

                else:

                        pass

        similars_names = sorted(similars, key=similars.get, reverse=False)

        result = []

        cont = 0

        for similar in similars_names:

                if cont >4:

                        break

                euc_dist = similars[similar]

                result.append((similar,euc_dist))

                cont +=1 

        return result

    

similars_ed = euc_dist("Cristiano Ronaldo")

print ("======================EUCLIDEAN DISTANCE========================")

print ("First: ", similars_ed[0][0], ". Distance: ",similars_ed[0][1])

print ("Second: ", similars_ed[1][0], ". Distance: ",similars_ed[1][1])

print ("Third: ", similars_ed[2][0], ". Distance: ",similars_ed[2][1])

print ("Fourth: ", similars_ed[3][0], ". Distance: ",similars_ed[3][1])

print ("Fifth: ", similars_ed[4][0], ". Distance: ",similars_ed[4][1])