#Getting the dataset
cs_file_path = '../input/mm_master_demos.csv'
cs_data = pd.read_csv(cs_file_path)
#Defining the columns wich will be used
cols = ['map','round','winner_side']
#Selecting only the rows that contains the "map_to_test"
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
#Test function
def get_winners_percentage(map_to_test, cs_data):
    ct = 0.0
    terror = 0.0
    prevRound = 0
    
    #Using .iterrows to access each row
    for index, row in cs_data.iterrows():
        if (row['round'] != prevRound):
            prevRound = row['round']
            if (row['winner_side'] == 'Terrorist'):
               terror+=1
            else:
                ct+=1

    total = int(terror + ct)

    terror = float((terror/total) * 100)
    ct = float((ct/total) * 100)
import pandas as pd

#Test function
def get_winners_percentage(map_to_test, cs_data):
    ct = 0.0
    terror = 0.0
    prevRound = 0
    
    #Using .iterrows to access each row
    for index, row in cs_data.iterrows():
        if (row['round'] != prevRound):
            prevRound = row['round']
            if (row['winner_side'] == 'Terrorist'):
               terror+=1
            else:
                ct+=1

    total = int(terror + ct)

    terror = float((terror/total) * 100)
    ct = float((ct/total) * 100)

    print ("from "+ str(total) +" rounds on map '"+ map_to_test +"', the winning percentage was:")
    print ("Terrorist: %.2f" %terror,"%")
    print ("Counter Terrorist: %.2f" %ct,"%")
    print ("")


#Getting the dataset
cs_file_path = '../input/mm_master_demos.csv'
cs_data = pd.read_csv(cs_file_path)
#Defining the columns wich will be used
cols = ['map','round','winner_side']


#Tests
#Defining the map to analyze
map_to_test = 'de_dust2'
#Selecting only the rows that contains the "map_to_test"
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
#Calling the test function
get_winners_percentage(map_to_test, cs_data1)

#Test 2 with a different map
map_to_test = 'de_cache'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 3 with a different map
map_to_test = 'de_inferno'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 4 with a different map
map_to_test = 'de_mirage'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 5 with a different map
map_to_test = 'de_overpass'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 6 with a different map
map_to_test = 'de_train'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 7 with a different map
map_to_test = 'de_dust'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 8 with a different map
map_to_test = 'de_cbble'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)
