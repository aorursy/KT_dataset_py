import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# check we have for the input dataset

print(os.listdir("../input"))
# "...may19.csv" is the latest one

fastfood_data = pd.read_csv('../input/fast-food-restaurants/Datafiniti_Fast_Food_Restaurants_May19.csv')
# shape of dataset

print("Dataset dimension:")

print(fastfood_data.shape)
# check the data type of the columns

print("Data types:")

print(fastfood_data.dtypes)
# take a peek

print("Top 10 rows:")

fastfood_data.head(10)
# the id, keys, sourceURLs, and websites are unique identifier

# we don't need them in our visualizations

fastfood_data = fastfood_data.drop(['id', 'keys', 'sourceURLs', 'websites'], axis=1)



# shape of dataset

print("Dimension after dropping columns:")

print(fastfood_data.shape)



# take a peek at the head to verify the drop is successful

fastfood_data.head()
# my personal reusable function for detecting missing data

def missing_value_describe(data):

    # check missing values in training data

    missing_value_stats = (data.isnull().sum() / len(data)*100)

    missing_value_col_count = sum(missing_value_stats > 0)

    missing_value_stats = missing_value_stats.sort_values(ascending=False)[:missing_value_col_count]

    print("Number of columns with missing values:", missing_value_col_count)

    if missing_value_col_count != 0:

        # print out column names with missing value percentage

        print("\nMissing percentage (desceding):")

        print(missing_value_stats)

    else:

        print("No misisng data!!!")

missing_value_describe(fastfood_data)
import matplotlib.pyplot as plt # import matplotlib for graphs
print("Number of unique restaurant:", fastfood_data['name'].nunique())
# top 20 restaurants recorded by count total

nameplot=fastfood_data['name'].value_counts()[:20].plot.bar(title='Top 20 mentioned restaurants')

nameplot.set_xlabel('name',size=20)

nameplot.set_ylabel('count',size=20)
import nltk
# # obtained unique names of restaurants

# restaurant_names = fastfood_data['name'].unique()



# # calculate similarity and record most-similar names together

# most_similar = []

# for i in range(len(restaurant_names)):

#     # temporary list to store the current similar words

#     temp_similar = [restaurant_names[i]]

    

#     # compare and save similar words

#     for j in range(len(restaurant_names)):

#         if restaurant_names[i] == restaurant_names[j]:

#             continue

#         if nltk.edit_distance(restaurant_names[i].lower(), restaurant_names[j].lower()) < 3:

#             temp_similar.append(restaurant_names[j])

            

#     # similar word(s) found

#     if len(temp_similar) > 1:

#         most_similar.append(temp_similar)

#     if i > 0 and i % 10 == 0:

#         print("index", i-10, "-", i, "checking finished| most similar size:", len(most_similar))

# print("similarity checking finished")



# # count number of similar words

# most_similar_word_count = 0

# for i in most_similar:

#     most_similar_word_count += len(i)

# print("size of the most similar list:", most_similar_word_count)

# most_similar
# let's remove the exceptions from the above most_similar's list

# since the list is small and we only need to do some minor changes, I will just do it manually

# to avoid running edit-distance again

most_similar_edited = [["Carl's Jr.", "Carl's Jr", 'Carls Jr'],

 ["McDonald's", "Mc Donald's", 'Mcdonalds', 'McDonalds'],

 ['Cook-Out', 'Cook Out', 'CookOut'],

 ["Steak 'n Shake",

  "STEAK 'N SHAKE",

  'Steak N Shake',

  'Steak n Shake',

  "Steak 'N Shake"],

 ['QDOBA Mexican Eats', 'Qdoba Mexican Eats'],

 ['Burger King', 'Burger King®'],

 ["Hardee's", 'Hardees'],

 ['Taco Time', 'TacoTime'],

 ["Arby's", 'Arbys'],

 ['Chick-fil-A', 'Chick-Fil-A', 'ChickfilA'],

 ['Subway', 'SUBWAY'],

 ['Kfc', 'KFC'],

 ["Jack's", 'Jacks'],

 ['Sonic Drive-In',

  'SONIC Drive-In',

  'SONIC Drive In',

  'Sonic DriveIn',

  'Sonic Drive-in'],

 ["Church's Chicken", 'Churchs Chicken'],

 ['Big Boys', 'Big Boy'],

 ['Dairy Queen', 'Dairy queen'],

 ['Guthries', "Guthrie's"],

 ['Chick-Fil-A', 'Chick-fil-A', 'ChickfilA'],

 ["Wendy's", 'Wendys'],

 ["Jimmy John's", 'Jimmy Johns'],

 ['Dairy Queen Grill Chill', 'Dairy Queen Grill & Chill'],

 ["Moe's Southwest Grill", 'Moes Southwest Grill'],

 ["Domino's Pizza", 'Dominos Pizza'],

 ["Rally's", 'Rallys'],

 ['Full Moon Bar-B-Que', 'Full Moon Bar B Que'],

 ["Guthrie's", 'Guthries'],

 ["McAlister's Deli", "Mcalister's Deli", 'McAlisters Deli'],

 ["Jason's Deli", 'Jasons Deli'],

 ['KFC', 'Kfc', 'KFC Kentucky Fried Chicken', 'KFC - Kentucky Fried Chicken'],

 ['Popeyes Louisiana Kitchen', "Popeye's Louisiana Kitchen"],

 ["Long John Silver's", 'Long John Silvers'],

 ['BLIMPIE', 'Blimpie'],

 ['Five Guys Burgers Fries', 'Five Guys Burgers & Fries'],

 ['SUBWAY', 'Subway'],

 ['Dairy Queen Grill & Chill', 'Dairy Queen Grill Chill'],

 ['Potbelly Sandwich Works', 'Pot Belly Sandwich Works'],

 ["Charley's Grilled Subs", 'Charleys Grilled Subs'],

 ["Jersey Mike's Subs", 'Jersey Mikes Subs'],

 ['In-N-Out Burger', 'InNOut Burger'],

 ["Culver's", "CULVER'S", 'Culvers'],

 ["Famous Dave's", 'Famous Daves'],

 ["Freddy's Frozen Custard Steakburgers",

  'Freddys Frozen Custard Steakburgers',

  "Freddy's Frozen Custard & Steakburgers"],

 ['Cook Out', 'Cook-Out', 'CookOut'],

 ['TacoTime', 'Taco Time'],

 ['Hooters', 'Roosters'],

 ['BurgerFi', 'Burgerfi'],

 ["Chen's Restaurant", "Chan's Restaurant"],

 ['Taco Del Mar', 'Taco del Mar'],

 ['SONIC Drive-In',

  'Sonic Drive-In',

  'SONIC Drive In',

  'Sonic DriveIn',

  'Sonic Drive-in'],

 ['Ciscos Taqueria', "Cisco's Taqueria"],

 ['China King', 'China Lin'],

 ["Bojangles' Famous Chicken 'n Biscuits",

  'Bojangles Famous Chicken n Biscuits'],

 ["Dominic's of New York", 'Dominics of New York'],

 ["Papa John's Pizza", 'Papa Johns Pizza'],

 ['Chanellos Pizza', 'Chanello’s Pizza'],

 ["Fazoli's", 'Fazolis'],

 ['Wing Street', 'Wingstreet'],

 ["George's Gyros Spot", "George's Gyros Spot 2"],

 ['Taco Johns', "Taco John's"],

 ['RUNZA', 'Runza'],

 ['Bru Burger Bar', 'Grub Burger Bar'],

 ["Taco John's", 'Taco Johns'],

 ["Bob's Burger Brew", "Bob's Burgers Brew", "Bob's Burgers Brew", "Bob's Burger Brew"],

 ['Best Burgers', 'Best Burger'],

 ['Burgermaster', 'Burger Master'],

 ["Dick's Drive-In", "DK's Drive-In"],

 ["Charley's Grill Spirits", "Charley's Grill & Spirits"],

 ['Tom Drive-in', "Tom's Drive-In"],

 ["Fox's Pizza Den", 'Foxs Pizza Den'],

 ["Mc Donald's", "McDonald's", 'Mcdonalds', 'McDonalds'],

 ['Taco CASA', 'Taco Casa'],

 ["Mcalister's Deli", "McAlister's Deli", 'McAlisters Deli'],

 ['Saras Too', "Sara's Too"],

 ['Backyard Burgers', 'Back Yard Burgers'],

 ["CULVER'S", "Culver's", 'Culvers'],

 ["Simple Simon's Pizza", 'Simple Simons Pizza'],

 ['China Sea', 'China Star', 'China Bear'],

 ["Dino's Drive In", "Dan's Drive In"],

 ["STEAK 'N SHAKE",

  "Steak 'n Shake",

  'Steak N Shake',

  'Steak n Shake',

  "Steak 'N Shake"],

 ['Stanfields Steak House', "Stanfield's Steakhouse"],

 ['Wingstreet', 'Wing Street'],

 ["Big Billy's Burger Joint", 'Big Billys Burger Joint'],

 ['Big Boy', 'Big Boys'],

 ["Frisch's Big Boy Restaurant", "1 Frisch's Big Boy Restaurant", 

  "40 Frisch's Big Boy Restaurant", "1 Frisch's Big Boy Restaurant",

  "90 Frisch's Big Boy Restaurant"],

 ['Fireplace Restaurant Lounge', 'Fireplace Restaurant & Lounge'],

 ["Carl's Jr", "Carl's Jr.", 'Carls Jr'],

 ["Rick's on the River", 'Ricks on the River'],

 ['Grub Burger Bar', 'Bru Burger Bar'],

 ["Franky's", "Grandy's"],

 ['Gyro X-Press', 'Gyro Express'],

 ['Dominos Pizza', "Domino's Pizza"],

 ["Pietro's Pizza Gallery of Games", "Pietro's Pizza & Gallery of Games"],

 ['Burrtio Amigos', 'Burrito Amigos'],

 ["Albee's Ny Gyros", "Albee's NY Gyros"],

 ['Gyro Stop', 'Gyro Spot'],

 ['Nicholas Restaurant', "Nicholas' Restaurant"],

 ['Mcdonalds', "McDonald's", "Mc Donald's", 'McDonalds'],

 ['Burgerfi', 'BurgerFi'],

 ["Ryan's", 'Ryans'],

 ['Taste of Buffalo Pizzeria', 'Taste Of Buffalo Pizzeria'],

 ['Bad Daddys Burger Bar', "Bad Daddy's Burger Bar"],

 ["Zaxby's", "Arby's"],

 ["Topper's Pizza", 'Toppers Pizza'],

 ['C J Drive In', 'C & J Drive In'],

 ['Full Moon Bar B Que', 'Full Moon Bar-B-Que'],

 ['China Lin', 'China King'],

 ["Raising Cane's Chicken Fingers", 'Raising Canes Chicken Fingers'],

 ["Mary's Pizza Shack", 'Marys Pizza Shack'],

 ['Peking Chinese Restaurants', 'Peking Chinese Restaurant'],

 ['Arbys', "Arby's"],

 ['SONIC Drive In',

  'Sonic Drive-In',

  'SONIC Drive-In',

  'Sonic DriveIn',

  'Sonic Drive-in'],

 ['Hardees', "Hardee's"],

 ['McDonalds', "McDonald's", "Mc Donald's", 'Mcdonalds'],

 ['Wendys', "Wendy's"],

 ['Papa Johns Pizza', "Papa John's Pizza"],

 ["George's Gyros Spot 2", "George's Gyros Spot"],

 ['ChickfilA', 'Chick-fil-A', 'Chick-Fil-A'],

 ['Rallys', "Rally's"],

 ['C & J Drive In', 'C J Drive In'],

 ['Steak N Shake',

  "Steak 'n Shake",

  "STEAK 'N SHAKE",

  'Steak n Shake',

  "Steak 'N Shake"],

 ["Popeye's Louisiana Kitchen", 'Popeyes Louisiana Kitchen'],

 ["DJ's Drive-In", "DK's Drive-In"],

 ["Dan's Drive In", "Dino's Drive In"],

 ['Best Burger', 'Best Burgers', 'Beef Burger'],

 ['Jimmy Johns', "Jimmy John's"],

 ['BaskinRobbins', 'Baskin-Robbins', 'Baskin Robbins'],

 ['Carls Jr', "Carl's Jr.", "Carl's Jr"],

 ['WG Grinders', 'Wg Grinders'],

 ['McAlisters Deli', "McAlister's Deli", "Mcalister's Deli"],

 ['Fazolis', "Fazoli's"],

 ['Marys Pizza Shack', "Mary's Pizza Shack"],

 ['Bojangles Famous Chicken n Biscuits',

  "Bojangles' Famous Chicken 'n Biscuits"],

 ['Jacks', "Jack's"],

 ["Hardee's/red Burrito", 'Hardees Red Burrito', "Hardee's/Red Burrito"],

 ['Captain Ds', "Captain D'S"],

 ['Mr Hero', 'Mr. Hero'],

 ["Chan's Restaurant", "Chen's Restaurant"],

 ['Ritters Frozen Custard', "Ritter's Frozen Custard"],

 ['Hot Dog on a Stick', 'Hot Dog On A Stick'],

 ['Jersey Mikes Subs', "Jersey Mike's Subs"],

 ['AW Restaurants',

  'Aw Restaurants',

  'AWRestaurants',

  'A W Restaurant',

  'AW Restaurant',

  'Jam Restaurants'],

 ['Long John Silvers', "Long John Silver's"],

 ["Rally's Hamburgers", 'Rallys Hamburgers'],

 ['HomeTown Buffet', 'Hometown Buffet'],

 ['Back Yard Burgers', 'Backyard Burgers'],

 ['Hardees Red Burrito', "Hardee's/red Burrito", "Hardee's/Red Burrito"],

 ["DK's Drive-In", "Dick's Drive-In", "DJ's Drive-In", "K's Drive In"],

 ['Baskin-Robbins', 'BaskinRobbins', 'Baskin Robbins'],

 ['Churchs Chicken', "Church's Chicken"],

 ['Blimpie', 'BLIMPIE'],

 ['Foxs Pizza Den', "Fox's Pizza Den"],

 ['Steak n Shake',

  "Steak 'n Shake",

  "STEAK 'N SHAKE",

  'Steak N Shake',

  "Steak 'N Shake"],

 ['Rallys Hamburgers', "Rally's Hamburgers"],

 ['Sonic DriveIn',

  'Sonic Drive-In',

  'SONIC Drive-In',

  'SONIC Drive In',

  'Sonic Drive-in'],

 ['Famous Daves', "Famous Dave's"],

 ['Beef Burger', 'Best Burger'],

 ['Dominics of New York', "Dominic's of New York"],

 ['Z-Pizza', 'zpizza'],

 ['KFC - Kentucky Fried Chicken', 'KFC Kentucky Fried Chicken'],

 ["Rockne's", 'Rocknes'],

 ["Hardee's/Red Burrito", "Hardee's/red Burrito", 'Hardees Red Burrito'],

 ['Aw Restaurants',

  'AW Restaurants',

  'AWRestaurants',

  'A W Restaurant',

  'AW Restaurant',

  'Jam Restaurants'],

 ['AWRestaurants', 'AW Restaurants', 'Aw Restaurants', 'AW Restaurant'],

 ["Hardee's Restaurant", "Hardee's Restaurants"],

 ["Hardee's Restaurants", "Hardee's Restaurant"],

 ["Stanfield's Steakhouse", 'Stanfields Steak House'],

 ['Dunkin Donuts', "Dunkin' Donuts"],

 ['Einstein Bros. Bagels', 'Einstein Bros Bagels'],

 ['Simple Simons Pizza', "Simple Simon's Pizza"],

 ['A W Restaurant', 'AW Restaurants', 'Aw Restaurants', 'AW Restaurant'],

 ['Einstein Bros Bagels', 'Einstein Bros. Bagels'],

 ['Roosters', 'Hooters'],

 ['Culvers', "Culver's", "CULVER'S"],

 ['Slice of Life', 'Slice Of Life'],

 ['Jasons Deli', "Jason's Deli"],

 ['Wg Grinders', 'WG Grinders'],

 ['Charleys Grilled Subs', "Charley's Grilled Subs"],

 ['Freddys Frozen Custard Steakburgers',

  "Freddy's Frozen Custard Steakburgers"],

 ['Moes Southwest Grill', "Moe's Southwest Grill"],

 ['CookOut', 'Cook-Out', 'Cook Out'],

 ['Peking Chinese Restaurant', 'Peking Chinese Restaurants'],

 ['InNOut Burger', 'In-N-Out Burger'],

 ["Nicholas' Restaurant", 'Nicholas Restaurant'],

 ['Chanello’s Pizza', 'Chanellos Pizza'],

 ['Ryans', "Ryan's"],

 ['Burger King®', 'Burger King'],

 ['Toppers Pizza', "Topper's Pizza"],

 ["Albee's NY Gyros", "Albee's Ny Gyros"],

 ['Qdoba Mexican Eats', 'QDOBA Mexican Eats'],

 ['Runza', 'RUNZA'],

 ['Slice Of Life', 'Slice of Life'],

 ['Mai-Tai Restaurant', 'Mai Tai Restaurant'],

 ['Gyro Express', 'Gyro X-Press'],

 ['zpizza', 'Z-Pizza'],

 ['Raising Canes Chicken Fingers', "Raising Cane's Chicken Fingers"],

 ['Rocknes', "Rockne's"],

 ['LL Hawaiian Barbecue', 'L L Hawaiian Barbecue', 'L L Hawaiian Barbeque'],

 ['Dairy queen', 'Dairy Queen'],

 ['Blakes Lotaburger', "Blake's Lotaburger"],

 ['Emidio & Sons Italian Restaurant', 'Emidio Sons Italian Restaurant'],

 ['Taste Of Buffalo Pizzeria', 'Taste of Buffalo Pizzeria'],

 ['L L Hawaiian Barbecue',

  'LL Hawaiian Barbecue',

  'L L Hawaiian Barbeque',

  'L & L Hawaiian Barbecue'],

 ['Killer Burgers', 'Killer Burger'],

 ["Steak 'N Shake",

  "Steak 'n Shake",

  "STEAK 'N SHAKE",

  'Steak N Shake',

  'Steak n Shake'],

 ['Burrito Amigos', 'Burrtio Amigos'],

 ["Zack's Hamburgers", "Jack's Hamburgers"],

 ['AW Restaurant',

  'AW Restaurants',

  'Aw Restaurants',

  'AWRestaurants',

  'A W Restaurant'],

 ['Jam Restaurants', 'AW Restaurants', 'Aw Restaurants'],

 ['Big Billys Burger Joint', "Big Billy's Burger Joint"],

 ['L L Hawaiian Barbeque', 'LL Hawaiian Barbecue', 'L L Hawaiian Barbecue'],

 ["Ritter's Frozen Custard", 'Ritters Frozen Custard'],

 ["Pietro's Pizza & Gallery of Games", "Pietro's Pizza Gallery of Games"],

 ["K's Drive In", "DK's Drive-In"],

 ['Killer Burger', 'Killer Burgers'],

 ["Dunkin' Donuts", 'Dunkin Donuts'],

 ['Farlows on the Water', "Farlow's On The Water"],

 ['Hometown Buffet', 'HomeTown Buffet'],

 ["Blake's Lotaburger", 'Blakes Lotaburger'],

 ["Jack's Hamburgers", "Zack's Hamburgers"],

 ["Cisco's Taqueria", 'Ciscos Taqueria'],

 ["Grandy's", "Franky's"],

 ["Farlow's On The Water", 'Farlows on the Water'],

 ["Bad Daddy's Burger Bar", 'Bad Daddys Burger Bar'],

 ['Baskin Robbins', 'BaskinRobbins', 'Baskin-Robbins'],

 ["Sara's Too", 'Saras Too'],

 ['T & L Hotdogs', 'T & L Hot Dogs'],

 ["Tom's Drive-In", 'Tom Drive-in'],

 ['Sonic Drive-in',

  'Sonic Drive-In',

  'SONIC Drive-In',

  'SONIC Drive In',

  'Sonic DriveIn'],

 ['Taco Casa', 'Taco CASA'],

 ['Emidio Sons Italian Restaurant', 'Emidio & Sons Italian Restaurant'],

 ['Fireplace Restaurant & Lounge', 'Fireplace Restaurant Lounge'],

 ['Mai Tai Restaurant', 'Mai-Tai Restaurant'],

 ['Ricks on the River', "Rick's on the River"],

 ['Taco del Mar', 'Taco Del Mar'],

 ['Five Guys Burgers & Fries', 'Five Guys Burgers Fries'],

 ['Mr. Hero', 'Mr Hero'],

 ["Captain D'S", 'Captain Ds'],

 ['Gyro Spot', 'Gyro Stop'],

 ["Charley's Grill & Spirits", "Charley's Grill Spirits"],

 ['Hot Dog On A Stick', 'Hot Dog on a Stick'],

 ['L & L Hawaiian Barbecue', 'L L Hawaiian Barbecue'],

 ['Pot Belly Sandwich Works', 'Potbelly Sandwich Works'],

 ['Burger Master', 'Burgermaster'],

 ["Freddy's Frozen Custard & Steakburgers",

  "Freddy's Frozen Custard Steakburgers"]]
# let's sort them by the first element of each sublist

def sortFirst(val): 

    return val[0]  



# sorts the array in ascending according to 1st element 

most_similar_edited.sort(key = sortFirst)  

most_similar_edited
# let's process the data second time manually to make the data cleaner

most_similar_sorted = [

 ['AW Restaurant', 'AW Restaurants', 'Aw Restaurants', 'A W Restaurant', 'AWRestaurants'],

 ["Albee's NY Gyros", "Albee's Ny Gyros"],

 ["Arby's", 'Arbys'],

 ['BLIMPIE', 'Blimpie'],

 ['Back Yard Burgers', 'Backyard Burgers'],

 ["Bad Daddy's Burger Bar", 'Bad Daddys Burger Bar'],

 ['Baskin Robbins', 'BaskinRobbins', 'Baskin-Robbins'],

 ['Best Burgers', 'Best Burger'],

 ["Big Billy's Burger Joint", 'Big Billys Burger Joint'],

 ['Big Boy', 'Big Boys'],

 ["Blake's Lotaburger", 'Blakes Lotaburger'],

 ['Blimpie', 'BLIMPIE'],

 ["Bob's Burger Brew",

  "Bob's Burgers Brew"],

 ['Bojangles Famous Chicken n Biscuits',

  "Bojangles' Famous Chicken 'n Biscuits"],

 ['Burger King', 'Burger King®'],

 ['Burger Master', 'Burgermaster'],

 ['BurgerFi', 'Burgerfi'],

 ['Burgermaster', 'Burger Master'],

 ['Burrito Amigos', 'Burrtio Amigos'],

 ['C & J Drive In', 'C J Drive In'],

 ["CULVER'S", "Culver's", 'Culvers'],

 ["Captain D'S", 'Captain Ds'],

 ["Carl's Jr", "Carl's Jr.", 'Carls Jr'],

 ["Chan's Restaurant", "Chen's Restaurant"],

 ['Chanellos Pizza', 'Chanello’s Pizza'],

 ["Charley's Grill & Spirits", "Charley's Grill Spirits"],

 ["Charley's Grilled Subs", 'Charleys Grilled Subs'],

 ["Chen's Restaurant", "Chan's Restaurant"],

 ['Chick-Fil-A', 'Chick-fil-A', 'ChickfilA'],

 ['China Sea', 'China Star', 'China Bear'],

 ["Church's Chicken", 'Churchs Chicken'],

 ["Cisco's Taqueria", 'Ciscos Taqueria'],

 ['Cook Out', 'Cook-Out', 'CookOut'],

 ["Culver's", "CULVER'S", 'Culvers'],

 ['Dairy Queen', 'Dairy queen'],

 ['Dairy Queen Grill & Chill', 'Dairy Queen Grill Chill'],

 ["Dominic's of New York", 'Dominics of New York'],

 ["Domino's Pizza", 'Dominos Pizza'],

 ['Dunkin Donuts', "Dunkin' Donuts"],

 ['Einstein Bros Bagels', 'Einstein Bros. Bagels'],

 ['Emidio & Sons Italian Restaurant', 'Emidio Sons Italian Restaurant'],

 ["Famous Dave's", 'Famous Daves'],

 ["Farlow's On The Water", 'Farlows on the Water'],

 ["Fazoli's", 'Fazolis'],

 ['Fireplace Restaurant & Lounge', 'Fireplace Restaurant Lounge'],

 ['Five Guys Burgers & Fries', 'Five Guys Burgers Fries'],

 ["Fox's Pizza Den", 'Foxs Pizza Den'],

 ["Freddy's Frozen Custard & Steakburgers",

  'Freddys Frozen Custard Steakburgers',

  "Freddy's Frozen Custard Steakburgers"],

 ["Frisch's Big Boy Restaurant",

  "1 Frisch's Big Boy Restaurant",

  "40 Frisch's Big Boy Restaurant",

  "1 Frisch's Big Boy Restaurant",

  "90 Frisch's Big Boy Restaurant"],

 ['Full Moon Bar B Que', 'Full Moon Bar-B-Que'],

 ["George's Gyros Spot", "George's Gyros Spot 2"],

 ['Grub Burger Bar', 'Bru Burger Bar'],

 ["Guthrie's", 'Guthries'],

 ['Gyro Express', 'Gyro X-Press'],

 ['Gyro Spot', 'Gyro Stop'],

 ["Hardee's", 'Hardees'],

 ["Hardee's Restaurant", "Hardee's Restaurants"],

 ["Hardee's/Red Burrito", "Hardee's/red Burrito", 'Hardees Red Burrito'],

 ['HomeTown Buffet', 'Hometown Buffet'],

 ['Hooters', 'Roosters'],

 ['Hot Dog On A Stick', 'Hot Dog on a Stick'],

 ['In-N-Out Burger', 'InNOut Burger'],

 ["Jack's", 'Jacks'],

 ["Jack's Hamburgers", "Zack's Hamburgers"],

 ["Jason's Deli", 'Jasons Deli'],

 ["Jersey Mike's Subs", 'Jersey Mikes Subs'],

 ["Jimmy John's", 'Jimmy Johns'],

 ['KFC', 'Kfc', 'KFC Kentucky Fried Chicken', 'KFC - Kentucky Fried Chicken'],

 ['Killer Burger', 'Killer Burgers'],

 ['L & L Hawaiian Barbecue', 'L L Hawaiian Barbecue',

  'LL Hawaiian Barbecue'],

 ["Long John Silver's", 'Long John Silvers'],

 ['Mai Tai Restaurant', 'Mai-Tai Restaurant'],

 ["Mary's Pizza Shack", 'Marys Pizza Shack'],

 ["Mc Donald's", "McDonald's", 'Mcdonalds', 'McDonalds'],

 ["McAlister's Deli", "Mcalister's Deli", 'McAlisters Deli'],

 ["Moe's Southwest Grill", 'Moes Southwest Grill'],

 ['Mr Hero', 'Mr. Hero'],

 ['Nicholas Restaurant', "Nicholas' Restaurant"],

 ["Papa John's Pizza", 'Papa Johns Pizza'],

 ['Peking Chinese Restaurant', 'Peking Chinese Restaurants'],

 ["Pietro's Pizza & Gallery of Games", "Pietro's Pizza Gallery of Games"],

 ["Popeye's Louisiana Kitchen", 'Popeyes Louisiana Kitchen'],

 ['Pot Belly Sandwich Works', 'Potbelly Sandwich Works'],

 ['QDOBA Mexican Eats', 'Qdoba Mexican Eats'],

 ['RUNZA', 'Runza'],

 ["Raising Cane's Chicken Fingers", 'Raising Canes Chicken Fingers'],

 ["Rally's", 'Rallys'],

 ["Rally's Hamburgers", 'Rallys Hamburgers'],

 ["Rick's on the River", 'Ricks on the River'],

 ["Ritter's Frozen Custard", 'Ritters Frozen Custard'],

 ["Rockne's", 'Rocknes'],

 ['Roosters', 'Hooters'],

 ['Runza', 'RUNZA'],

 ["Ryan's", 'Ryans'],

 ['SONIC Drive In',

  'Sonic Drive-In',

  'SONIC Drive-In',

  'Sonic DriveIn',

  'Sonic Drive-in'],

 ["STEAK 'N SHAKE",

  "Steak 'n Shake",

  'Steak N Shake',

  'Steak n Shake',

  "Steak 'N Shake"],

 ['SUBWAY', 'Subway'],

 ["Sara's Too", 'Saras Too'],

 ["Simple Simon's Pizza", 'Simple Simons Pizza'],

 ['Slice Of Life', 'Slice of Life'],

 ["Stanfield's Steakhouse", 'Stanfields Steak House'],

 ['T & L Hotdogs', 'T & L Hot Dogs'],

 ['Taco CASA', 'Taco Casa'],

 ['Taco Del Mar', 'Taco del Mar'],

 ["Taco John's", 'Taco Johns'],

 ['Taco Time', 'TacoTime'],

 ['Taste Of Buffalo Pizzeria', 'Taste of Buffalo Pizzeria'],

 ['Tom Drive-in', "Tom's Drive-In"],

 ["Topper's Pizza", 'Toppers Pizza'],

 ['WG Grinders', 'Wg Grinders'],

 ["Wendy's", 'Wendys'],

 ['Wg Grinders', 'WG Grinders'],

 ['Wing Street', 'Wingstreet'],

 ['Z-Pizza', 'zpizza'],

 ["Zack's Hamburgers", "Jack's Hamburgers"]]

print("cleaned, matched restaurant name count:", len(most_similar_sorted))
# let's create a dictionary to make name replace easier

match_name_dict = {}

for row in most_similar_sorted:

    for similar_word in row:

        match_name_dict[similar_word] = row[0]

match_name_dict
# let's use the match_name_dict to replace names in the dataset to make it cleaner

names = fastfood_data['name'].values

print("size:", len(names))



# replace names with their dictionary value

for i in range(len(names)):

    if match_name_dict.get(names[i]) != None:

        names[i] = match_name_dict[names[i]]



fastfood_data['names'] = names
# top 20 restaurants recorded by count total

print("Number of unique restaurant:", fastfood_data['name'].nunique())

nameplot = fastfood_data['name'].value_counts()[:20].plot.bar(title='Top 20 mentioned restaurants')

nameplot.set_xlabel('name',size=20)

nameplot.set_ylabel('count',size=20)
# take a peek at the dateAdded column

fastfood_data['dateAdded'].head()
# let's parse the dates for dateAdded and dateUpdated

from datetime import datetime

from dateutil.parser import parse
# let's convert dateAdded and dateUpdated column data to 

# datetime object with apply function

fastfood_data['dateAdded'] = fastfood_data['dateAdded'].apply(

        lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

fastfood_data['dateUpdated'] = fastfood_data['dateUpdated'].apply(

        lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
# check the top 6 rows again to verify the datatype has changed

print(fastfood_data['dateAdded'].head())

print()

print(fastfood_data['dateUpdated'].head())
# we can see the data type changed for the date related columns

fastfood_data.dtypes
# histogram of dateAdded column

plt.hist(fastfood_data['dateAdded'])

plt.xlabel("date")

plt.ylabel("frequency (count)")

plt.title("Histogram of dateAdded column")

plt.show()



# histogram of dateUpdated column

plt.hist(fastfood_data['dateUpdated'], color="orange")

plt.xlabel("date")

plt.ylabel("frequency (count)")

plt.title("Histogram of dateUpdated column")

plt.show()
# what types of category do we have for primary categories?

fastfood_data['primaryCategories'].value_counts()
# drop primaryCategory column

fastfood_data = fastfood_data.drop(['primaryCategories'], axis=1)



# shape of dataset

print("Dimension after dropping columns:")

print(fastfood_data.shape)



# take a peek at the head to verify the drop is successful

fastfood_data.head()
# let's split the categories string data by ","

categories = fastfood_data['categories'].values

for i in range(len(categories)):

    categories[i] = categories[i].split(",")
# update the "categories" column in the fastfood_data

fastfood_data['categories'] = categories

fastfood_data.head()
# top 50 most appeared restaurants in the dataset and their corresponding category

top50_name_by_number = fastfood_data['name'].value_counts()[:50].index.tolist()

top50_rest_categories = []

for name in top50_name_by_number:

    category = fastfood_data[fastfood_data.name == name]["categories"][:1].values[0]

    top50_rest_categories.append(category)
# remove stopword "Restaurants", "Restaurant", and "Fast Food Restaurants" in categories

stopwords = ["Fast Food Restaurants","Fast Food restaurants", "Fast Food", "Restaurants", "Restaurant", "restaurants", "restaurant"]

for i in range(len(top50_rest_categories)):

    for j in range(len(top50_rest_categories[i])):

        word = top50_rest_categories[i][j]

        # remove stopword as the only word given

        for stopword in stopwords:

            isRemoved = False

            if isRemoved == False and word == stopword:

                top50_rest_categories[i][j] = top50_rest_categories[i][j].replace(stopword, "")

                isRemoved = True



        # replace the stopword within a group of words

        for stopword in stopwords:

            if isRemoved == False and stopword in word:

                top50_rest_categories[i][j] = top50_rest_categories[i][j].replace(stopword, "")
top50_rest_categories
from wordcloud import WordCloud



# empty string is declare 

text = "" 



# iterating through list of rows 

for row in top50_rest_categories : 

    # iterating through words in the row 

    for word in row:

        if len(word) == 0:

            continue

        # concatenate the words 

        if word[-1] == " ":

            word = word[:-1] # remove the last space character

        text = text + " " + word.replace(" ", "_") 

print("Vocabulary of our processed categories data:\n")

print(text)
# generate wordcloud

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white',

                min_font_size = 10).generate(text) 

  

# plot the WordCloud image                        

plt.figure(figsize = (16, 16), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
# top 20 fast food populated cities recorded by count total

print("Number of unique cities:", fastfood_data['city'].nunique())

nameplot = fastfood_data['city'].value_counts()[:20].plot.bar(title='Top 20 mentioned cities')

nameplot.set_xlabel('city',size=20)

nameplot.set_ylabel('count',size=20)
# let's run pandas' .descirbe() function 

rest_count_by_city = fastfood_data['city'].value_counts()

rest_count_by_city.describe()
print(len(rest_count_by_city[rest_count_by_city < 4]), "cities opened less than 4 fast food restaurants")

print("They are", round(100*len(rest_count_by_city[rest_count_by_city < 4])/len(rest_count_by_city),2), "%"),

print("of the total cities recorded in the dataset")
# plot pie chart

fig, ax = plt.subplots()

total_cities_with_less_than_4_rests = len(rest_count_by_city[rest_count_by_city < 4])

total_cities_with_greater_equal_4_rests = fastfood_data['city'].nunique() - total_cities_with_less_than_4_rests

values = [total_cities_with_less_than_4_rests, total_cities_with_greater_equal_4_rests]

ax.pie(values, 

       labels=["city opened 1-3 shops", "city opened more than 3 shops"], autopct='%.1f%%', radius=1, 

       explode = (0.1, 0))

ax.set_aspect('equal')

ax.set_title("US Cities' Fast Food Restaurants Number")

plt.show()
# input 2 letter code name for easier conversion

us_state_names = pd.read_csv('../input/us2letterstatecodecsv/US-2-letter-state-code.csv')

us_state_names.head()
us_state_names.columns
# convert the names of the states

def convert_state_names(state_code):

    return us_state_names[us_state_names['state_code']== state_code].values[0][0]



# use the above function to convert code to name

fastfood_data['province_full'] = ""

state_names = [None] * len(fastfood_data['province'].values)

state_codes = fastfood_data['province'].values

for i in range(len(state_codes)):

    state_names[i] = convert_state_names(state_codes[i])

print(state_names[:20])



# # let's replace the province column values with full state names

fastfood_data['province_full'] = state_names
fastfood_data['province_full'].value_counts()
fastfood_data['province'].head()
# top 20 fast food populated states recorded by count total

print("Number of unique provinces:", fastfood_data['province_full'].nunique())

nameplot = fastfood_data['province_full'].value_counts()[:20].plot.bar(title='Top 20 mentioned province')

nameplot.set_xlabel('state',size=20) # changed the label to state; it's the US

nameplot.set_ylabel('count',size=20)
fastfood_data['province_full'].value_counts().describe()
fastfood_data['province_full'].value_counts()[fastfood_data['province_full'].value_counts() == 1]
# get top 3 fast food populated cities

top3_fastfood_populated_cities = fastfood_data['city'].value_counts()[:3].index.tolist()

top3_fastfood_populated_cities[:3]
# take a look at the restaurant numbers

fastfood_data[fastfood_data["city"]=="Columbus"]["names"].value_counts()
# get the restaurant number count by unique restaurant name

def get_restaurants_counts(city_name):

    return fastfood_data[fastfood_data["city"]==city_name]["names"].value_counts()



# print the restaurants that opened more than 1 shop

def print_more_than_1_shop_rest(restaurants, city_name):

    more_than_1_shops = restaurants[restaurants > 1]

    print("\nAmong", len(restaurants), "unique fastfood restaurant brands in", city_name+",")

    print(len(more_than_1_shops), "brands opened more than 1 shops.")

    print("They occupied", str(round(len(more_than_1_shops)/sum(restaurants)*100,2))+

          "% of total restaurants by number")

    print(more_than_1_shops)
# let's generate a quick summary of the shops

for city in top3_fastfood_populated_cities:

    print_more_than_1_shop_rest(get_restaurants_counts(city), city)
# reusable function to create data needed for plotting pie chart

def create_pie_chart_data(city_name):

    counts =  fastfood_data[fastfood_data["city"]==city_name]["names"].value_counts()[:10].values

    labels = fastfood_data[fastfood_data["city"]==city_name]["names"].value_counts().index.tolist()[:10]

    return counts, labels
# reusable function to plot pie chart for our EDA

def plot_pie_chart(counts, labels, city_name):

    fig, ax = plt.subplots()

    ax.pie(counts, labels=labels, autopct='%.1f%%', radius=1.1, 

          explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    ax.set_aspect('equal')

    ax.set_title("Top 10 Restaurants in " + city_name)

    plt.show()
# for each city, let's create the data needed and plot them in pie chart

for city in top3_fastfood_populated_cities:

    counts, labels = create_pie_chart_data(city)

    print(labels[:3], "are the top 3 populated restaurants in", city)

    plot_pie_chart(counts, labels, city)
from plotly.offline import init_notebook_mode, iplot

from plotly.graph_objs import *

init_notebook_mode(connected=True)

import plotly.graph_objs as go
# create data needed

state_codes = fastfood_data['province'].value_counts().index.tolist()

value_counts_by_states = fastfood_data['province'].value_counts()



# plotly choropleth

data= [dict(type='choropleth',

            locations = state_codes, # Spatial coordinates

            z = value_counts_by_states, # Data to be color-coded

            locationmode = 'USA-states', # set of locations match entries in `locations`

            colorscale = 'Reds',

            marker_line_color = 'white',

            colorbar_title = "Number of Fast Fast Restaurants"

        )]

layout = dict(title = '2019 US Fast Food Restaurants by State', 

              geo = dict(scope='usa'))

iplot(dict(data=data, layout=layout))
data = [dict(type = 'scattergeo',

            locationmode = 'USA-states',

            lon = fastfood_data['longitude'],

            lat = fastfood_data['latitude'],

            mode = 'markers',

            marker = dict(size = 3,

                opacity = 0.7,

                reversescale = True,

                autocolorscale = False,

                symbol = 'circle',

                line = dict(width=.5, color='black'),

                color = 'red'

            )

        )]



layout = dict(title = '2019 US Fast Food Restaurants by GPS location',

            geo = dict(scope='usa',

                showland = True,

                landcolor = "rgb(250, 250, 250)",

                subunitcolor = "rgb(217, 217, 217)"

            )

        )

iplot(dict(data=data, layout=layout ))
# input census data with "|" as separator

us_census_data = pd.read_csv('../input/us-census-2017/US-Census-2017.txt', sep="|", names=["state", "population"])

us_census_data.head()
# we have number of restaurants per state before

value_counts_by_states.head()
test = int(us_census_data[us_census_data['state']=="California"].iloc[0,1].replace(",",""))

print(test-1)
# let's divide the population by people in each state

per_capita_count = [None] * len(state_codes)

for i in range(len(state_codes)):

    state = fastfood_data[fastfood_data['province']==state_codes[i]]['province_full'][:1]

    per_capita_count[i] = value_counts_by_states[i] / int(us_census_data[us_census_data['state']=="California"].iloc[0,1].replace(",",""))

per_capita_count[:10]
# plotly choropleth

data= [dict(type='choropleth',

            locations = state_codes, # Spatial coordinates

            z = per_capita_count, # Data to be color-coded

            locationmode = 'USA-states', # set of locations match entries in `locations`

            colorscale = 'Reds',

            marker_line_color = 'white',

            colorbar_title = "Restaurants opened / population"

        )]

layout = dict(title = 'Ratio between number of fast food Restaurants and population', 

              geo = dict(scope='usa'))

iplot(dict(data=data, layout=layout))