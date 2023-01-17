from csv import reader
import matplotlib.pyplot as plt
import numpy as np
#____Basic CSV Open Function____
def csv_to_list(file_name, encode=None):
    if encode =="utf8":
        csv_file = open(file_name, encoding="utf8")
        reade_file = reader(csv_file)
        new_list = list(reade_file)
        list_header = new_list[0]
        list_body = new_list[1:]
        return list_header, list_body
    elif encode =="Latin-1":
        csv_file = open(file_name, encoding="latin-1")
        reade_file = reader(csv_file)
        new_list = list(reade_file)
        list_header = new_list[0]
        list_body = new_list[1:]
        return list_header, list_body
    else:
        csv_file = open(file_name)
        reade_file = reader(csv_file)
        new_list = list(reade_file)
        list_header = new_list[0]
        list_body = new_list[1:]
        return list_header, list_body        
    
#____Opening AppleStore Dataset____
appstore = "../input/app-store-apple-data-set-10k-apps/AppleStore.csv"
encode = "utf8"
apple_header, apple_dataset = csv_to_list(appstore, encode)
print("AppStore dataset size: ", len(apple_dataset))

#____Opening GoogleAppStore Dataset____
iosStore = "../input/google-play-store-apps/googleplaystore.csv"
encode = "utf8"
ios_header, ios_dataset = csv_to_list(iosStore, encode)
print("GooglePlay dataset size: ", len(ios_dataset))
print(ios_dataset[10472])
# this entry has missing 'Rating' 
del ios_dataset[10472]
print('Test:\n', ios_dataset[10472])
name_list = []
cnt = 0
repeated_index_list = []

for name in ios_dataset:
    if name[0] in name_list:
        repeated_index_list.append(cnt)
    else:
        name_list.append(name[0])
    cnt+=1

print('There is ', len(repeated_index_list), 'duplicated apps.')
#___Changing the order of the repeated index, so we can delete without changing the others
repeated_index_list.sort(key=int, reverse=True)

for n in repeated_index_list:
    del ios_dataset[n]
#___Locating___
cnt = 0
non_english_index = []

for name in ios_dataset:
    for character in name[0]:
        if ord(character) > 127 and ord(character) != 8211 and ord(character) != 8482 and ord(character) != 174 and ord(character) != 8212:
            non_english_index.append(cnt)
            break
    cnt+=1

cnt = 0
print('Examples of apps tracedown:')
for n in non_english_index:
    print(ios_dataset[n][0])
    cnt+=1
    if cnt>5:
        break

#___Deleting___
non_english_index.sort(key=int,reverse=True)
for index in non_english_index:
    del ios_dataset[index]
    
#___Confirming___
cnt = 0
test_non_english_index = []

for name in ios_dataset:
    for character in name[0]:
        if ord(character) > 127 and ord(character) != 8211 and ord(character) != 8482 and ord(character) != 174 and ord(character) != 8212:
            test_non_english_index.append(cnt)
            break
    cnt+=1

print('\nTest: Applications not deleted:', test_non_english_index)

#___Locating___
cnt = 0
non_english_index = []

for name in apple_dataset:
    for character in name[2]:
        if ord(character) > 127 and ord(character) != 8211 and ord(character) != 8482 and ord(character) != 174 and ord(character) != 8212:
            non_english_index.append(cnt)
            break
    cnt+=1

print('Examples of apps tracedown:')
cnt = 0
for n in non_english_index:
    print(apple_dataset[n][2])
    cnt+=1
    if cnt>5:
        break

#___Deleting___
non_english_index.sort(key=int,reverse=True)
for index in non_english_index:
    del apple_dataset[index]
    
#___Confirming___
cnt = 0
test_non_english_index = []

for name in ios_dataset:
    for character in name[2]:
        if ord(character) > 127 and ord(character) != 8211 and ord(character) != 8482 and ord(character) != 174 and ord(character) != 8212:
            test_non_english_index.append(cnt)
            break
    cnt+=1

print('\nTest: Applications not deleted:', test_non_english_index)



apple_dataset_free = []
for row in apple_dataset:
    if row[5] == '0':
        apple_dataset_free.append(row)

ios_dataset_free = []
for row in ios_dataset:
    if row[6] == 'Free':
        ios_dataset_free.append(row)
        
print("AppleAppStore dataset size: ", len(apple_dataset_free))
print("GooglePlay dataset size: ", len(ios_dataset_free))
apple_genre_dictionarie = {}

for genre in apple_dataset_free:
    if genre[12] in apple_genre_dictionarie:
        apple_genre_dictionarie[genre[12]] +=1
    else:
        apple_genre_dictionarie[genre[12]] = 1

apple_dataset_free_size = len(apple_dataset_free)
for key in apple_genre_dictionarie:
    apple_genre_dictionarie[key] = round((apple_genre_dictionarie[key]/apple_dataset_free_size)*100, 2)

print("Percentage of apps in the store per genre in AppStore:\n")
sorted_by_value = sorted(apple_genre_dictionarie.items(), key=lambda kv: kv[1], reverse = True)

for item in sorted_by_value:
    print(item)

ios_genre_dictionarie = {}

for genre in ios_dataset_free:
    if genre[1] in ios_genre_dictionarie:
        ios_genre_dictionarie[genre[1]] += 1
    else:
        ios_genre_dictionarie[genre[1]] = 1

ios_dataset_free_size = len(ios_dataset_free)
for key in ios_genre_dictionarie:
    ios_genre_dictionarie[key] = round((ios_genre_dictionarie[key]/ios_dataset_free_size)*100, 2) 

print("Percentage of apps in the store per genre in PlayStore:\n")
sorted_by_value = sorted(ios_genre_dictionarie.items(), key=lambda kv: kv[1], reverse = True)

for item in sorted_by_value:
    print(item)
for row in apple_dataset_free:
    if row[12] == 'Navigation':
        print(row[2], ':', row[6]) 
cnt = 0
for row in apple_dataset_free:
    cnt +=1
    if row[12] == 'Social Networking':
        print(cnt, " ", row[2], ':', row[6])
    if cnt>300 :
        break
#LOCATING MOST RATED APPS

#___Creating a list of Genre___
apple_genre_list = []
for genre in apple_dataset_free:
    if genre[12] not in apple_genre_list:
        apple_genre_list.append(genre[12])
        
        
#___Initializing Variables___
apple_onepercent_list = ['Weather', 'Food & Drink', 'Reference','Business','Book', 'Navigation','Medical','Catalogs']
cnt_index = 0
cnt_first_apps = 0
list_position = 0
list_of_index = []
list_of_ratings = []
list_position = 0



#Looping throught genres
for genre in apple_genre_list:
    cnt_first_apps = 0
    cnt_index = 0
    
    #Looping throught dataset for each genre
    for rating in apple_dataset_free:
               
        if rating[12] == genre:
            cnt_first_apps += 1
            
            #Treating items with one percent
            if rating[12] in apple_onepercent_list:
                if cnt_first_apps == 1:
                    list_of_index.append(cnt_index)
                    list_of_ratings.append(int(rating[6]))
                else:
                    if int(rating[6]) > list_of_ratings[-1]:
                        list_of_ratings[-1] = int(rating[6])
                        list_of_index[-1] = cnt_index
            #Treating items with more than one percent   
            else:
                if cnt_first_apps < 5:
                    list_of_index.append(cnt_index)
                    list_of_ratings.append(int(rating[6]))
                    
                else:
                    for i in range(4):
                        if list_of_ratings[list_position + i] < int(rating[6]):
                            list_of_ratings[list_position + i] = int(rating[6])
                            list_of_index[list_position + i] = cnt_index
                            break
        cnt_index += 1
    if genre in apple_onepercent_list:
        list_position += 1
        
    else:
        list_position += 4
            
#DELETING THEN

apple_dataset_free_nobigapps = apple_dataset_free.copy()

list_of_index.sort(key=int, reverse=True)

for n in list_of_index:
    del apple_dataset_free_nobigapps[n]

#MAKING THE ANALYZES:
apple_genre_rating_dictionarie = {}
sum_per_genre = 0
cnt = 0

for genre in apple_genre_list:
    sum_per_genre = 0 
    cnt = 0
    
    for rating in apple_dataset_free_nobigapps:
        
        if genre == rating[12]:
            sum_per_genre += int(rating[6])
            cnt +=1 
    
    apple_genre_rating_dictionarie[genre] = round(sum_per_genre /cnt, 2)
        

print("Download average per app in diferent genres in AppStore:\n")
sorted_by_value = sorted(apple_genre_rating_dictionarie.items(), key=lambda kv: kv[1], reverse = True)

for item in sorted_by_value:
    print(item)
    
#The Install column must be treated:
a ='10,000+'
print(a[:-1].replace(",",""))
#LOCATING MOST RATED APPS

#___Creating a list of Genre___
ios_genre_list = []
for genre in ios_dataset_free:
    if genre[1] not in ios_genre_list:
        ios_genre_list.append(genre[1])
        
        
#___Initializing Variables___
ios_onepercent_list = ['AUTO_AND_VEHICLES', 'LIBRARIES_AND_DEMO','HOUSE_AND_HOME','WEATHER','EVENTS','ART_AND_DESIGN','PARENTING','BEAUTY','COMICS']
cnt_index = 0
cnt_first_apps = 0
list_position = 0
list_of_index = []
list_of_ratings = []
list_position = 0



#Looping throught genres
for genre in ios_genre_list:
    cnt_first_apps = 0
    cnt_index = 0
    
    #Looping throught dataset for each genre
    for rating in ios_dataset_free:
               
        if rating[1] == genre:
            cnt_first_apps += 1
            
            #Treating items with one percent
            if rating[1] in ios_onepercent_list:
                if cnt_first_apps == 1:
                    list_of_index.append(cnt_index)
                    list_of_ratings.append(int(rating[5][:-1].replace(",","")))
                else:
                    if int(rating[5][:-1].replace(",","")) > list_of_ratings[-1]:
                        list_of_ratings[-1] = int(rating[5][:-1].replace(",",""))
                        list_of_index[-1] = cnt_index
            #Treating items with more than one percent   
            else:
                if cnt_first_apps < 5:
                    list_of_index.append(cnt_index)
                    list_of_ratings.append(int(rating[5][:-1].replace(",","")))
                    
                else:
                    for i in range(4):
                        if list_of_ratings[list_position + i] < int(rating[5][:-1].replace(",","")):
                            list_of_ratings[list_position + i] = int(rating[5][:-1].replace(",",""))
                            list_of_index[list_position + i] = cnt_index
                            break
        cnt_index += 1
    if genre in ios_onepercent_list:
        list_position += 1
        
    else:
        list_position += 4
            
            
#DELETING THEN

ios_dataset_free_nobigapps = ios_dataset_free.copy()

list_of_index.sort(key=int, reverse=True)

for n in list_of_index:
    del ios_dataset_free_nobigapps[n]

#MAKING THE ANALYZES:
ios_genre_rating_dictionarie = {}
sum_per_genre = 0
cnt = 0

for genre in ios_genre_list:
    sum_per_genre = 0 
    cnt = 0
    
    for rating in ios_dataset_free_nobigapps:
        
        if genre == rating[1]:
            sum_per_genre += int(rating[5][:-1].replace(",",""))
            cnt +=1 
    
    ios_genre_rating_dictionarie[genre] = round(sum_per_genre /cnt, 2)
        

print("Download average per app in diferent genres in PlayStore:\n")
sorted_by_value = sorted(ios_genre_rating_dictionarie.items(), key=lambda kv: kv[1], reverse = True)

for item in sorted_by_value:
    print(item)
    
#PlayStore
ios_genre_popular_and_low_competition = []

for genre in ios_genre_list:
    if ios_genre_rating_dictionarie[genre] > 2000000 and ios_genre_dictionarie[genre] < 3:
        ios_genre_popular_and_low_competition.append(genre)

#AppStore
apple_genre_popular_and_low_competition = []

for genre in apple_genre_list:
    if apple_genre_rating_dictionarie[genre] > 15000 and apple_genre_dictionarie[genre] < 3:
        apple_genre_popular_and_low_competition.append(genre)
        
print(ios_genre_popular_and_low_competition)
print(apple_genre_popular_and_low_competition)