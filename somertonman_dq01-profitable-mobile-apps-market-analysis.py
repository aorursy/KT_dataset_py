import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from csv import reader
def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]    

    for row in dataset_slice:

        print(row)

        print('\n') # adds a new (empty) line after each row



    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))
appstore="/kaggle/input/sample-app-store-for-apps-analysis/AppleStore.csv"

googlestore="/kaggle/input/sample-google-play-for-apps-analysis/googleplaystore.csv"



apple = list(reader(open(appstore)))

android = list(reader(open(googlestore)))
columns_apple = explore_data(apple,0,1)

columns_apple
columns_android = explore_data(android,0,1)

columns_android
android[10473]
del android[10473]
## Android DS

duplicate_apps=[]

unique_apps=[]



for i in android[1:]:

    name= i[0]

    if name in unique_apps:

        duplicate_apps.append(name)

    else:

        unique_apps.append(name)

print("Google store")

print("Duplicated apps: ", len(duplicate_apps))

print("Unique apps: ", len(unique_apps))
## Apple DS

duplicate_apps=[]

unique_apps=[]



for i in apple[1:]:

    name= i[1]

    if name in unique_apps:

        duplicate_apps.append(name)

    else:

        unique_apps.append(name)

print("Apple store")

print("Duplicated apps: ", len(duplicate_apps))

print("Unique apps: ", len(unique_apps))

reviews_max_android={}

for i in android[1:]:

    name=i[0]

    n_reviews= float(i[3])

    if name in reviews_max_android and reviews_max_android[name] < n_reviews:

        reviews_max_android[name]=n_reviews

    elif name not in reviews_max_android:

        reviews_max_android[name]=n_reviews

        

android_clean=[]

already_added_android=[]



for i in android[1:]:

    name=i[0]

    n_reviews= float(i[3])

    if n_reviews == reviews_max_android[name] and name not in already_added_android:

        android_clean.append(i)

        already_added_android.append(name)

        

print("Google store")

print ("Unique names with the highest reviews number: ",len(reviews_max_android))

print("Records in cleaned dataset for Google store: ",len(android_clean)," (should match)" )
def is_english_chars_in_string(string):

    non_english_chars_number=0

    for i in string:

        if ord(i)>127:

            non_english_chars_number+=1

    if non_english_chars_number>3:

        return False

    else:

        return True



is_english_chars_in_string('爱奇艺PPS -《欢乐颂2》电视剧热播')

is_english_chars_in_string('Instagram')
android_clean_english=[]

for i in android_clean:

    name = i[0]

    if is_english_chars_in_string(name)==True:

        android_clean_english.append(i)

        

print ("Google Store, English apps only number: ",len(android_clean_english))
apple_clean_english=[]

for i in apple[1:]:

    name = i[1]

    if is_english_chars_in_string(name)==True:

        apple_clean_english.append(i)

        

print ("Apple Store, English apps only number: ",len(apple_clean_english))
## possible values of Price field in dataset

unique_prices=set()

for i in android_clean_english:

    price = i[6]

    unique_prices.add(price)

print(unique_prices)



## we need Free apps only for this research



android_clean_english_free=[]



for i in android_clean_english:

    price = i[7]

    

    if price =='0':

        android_clean_english_free.append(i)

    

len(android_clean_english_free)   
android_final = []

ios_final = []



for i in android_clean_english:

    price = i[7]

    if price == '0':

        android_final.append(i)

        

for i in apple_clean_english:

    price = i[4]

    if price == '0.0':

        ios_final.append(i)

        

print("Google Store total free English apps number: ",len(android_final))

print("Apple Store total free English apps number: ",len(ios_final))
explore_data(android,0,1)
explore_data(apple,0,1)
## creating a frequency table from a Dataset for a column of Index



def freq_table(dataset,index):

    d={}

    number_of_records=0

    for i in dataset:

        number_of_records+=1

        par=i[index]

        if par not in d:

            d[par]=1

        else:

            d[par]+=1

    for i in d:

        d[i]=round(d[i]/number_of_records*100,2)

    return d



## helper function to show statistics from data

def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key)

        table_display.append(key_val_as_tuple)



    table_sorted = sorted(table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
## display top results, percentage



print("Google store. Frequency table for Category")

android_category_table = display_table(android_final,1)

print("_"*50)

print("Google store. Frequency table for Genres")

android_genres_table = display_table(android_final,9)

print("_"*50)

print("Apple store. Frequency table for prime_genre")

apple_genres_table = display_table(ios_final,11) 

apple_freq_prime_genre = freq_table(ios_final,11) 
apple_installs_by_genre={}

for genre in apple_freq_prime_genre:

    total=0

    len_genre=0

    for  i in ios_final:

        genre_app=i[11]

        if genre_app == genre:

            num_of_ratings=float(i[5])

            total+=num_of_ratings

            len_genre+=1



    avg_num_of_ratings=total/len_genre

    apple_installs_by_genre[genre]=avg_num_of_ratings

    #print(genre," : ",avg_num_of_ratings)



## let's sort the list:



import operator

apple_installs_by_genre_sorted = sorted(apple_installs_by_genre.items(), key=operator.itemgetter(1),reverse=True)

apple_installs_by_genre_sorted
print("Google store. Frequency table for Category")

android_category_table = display_table(android_final,1)

print("_"*50)

print("Google store. Frequency table for Genres")

android_genres_table = display_table(android_final,9)
android_freq_category = freq_table(android_final,1) 
android_installs_by_genre={}

for category in android_freq_category:

    total=0

    len_genre=0

    #print(category)

    for  i in android_final:

        category_app=i[1]

        if category_app == category:

            num_of_installs=i[5]

            num_of_installs=num_of_installs.replace(",","")

            num_of_installs=num_of_installs.replace("+","")

            num_of_installs=float(num_of_installs)

            total+=num_of_installs

            len_genre+=1



    avg_num_of_installs=total/len_genre

    #print(avg_num_of_installs)

    android_installs_by_genre[category]=avg_num_of_installs

    

import operator

android_installs_by_genre_sorted = sorted(android_installs_by_genre.items(), key=operator.itemgetter(1),reverse=True)

android_installs_by_genre_sorted