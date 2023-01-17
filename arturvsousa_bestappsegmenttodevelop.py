from csv import reader



opened_file_apple = open('../input/google-and-apple-store/AppleStore.csv')

read_apple = reader(opened_file_apple)

apple_apps_data = list(read_apple)



opened_file_android = open('../input/google-and-apple-store/googleplaystore.csv')

read_android = reader(opened_file_android)

android_apps_data = list(read_android)
from collections import OrderedDict
android_header = (android_apps_data[0])

print (android_header)
apple_header = (apple_apps_data[0])

print (apple_header)
def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]    

    for row in dataset_slice:

        print(row)

        print('\n') # adds a new (empty) line after each row



    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))
explore_data(apple_apps_data, 1, 4, rows_and_columns=True)
explore_data(android_apps_data, 1, 4, rows_and_columns=True)
index = 0

for row in android_apps_data[1:]:

    index += 1

    app_name = row[0]

    if app_name == 'Life Made WI-Fi Touchscreen Photo Frame':

        print (row)

        print (index)
del android_apps_data[10473]
unique_android = []

duplicated_android = []



for row in android_apps_data[1:]:

    name = row[0]

    if name in unique_android:

        duplicated_android.append(name)

    else:

        unique_android.append(name)
unique_ios = []

duplicated_ios = []



for row in apple_apps_data[1:]:

    name = row[1]

    if name in unique_ios:

        duplicated_ios.append(name)

    else:

        unique_ios.append(name)
and_dupl_num = len (duplicated_android)

and_uniq_num = len (unique_android)

ios_dupl_num = len (duplicated_ios)

ios_uniq_num = len (unique_ios)



print (and_dupl_num)

print (and_uniq_num)

print (ios_dupl_num)

print (ios_uniq_num)
print (duplicated_android[:10])
for row in android_apps_data[1:]:

    name = row[0]

    if name == 'Instagram':

        print (row)
reviews_max = {}



for row in android_apps_data[1:]:

    name = row[0]

    n_reviews = float(row[3])

    if name not in reviews_max:

        reviews_max[name] = n_reviews

    if name in reviews_max:

        if n_reviews > reviews_max[name]:

            reviews_max[name] = n_reviews
len (reviews_max)
android_clean = []

apple_clean = apple_apps_data[1:]

already_added = []



for row in android_apps_data[1:]:

    name = row[0]

    n_reviews = float(row[3])

    if n_reviews == reviews_max[name]:

        if name not in already_added:

            android_clean.append(row)

            already_added.append(name)

            
len (android_clean)
explore_data(android_clean, 1, 7)
len (already_added)

print (already_added[1:6])
def check_latin (app_name):

    latin_char = []

    non_latin_char = []

    for char in app_name:

        if ord(char) < 128:

            latin_char.append(char)

        if ord(char) > 127:

            non_latin_char.append(char)

    

    len_latin = len (latin_char)

    len_non_latin = len (non_latin_char)

    if len_latin > len_non_latin:

        return True

    else:

        return False



real_clean_googleapps = []

for row in android_clean[1:]:

    ck = check_latin(row[0])

    if ck == True:

        real_clean_googleapps.append(row)

        

real_clean_appleapps = []

for row in apple_clean[1:]:

    chk = check_latin(row[1])

    if chk == True:

        real_clean_appleapps.append(row)
goog_len = len (real_clean_googleapps)

appl_len = len (real_clean_appleapps)

print (goog_len)

print (appl_len)
apple_apps_prices = {'Free': 0, 'Non_free': 0}

apple_apps_downloads = {'Free': 0, 'Non_free': 0}

for row in real_clean_appleapps:

    price = float (row[4])

    downloaded_times = float (row[5])

    if price == 0.0:

        apple_apps_prices['Free'] += 1

        apple_apps_downloads['Free'] += downloaded_times

    else:

        apple_apps_prices['Non_free'] += 1

        apple_apps_downloads['Non_free'] += downloaded_times



total_apps = apple_apps_prices['Free'] + apple_apps_prices['Non_free']

total_downloads = apple_apps_downloads['Free'] + apple_apps_downloads['Non_free']
print (apple_apps_prices)
print (apple_apps_downloads)
apple_free_apps_propor = (apple_apps_prices['Free']/total_apps)*100

apple_free_downloads = (apple_apps_downloads['Free']/total_downloads)*100
proporcao_apps = round(apple_free_apps_propor)

proporcao_downloads = round (apple_free_downloads)

print (proporcao_apps)

print (proporcao_downloads)
print (android_apps_data[0])

print (apple_apps_data[0])
print (real_clean_googleapps[0])

print (real_clean_appleapps[0])
android_free = []

android_non_free = []

ios_free = []

ios_non_free = []



for row in real_clean_googleapps:

    price = row[6]

    if price == 'Free':

        android_free.append(row)

    else:

        android_non_free.append(row)

    

    

for row in real_clean_appleapps:

    price = row[4]

    if price == '0.0':

        ios_free.append(row)

    else:

        ios_non_free.append(row)
explore_data(android_free, 0, 5)
explore_data(ios_free, 0, 5)
len_free_ios = len (ios_free)

print (len_free_ios)



len_non_free_ios = len (ios_non_free)

print (len_non_free_ios)
len_free_and = len (android_free)

print (len_free_and)



len_non_free_and = len (android_non_free)

print (len_non_free_and)
def freq_table (dataset, index, frequency=True):

    table = {}

    total = 0

    

    for row in dataset:

        total += 1

        

        value = row[index]

        if value not in table:

            table[value] = 1

        if value in table:

            table[value] += 1

    

    percent_tab = {}

    for key in table:

        percentage = (float (table[key])/total)*100

        percent_tab[key] = percentage

        

    if frequency == True:

        return (table)

    if frequency == False:

        return (percent_tab)
def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    

    for key in table:

        key_val_as_tuple = (table[key], key)

        table_display.append(key_val_as_tuple)

    

    table_sorted = sorted (table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
display_table(android_free, 1)
android_genre_qt = display_table(android_free, 9)
ios_genre_qt = display_table (ios_free, 11)
explore_data(android_free, 0, 5)
android_installs_genre = {}



for row in android_free:

    genre = row[9]

    

    installs = row[5]

    installs = installs.replace(',','')

    installs = installs.replace('+','')

    installs = float(installs)

    

    if genre not in android_installs_genre:

        android_installs_genre[genre] = installs

    

    if genre in android_installs_genre:

        android_installs_genre[genre] += installs
sorted(android_installs_genre.items(), key=lambda x: (-x[1], x[0]))
android_category_installs = {}



for row in android_free:

    category = row[1]

    

    installs = row[5]

    installs = installs.replace('+','')

    installs = installs.replace(',','')

    installs = float(installs)

    

    if category not in android_category_installs:

        android_category_installs[category] = installs

        

    if category in android_category_installs:

        android_category_installs[category] += installs
sorted(android_category_installs.items(), key=lambda x: (-x[1], x[0]))
android_freq_gen = freq_table(android_free, 1)

sorted(android_freq_gen.items(), key=lambda x: (-x[1], x[0]))
avg_inst_gen = {k:v1/android_freq_gen.get(k,0) for k,v1 in android_category_installs.items()}
sorted(avg_inst_gen.items(), key=lambda x: (-x[1], x[0]))
communication_apps = []



for row in android_free:

    genre = row[1]

    if genre == 'COMMUNICATION':

        communication_apps.append(row)
sorted (communication_apps)
video_players_apps = []



for row in android_free:

    genre = row[1]

    if genre == 'VIDEO_PLAYERS':

        video_players_apps.append(row)
sorted (video_players_apps)
social_apps = []



for row in android_free:

    genre = row[1]

    if genre == 'SOCIAL':

        social_apps.append(row)
sorted (social_apps)
comlen = len (communication_apps)

vilen = len (video_players_apps)

soclen = len (social_apps)



print (comlen)

print (vilen)

print (soclen)