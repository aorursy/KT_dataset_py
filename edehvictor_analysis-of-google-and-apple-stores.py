opened_1 = open('../input/google-and-apple-store/AppleStore.csv', encoding='utf8')

opened_2 = open('../input/google-and-apple-store/googleplaystore.csv', encoding='utf8')

from csv import reader

read_file1= reader(opened_1)

read_file2 = reader(opened_2)

ios = list(read_file1)

google_play = list(read_file2)
def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]

    for row in dataset_slice:

        print(row)

        print('\n')

        

    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))

        print('----------------------------------------------------------------------------')

        

        

print('For Apple Store: ')

explore_data(ios[1:], 0, 2, rows_and_columns=True)

print('For Google Play: ')

explore_data(google_play[1:], 0, 2, rows_and_columns=True)
ios_column_names = ios[0]

google_play_column_names = google_play[0]



print(ios_column_names)

print('\n')

print(google_play_column_names)
print(google_play[10473])
del google_play[10473] #DO NOT RUN DEL STATEMENT MORE THAN ONCE!
def duplicate_and_unique_entries(dataset, index):

    unique_apps = []

    duplicate_apps = []

    

    for apps in dataset:

        name = apps[index]

        if name in unique_apps:

            duplicate_apps.append(name)

        else:

            unique_apps.append(name)

            

    print('Number of duplicate apps: ', len(duplicate_apps)) 

    print('Number of unique apps: ', len(unique_apps))

    print('\n')

    print('Examples of duplicate apps: ', duplicate_apps[:3])

   

 

print('For Google Play: ')

duplicate_and_unique_entries(google_play[1:], 0)

print('--------------------------')

print('For ios:')

duplicate_and_unique_entries(ios[1:], 1)
print('For Google Play: ')

for app in google_play[1:]: 

    name = app[0]

    if name == 'Quick PDF Scanner + OCR FREE':

        print(app)

        

        

print('\n')

print('For ios: ')

for app in ios[1:]:

    name = app[1]

    if name == 'Mannequin Challenge':

        print(app)
print('Expected length: ', len(google_play[1:]) - 1181)

print('Expected length for ios: ', len(ios[1:]) - 2)
reviews_max = {}

for apps in google_play[1:]:

    name = apps[0]

    n_reviews = float(apps[3])

    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews

    elif name not in reviews_max:

        reviews_max[name] = n_reviews







google_play_cleaned = []

already_added = []



for app in google_play[1:]:

    name = app[0]

    n_reviews = float(app[3])

    if n_reviews == reviews_max[name] and name not in already_added:

        google_play_cleaned.append(app)

        already_added.append(name)

print('Length of cleaned google dataset is: ', len(reviews_max))



print('-----------------------------------------------------------------------------------------------')

ios_rating_max = {}

for apps in ios[1:]:

    name = apps[1]

    n_ratings = float(apps[5])

    if name in ios_rating_max and ios_rating_max[name] < n_ratings:

        ios_rating_max[name] = n_ratings

    elif name not in ios_rating_max:

        ios_rating_max[name] = n_ratings





ios_cleaned = []

ios_already_added = []

for App in ios[1:]:

    app_name = App[1]

    N_ratings = float(App[5])

    if N_ratings == ios_rating_max[app_name] and app_name not in ios_already_added:

        ios_cleaned.append(App)

        ios_already_added.append(app_name)

print('Length of cleaned  ios dataset is: ', len(ios_cleaned))
def is_english(name):

    count = 0

    for char in name:

        if ord(char) > 127:

            count += 1

            if count > 3:

                return False

        

    return True

    

is_english('爱奇艺PPS -《欢乐颂2》电视剧热播')
google_play_eng_apps = []

app_store_eng_apps = []



for app in ios_cleaned:

    name = app[1]

    if is_english(name):

        app_store_eng_apps.append(app)

        

for app in google_play_cleaned:

    name = app[0]

    if is_english(name):

        google_play_eng_apps.append(app)

        

print('The number of apps left in google_play dataset after removing the duplicates and non english apps are: ', len(google_play_eng_apps))

print('The number of apps left in ios afer removing non english apps are: ', len(app_store_eng_apps))        
free_google_apps = []

free_ios_apps = []



for apps in google_play_eng_apps:

    price = apps[7]

    if price == '0':

        free_google_apps.append(apps)



for apps in app_store_eng_apps:

    price = apps[4]

    if price == '0.0':

        free_ios_apps.append(apps)



print('Google apps left: ', len(free_google_apps))

print('ios apps left: ', len(free_ios_apps))

def freq_table(dataset, index):

    freq_dict = {}

    for apps in dataset:

        genre = apps[index]

        if genre not in freq_dict:

            freq_dict[genre] = 1

        elif genre in freq_dict:

            freq_dict[genre] += 1



    for genre in freq_dict:

        freq_dict[genre] /= len(dataset)

        freq_dict[genre] *= 100

    return freq_dict

        



def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_value_as_tuple = (table[key], key)

        table_display.append(key_value_as_tuple)

    

    table_sorted = sorted(table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':',entry[0])



genres = []

genre_avg_user_ratings = {}



for key in freq_table(free_ios_apps, 11):

    genres.append(key)



for genre in genres:

    total = 0 #stores the sum of the number of ratings specific to each genre

    len_genre = 0 #stores the number of apps specific to each genre

    for app in free_ios_apps:

        genre_app = app[11]

        if genre_app == genre:

            no_of_user_rating = float(app[5])

            total += no_of_user_rating

            len_genre += 1

    average_user_rating = total / len_genre

    genre_avg_user_ratings[genre] = average_user_rating



genre_avg_user_ratings_display = []

for key in genre_avg_user_ratings:

    key_value_as_tuple = (genre_avg_user_ratings[key], key)

    genre_avg_user_ratings_display.append(key_value_as_tuple)

    

genre_avg_user_ratings_display = sorted(genre_avg_user_ratings_display, reverse=True)

print('The averager ratings for apps in ios store is: ')

for entry in genre_avg_user_ratings_display:

    print(entry[1], ': ', entry[0])
google_app_categories = []

google_app_avg_no_installs = {}



for key in freq_table(free_google_apps, 1):

    google_app_categories.append(key)





for category in google_app_categories:

    total = 0

    len_category = 0

    for app in free_google_apps:

        category_app = app[1]

        if category_app == category:

            no_of_installs = app[5]

            no_of_installs = no_of_installs.replace('+', '')

            no_of_installs = no_of_installs.replace(',', '')

            no_of_installs = float(no_of_installs)

            total += no_of_installs

            len_category += 1

    avg_no_installs = total / len_category

    google_app_avg_no_installs[category] = avg_no_installs



    

Google_avg_installs = []

for key in google_app_avg_no_installs:

    keyValue_as_tuple = (google_app_avg_no_installs[key], key)

    Google_avg_installs.append(keyValue_as_tuple)



Google_avg_installs = sorted(Google_avg_installs, reverse=True)

print("The average number of installs for each app category in Google Play Store is: ")

for entry in Google_avg_installs:

    print(entry[1], ': ', entry[0])