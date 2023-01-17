from csv import reader



### The Google Play data set ###

opened_file = open('googleplaystore.csv')

read_file = reader(opened_file)

android = list(read_file)

android_header = android[0]

android = android[1:]



### The App Store data set ###

opened_file = open('AppleStore.csv')

read_file = reader(opened_file)

ios = list(read_file)

ios_header = ios[0]

ios = ios[1:]
def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]    

    for row in dataset_slice:

        print(row)

        print('\n') # adds a new (empty) line between rows

        

    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))



print(android_header)

print('\n')

explore_data(android, 0, 3, True)

print('\n')

print(ios_header)

print('\n')

explore_data(ios, 0, 3, True)
##Data cleaning - removing incorrect information

print(android[10472])  # incorrect row

print('\n')

print(android_header)  # header

print('\n')

print(android[0])      # correct row
print(len(android))

del android[10472]  

print(len(android))
##Removing duplicate entries

duplicate_apps = []

unique_apps = []



for app in android:

    name = app[0]

    if name in unique_apps:

        duplicate_apps.append(name)

    else:

        unique_apps.append(name)

    

print('Number of duplicate apps:', len(duplicate_apps))

print('\n')

print('Examples of duplicate apps:', duplicate_apps[:15])
##Selecting only one entry per app and which has highest number of reviews

reviews_max = {}



for app in android:

    name = app[0]

    n_reviews = float(app[3])

    

    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews

        

    elif name not in reviews_max:

        reviews_max[name] = n_reviews
print('Expected length:', len(android) - 1181)

print('Actual length:', len(reviews_max))
#Removing duplicates final



android_clean = []

already_added = []



for app in android:

    name = app[0]

    n_reviews = float(app[3])

    

    if (reviews_max[name] == n_reviews) and (name not in already_added):

        android_clean.append(app)

        already_added.append(name) 
explore_data(android_clean, 0, 3, True)
#Removing apps with Non-english names - particularly apps names having more than 3 ASCII characters



def is_english(string):

    non_ascii = 0

    

    for character in string:

        if ord(character) > 127:

            non_ascii += 1

    

    if non_ascii > 3:

        return False

    else:

        return True



print(is_english('CamScanner'))

print(is_english('Wechat 😜😜😜😜' ))
# Use the above function to filter out the datasets

android_english = []

ios_english = []



for app in android_clean:

    name = app[0]

    if is_english(name):

        android_english.append(app)

        

for app in ios:

    name = app[1]

    if is_english(name):

        ios_english.append(app)

        

explore_data(android_english, 0, 3, True)

print('\n')

explore_data(ios_english, 0, 3, True)
## In this project we want to analyse only the apps which are free, hence filtering the free apps



android_final = []

ios_final = []



for app in android_english:

    price = app[7]

    if price == '0':

        android_final.append(app)

        

for app in ios_english:

    price = app[4]

    if price == '0.0':

        ios_final.append(app)

        

print(len(android_final))

print(len(ios_final))
##To know the most common genres of apps, let's build a frequency table for the prime_genre column of the App Store data set, and the Genres and Category columns of the Google Play data set.

def freq_table(dataset, index):

    table = {}

    total = 0

    

    for row in dataset:

        total += 1

        value = row[index]

        if value in table:

            table[value] += 1

        else:

            table[value] = 1

    

    table_percentages = {}

    for key in table:

        percentage = (table[key] / total) * 100

        table_percentages[key] = percentage 

    

    return table_percentages





def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key)

        table_display.append(key_val_as_tuple)

        

    table_sorted = sorted(table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
#For app store data



display_table(ios_final, -5)
display_table(android_final, -4)
## Most popular apps by genre in app store



genres_ios = freq_table(ios_final, -5)



for genre in genres_ios:

    total = 0

    len_genre = 0

    for app in ios_final:

        genre_app = app[-5]

        if genre_app == genre:            

            n_ratings = float(app[5])

            total += n_ratings

            len_genre += 1

    avg_n_ratings = total / len_genre

    print(genre, ':', avg_n_ratings)
#For the Google Play market, we actually have data about the number of installs, so we should be able to get a clearer picture about genre popularity. 

display_table(android_final, 5)
#average number of installs for each genre (category).

categories_android = freq_table(android_final, 1)



for category in categories_android:

    total = 0

    len_category = 0

    for app in android_final:

        category_app = app[1]

        if category_app == category:            

            n_installs = app[5]

            n_installs = n_installs.replace(',', '')

            n_installs = n_installs.replace('+', '')

            total += float(n_installs)

            len_category += 1

    avg_n_installs = total / len_category

    print(category, ':', avg_n_installs)