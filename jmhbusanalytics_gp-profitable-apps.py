#install package

#! pip install reader
# Open both data sets and seperate header from data

# This function allows us to open a csv file, given the file_name paramater.

def opening_csv(file_name):

    opened_file = open(file_name, encoding = "utf8")

    from csv import reader

    read_file = reader(opened_file)

    data = list(read_file)

    return data

    

#Saving googleplaystore.csv, AppleStore.csv

google_play = opening_csv('../input/google-play-store-apps/googleplaystore.csv')

apple_store = opening_csv('../input/app-store-apple-data-set-10k-apps/AppleStore.csv')



#Seperating header and data from csv file

google_play_header = google_play[0]

google_play_data = google_play[1:]



apple_store_header = apple_store[0]

apple_store_data = apple_store[1:]











#Create a function called explore_data to help us look at data set.

def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]    

    for row in dataset_slice:

        print(row)

        print('\n') # adds a new (empty) line after each row

        

    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))
#Explore the google_play_data and apple_store_data with the explore_data() function

print(google_play_header)

print('\n')

explore_data(google_play_data,0,2,True)



print('\n')



print(apple_store_header)

print('\n')

explore_data(apple_store_data,0,2,True)
# A function that checks missing data points in rows



def missing_row(data_set, data_header):

    for row in data_set:

        header_length = len(data_header)

        row_length = len(row)

        if row_length != header_length:

            print(row)

            print('\n')

            print(data_set.index(row))
#Using missing row() function on google data and apple data

print(google_play_header)

print('\n')

print(missing_row(google_play_data, google_play_header))

print('\n')

print(missing_row(apple_store_data, apple_store_header))





#Deleting the row with a missing data point

del google_play_data[10472]

#check if we deleted the row successfully.

print(missing_row(google_play_data, google_play_header))

print(google_play_data[10472])
#create a function that will check for duplicated data points



def check_duplicates(data_set, a_number):

    duplicated_data = []

    unique_data = []

    for row in data_set:

        data = row[a_number]

        if data in unique_data:

            duplicated_data.append(data)

        else:

            unique_data.append(data)

    print('Number of duplicated data points:', len(duplicated_data))

    print('\n')

    print('Examples of duplicated data points:', duplicated_data[:5])

    print('\n')

    print('Example of unique data points:', unique_data[:5])

    
#Check data sets for duplicates.



google_play_app_dup = check_duplicates(google_play_data, 0)

print(google_play_app_dup)
for row in google_play_data:

    name = row[0]

    if name == 'Instagram':

        print(row)
#Cleaning out the duplicates by keeping the rows with the most reviews

reviews_max = {}



for row in google_play_data:

    name = row[0]

    n_reviews = float(row[3])

    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews

    elif name not in reviews_max:

        reviews_max[name] = n_reviews

        

print(len(reviews_max))
#creating new data set with removed duplicates

google_play_clean = []

already_added = []



for row in google_play_data:

    name = row[0]

    n_reviews = float(row[3])

    if reviews_max[name] == n_reviews and name not in already_added:

        google_play_clean.append(row)

        already_added.append(name)

        
explore_data(google_play_clean,0,3,True)
#ord() Test

print('The letter "a" is also', ord('a'))

print('The letter "A" is also', ord('A'))



# function checks if a string goes beyond the english character list, (a - z, A - Z)

def english_text(string):

    for character in string:

        if ord(character) > 127:

            return False

    return True

            

#checking if function works

print(english_text('Instagram'))

print(english_text('爱奇艺PPS -《欢乐颂2》电视剧热播'))

        

#apps with non-english letters

print(english_text('Docs To Go™ Free Office Suite'))

print(english_text('Instachat 😜'))



print(ord('™'))

print(ord('😜'))
#updated version of english text check with added strike 3 rule.

def updated_english_text(string):

    non_ascii = 0

    for character in string:

        if ord(character) > 127:

            non_ascii +=1

    if non_ascii > 3:

        return False

    else:

        return True

    

print(updated_english_text('Docs To Go™ Free Office Suite'))

print(updated_english_text('Instachat 😜'))

print(updated_english_text('爱奇艺PPS -《欢乐颂2》电视剧热播'))
google_english = []

apple_english = []



for app in google_play_clean:

    name = app[0]

    if updated_english_text(name):

        google_english.append(app)

        

for app in apple_store_data:

    name = app[2]

    if updated_english_text(name):

        apple_english.append(app)

        

print('Google Play')        

explore_data(google_english,0,2,True)

print('\n')

print('Apple Store')

explore_data(apple_english,0,2,True)

        
# Filter data set for rows that have a price = $0.

google_free = []

apple_free = []



for row in google_english:

    price = row[7]

    if price == '0':

        google_free.append(row)

        

for row in apple_english:

    price = row[5]

    if price == '0':

        apple_free.append(row)

        

print(len(google_free))

print(len(apple_free))
print('Google Play')

print(google_play_header)

print('\n')

print('Apple Store')

print(apple_store_header)
# Create a frequency table that will show percentage of data

def freq_table(dataset, index):

    frequency = {}

    total = 0

    for row in dataset:

        genre = row[index]

        total += 1

        if genre in frequency:

            frequency[genre] += 1

        else:

            frequency[genre] = 1

    table_percentages = {}

    for key in frequency:

        percentage = (frequency[key]/total) * 100

        table_percentages[key] = percentage

    return table_percentages



# Display percentages in descending order

def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key)

        table_display.append(key_val_as_tuple)



    table_sorted = sorted(table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
#Apple Store Frequency: Prime Genre

print('Apple Store Frequency: Prime Genre')

print('\n')

display_table(apple_free, -5)

#Google Play Frequency: Category

print('Google Play Frequency: Category')

print('\n')

display_table(google_free, 1)
#Google Play Frequency: Genres

print('Google Play Frequency: Genres')

print('\n')

display_table(google_free, -5)
#Find the frequency for genre(iOS)

apple_genre = freq_table(apple_free, -5)



for genre in apple_genre:

    total = 0 

    len_genre = 0

    for app in apple_free:

        genre_app = app[-5]

        if genre_app == genre:

            user_ratings = float(app[6])

            total += user_ratings

            len_genre += 1

    avg_genre_rating = total / len_genre

    print(genre, ':', avg_genre_rating)

    
# Find the Frequency for Category(Andriod)

google_categories = freq_table(google_free,1)



for category in google_categories:

    total = 0

    len_category = 0

    for app in google_free:

        category_app = app[1]

        if category_app == category:

            n_installs = app[5]

            n_installs = n_installs.replace('+','')

            n_installs = n_installs.replace(',','')

            total += float(n_installs)

            len_category += 1

    avg_num_installs = total / len_category

    print(category, ':', avg_num_installs)
#Segmenting app names by number installs

for app in google_free:

    if app[1] == "SHOPPING" and (app[5] == '1,000,000,000+' or app[5] == '5,000,000+' or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])
#segmenting app names by number of installs

for app in google_free: 

    if app[1] == "EDUCATION" and (app[5] == '1,000,000,000+' or app[5] == '5,000,000+' or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])
#Finding the average number of installs for EDUCATION category

under_100_m = []



for app in google_free:

    n_installs = app[5]

    n_installs = n_installs.replace(',', '')

    n_installs = n_installs.replace('+', '')

    if (app[1] == 'SHOPPING') and (float(n_installs) < 100000000):

        under_100_m.append(float(n_installs))

        

sum(under_100_m) / len(under_100_m)
#looking at the EDUCATION category and the number of installs.

for app in google_free:

    if app[1] == 'SHOPPING':

        print(app[0], ':', app[5])