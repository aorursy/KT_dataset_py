opened_file = open('../input/AppleStore.csv', encoding='utf8')

from csv import reader

read_file = reader(opened_file)

apple = list(read_file)

apple_header = apple[0]

apple = apple[1:]



opened_file = open('../input/googleplaystore.csv', encoding='utf8')

read_file = reader(opened_file)

google = list(read_file)

google_header = google[0]

google = google[1:]
def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]    

    for row in dataset_slice:

        print(row)

        print('\n') # adds a new (empty) line after each row



    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))
explore_data(google, 10472, 10474, False)
del google[10472]
duplicate_apps = []

unique_apps = []



for app in google:

    name = app[0]

    if name in unique_apps:

        duplicate_apps.append(name)

    else:

        unique_apps.append(name)

        

print('Number of duplicate apps: ', len(duplicate_apps))

print('\n')

print('Examples of duplicate apps: ', duplicate_apps[:15]) #print a small handful of the duplicate result set
#Expected length of Google data set (9659):

print('Expected length:', len(google) - 1181)
name_and_reviews = {'Instagram': 66577313, 'Facebook': 78158306}

print('LinkedIn' not in name_and_reviews) #True, it is NOT in the dictionary.

print('Instagram' not in name_and_reviews) #False, it IS in the dictionary.
reviews_max = {}



for app in google:

    name = app[0] #app name

    n_reviews = float(app[3]) #number of reviews

    

    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews

        

    if name not in reviews_max:

        reviews_max[name] = n_reviews
print('Expected length:', len(google) - 1181) #9659

print('Actual length:', len(reviews_max)) #9659
android_clean = [] #holds cleaned data set

already_added = [] #holds app names



for app in google:

    name = app[0] #app name

    n_reviews = float(app[3]) #number of reviews

    

    if name not in already_added and reviews_max[name] == n_reviews:

        android_clean.append(app)

        already_added.append(name)
print('Actual length:', len(android_clean)) #9659
def determine_english(string):

    for char in string:

        if ord(char) > 127:

            return False

    return True



determine_english('Instagram') #True

determine_english('爱奇艺PPS -《欢乐颂2》电视剧热播') #False

determine_english('Docs To Go™ Free Office Suite') #False

determine_english('Instachat 😜') #False
def determine_english(string):

    char_counter = 0

    for char in string:

        if ord(char) > 127:

            char_counter += 1

            

    if char_counter > 3:

        return False

    else:

        return True

    

determine_english('Instagram') #True

determine_english('爱奇艺PPS -《欢乐颂2》电视剧热播') #False

determine_english('Docs To Go™ Free Office Suite') #True

determine_english('Instachat 😜') #True

determine_english('Instachat 😜😜') #True

determine_english('Instachat 😜😜😜') #True

determine_english('Instachat 😜😜😜😜') #False
android_english = []

apple_english = []



for app in android_clean:

    name = app[0]

    if determine_english(name): 

        android_english.append(app)

        

for app in apple:

    name = app[2]

    if determine_english(name): 

        apple_english.append(app)

        

explore_data(android_english, 0, 3, True)

print('\n')

explore_data(apple_english, 0, 3, True)
android_free = []

apple_free = []



for app in android_english:

    price = app[7]

    if price == '0':

        android_free.append(app)

        

for app in apple_english:

    price = app[5]

    if price == '0':

        apple_free.append(app)

        

print('Android: ', len(android_free))

print('Apple: ', len(apple_free))
def freq_table(dataset, index):

    freq_dictionary = {}

    freq_dictionary_percent = {}

    total = 0

    

    #Add genres to dictionary freq_dictionary:

    for row in dataset:

        total += 1

        genre = row[index]

        if genre in freq_dictionary:

            freq_dictionary[genre] += 1

        else:

            freq_dictionary[genre] = 1

            

    #Turn values into percentages:

    for key in freq_dictionary:

        percentage = (freq_dictionary[key] / total) * 100

        freq_dictionary_percent[key] = percentage

        

    return freq_dictionary_percent





def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key) #turn the dictionary row into tuple

        table_display.append(key_val_as_tuple)



    table_sorted = sorted(table_display, reverse = True) #use sorted() to sort the table_display tuple

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
display_table(apple_english, 12) #prime_genre column
display_table(android_english, 1) #category column
display_table(android_english, 9) #Genres