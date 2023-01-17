from csv import reader

#opening applestore data

df_apple_store=open('../input/app-dataset/AppleStore.csv',encoding="utf8")

df_apple_read=reader(df_apple_store)

df_list_apple=list(df_apple_read)

df_header_apple=df_list_apple[0]

df_data_apple=df_list_apple[1:]

#opening google playstore data

df_googleplaystore=open('../input/app-dataset/googleplaystore.csv',encoding="utf8")

df_google_read=reader(df_googleplaystore)

df_list_google=list(df_google_read)

df_header_google=df_list_google[0]

df_data_google=df_list_google[1:]

def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]    

    for row in dataset_slice:

        print(row)

        print('\n') 



    if rows_and_columns:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))

explore_data(df_data_google,0,6,True)
#it is findout that there is a missing data at index number 10472

#for the sake of accurate analysis that data is removed

del df_data_google[10472]

unique_data=[]

duplicate_data=[]

for row in df_list_apple:

    name=row[0]

    if name in unique_data:

        duplicate_data.append(name)

    else:

        unique_data.append(name)

print(unique_data)

print(duplicate_data)#we get an empty list as result which means 

#there is no duplicate data

unique_data=[]

duplicate_data=[]

for row in df_list_google:

    name=row[0]

    if name in unique_data:

        duplicate_data.append(name)

    else:

        unique_data.append(name)



print(duplicate_data)#we get 1181 duplicate 

                          #data on googleplaystore dataset

print(len(duplicate_data))

df_header_google
reviews_max={}#creating a dictonary for storing highest review values for app keyword

for row in df_data_google:

    app_name=row[0]

    n_reviews=float(row[3])

    if app_name in reviews_max and reviews_max[app_name]<n_reviews:

        reviews_max[app_name]=n_reviews

    if app_name not in reviews_max:

        reviews_max[app_name]=n_reviews

        

print(len(reviews_max))
print((reviews_max))
android_clean = []

already_added = []



for app in df_data_google:

    app_name = app[0]

    n_reviews = float(app[3])

    

    if (reviews_max[app_name] == n_reviews) and (app_name not in already_added):

        android_clean.append(app)

        already_added.append(app_name)
print(android_clean)
print(already_added)
def is_english(string):

    non_ascii = 0

    

    for character in string:

        if ord(character) > 127:

            non_ascii += 1

    

    if non_ascii > 3: #ASCII value of english charactors are below 127.if more than 3 non english charactors are detected 

        return False #remove that app from analysis

    else:

        return True



print(is_english('Docs To Go™ Free Office Suite'))#only have one non-english charactor.so it takes into account

print(is_english('Instachat 😜'))#only have one non-english charactor.so it takes into account
android_english = []

ios_english = []



for app in android_clean:#iterate through anaroid apps for finding english app names only 

    name = app[0]

    if is_english(name):

        android_english.append(app)

        

for app in df_data_apple:#iterate through ios apps for finding english app names only 

    name = app[1]

    if is_english(name):

        ios_english.append(app)

        

explore_data(android_english, 0, 3, True)

print('\n')

explore_data(ios_english, 0, 3, True)
android_final = []

ios_final = []



for app in android_english:#iterate through android apps for finding free apps only

    price = app[7]

    if price == '0':

        android_final.append(app)

        

for app in ios_english:#iterate through ios apps for finding free apps only

    price = app[4]

    if price == '0.0':

        ios_final.append(app)

        

print(len(android_final))

print(len(ios_final))
df_header_google
def freq_table(dataset, index):

    table = {}

    total = 0

    

    for row in dataset:

        total += 1 #counting total number of data

        value = row[index]

        if value in table:

            table[value] += 1

        else:

            table[value] = 1

    

    table_percentages = {}

    for key in table:

        percentage = (table[key] / total) * 100 #finding percentage

        table_percentages[key] = percentage  #adding percentage value to key in new dictonary

    

    return table_percentages





def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key)#creating tuple data type from dictonary data type

        table_display.append(key_val_as_tuple)

        

    table_sorted = sorted(table_display, reverse = True)#sorting in ascending order

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
display_table(android_final,-4)




display_table(ios_final, -5)



df_header_apple
genres_ios = freq_table(ios_final, -5)



for genre in genres_ios:

    total = 0

    len_genre = 0

    for app in ios_final:

        genre_app = app[-5]

        if genre_app == genre:            

            n_ratings = float(app[5])

            total += n_ratings #adding number of user reviews together 

            len_genre += 1

    avg_n_ratings = total / len_genre

    print(genre, ':', avg_n_ratings)

for app in ios_final:

    if app[-5] == 'Navigation':

        print(app[1], ':', app[5])
df_header_google
display_table(android_final, 5)
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
for app in android_final:

    if app[1] == 'COMMUNICATION' and (app[5] == '1,000,000,000+'

                                      or app[5] == '500,000,000+'

                                      or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])