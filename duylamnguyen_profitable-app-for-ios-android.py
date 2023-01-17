from csv import reader



#Google Play data set:

opened_file = open('../input/google-play-store-apps/googleplaystore.csv')

read_file = reader(opened_file)

android = list(read_file)

android_header = android[0]

android_body = android[1:]



#Apple Store data set:

opened_file = open('../input/app-store-apple-data-set-10k-apps/AppleStore.csv')

read_file = reader(opened_file)

ios = list(read_file)

ios_header = ios[0]

ios_body = ios[1:]
def f_explore_data(dataset, start, end, no_rows_columns = False):

    data_slice = dataset[start:end]

    for row in data_slice:

        print(row)

        print('\n')

    if no_rows_columns:

        print('Number of rows: ', len(dataset))

        print('Number of columns: ', len(dataset[0]))
print(android_header)

print('\n')

f_explore_data(android_body, 0, 3, True)
print(ios_header)

print('\n')

f_explore_data(ios_body, 0, 3, True)
print(android_header)

print('\n')

print(android_body[10472])
del android_body[10472]
print(android_header)

print('\n')

for row in android_body:

    v_app_name = row[0]

    if v_app_name == 'Instagram':

        print(row)
def f_find_duplication(dataset, index):

    l_duplicate_app = []

    l_unique_app = []

    

    for row in dataset:

        v_app_name = row[index]

        if v_app_name in l_unique_app:

            l_duplicate_app.append(v_app_name)

        else:

            l_unique_app.append(v_app_name)

    

    print('Number of duplicate apps: ', len(l_duplicate_app))

    print('Number of unique apps: ', len(l_unique_app))
f_find_duplication(android_body, 0)
f_find_duplication(ios_body, 1)
d_reviews_max = {}



for row in android_body:

    v_app_name = row[0]

    v_reviews_max = int(row[3])

    if (v_app_name in d_reviews_max) and (v_reviews_max > d_reviews_max[v_app_name]):

        d_reviews_max[v_app_name] = v_reviews_max

    elif v_app_name not in d_reviews_max:

        d_reviews_max[v_app_name] = v_reviews_max



print('Number of unique apps in d_reviews_max dictionary: ', len(d_reviews_max))
print('Number of Instagram\'s reviews: ',d_reviews_max['Instagram'])
l_android_clean = []

already_added = []



for row in android_body:

    v_app_name = row[0]

    v_reviews_max = int(row[3])

    if (v_app_name not in already_added) and (v_reviews_max == d_reviews_max[v_app_name]):

        l_android_clean.append(row)

        already_added.append(v_app_name)

        

print('Number of unique apps in l_android_clean: ',len(l_android_clean))
print(android_header)

print('\n')

for row in l_android_clean:

    v_app_name = row[0]

    if v_app_name == 'Instagram':

        print(row)
def f_is_english(v_string):

    for v_character in v_string:

        if ord(v_character) > 127:

            return False

    return True
f_is_english('Instagram')
f_is_english('爱奇艺PPS -《欢乐颂2》电视剧热播')
f_is_english('Docs To Go™ Free Office Suite')
f_is_english('Instachat 😜')
def f_is_english(v_string):

    v_count = 0

    

    for character in v_string:

        if ord(character) > 127:

            v_count += 1

            if v_count > 3:

                return False

    return True
f_is_english('Docs To Go™ Free Office Suite')
f_is_english('Instachat 😜')
f_is_english('爱奇艺PPS -《欢乐颂2》电视剧热播')
l_android_english = []

l_ios_english = []



for row in l_android_clean:

    v_app_name = row[0]

    if f_is_english(v_app_name):

        l_android_english.append(row)

        

for row in ios_body:

    v_app_name = row[2]

    if f_is_english(v_app_name):

        l_ios_english.append(row)

        

print('Number of english Android apps: ', len(l_android_english))

print('Number of english iOS apps: ', len(l_ios_english))
l_android_final = []

l_ios_final = []



for row in l_android_english:

    price = row[7]

    price = price.replace('$','')

    price = price.replace('Everyone', '0')

    price = float(price)

    if price == 0.0:

        l_android_final.append(row)

        

for row in l_ios_english:

    price = float(row[5])

    if price == 0.0:

        l_ios_final.append(row)

        

print('Number of free Android apps: ',len(l_android_final))

print('Number of free iOS apps: ',len(l_ios_final))
def f_freq_table(dataset, index):

    d_freq_table = {}

    

    for row in dataset:

        v_genre = row[index]

        if v_genre in d_freq_table:

            d_freq_table[v_genre] += 1

        else:

            d_freq_table[v_genre] = 1

    

    d_freq_table_percent = {}

    for key in d_freq_table:

        v_genre_percent = (d_freq_table[key] / len(dataset)) * 100

        d_freq_table_percent[key] = v_genre_percent

        

    return d_freq_table_percent



def f_freq_table_desc(dataset, index):

    d_table = f_freq_table(dataset, index)

    l_table_display = []

    

    for key in d_table:

        v_tuple = (float(d_table[key]), key)

        l_table_display.append(v_tuple)

        

    l_table_display = sorted(l_table_display, reverse = True)

    for row in l_table_display:

        print(row[1], ':', row[0])
f_freq_table_desc(l_android_final, 1)
f_freq_table_desc(l_ios_final, 12)
d_popular_genre = {}



for row in l_android_final:

    v_genre = row[1]

    v_installs = row[5]

    v_installs = v_installs.replace('+','')

    v_installs = v_installs.replace(',','')

    v_installs = int(v_installs)

    if v_genre in d_popular_genre:

        d_popular_genre[v_genre] += v_installs

    else:

        d_popular_genre[v_genre] = v_installs

        

l_table_display = []



for key in d_popular_genre:

    v_tuple = (d_popular_genre[key], key)

    l_table_display.append(v_tuple)

    

l_table_display = sorted(l_table_display, reverse = True)

for row in l_table_display:

    print(row[1], ':', row[0])
d_popular_genre = {}



for row in l_ios_final:

    v_genre = row[12]

    v_ratings = int(row[6])

    if v_genre in d_popular_genre:

        d_popular_genre[v_genre] += v_ratings

    else:

        d_popular_genre[v_genre] = v_ratings

        

l_table_display = []



for key in d_popular_genre:

    v_tuple = (d_popular_genre[key], key)

    l_table_display.append(v_tuple)

    

l_table_display = sorted(l_table_display, reverse = True)

for row in l_table_display:

    print(row[1], ':', row[0])
##Google Play

d_popular_game = {}



for row in l_android_final:

    v_app_name = row[0]

    v_genre = row[1]

    v_installs = row[5]

    v_installs = v_installs.replace('+','')

    v_installs = v_installs.replace(',', '')

    v_installs = int(v_installs)

    if v_genre == 'GAME':

        d_popular_game[v_app_name] = v_installs

        

l_popular_game = []



for key in d_popular_game:

    v_tuple = (d_popular_game[key], key)

    l_popular_game.append(v_tuple)



l_popular_game = sorted(l_popular_game, reverse = True)



for row in l_popular_game[:6]:

    print(row[1], ':',row[0])
##Apple Store

d_popular_game = {}



for row in l_ios_final:

    v_app_name = row[2]

    v_genre = row[12]

    v_ratings = int(row[6])

    if v_genre == 'Games':

        d_popular_game[v_app_name] = v_ratings

        

l_popular_game = []



for key in d_popular_game:

    v_tuple = (d_popular_game[key], key)

    l_popular_game.append(v_tuple)



l_popular_game = sorted(l_popular_game, reverse = True)



for row in l_popular_game[:6]:

    print(row[1], ':',row[0])