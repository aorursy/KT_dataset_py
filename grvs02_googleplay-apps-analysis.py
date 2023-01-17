from csv import reader

opened_file = open('Desktop/googleplaystore.csv', encoding = 'utf-8')

read_file = reader(opened_file)

android = list(read_file)

andr_header = android[0]

android = android[1:]

andr_header

print('\n')

android
#We are now defining a general function for exploring the datasets.let see first how a function works over a dataset.giving example her 

for row in android[1:25]:

    print(row)

    print('\n')

print(len(row))    

    
#Here 'row' is a variable which itereates over a column as there is 13 column so it iterates 13 times and exploring the row details.

#now for exploring a function for dataset opening or exploring.we need certain parameters as an arguments in funtion explore_data.



def explore_data(dataset, start, end, rows_column = False):

    dataset_slice = dataset[start:end]

    for row in dataset_slice:

        print(row)

        print('\n')

        

    if rows_column:

        print('Number of rows:', len(dataset))

        print('Number of columns:', len(dataset[0]))

        

explore_data(android,0,3,True)        

        

    

        
#opening ios dataset,first open/read the dataset

opened_file = open('Desktop/AppleStore.csv', encoding='utf-8')

read_file = reader(opened_file)

ios= list(read_file)

ios_header = ios[0]

ios = ios[1:]

ios_header

print('\n')

ios
explore_data(ios,0,3,True)
#now,Detecting wrong rows which has wrong entries 

print(android[10472])

print('\n')

print(andr_header)
#here as per category details are wrongly filled such as rating can't be increase that 5 and so on.so, we will use del function for dat cleaning procrss here.

print(len(android))

del android[10472]

print(len(android))

#remove duplicate entries, if we see googleplaystore dataset,we find that some entries are repeated and repeated again.first explore duplicate entries.

for app in android:

    name = app[0]

    if name == 'Instagram':

        print(app)

#In total,1,181 cases are duplicates.lets see the genralized function for the same.

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

print('examples of duplicate  apps:', duplicate_apps[:20])



        

#We don't want to count certain apps more than once when we analyze data, so we need to remove the duplicate entries and keep only one entry per app. One thing we could do is remove the duplicate rows randomly, but we could probably find a better way.



#If you examine the rows we printed two cells above for the Instagram app, the main difference happens on the fourth position of each row, which corresponds to the number of reviews. The different numbers show that the data was collected at different times. We can use this to build a criterion for keeping rows. We won't remove rows randomly, but rather we'll keep the rows that have the highest number of reviews because the higher the number of reviews, the more reliable the ratings.



#To do that, we will:

#Create a dictionary where each key is a unique app name, and the value is the highest number of reviews of that app

#Use the dictionary to create a new data set, which will have only one entry per app (and we only select the apps with the highest number of reviews)

#Part Two







reviews_max = {}



for app in android:

    name = app[0]

    n_reviews = float(app[3])

    

    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews

        

    elif name not in reviews_max:

        reviews_max[name] = n_reviews

        

        

        
#In a previous code cell, we found that there are 1,181 cases where an app occurs more than once, so the length of our dictionary (of unique apps) should be equal to the difference between the length of our data set and 1,181.



print('expected length:', len(android) - 1181)

print('actual length:', len(reviews_max))

#Now, let's use the reviews_max dictionary to remove the duplicates. For the duplicate cases, we'll only keep the entries with the highest number of reviews. In the code cell below:



#We start by initializing two empty lists, android_clean and already_added.

#We loop through the android data set, and for every iteration:

#We isolate the name of the app and the number of reviews.

#We add the current row (app) to the android_clean list, and the app name (name) to the already_cleaned list if:

#The number of reviews of the current app matches the number of reviews of that app as described in the reviews_max dictionary; and

#The name of the app is not already in the already_added list. We need to add this supplementary condition to account for those cases where the highest number of reviews of a duplicate app is the same for more than one entry (for example, the Box app has three entries, and the number of reviews is the same). If we just check for reviews_max[name] == n_reviews, we'll still end up with duplicate entries for some apps.

android_clean = []

already_added = []



for app in android:

    name = app[0]

    n_reviews = float(app[3])

    

    if (reviews_max[name] == n_reviews) and (name not in already_added):

        android_clean.append(app)

        already_added.append(name)

        

      

explore_data(android_clean,0,3,True)
#Removing non english data as it has no use in our valuation process

#every character has coressponding ascii value and we can take the benefit of this as limitation stands from 0 to 127.

def is_english(string):

    for char in string:

        if ord(char)>127:

            return False

    else:

        return True

print(is_english('Instagram')) 

print(is_english('爱奇艺PPS -《欢乐颂2》电视剧热播'))

        
#The function seems to work fine, but some English app names use emojis or other symbols (™, — (em dash), – (en dash), etc.) that fall outside of the ASCII range. Because of this, we'll remove useful apps if we use the function in its current form.



print(is_english('Docs To Go™ Free Office Suite'))

print(is_english('Instachat 😜'))



print(ord('™'))

print(ord('😜'))
def is_english(string):

    non_ascii = 0

    for char in string:

        if ord(char)>127:

            non_ascii +=1

    if non_ascii > 3:

        return False

    else:

        return True

    

print(is_english('Docs To Go™ Free Office Suite') )

print(is_english('Instachat 😜'))



        
#The function is still not perfect, and very few non-English apps might get past our filter, but this seems good enough at this point in our analysis — we shouldn't spend too much time on optimization at this point.



#Below, we use the is_english() function to filter out the non-English apps for both data sets



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

explore_data(android_english,0,3,True)



#As we mentioned in the introduction, we only build apps that are free to download and install, and our main source of revenue consists of in-app ads. Our data sets contain both free and non-free apps, and we'll need to isolate only the free apps for our analysis. Below, we isolate the free apps for both our data sets.

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



        
#Most Common Apps by Genre

#Part One

'''As we mentioned in the introduction, our aim is to determine the kinds of apps that are likely to attract more users because our revenue is highly influenced by the number of people using our apps.



To minimize risks and overhead, our validation strategy for an app idea is comprised of three steps:



Build a minimal Android version of the app, and add it to Google Play.

If the app has a good response from users, we then develop it further.

If the app is profitable after six months, we also build an iOS version of the app and add it to the App Store.

Because our end goal is to add the app on both the App Store and Google Play, we need to find app profiles that are successful on both markets. For instance, a profile that might work well for both markets might be a productivity app that makes use of gamification.



Let's begin the analysis by getting a sense of the most common genres for each market. For this, we'll build a frequency table for the prime_genre column of the App Store data set, and the Genres and Category columns of the Google Play data set.



Part Two

We'll build two functions we can use to analyze the frequency tables:



One function to generate frequency tables that show percentages

Another function that we can use to display the percentages in a descending order'''

def freq_table(dataset, index):

    table = {}

    total = 0

    for row in dataset:

        total +=1

        value = row[index]

        if value in table:

            table[value] +=1

        else:

            table[value] = 1

            

    table_percentages = {} 

    for key in table:

        percentage = (table[key]/total)* 100

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

        

      



                
display_table(android_final, 1)
display_table(android_final, -4)
'''The difference between the Genres and the Category columns is not crystal clear, but one thing we can notice is that the Genres column is much more granular (it has more categories). We're only looking for the bigger picture at the moment, so we'll only work with the Category column moving forward.

Up to this point, we found that the App Store is dominated by apps designed for fun, while Google Play shows a more balanced landscape of both practical and for-fun apps. Now we'd like to get an idea about the kind of apps that have most users.



Most Popular Apps by Genre on the App Store

One way to find out what genres are the most popular (have the most users) is to calculate the average number of installs for each app genre. For the Google Play data set, we can find this information in the Installs column, but for the App Store data set this information is missing. As a workaround, we'll take the total number of user ratings as a proxy, which we can find in the rating_count_tot app.'''

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
#On average, navigation apps have the highest number of user reviews, but this figure is heavily influenced by Waze and Google Maps, which have close to half a million user reviews together:



for app in ios_final:

    if app[-5] == 'Navigation':

        print(app[1], ':', app[5])

        
#The same pattern applies to social networking apps, where the average number is heavily influenced by a few giants like Facebook, Pinterest, Skype, etc. Same applies to music apps, where a few big players like Pandora, Spotify, and Shazam heavily influence the average number.

#Our aim is to find popular genres, but navigation, social networking or music apps might seem more popular than they really are. The average number of ratings seem to be skewed by very few apps which have hundreds of thousands of user ratings, while the other apps may struggle to get past the 10,000 threshold. We could get a better picture by removing these extremely popular apps for each genre and then rework the averages, but we'll leave this level of detail for later.



for app in ios_final:

    if app[-5] == 'Reference':

        print(app[1], ':', app[5])
'''However, this niche seems to show some potential. One thing we could do is take another popular book and turn it into an app where we could add different features besides the raw version of the book. This might include daily quotes from the book, an audio version of the book, quizzes about the book, etc. On top of that, we could also embed a dictionary within the app, so users don't need to exit our app to look up words in an external app.



This idea seems to fit well with the fact that the App Store is dominated by for-fun apps. This suggests the market might be a bit saturated with for-fun apps, which means a practical app might have more of a chance to stand out among the huge number of apps on the App Store.



Other genres that seem popular include weather, book, food and drink, or finance. The book genre seem to overlap a bit with the app idea we described above, but the other genres don't seem too interesting to us:



Weather apps — people generally don't spend too much time in-app, and the chances of making profit from in-app adds are low. Also, getting reliable live weather data may require us to connect our apps to non-free APIs.



Food and drink — examples here include Starbucks, Dunkin' Donuts, McDonald's, etc. So making a popular food and drink app requires actual cooking and a delivery service, which is outside the scope of our company.



Finance apps — these apps involve banking, paying bills, money transfer, etc. Building a finance app requires domain knowledge, and we don't want to hire a finance expert just to build an app.



Now let's analyze the Google Play market a bit.'''



'''Most Popular Apps by Genre on Google Play

For the Google Play market, we actually have data about the number of installs, so we should be able to get a clearer picture about genre popularity. However, the install numbers don't seem precise enough — we can see that most values are open-ended (100+, 1,000+, 5,000+, etc.):'''



display_table(android_final, 5)
'''One problem with this data is that is not precise. For instance, we don't know whether an app with 100,000+ installs has 100,000 installs, 200,000, or 350,000. However, we don't need very precise data for our purposes — we only want to get an idea which app genres attract the most users, and we don't need perfect precision with respect to the number of users.



We're going to leave the numbers as they are, which means that we'll consider that an app with 100,000+ installs has 100,000 installs, and an app with 1,000,000+ installs has 1,000,000 installs, and so on.



To perform computations, however, we'll need to convert each install number to float — this means that we need to remove the commas and the plus characters, otherwise the conversion will fail and raise an error. We'll do this directly in the loop below, where we also compute the average number of installs for each genre (category).'''



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

#On average, communication apps have the most installs: 38,456,119. This number is heavily skewed up by a few apps that have over one billion installs (WhatsApp, Facebook Messenger, Skype, Google Chrome, Gmail, and Hangouts), and a few others with over 100 and 500 million installs:

for app in android_final:

    if app[1] == 'COMMUNICATION' and (app[5] == '1,000,000,000+'

                                      or app[5] == '500,000,000+'

                                      or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])

under_100_m = []



for app in android_final:

    n_installs = app[5]

    n_installs = n_installs.replace(',', '')

    n_installs = n_installs.replace('+', '')

    if (app[1] == 'COMMUNICATION') and (float(n_installs) < 100000000):

        under_100_m.append(float(n_installs))



sum(under_100_m) / len(under_100_m)

'''We see the same pattern for the video players category, which is the runner-up with 24,727,872 installs. The market is dominated by apps like Youtube, Google Play Movies & TV, or MX Player. The pattern is repeated for social apps (where we have giants like Facebook, Instagram, Google+, etc.), photography apps (Google Photos and other popular photo editors), or productivity apps (Microsoft Word, Dropbox, Google Calendar, Evernote, etc.).





Again, the main concern is that these app genres might seem more popular than they really are. Moreover, these niches seem to be dominated by a few giants who are hard to compete against.



The game genre seems pretty popular, but previously we found out this part of the market seems a bit saturated, so we'd like to come up with a different app recommendation if possible.



The books and reference genre looks fairly popular as well, with an average number of installs of 8,767,811. It's interesting to explore this in more depth, since we found this genre has some potential to work well on the App Store, and our aim is to recommend an app genre that shows potential for being profitable on both the App Store and Google Play.



Let's take a look at some of the apps from this genre and their number of installs:'''



for app in android_final:

    if app[1] == 'BOOKS_AND_REFERENCE':

        print(app[0], ':', app[5])



#The book and reference genre includes a variety of apps: software for processing and reading ebooks, various collections of libraries, dictionaries, tutorials on programming or languages, etc. It seems there's still a small number of extremely popular apps that skew the average:

for app in android_final:

    if app[1] == 'BOOKS_AND_REFERENCE' and (app[5] == '1,000,000,000+'

                                            or app[5] == '500,000,000+'

                                            or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])

#However, it looks like there are only a few very popular apps, so this market still shows potential. Let's try to get some app ideas based on the kind of apps that are somewhere in the middle in terms of popularity (between 1,000,000 and 100,000,000 downloads):

for app in android_final:

    if app[1] == 'BOOKS_AND_REFERENCE' and (app[5] == '1,000,000+'

                                            or app[5] == '5,000,000+'

                                            or app[5] == '10,000,000+'

                                            or app[5] == '50,000,000+'):

        print(app[0], ':', app[5])

'''This niche seems to be dominated by software for processing and reading ebooks, as well as various collections of libraries and dictionaries, so it's probably not a good idea to build similar apps since there'll be some significant competition.



We also notice there are quite a few apps built around the book Quran, which suggests that building an app around a popular book can be profitable. It seems that taking a popular book (perhaps a more recent book) and turning it into an app could be profitable for both the Google Play and the App Store markets.



However, it looks like the market is already full of libraries, so we need to add some special features besides the raw version of the book. This might include daily quotes from the book, an audio version of the book, quizzes on the book, a forum where people can discuss the book, etc.



Conclusions

In this project, we analyzed data about the App Store and Google Play mobile apps with the goal of recommending an app profile that can be profitable for both markets.



We concluded that taking a popular book (perhaps a more recent book) and turning it into an app could be profitable for both the Google Play and the App Store markets. The markets are already full of libraries, so we need to add some special features besides the raw version of the book. This might include daily quotes from the book, an audio version of the book, quizzes on the book, a forum where people can discuss the book, etc.'''
