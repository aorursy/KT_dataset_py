import numpy as np # linear algebra

import matplotlib.pyplot as plt # plot data

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# reads all data from file

data = pd.read_table("../input/amazon-alexa-reviews/amazon_alexa.tsv")



# highest rated devices

highest_rated = data[(data['rating'] == 5)]



# lowest rated devices

lowest_rated = data[(data['rating'] <= 2)]



#calculate median rating



# average rating of devices

rating_sum = 0



for rating in data.rating:

    rating_sum += rating



median_rating = rating_sum / len(data.rating)



average_rated = data[data.rating == median_rating]





# highest rated reviews

highest_rated_reviews = highest_rated['verified_reviews']



# lowest rated reviews

lowest_rated_reviews = lowest_rated['verified_reviews']



# returns object with filtered words from reviews data

def countWordOccurance(reviews, count_range):

    word_count = {}

    # populates review word count dictionary of highest rated

    for review in reviews:

        # splits words in review per space

        words = review.split()

        # counts the occurance of each word

        for word in words:

            w = word.lower()

            if w not in word_count:

                word_count[w] = 1

            else:

                word_count[w] += 1



    filtered = {}

    generic = ('i','the','and','to','it', 'a', 'you', 'my', 'it.', 'is')

    

    # filter based on words that occured more than 300 times

    # ommits generic words that don't reflect sentiment



    for (key, value) in word_count.items():

       # Check if word count is greater than count range

       if value >= count_range and key not in generic:

            filtered[key] = value

            

    # returns filtered dict

    return filtered



# Word occurences based on review data

words_in_highest_r = countWordOccurance(highest_rated_reviews, 250)

words_in_lowest_r = countWordOccurance(lowest_rated_reviews, 30)

# plot chart of filtered words occurance

plt.figure(figsize=(15,10))

plt.bar(range(len(words_in_highest_r)), list(words_in_highest_r.values()), align='center', color="green")

plt.xticks(range(len(words_in_highest_r)), list(words_in_highest_r.keys()))

plt.title('Frequently Used Words in 5 Star Reviews')

plt.xlabel('Words')

plt.ylabel('# of Times Words Appeared')

plt.show()
plt.figure(figsize=(20,10))

plt.bar(range(len(words_in_lowest_r)), list(words_in_lowest_r.values()), align='center', color="violet")

plt.xticks(range(len(words_in_lowest_r)), list(words_in_lowest_r.keys()))

plt.title('Frequently Used Words in 1-2 Star Reviews')

plt.xlabel('Words')

plt.ylabel('# of Times Words Appeared')

plt.show()
plt.figure(figsize=(10,10))

plt.hist(data['rating'], bins=10, color="gold", edgecolor="black")

plt.plot(median_rating)

plt.title('Number of Ratings of Alexa Products')

plt.xlabel('Ratings')

plt.ylabel('# of Reviews')

plt.show()
plt.figure(figsize=(30,10))

plt.hist(highest_rated['variation'],edgecolor="black", bins=60)

plt.hist(lowest_rated['variation'], edgecolor="black", bins=60)

plt.title('5 Star Reviews vs 1 Star Reviews By Device Variation')

plt.xlabel('Device Variant')

plt.ylabel('# of Reviews')

plt.legend()

plt.show()
plt.figure(figsize=(30,10))

plt.hist(highest_rated.variation, edgecolor="black", color="teal", bins=50)

plt.title('Alexa Variants Highest Rated')

plt.ylabel('# of 5 Star Reviews')

plt.xlabel('Variants')

plt.show()
plt.figure(figsize=(30,10))

plt.hist(lowest_rated.variation, edgecolor="black", color="red", bins=50)

plt.title('Alexa Variants Lowest Rated')

plt.ylabel('# of 1 Star Reviews')

plt.xlabel('Variants')

plt.show()