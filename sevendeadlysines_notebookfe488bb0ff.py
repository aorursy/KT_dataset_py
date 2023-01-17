# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#I used a helpful tutorial from SRK to learn how to organize the XGB



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb 

import scipy as sp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print('Files available as inputs')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



print('Reading in data...')

input_path = '../input/'

train_file = input_path + 'train.json'

test_file = input_path + 'test.json'



train_data = pd.read_json(train_file)

test_data = pd.read_json(test_file)



#Print out the first row to look at the headers for the data

print('                  ')

print('Here is an example of our data:')

print(train_data.iloc[0,:])



#Convert out labels from strings into a NumPy Array of integers

label_dictionary = {'high':2, 'medium':1, 'low':0 }

train_labels = np.array(train_data['interest_level'].apply(lambda x: label_dictionary[x]))



#Grab a few super elementary features

print(' ')

# #1: Number of features listed

print('Calculating the number of features...')

train_data['how_many_features'] = train_data['features'].apply(len)

test_data['how_many_features'] = test_data['features'].apply(len)



# #2: Number of photos included

print('Calculating the number of photos...')

train_data['how_many_photos'] = train_data['photos'].apply(len)

test_data['how_many_photos'] = test_data['photos'].apply(len)



# #3: Length of the description (as word count)

print('Calculating description lengths...')

#Do this for the training data

desc_lengths = np.zeros(len(train_data['description']))

for this_desc in range(0,len(desc_lengths)):

    desc_lengths[this_desc] = len(train_data['description'].iloc[this_desc].split())



train_data['description_length'] = desc_lengths



#Do this for the testing data

desc_lengths = np.zeros(len(test_data['description']))

for this_desc in range(0,len(desc_lengths)):

    desc_lengths[this_desc] = len(test_data['description'].iloc[this_desc].split())



test_data['description_length'] = desc_lengths



print('All features added!')
#Remove outliers from the prices using the IQR method

Q1 = np.percentile(train_data.price.values,25)

Q3 = np.percentile(train_data.price.values,75)

IQR = Q3-Q1

price_outlier_cutoff = Q3+(IQR*3)

print(['Price Outlier Cutoff: ' + str(price_outlier_cutoff)])

#Replace anything over the cutoff with the cutoff

train_data.loc[(train_data['price']>price_outlier_cutoff),'price'] = price_outlier_cutoff

test_data.loc[(test_data['price']>price_outlier_cutoff),'price'] = price_outlier_cutoff



sns.distplot(train_data.price.values, bins=50, kde=True,color='c')

plt.xlabel('price', fontsize=12)

plt.title('Price Distribution')

plt.show()
#Remove outliers from the latitudes and longitudes using the IQR method



#Calculate the latitude outlier cutoffs

Q1 = np.percentile(train_data.latitude.values,25)

Q3 = np.percentile(train_data.latitude.values,75)

IQR = Q3-Q1

upper_lat_outlier_cutoff = Q3+(IQR*3)

lower_lat_outlier_cutoff = Q1-(IQR*3)



#Adjust the latitudes

train_data.loc[(train_data['latitude']>upper_lat_outlier_cutoff),'latitude'] = upper_lat_outlier_cutoff

train_data.loc[(train_data['latitude']<lower_lat_outlier_cutoff),'latitude'] = lower_lat_outlier_cutoff

test_data.loc[(test_data['latitude']>upper_lat_outlier_cutoff),'latitude'] = upper_lat_outlier_cutoff

test_data.loc[(test_data['latitude']<lower_lat_outlier_cutoff),'latitude'] = lower_lat_outlier_cutoff



#Calculate the longitude outlier cutoffs

Q1 = np.percentile(train_data.longitude.values,25)

Q3 = np.percentile(train_data.longitude.values,75)

IQR = Q3-Q1

upper_long_outlier_cutoff = Q3+(IQR*3)

lower_long_outlier_cutoff = Q1-(IQR*3)





#Adjust the longitudes

train_data.loc[(train_data['longitude']>upper_long_outlier_cutoff),'longitude'] = upper_long_outlier_cutoff

train_data.loc[(train_data['longitude']<lower_long_outlier_cutoff),'longitude'] = lower_long_outlier_cutoff

test_data.loc[(test_data['longitude']>upper_long_outlier_cutoff),'longitude'] = upper_long_outlier_cutoff

test_data.loc[(test_data['longitude']<lower_long_outlier_cutoff),'longitude'] = lower_long_outlier_cutoff



sns.distplot(train_data.latitude.values, bins=50, kde=True,color='c')

plt.xlabel('latitudes', fontsize=12)

plt.show()

sns.distplot(train_data.longitude.values, bins=50, kde=True,color='c')

plt.xlabel('longitudes', fontsize=12)

plt.show()
#Calculate each the mean price and interest in the neighborhood (i.e. K neighbors) to see if its over- or under- priced

k_neighbors = 20



prices = np.array(train_data.price.values)

lats = np.array(train_data.latitude.values)

longs = np.array(train_data.longitude.values)

train_coords = np.transpose(np.array(np.append([lats], [longs],axis=0)))



#Loop through each point, use a KDtree to get K nearest neighbors

tree = sp.spatial.KDTree(train_coords)

print('Created the KD Tree!')



print('Calculating neighborhood price and neighborhood interest for training data...')

neighborhood_price = np.zeros(len(train_coords))

neighborhood_interest = np.zeros(len(train_coords))

interest_differential = np.zeros(len(train_coords))

for this_apt in range(0,len(train_coords)):

    dists, nearneighbors = tree.query(train_coords[this_apt,:], k=20, eps=0, p=1)

    neighborhood_price[this_apt] = np.mean(prices[nearneighbors])

    neighborhood_interest[this_apt] = np.mean(train_labels[nearneighbors])

    interest_differential[this_apt] = np.array(train_labels[this_apt] - neighborhood_interest[this_apt])



train_data['neighborhood_price'] = neighborhood_price

train_data['neighborhood_interest'] = neighborhood_interest

    

print('Calculating neighborhood price and neighborhood interest for testing data...')

#Now do the same calculation for the testing data (using the tree we built from training)

#test_prices = np.array(test_data.price.values)

lats = np.array(test_data.latitude.values)

longs = np.array(test_data.longitude.values)

test_coords = np.transpose(np.array(np.append([lats], [longs],axis=0)))



neighborhood_price = np.zeros(len(test_coords))

neighborhood_interest = np.zeros(len(test_coords))

for this_apt in range(0,len(test_coords)):

    dists, nearneighbors = tree.query(test_coords[this_apt,:], k=20, eps=0, p=1)

    neighborhood_price[this_apt] = np.mean(prices[nearneighbors])

    neighborhood_interest[this_apt] = np.mean(train_labels[nearneighbors])

#    test_price_differential[this_apt] = np.array(test_prices[this_apt] - area_price)



test_data['neighborhood_price'] = neighborhood_price

test_data['neighborhood_interest'] = neighborhood_interest



sns.distplot(train_data.neighborhood_price.values, bins=50, kde=True,color='c')

plt.xlabel('neighborhood price', fontsize=12)

plt.title(['Mean Price of ' + str(k_neighbors) + ' Neighbors' ])

plt.show()



sns.distplot(train_data.neighborhood_interest.values, bins=50, kde=True,color='c')

plt.xlabel('neighborhood interest', fontsize=12)

plt.title(['Mean Interest in ' + str(k_neighbors) + ' Neighbors' ])

plt.show()



#Note we can't use differential interest as a feature, since there's no way to calculate it on the test set

#But we might want to find "gems" which are places that attract much more interest better than their surroundings

sns.distplot(interest_differential, bins=50, kde=True,color='c')

plt.xlabel('Differential Interest', fontsize=12)

plt.title(['Interest - Mean Interest in ' + str(k_neighbors) + ' Neighbors' ])

plt.show()
#Parse all of the words in the descriptions



#Get all of the descriptions for high interest apartments

high_interest = train_data.loc[train_data['interest_level']=='high']



high_descriptions = ''

for ind, row in high_interest.iterrows(): #train_data.iterrows():

    #Add each new discription on to the last, adding one white space to separate them

    high_descriptions = " ".join([high_descriptions , row['description']])

    

print('Parsed all High Interest descriptions')

#Get all of the descriptions for medium interest apartments

medium_interest = train_data.loc[train_data['interest_level']=='medium']



medium_descriptions = ''

for ind, row in medium_interest.iterrows(): #train_data.iterrows():

    #Add each new discription on to the last, adding one white space to separate them

    medium_descriptions = " ".join([medium_descriptions , row['description']])



print('Parsed all Medium Interest descriptions')



#Get all of the descriptions for low interest apartments

low_interest = train_data.loc[train_data['interest_level']=='low']



low_descriptions = ''

for ind, row in low_interest.iterrows(): #train_data.iterrows():

    #Add each new discription on to the last, adding one white space to separate them

    low_descriptions = " ".join([low_descriptions , row['description']])



print('Parsed all Low Interest descriptions')

#Now compile them all together into a single "all_descriptions"



all_descriptions = " ".join([low_descriptions, medium_descriptions, high_descriptions])

low_descriptions = low_descriptions.strip()

medium_descriptions = medium_descriptions.strip()

high_descriptions = high_descriptions.strip()

all_descriptions = all_descriptions.strip()
#Define functions that let us process the text in the descriptions 



#We need one to remove filler words

def RemoveUninformativeWords(word_freq_counter):



    del word_freq_counter['a']

    del word_freq_counter['at']

    del word_freq_counter['an']

    del word_freq_counter['as']

    del word_freq_counter['are']

    del word_freq_counter['and']

    del word_freq_counter['be']        

    del word_freq_counter['by']

    del word_freq_counter['for']

    del word_freq_counter['from']

    del word_freq_counter['has']

    del word_freq_counter['have']

    del word_freq_counter['that']

    del word_freq_counter['this']        

    del word_freq_counter['the']

    del word_freq_counter['you']

    del word_freq_counter['your']

    del word_freq_counter['will']

    del word_freq_counter['with']

    del word_freq_counter['i']

    del word_freq_counter['is']

    del word_freq_counter['in'] 

    del word_freq_counter['it']

    del word_freq_counter['on']

    del word_freq_counter['of']

    del word_freq_counter['or']

    del word_freq_counter['to']

    del word_freq_counter['me']       

    del word_freq_counter['1']

    del word_freq_counter['2']

    del word_freq_counter['3']

    del word_freq_counter['one']

    del word_freq_counter['two']

    del word_freq_counter['three']

    del word_freq_counter['website_redacted']

    del word_freq_counter['kagglemanager@renthop.com']

    del word_freq_counter['kagglemanager@renthop.com<br']

    del word_freq_counter['opportunity.<p><a']

    del word_freq_counter['/><p><a']

    del word_freq_counter['/><br']

    del word_freq_counter['&']

    del word_freq_counter['_']

    del word_freq_counter['-']

    

    return word_freq_counter



#And we need one to count how often these words occcur so we can pick informative features

def WordCountAnalyzer(all_descriptions, how_many_words=100, plots=0, titlestring='Descriptions'):



    #Now plot a histogram of word frequencies from the dictionary so we can make decisions about which to include or not

    from wordcloud import WordCloud

    from collections import Counter



    #Count all of the words in the description and remove bad strings (i.e. ',') and entries (i.e. "a")

    word_freq_counter = Counter(all_descriptions.lower().replace(',','').split())

    word_freq_counter = RemoveUninformativeWords(word_freq_counter)



    #Now make a dictionary of the top N words

    most_frequent = dict(word_freq_counter.most_common(how_many_words))



    if plots == 0:

        return word_freq_counter

    else:

        unique_words = most_frequent.keys()

        word_counts = most_frequent.values()



        plt.figure(figsize=(20,10))

        plt.bar(np.arange(len(unique_words)),word_counts,.75,edgecolor='black',linewidth=1,color=(0.6, 0, 0.6))

        plt.xticks(np.arange(len(unique_words)),unique_words,rotation=90,fontsize=14)

        plt.title(["Histogram of " + titlestring], fontsize=30)

        plt.show()



        words_by_frequency = list(Counter(most_frequent).elements())



        wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=20).generate_from_frequencies(most_frequent)

        wordcloud.recolor(random_state=0)

        plt.imshow(wordcloud)

        plt.title(["Wordcloud of " + titlestring], fontsize=30)

        plt.axis("off")

        plt.show()
#WordCountAnalyzer(all_descriptions, how_many_words=100, plots=1, titlestring='All Descriptions')

WordCountAnalyzer(low_descriptions, how_many_words=100, plots=1, titlestring='Low Interest Descriptions')

WordCountAnalyzer(medium_descriptions, how_many_words=100, plots=1, titlestring='Medium Interest Descriptions')

WordCountAnalyzer(high_descriptions, how_many_words=100, plots=1, titlestring='High Interest Descriptions')



high_word_counter = WordCountAnalyzer(high_descriptions, how_many_words=200, plots=0, titlestring='High Interest Descriptions')

low_word_counter = WordCountAnalyzer(low_descriptions, how_many_words=200, plots=0, titlestring='High Interest Descriptions')



differential_word_counter = high_word_counter

differential_word_counter.subtract(low_word_counter)



differential_word_disctionary = dict(differential_word_counter)      

words = differential_word_disctionary.keys()

differential_frequency = differential_word_disctionary.values()



plt.figure(figsize=(20,10))

plt.bar(np.arange(len(words)),differential_frequency,.75,edgecolor='black',linewidth=1,color=(0.6, 0, 0.6))

plt.xticks(np.arange(len(words)),words,rotation=90,fontsize=14)

plt.title(['Difference in frequency between High and Low Interest'], fontsize=30)

plt.show()