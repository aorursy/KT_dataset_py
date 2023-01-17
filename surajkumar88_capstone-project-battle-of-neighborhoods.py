#Importing required libraries

import numpy as np

import pandas as pd



from geopy.geocoders import Nominatim

try:

    import geocoder

except:

    !pip install geocoder

    import geocoder



import requests

from bs4 import BeautifulSoup



try:

    import folium

except:

    !pip install folium

    import folium

    

from sklearn.cluster import KMeans



from sklearn import preprocessing



from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier



from sklearn import metrics



import matplotlib as mpl

import matplotlib.pyplot as plt
# install wordcloud

!pip install wordcloud

# import package and its set of stopwords

from wordcloud import WordCloud, STOPWORDS



print ('Wordcloud is installed and imported!')
#Getting the location of Bangalore city using the geocoder package

g = geocoder.arcgis('Bangalore, India')

blr_lat = g.latlng[0]

blr_lng = g.latlng[1]

print("The Latitude and Longitude of Bangalore is {} and {}".format(blr_lat, blr_lng))
#Scraping the Wikimedia webpage for list of localities present in Bangalore city

neig = requests.get("https://commons.wikimedia.org/wiki/Category:Suburbs_of_Bangalore").text
#parsing the scraped content

soup = BeautifulSoup(neig, 'html.parser')
#Creating a list to store neighborhood data

neighborhoodlist = []
#Searching the localities using class labels and appending it to the neighborhood list

for i in soup.find_all('div', class_='mw-category')[0].find_all('a'):

    neighborhoodlist.append(i.text)



#Creating a dataframe from the list

neig_df = pd.DataFrame({"Locality": neighborhoodlist})

neig_df.head()
#Shape of dataframe neig_df

neig_df.shape
#Defining a function to get the location of the localities

def get_location(localities):

    g = geocoder.arcgis('{}, Bangalore, India'.format(localities))

    get_latlng = g.latlng

    return get_latlng
#Creating an empty list

co_ordinates = []

#Getting the co-ordinates of each locality using the function defined above

for i in neig_df["Locality"].tolist():

    co_ordinates.append(get_location(i))

print(co_ordinates)
co_ordinates[:5]
#Creating a dataframe from the list of location co-ordinates

co_ordinates_df = pd.DataFrame(co_ordinates, columns=['Latitudes', 'Longitudes'])
#Adding co-ordinates of localities to neig_df dataframe

neig_df["Latitudes"] = co_ordinates_df["Latitudes"]

neig_df["Longitudes"] = co_ordinates_df["Longitudes"]
print("The shape of neig_df is {}".format(neig_df.shape))

neig_df.head()
#Creating a map

blr_map = folium.Map(location=[blr_lat, blr_lng],zoom_start=11)



#adding markers to the map for localities

#marker for Bangalore

folium.Marker([blr_lat, blr_lng], popup='<i>Bangalore</i>', color='red', tooltip="Click to see").add_to(blr_map)



#markers for localities

for latitude,longitude,name in zip(neig_df["Latitudes"], neig_df["Longitudes"], neig_df["Locality"]):

    folium.CircleMarker(

        [latitude, longitude],

        radius=6,

        color='blue',

        popup=name,

        fill=True,

        fill_color='#3186ff'

    ).add_to(blr_map)



blr_map
#Foursquare Credentials

# @hidden_cell

CLIENT_ID = 'CLIENT_ID'

CLIENT_SECRET = 'CLIENT_SECRET'

VERSION = '20180605' # Foursquare API version
#Getting the top 100 venues in each locality

radius = 2000

LIMIT = 100



venues = []



for lat, lng, locality in zip(neig_df["Latitudes"], neig_df["Longitudes"], neig_df["Locality"]):

    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, lat, lng, VERSION, radius, LIMIT)

    results = requests.get(url).json()['response']['groups'][0]['items']



    for venue in results:

        venues.append((locality, lat, lng, venue['venue']['name'], venue['venue']['location']['lat'], venue['venue']['location']['lng'], venue['venue']['categories'][0]['name'], venue['venue']['id']))
#Looking at the first venue

venues[0]
#Convert the venue list into dataframe

venues_df = pd.DataFrame(venues)

venues_df.columns = ['Locality', 'Latitude', 'Longitude', 'Venue name', 'Venue Lat', 'Venue Lng', 'Venue Category', 'Venue ID']

venues_df.head()
venues_df.shape
#Getting the list of all the categories of all the restaurant present in venues_df dataframe

res_df = pd.DataFrame({'Venue Category': venues_df['Venue Category'], 'Strength': venues_df['Venue Category']})

res_df = res_df.groupby(['Venue Category']).count()

res_df = res_df.sort_values(['Strength'], ascending=False)

print(res_df.head())

print("We can see that most restaurants belongs to Indian Restaurant category i.e. {}".format(res_df['Strength'][0]))
res_df.shape
demo1_df = pd.DataFrame({'Venue Category':res_df.index[:50]})

category_strength=[]

for i in range(50):

    category_strength.append(res_df['Strength'][i])

demo2_df = pd.DataFrame(category_strength, columns=['Strength'])

demo_df = pd.DataFrame({'Venue Category': demo1_df['Venue Category'], 'Strength': demo2_df['Strength']})

demo_df.head()
word_string = ''

for i in range(50):

    tmp = demo_df['Venue Category'][i]

    if len(tmp.split(' ')) == 1:

        word_string = word_string + (tmp + ' ')*demo_df['Strength'][i]

    else:

        part = tmp.split(' ')

        tmp = part[0]+'_'+part[1]

        word_string = word_string + (tmp + ' ')*demo_df['Strength'][i]
wordcloud = WordCloud(width=800, height=400, collocations=False).generate(word_string)

print('Word cloud created!')

# Open a plot of the generated image.



plt.figure( figsize=(10,9), facecolor='k')

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#List of 50 most common categories of restuarants in Bangalore City

cat_res_list = res_df.index[0:50]

cat_res_list
#creating a dataframe from the list of common categories created above

venue_etables = venues_df[venues_df['Venue Category'].isin(['Indian Restaurant', 'Café', 'Ice Cream Shop', 'Fast Food Restaurant',

       'Pizza Place', 'Coffee Shop', 'Hotel', 'Chinese Restaurant', 'Lounge',

       'Italian Restaurant', 'Bakery', 'Pub', 'Restaurant',

       'Asian Restaurant', 'Breakfast Spot', 'Bar', 'Brewery', 'Burger Joint',

       'Shopping Mall', 'Sandwich Place', 'Vegetarian / Vegan Restaurant',

       'BBQ Joint', 'Snack Place', 'Park', 'Juice Bar',

       'South Indian Restaurant', 'Tea Room',

       'Middle Eastern Restaurant', 'Dessert Shop', 'Donut Shop', 'Bookstore',

       'Multiplex', 'Cocktail Bar',

       'Seafood Restaurant', 'Mexican Restaurant', 'French Restaurant',

       'Andhra Restaurant', 'Korean Restaurant', 'Cupcake Shop',

       'Karnataka Restaurant', 'Steakhouse', 'Boutique', 'Liquor Store',

       'Arcade', 'Deli / Bodega', 'Bus Station'])]
#Function for calculating the tipcount for each venue

def addingtip(venue_id):

    url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}&radius={}&limit={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION, radius, LIMIT)

    stats = requests.get(url).json()['response']

    tipcount = stats['venue']['stats']['tipCount']

    return tipcount
#calling the addingtip function for each venue

tip_count = []

for i in venue_etables["Venue ID"]:

    venue_id = venue_etables['Venue ID'].iloc[i]

    tipcount = addingtip(venue_id)

    tip_count.append(tipcount)
#Converting the list into a dataframe

df_tipcount = pd.DataFrame(tip_count)
#Changing the column name of df_tipcount dataframe

df_tipcount.columns = ["Tip count"]

df_tipcount.head()
#Attaching the tip_count to the venue_etables and creating a new dataframe

venue_etables["Tip count"] = df_tipcount["Tip count"]

venue_etables.head()
#creating one hot encoding

blr_onehot = pd.get_dummies(venues_df[['Venue Category']], prefix="", prefix_sep="")



blr_onehot['Locality'] = venues_df['Locality']



#moving the locality column to the front

blr_onehot = blr_onehot[ [ 'Locality' ] + [ col for col in blr_onehot.columns if col!='Locality' ] ]

blr_onehot.head()
blr_grouped = blr_onehot.groupby(['Locality']).mean().reset_index()

print(blr_grouped.shape)

blr_grouped.head()
#numbers of localities having Italian Restaurants

len(blr_grouped[blr_grouped['Italian Restaurant'] > 0])
blr_italian = blr_grouped[['Locality', 'Italian Restaurant']]

blr_italian.head()
#Creating a map

blr_map = folium.Map(location=[blr_lat, blr_lng],zoom_start=11)



#adding markers to the map for localities

#marker for Bangalore

folium.Marker([blr_lat, blr_lng], popup='<i>Bangalore</i>', color='red', tooltip="Click to see").add_to(blr_map)



#markers for localities

for latitude,longitude,name,strength in zip(neig_df["Latitudes"], neig_df["Longitudes"], neig_df["Locality"], blr_italian["Italian Restaurant"]):

    folium.CircleMarker(

        [latitude, longitude],

        radius=strength*300,

        color='green',

        popup=name,

        fill=True,

        fill_color='#3186ff'

    ).add_to(blr_map)



blr_map
#K-means clustering

cluster = 3 



#Dataframe for clustering

blr_clustering = blr_italian.drop(['Locality'], 1)



#run K-means clustering

k_means = KMeans(init="k-means++", n_clusters=cluster, n_init=12).fit(blr_clustering)



#getting the labels for first 10 locality 

print(k_means.labels_[0:10])
#Creating a copy of blr_italian dataframe

blr_labels = blr_italian.copy()



#adding label to blr_labels

blr_labels["Cluster Label"] = k_means.labels_



blr_labels.head()
#Merging the blr_labels and neig_df dataframes to get the latitude and longitudes for each locality

blr_labels = blr_labels.join(neig_df.set_index('Locality'), on='Locality')

blr_labels.head()
#Grouping the localities according to their Cluster Labels

blr_labels.sort_values(["Cluster Label"], inplace=True)

blr_labels.head()
#Cleaning the dataframe for mapping the localities according to their cluster labels

blr_only_labels = blr_labels.drop(columns=['Italian Restaurant','Latitudes','Longitudes'])

blr_only_labels.head()
#Plot the cluster on map

cluster_map = folium.Map(location=[blr_lat, blr_lng],zoom_start=11)



#marker for Bangalore

folium.Marker([blr_lat, blr_lng], popup='<i>Bangalore</i>', color='red', tooltip="Click to see").add_to(cluster_map)



#Getting the colors for the clusters

col = ['red', 'green', 'blue']



#markers for localities

for latitude,longitude,name,clus in zip(blr_labels["Latitudes"], blr_labels["Longitudes"], blr_labels["Locality"], blr_labels["Cluster Label"]):

    label = folium.Popup(name + ' - Cluster ' + str(clus+1))

    folium.CircleMarker(

        [latitude, longitude],

        radius=6,

        color=col[clus],

        popup=label,

        fill=False,

        fill_color=col[clus],

        fill_opacity=0.3

    ).add_to(cluster_map)

       

cluster_map
#Cluster 1

#Dataframe containing localities with cluster label 0, which corresponds to localities with no Italian Restaurant

cluster_1 = blr_labels[blr_labels['Cluster Label'] == 0]

print("There are {} localities in cluster-1".format(cluster_1.shape[0]))

mean_presence_1 = cluster_1['Italian Restaurant'].mean()

print("The mean occurence of Italian restaurant in cluster-1 is {0:.2f}".format(mean_presence_1))

cluster_1.head()
#Cluster 2

#Dataframe containing localities with cluster label 1, which corresponds to localities with high density of Italian Restaurant

cluster_2 = blr_labels[blr_labels['Cluster Label'] == 1]

print("There are {} localities in cluster-2".format(cluster_2.shape[0]))

mean_presence_2 = cluster_2['Italian Restaurant'].mean()

print("The mean occurence of Italian restaurant in cluster-2 is {0:.2f}".format(mean_presence_2))

cluster_2.head()
#Cluster 3

#Dataframe containing localities with cluster label 2, which corresponds to localities with low density of Italian Restaurant

cluster_3 = blr_labels[blr_labels['Cluster Label'] == 2]

print("There are {} localities in cluster-3".format(cluster_3.shape[0]))

mean_presence_3 = cluster_3['Italian Restaurant'].mean()

print("The mean occurence of Italian restaurant in cluster-3 is {0:.2f}".format(mean_presence_3))

cluster_3.head()
#Function for getting the cluster label of each locality

def get_clus_label(locality):

    loca = locality

    for i in range(blr_only_labels.shape[0]):

        value = blr_only_labels.iloc[i]

        value_locality = value[0]

        value_cluster_label = value[1]

        if value_locality == loca:

            return value_cluster_label
#Getting the cluster label and appending it to "cluster_label" list

cluster_labels = []

for locality in venue_eatables['Locality']:

    label = get_clus_label(locality)

    cluster_labels.append(label)
#adding the cluster_label to the venue_eatables dataframe

cluster_labels_df = pd.DataFrame(cluster_labels, columns=['Cluster Label'])

venue_eatables['Cluster Labels'] = cluster_labels_df['Cluster Label']

venue_eatables.head()
#creating a dataframe with empty columns for surrounding venues which will contain 30 surrounding venues

for i in range(30):

    tag = "SV "

    tag = tag + str(i+1) 

    venue_eatables[tag] = ""

venue_eatables.head()
#Separating the venue_eatables dataframe into two, one for the localities containing Italian restaurants and the other for localities 

#that does not contains Italian restaurants

venue_eatables_without_italian = venue_eatables[~venue_eatables['Venue Category'].isin(['Italian Restaurant'])]

venue_eatables_without_italian.reset_index(inplace=True, drop=True)

venue_eatables_with_italian = venue_eatables[venue_eatables['Venue Category'].isin(['Italian Restaurant'])]

venue_eatables_with_italian.reset_index(inplace=True, drop=True)
#Setting the radius and LIMIT of the results from foursquare API

radius = 2000

LIMIT = 30
# Getting the 30 nearest venues around the given venue in localities with Italian restaurant

# and adding it to the venue_eatables_without_italian dataframe

for i in range(venue_eatables_with_italian.shape[0]):

    venue = venue_eatables_with_italian.loc[i,'Venue name']

    venue_lat = venue_eatables_with_italian.loc[i,'Venue Lat']

    venue_lng = venue_eatables_with_italian.loc[i,'Venue Lng']

    

    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, venue_lat, venue_lng, VERSION, radius, LIMIT)

    results = requests.get(url).json()

    

    for j in range(30):

        sv_index = "SV " + str(j+1)

        try:

            cat = results['response']['groups'][0]['items'][j]['venue']['categories'][0]['name']

        except:

            cat = np.nan

        venue_eatables_with_italian.loc[i, sv_index] = cat
pd.set_option('display.max_columns', 40)

pd.set_option('display.max_rows', 5000)
# Getting the 30 nearest venues around the given venue in localities without Italian restaurant

# and adding it to the venue_eatables_without_italian dataframe

for i in range(venue_eatables_without_italian.shape[0]):

    venue = venue_eatables_without_italian.loc[i,'Venue name']

    venue_lat = venue_eatables_without_italian.loc[i,'Venue Lat']

    venue_lng = venue_eatables_without_italian.loc[i,'Venue Lng']

    

    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, venue_lat, venue_lng, VERSION, radius, LIMIT)

    results = requests.get(url).json()

    

    for j in range(30):

        sv_index = "SV " + str(j+1)

        try:

            cat = results['response']['groups'][0]['items'][j]['venue']['categories'][0]['name']

        except:

            cat = np.nan

        venue_eatables_without_italian.loc[i, sv_index] = cat
#Creating the dataframe neig_df with empty columns for 30 venues in that locality

for i in range(30):

    tag = "SV "

    tag = tag + str(i+1) 

    neig_df[tag] = ""

neig_df.head()
# Getting the list of 30 venues in each locality in Bangalore

for i in range(neig_df.shape[0]):

    venue = neig_df.loc[i,'Locality']

    venue_lat = neig_df.loc[i,'Latitudes']

    venue_lng = neig_df.loc[i,'Longitudes']

    

    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, venue_lat, venue_lng, VERSION, radius, LIMIT)

    results = requests.get(url).json()

    

    for j in range(30):

        sv_index = "SV " + str(j+1)

        try:

            cat = results['response']['groups'][0]['items'][j]['venue']['categories'][0]['name']

        except:

            cat = np.nan

        neig_df.loc[i, sv_index] = cat
#concatenatig the two dataframes "venue_eatables_with_italian" and "venue_eatables_without_italian" to create one dataframe

venue_eatables_with_sv = pd.concat([venue_eatables_with_italian, venue_eatables_without_italian])

print(venue_eatables_with_sv.shape)

venue_eatables_with_sv.head()
#cleaning the train_df for only relevant columns for training the model

venue_eatables_with_sv = venue_eatables_with_sv.drop(columns=['Locality','Latitude','Longitude','Venue name','Venue Lat','Venue Lng','Venue ID','Tip count','Cluster Labels'])

venue_eatables_with_sv.reset_index(inplace = True, drop = True) 

venue_eatables_with_sv.head()
#cleaning the neig_df_with_sv for only relevant columns for prediction.

neig_df_with_sv = neig_df_with_sv.drop(columns=['Latitudes','Longitudes'])

#saving this intermediate dataframe for using to compare the prediction 

neig_df_with_sv_interm = neig_df_with_sv

neig_df_with_sv = neig_df_with_sv.drop(columns=['Locality'])

neig_df_with_sv.head()
#Getting the categories of venues present in the datasets and indexing them 

#so that categories can be replaced with respective index for training the model

category_list_df = pd.DataFrame({'Venue Category': venues_df['Venue Category'], 'Venue Cat': venues_df['Venue Category']})

category_list_df = category_list_df['Venue Category'].unique()

category_list_df = category_list_df.tolist()

category_dict = { category_list_df[i] : i for i in range(len(category_list_df))}
#Replacing the categories in dataframe "venue_eatables_with_sv" with their respective index

venue_eatables_with_sv = venue_eatables_with_sv.applymap(lambda x: category_dict.get(x) if x in category_dict else x)

venue_eatables_with_sv.head()
#finding missing categories in venue_eatables_with_sv

missing_cat = []

for i in range(venue_eatables_with_sv.shape[0]):

    for j in range(venue_eatables_with_sv.shape[1]):

        if isinstance(venue_eatables_with_sv.iloc[i][j], str) and venue_eatables_with_sv.iloc[i][j] not in missing_cat:

            missing_cat.append(venue_eatables_with_sv.iloc[i][j])



print(missing_cat)
#Appending the missing venue categories with index to the category_dict dictionary

missing_cat.append('Wings Joint')

category_list_df = category_list_df+missing_cat

category_dict = { category_list_df[i] : i for i in range(len(category_list_df))}

print(category_dict)
#Replacing the categories in dataframe "venue_eatables_with_sv" with their respective code

venue_eatables_with_sv = venue_eatables_with_sv.applymap(lambda x: category_dict.get(x) if x in category_dict else x)

venue_eatables_with_sv = venue_eatables_with_sv.fillna(300)

venue_eatables_with_sv.head()
#Replacing the categories in dataframe "neig_df_with_sv" with their respective code

neig_df_with_sv = neig_df_with_sv.applymap(lambda x: category_dict.get(x) if x in category_dict else x)

neig_df_with_sv = neig_df_with_sv.fillna(300)

neig_df_with_sv.head()
#creating X and Y array for input and output values

X = venue_eatables_with_sv[['SV 1', 'SV 2', 'SV 3', 'SV 4', 'SV 5', 'SV 6', 'SV 7', 'SV 8', 'SV 9', 'SV 10', 'SV 11', 'SV 12', 'SV 13', 'SV 14', 'SV 15', 'SV 16', 'SV 17', 'SV 18', 'SV 19', 'SV 20', 'SV 21', 'SV 22', 'SV 23', 'SV 24', 'SV 25', 'SV 26', 'SV 27', 'SV 28', 'SV 29', 'SV 30']].values

Y = venue_eatables_with_sv[['Venue Category']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

print('Train Set:',X_train.shape, Y_train.shape)

print('Test Set:',X_test.shape, Y_test.shape)
#Cheking accuracy for different value of Ks.

ks=11

mean_acc = np.zeros(ks-1)

std_acc = np.zeros(ks-1)



for i in range(1, ks):

    

    #training model and predicting

    clf = KNeighborsClassifier(n_neighbors=i).fit(X_train, Y_train.ravel())

    yhat = clf.predict(X_test)

    mean_acc[i-1] = metrics.accuracy_score(Y_test, yhat)

    std_acc[i-1] = np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])
print(mean_acc)
#Plotting the accuracy for different values of K

plt.plot(range(1,ks),mean_acc,'g')
#To avoid overfitting of the model we are taking K=3

#Creating the model and training it with train data

k=3

clf = KNeighborsClassifier(n_neighbors=k).fit(X_train,Y_train.ravel())

clf
#predicting

yhat = clf.predict(X_test)
#accuracy evaluation

print("Test set accuracy : ", metrics.accuracy_score(Y_test, yhat))
#Training the model with whole dataset

k=3

clf1 = KNeighborsClassifier(n_neighbors=k).fit(X,Y.ravel())

clf1
#predicting the probable localities usnig the above trained model

pred = clf1.predict(neig_df_with_sv)

print(pred)
#defining the function to get the keys from the values, from category_dict dictionary

def get_key(val):

    for key, value in category_dict.items():

        if value == val:

            return key

#Using the dictionary "category_dict", we'll change the indices predicted by the model back to the name of the categories

#Creating new variable "pred_cat" for predicted categories

pred_cat = []

for i in range(len(pred)):

    pred_cat.append(get_key(pred[i]))
#changing the list pred_cat to a dataframe

pred_cat_df = pd.DataFrame(pred_cat, columns=['Prediction'])

#adding the prediction column from pred_cat_df to neig_df_with_sv_interm

neig_df_with_sv_interm["Prediction"] = pred_cat_df["Prediction"]

#moving the prediction column to the front

neig_df_with_sv_interm = neig_df_with_sv_interm[ [ 'Prediction' ] + [ col for col in neig_df_with_sv_interm.columns if col!='Prediction' ] ]

neig_df_with_sv_interm[50:55]
#Plot the cluster on map

cluster_map = folium.Map(location=[blr_lat, blr_lng],zoom_start=11)



#marker for Bangalore

folium.Marker([blr_lat, blr_lng], popup='<i>Bangalore</i>', color='red', tooltip="Click to see").add_to(cluster_map)



#predicted locality

pred_locality = None

for locality, prediction in zip(neig_df_with_sv_interm['Locality'], neig_df_with_sv_interm['Prediction']):

    if prediction == "Italian Restaurant":

        pred_locality = locality



#Getting the colors for the clusters

col = ['red', 'green', 'blue']



#markers for localities

for latitude,longitude,name,clus in zip(blr_labels["Latitudes"], blr_labels["Longitudes"], blr_labels["Locality"], blr_labels["Cluster Label"]):

    label = folium.Popup(name + ' - Cluster ' + str(clus+1))

    if name==pred_locality:

        folium.Marker([latitude, longitude], popup=name, color='orange', tooltip="This is the predicted locality for opening a new Italian Restaurant.").add_to(cluster_map)

    else:

        folium.CircleMarker(

            [latitude, longitude],

            radius=6,

            color=col[clus],

            popup=label,

            fill=False,

            fill_color=col[clus],

            fill_opacity=0.3

        ).add_to(cluster_map)

       

cluster_map