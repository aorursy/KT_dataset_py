#Basics
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Plotting libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image as im
from IPython.display import Image
from scipy import stats

#Wordcloud packages
from os import path
from PIL import Image
from IPython.display import SVG, display
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re 
import collections as c

#Modelling
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#Importing data
ny_df = pd.read_csv('/Users/emilykasa/Desktop/repos/Capstone_Project/capstone_data/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
ny_df.head()
#ny_df.info()
#Take a look at naan value percentages
missing_data_summary = (ny_df.isnull().sum()/len(ny_df))*100
missing_data_summary
#looks like if there are no reviews then last review and reviews per month are both NaN
#dropping rows with nan would mean losing 20% of our data, but dropping the columns would be a loss of a possibly important predictor
ny_df[ny_df.isna()["reviews_per_month"]]
reviews_per_month_no_nan=ny_df[ny_df.notnull()["reviews_per_month"]]
reviews_per_month_no_nan.head()
#Lets replace the nan values in the reviews_per_month with 0 and drop the last_review  column
ny_df["reviews_per_month"] = ny_df["reviews_per_month"].fillna(0)
ny_df.drop('last_review', axis=1, inplace=True)
#Now lets deal with the missing names and host names. Since we might want to use names later, we wont drop
#We also want to get rid of NaN values, so we'll replace NaN with Unkown 

ny_df["name"] = ny_df["name"].fillna("Unknown")
ny_df["host_name"] = ny_df["name"].fillna("Unknown")

#making sure we don't have any nan values left 
missing_data_summary = (ny_df.isnull().sum()/len(ny_df))*100
missing_data_summary
#Moving price to the end of the dataframe
ny_df.columns
ny_df=ny_df[['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
       'neighbourhood', 'latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365','price']]
#Take a look at the continuous data in our dataframe
display(ny_df.describe())

#Lets just take a look at our target, price, to see what kind of distribution it has
fig = px.histogram(ny_df, x="price", nbins=30, title='Histogram of Price (Whole Dataset)')

print(f'The mean price is {ny_df["price"].mean()}')
print(f'The median price is {ny_df["price"].median()}')
print(f'The max price is {ny_df["price"].max()}')
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
#Lets just take a look at our target, price, to see what kind of distribution it has
fig = px.histogram(ny_df[ny_df["price"]<1000], x="price", nbins=30, title='Histogram of Price (Price Less Than 1000)')

print(f'The mean price is {ny_df[ny_df["price"]<1000]["price"].mean()}')
print(f'The median price is {ny_df[ny_df["price"]<1000]["price"].median()}')
print(f'The max price is {ny_df[ny_df["price"]<1000]["price"].max()}')
print(f'Dropping {len(ny_df[ny_df["price"]>1000])} rows, {round((len(ny_df[ny_df["price"]>1000]))/len(ny_df), 3)*100} percent')
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
#Lets just take a look at our target, price, to see what kind of distribution it has
fig = px.histogram(ny_df[ny_df["price"]<250], x="price", nbins=30, title='Histogram of Price (Price Less Than 250)')

print(f'The mean price is {ny_df[ny_df["price"]<250]["price"].mean()}')
print(f'The median price is {ny_df[ny_df["price"]<250]["price"].median()}')
print(f'The max price is {ny_df[ny_df["price"]<250]["price"].max()}')
print(f'Dropping {len(ny_df[ny_df["price"]>250])} rows, {round((len(ny_df[ny_df["price"]>250]))/len(ny_df), 3)*100} percent')
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
#Dropping rows with price above $250
ny_df=ny_df[ny_df["price"]<250]

#Resetting index after dropping prices above $250
ny_df=ny_df.reset_index()
ny_df.drop("index", inplace=True, axis=1)
#Lets just check some basic correllation properties
ny_df.corr().style.background_gradient(cmap='coolwarm')
#Lets build a wordcloud of common words used in the "name" column

text = " ".join(name for name in ny_df.name)

whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

names_concat = ''.join(filter(whitelist.__contains__, text))
names_concat=names_concat.split(" ")

#Creating dictionary of word counts
word_counts={}
for i in names_concat:
    if i not in word_counts:
        word_counts[i]=0
    word_counts[i]+=1

#Get rid of words with little meaning
stopwords = set(STOPWORDS)
stopwords.update(["Room", "Bedroom", "Private", "In", "in", "NYC", "apartment", "room", "bedroom", "br", "Apartment", "BR", " "])

#Get rid of stopwords in our dictionary
for i in stopwords:
    if i in word_counts:
        del word_counts[i]

#Use dictionary to build wordcloud 
wordcloud = WordCloud(background_color="white", width=5000, height=3000, max_words=50).generate_from_frequencies(word_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#trying to see if we can get info on number of bedrooms from name

bedrooms=[]
for i in ny_df["name"]:
    m=re.match("^\d+\s(?i)(br|bedroom)", i)
    bedrooms.append(m)    

ny_df["number_of_bedrooms"]=bedrooms
ny_df["number_of_bedrooms"] = ny_df["number_of_bedrooms"].fillna("Unknown")
ny_df["number_of_bedrooms"] = ny_df["number_of_bedrooms"].astype('category')


ny_df.groupby("number_of_bedrooms")["price"].mean()



#Dropping this column since it won't be useful
ny_df.drop(["number_of_bedrooms"], inplace=True, axis=1)
#sorting words_count dict by values (word counts)

sorted_keys = sorted(word_counts, key=word_counts.get, reverse=True)
sorted_keys

sorted_list_wc=[]
for r in sorted_keys:
    sorted_list_wc.append((r, word_counts[r]))

sorted_list_wc[1:20]
#Lets look at prices based on room type

fig = px.histogram(ny_df, x="price", color="room_type", nbins=30, title='Histogram of Price By Room Type')
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))

ax = sns.swarmplot(x="room_type", y="price", data=ny_df)


fig = px.histogram(ny_df[ny_df["minimum_nights"]<10], x="minimum_nights", nbins=20, title='Histogram of Min Nights, 0-10 Min Nights')
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))

fig = px.histogram(ny_df[ny_df["minimum_nights"]<30], x="minimum_nights", nbins=20, title='Histogram of Min Nights, 0-30 Min Nights')
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))

fig = px.histogram(ny_df, x="minimum_nights", nbins=20, title='Histogram of Min Nights, Entire Data')
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))

print('the mode is ', ny_df["minimum_nights"].mode())
print('the median is ', ny_df["minimum_nights"].median())
print('the mean is ', ny_df["minimum_nights"].mean())







#Lets look at prices based on latitude (note that Mahattan has greater Longitude values than Brooklyn/Williamsburg)

fig = px.scatter(ny_df, x="longitude", y="price", title='Longitude vs Price', trendline='ols')
results = px.get_trendline_results(fig)
results = px.get_trendline_results(fig)
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
#Lets look at prices based on latitude (note that Mahattan has higher latitude values)

fig = px.scatter(ny_df[ny_df["price"]<1000], x="latitude", y="price", title='Latitude vs Price', trendline='ols')
results = px.get_trendline_results(fig)
results = px.get_trendline_results(fig)
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
ax = sns.violinplot(x="neighbourhood_group", y="price", data=ny_df)
#Map looking at most expensive neighborhoods

neighborhoods=ny_df.groupby(["neighbourhood", "room_type"])[["price", "latitude", "longitude", "minimum_nights"]].mean().reset_index()
neighborhoods.sort_values('price', ascending=False, inplace=True)
rich_neighborhoods=neighborhoods


fig = px.scatter_mapbox(rich_neighborhoods, lat="latitude", lon="longitude",  size="price", color="room_type", size_max=15, zoom=10, hover_name="neighbourhood", hover_data=["price", "minimum_nights"], title="Map of Prices By Neighborhood and Roomtype")
fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

img_bytes = fig.to_image(format="png")
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
#Maps looking at prices for each Airbnb 


fig = px.scatter_mapbox(ny_df, lat="latitude", lon="longitude",  color="price", size_max=15, zoom=10)
fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))
#Neighborhood Group Map
fig = px.scatter_mapbox(ny_df, lat="latitude", lon="longitude",  color="neighbourhood_group", size="price", size_max=15, zoom=10)
fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
img_bytes = pio.to_image(fig, format="svg")
display(SVG(img_bytes))

#Checking the median and mean price
print(ny_df["price"].median())
print(ny_df["price"].mean())
#adding dummies for roomtype
room_type_dummies=pd.get_dummies(ny_df['room_type'], prefix='room_type', drop_first=True).reset_index()
room_type_dummies.drop("index", axis=1, inplace=True)
ny_df_added_features = pd.concat([room_type_dummies, ny_df], axis=1)

#adding dist_from_midtown feature. After doing EDA on prices and location, a lot of the higher prices appeared to be very close to midtown. Lets construct a feature for this.
ny_df_added_features["dist_from_midtown"]=abs(ny_df_added_features["latitude"]-40.7069)+abs(ny_df_added_features["latitude"]+74.0031)

#adding dummies for neighborhood group
ny_df_added_features = pd.concat([ny_df_added_features ,pd.get_dummies(ny_df_added_features['neighbourhood_group'], prefix='neighbourhood_group', drop_first=True)],axis=1)

#adding dummies for neighborhood
ny_df_added_features = pd.concat([ny_df_added_features ,pd.get_dummies(ny_df_added_features['neighbourhood'], prefix='neighbourhood', drop_first=True)],axis=1)


#Dropping the original columns we just one hot encoded
ny_df_added_features.drop(["neighbourhood_group", "neighbourhood", "room_type", "host_id", "host_name", "id"], axis=1, inplace=True)
ny_df_added_features.head()
#Splitting data into remainder (train and validation) and test split of 20% (calling X datarame "no words" because I haven't count vectorized the name column yet).

#I want all the columns except price in my X dataframe
X=ny_df_added_features.loc[:, ny_df_added_features.columns != 'price']

#Price is the target
y=ny_df_added_features["price"]
split = 0.2
X_remainder_no_words, X_test_no_words, y_remainder, y_test = train_test_split(X, y, test_size=split, random_state=6)
for i in [X_remainder_no_words, X_test_no_words, y_remainder, y_test]:
    print(i.shape)


#Resettig axis and dropping index columns
X_remainder_no_words=X_remainder_no_words.reset_index()
X_remainder_no_words.drop("index", axis=1, inplace=True)

y_remainder=y_remainder.reset_index()
y_remainder.drop("index", axis=1, inplace=True)


X_test_no_words=X_test_no_words.reset_index()
X_test_no_words.drop("index", axis=1, inplace=True)

y_test=y_test.reset_index()
y_test.drop("index", axis=1, inplace=True)
#Scaling X data so we can compare the effect different coefficients are having

X_remainder_no_words_scaled=X_remainder_no_words.copy()
X_test_no_words_scaled=X_test_no_words.copy()


scalerx = StandardScaler()
scalerx.fit(X_remainder_no_words_scaled[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])
X_remainder_no_words_scaled[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] = scalerx.transform(X_remainder_no_words_scaled[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])
X_test_no_words_scaled[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] = scalerx.transform(X_test_no_words_scaled[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])



#Baseline linear regression using scaled data so we can evaluate coefficients (not including latitude/longitude/dist_from_midtown)
X = X_remainder_no_words_scaled[[ "room_type_Private room",
"room_type_Shared room",
"neighbourhood_group_Manhattan",
"neighbourhood_group_Brooklyn",
"neighbourhood_group_Queens",
"neighbourhood_group_Staten Island"]]
y = y_remainder
X_withconstant = sm.add_constant(X)

# 1. Instantiate Model
myregression = sm.OLS(y,X_withconstant)

# 2. Fit Model (this returns a seperate object with the parameters)
myregression_results = myregression.fit()

# Looking at the summary
myregression_results.summary()
 
print(X_remainder_no_words.shape)
print(X_test_no_words.shape)
print(y_remainder.shape)
print(y_test.shape)
#feed in the X_test dataframe, y_test, and the model
def get_errors(X, y_test, model):
    
    #predict the models 
    y_pred=model.predict(X)
    
    #create a dataframe to store the predictions, actual values, and errors
    y_test_df=y_test.copy()
    
    #add predictions to the dataframe
    y_test_df["pred"]=y_pred
    
    #add the squared error of each prediction from the actual value to the dataframe
    y_test_df["error"]=(y_test_df["price"]-y_test_df["pred"])**(2)
    
    #add the absolute squared error to the dataframe
    y_test_df["abs_error"]=abs((y_test_df["price"]-y_test_df["pred"]))


    #take the square root of the sum of the squared error column of the dataframe to get RMSE
    print(f'RMSE={(((y_test_df["error"].sum()))/len(y_test_df))**(1/2)}')
    
    #Plot the absolute errors in a histogram
    fig=px.histogram(x=y_test_df['abs_error'], title="Abs Error", nbins=30)


    img_bytes = pio.to_image(fig, format="svg")
    display(SVG(img_bytes))
#Try linear model using continuous data plust one hot encoded data for room_type and neighborhood_group


X_remainder_baseline=X_remainder_no_words[['room_type_Private room', 'room_type_Shared room', 'latitude', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365', 'dist_from_midtown', 'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island']]
X_test_baseline=X_test_no_words[['room_type_Private room', 'room_type_Shared room', 'latitude', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365', 'dist_from_midtown', 'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island']]


# 1. Instantiate the model
linear_regression_model = LinearRegression()

# 2. Fit the model
linear_regression_model.fit(X_remainder_baseline, y_remainder)

#The intercept
intercept = linear_regression_model.intercept_

#The coefficient, notice it returns an array with one spot for each feature
coefficient = linear_regression_model.coef_


model_prediction_train = linear_regression_model.predict(X_remainder_baseline)
model_prediction_test = linear_regression_model.predict(X_test_baseline)



# Evaluate the model on each set
print(f'The R2 score on the training set: {r2_score(y_remainder,model_prediction_train)}')
print(f'The R2 score on the testing set: {r2_score(y_test,model_prediction_test)}')

print(f'The RMSE score on the training set: {(mean_squared_error(y_remainder,model_prediction_train))**(1/2)}')
print(f'The RMSE score on the testing set: {(mean_squared_error(y_test,model_prediction_test))**(1/2)}')


#looks like we are massively overfitting- negative R2 on test 
X_remainder_added=X_remainder_no_words.loc[:, X_remainder_no_words.columns != 'name']
X_test_added=X_test_no_words.loc[:, X_test_no_words.columns != 'name']

# 1. Instantiate the model
linear_regression_model = LinearRegression()

# 2. Fit the model
linear_regression_model.fit(X_remainder_added, y_remainder)

#The intercept
intercept = linear_regression_model.intercept_

#The coefficient, notice it returns an array with one spot for each feature
coefficient = linear_regression_model.coef_


model_prediction_train = linear_regression_model.predict(X_remainder_added)
model_prediction_test = linear_regression_model.predict(X_test_added)



# Evaluate the model on each set
print(f'The R2 score on the training set: {r2_score(y_remainder,model_prediction_train)}')
print(f'The R2 score on the testing set: {r2_score(y_test,model_prediction_test)}')

print(f'The RMSE score on the training set: {(mean_squared_error(y_remainder,model_prediction_train))**(1/2)}')
print(f'The RMSE score on the testing set: {(mean_squared_error(y_test,model_prediction_test))**(1/2)}')


#looks like we are massively overfitting- negative R2 on test 
get_errors(X_test_added, y_test, linear_regression_model)
print(X_remainder_added.shape)
print(X_test_added.shape)
print(y_remainder.shape)
print(y_test.shape)




#Now looking at the count vectorized name data 


# 1. Instantiate, setting min_df to 10 which means that the minimum times the word must appear in the corpus is 10
bagofwords = CountVectorizer(min_df=10)

# 2. Fit 
bagofwords.fit(X_remainder_no_words["name"])

# 3. Transform 
X_remainder_bagofwords = bagofwords.transform(X_remainder_no_words["name"])
X_test_bagofwords = bagofwords.transform(X_test_no_words["name"])



#Adding the bag of words data to a dataframe from the array that the transform method produced
bagofwords.get_feature_names()
X_remainder_bagofwords.toarray()
X_test_bagofwords.toarray()


bag_of_words_train = pd.DataFrame(columns=bagofwords.get_feature_names(), data=X_remainder_bagofwords.toarray())
bag_of_words_test = pd.DataFrame(columns=bagofwords.get_feature_names(), data=X_test_bagofwords.toarray())



#Now we concat the count vectorized words to our X dataframes to get X_added_features
X_remainder_added_features=pd.concat([bag_of_words_train, X_remainder_no_words], axis=1)
X_test_added_features=pd.concat([bag_of_words_test, X_test_no_words], axis=1)

print(X_remainder_added_features.shape)
print(X_test_added_features.shape)


X_remainder_added_features.drop(["name"], axis=1, inplace=True)
X_test_added_features.drop(["name"], axis=1, inplace=True)
print(X_remainder_added_features.shape)
print(X_test_added_features.shape)


# Create the lasso and ridge models
lasso = Lasso()
lasso.fit(X_remainder_added_features,y_remainder)
ridge = Ridge(alpha=20)
ridge.fit(X_remainder_added_features,y_remainder)

print("Coefficients:")
print("Lasso:", lasso.coef_)
print("Ridge:", ridge.coef_)
print("")


# Compare R-squared
print("R-squared:")
print("Lasso train:", lasso.score(X_remainder_added_features,y_remainder))
print("Ridge train:", ridge.score(X_remainder_added_features,y_remainder))
print("Lasso test:", lasso.score(X_test_added_features,y_test))
print("Ridge test:", ridge.score(X_test_added_features,y_test))


print(f'The RMSE score on the training set: {(mean_squared_error(y_remainder,model_prediction_train))**(1/2)}')
print(f'The RMSE score on the testing set: {(mean_squared_error(y_test,model_prediction_test))**(1/2)}')

print("Ridge Train and Test RMSE:")

print("Ridge Train")
get_errors(X_remainder_added_features, y_remainder, ridge)

print("Ridge Test")
get_errors(X_test_added_features, y_test, ridge)


print("Lasso Train")
get_errors(X_remainder_added_features, y_remainder, lasso)

print("Lasso Test")
get_errors(X_test_added_features, y_test, lasso)
#Running a logistic regression of bag of words on price to see which words have the most predictive power

#Fit a logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(bag_of_words_train, y_remainder)

print(f'score on training: {logreg_model.score(X_remainder_bagofwords, y_remainder)}')
print(f'score on testing: {logreg_model.score(X_test_bagofwords, y_test)}')


bag_of_words_train.head()
#Getting list of indicies for the words
indicies=[]
for i in bag_of_words_train:
    index=bag_of_words_train.columns.get_loc(i)
    indicies.append(index)

#making a list of coefficients from the logistic regression    
coef_lst=logreg_model.coef_.tolist()

#merging indicies and coefficients into a list of tuple 
tuples=list(zip(indicies, coef_lst[0]))
tuples

#sorting the tuples 
def Sort_Tuple(tup):  
  
    # reverse = None (Sorts in Ascending order)  
    # key is set to sort using second element of  
    # sublist lambda has been used  
    tup.sort(key = lambda x: x[1], reverse=True)  
    return tup 

Sorted_words=Sort_Tuple(tuples)

indicies_2=[]
for i in Sorted_words:
    indicies_2.append(i[0])

words_sorted=bag_of_words_train.columns[indicies_2]

#Bottom 30 words 
for i in words_sorted[0:30]:
    print(i)
#redefine the sort function to sort in decending order  
def Sort_Tuple(tup):  
  
    # reverse = None (Sorts in Ascending order)  
    # key is set to sort using second element of  
    # sublist lambda has been used  
    tup.sort(key = lambda x: x[1])  
    return tup 

Reverse_Sorted_words=Sort_Tuple(tuples)

indicies_2=[]
for i in Reverse_Sorted_words:
    indicies_2.append(i[0])

words_sorted_reverse=bag_of_words_train.columns[indicies_2]

#Top 30 predictive words of high price
for i in words_sorted_reverse[0:30]:
    print(i)
X_train_added_features, X_validation_added_features, y_train, y_validation = \
    train_test_split(X_remainder_added_features, y_remainder, test_size = 0.3,
                     random_state=1)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

cross_validation_scores_val = []
cross_validation_scores_train = []




depths=range(1,10)
for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X_train_added_features, y_train)
    cv_score_val = np.mean(cross_val_score(tree, X_validation_added_features, y_validation, cv = 5))
    cross_validation_scores_val.append(cv_score_val)
    cv_score_train = np.mean(cross_val_score(tree, X_train_added_features, y_train, cv = 5))
    cross_validation_scores_train.append(cv_score_train)


    print(f'depth = {depth}')
    print(f"DT R^2 score on training set: {tree.score(X_train_added_features, y_train):0.3f}")
    print(f"DT R^2 score on validation set: {tree.score(X_validation_added_features, y_validation):0.3f}")
    

plt.figure()
plt.plot(depths, cross_validation_scores_val,label="Cross Validation Score Val",marker='.')
plt.plot(depths, cross_validation_scores_train,label="Cross Validation Score Train",marker='.')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Cross Validation Score')
plt.grid()
plt.show()
tree = DecisionTreeRegressor(max_depth=8)
tree.fit(X_train_added_features, y_train)
print(f'Validation score with optimal max depth of 8: {tree.score(X_validation_added_features, y_validation)}')
print(f'Test score with optimal max depth of 8: {tree.score(X_test_added_features, y_test)}')
get_errors(X_test_added_features, y_test, tree)
cross_validation_scores_val = []
cross_validation_scores_train = []

max_depth_lst=range(1,10)
for max_depth in max_depth_lst:
    my_random_forest = RandomForestRegressor(n_estimators=100, max_depth=max_depth)
    my_random_forest.fit(X_train_added_features, y_train.values.ravel())
    cv_score_val = np.mean(cross_val_score(tree, X_validation_added_features, y_validation, cv = 5))
    cross_validation_scores_val.append(cv_score_val)
    cv_score_train = np.mean(cross_val_score(tree, X_train_added_features, y_train, cv = 5))
    cross_validation_scores_train.append(cv_score_train)
    

    print(f'depth = {max_depth}')
    print(f"DT R^2 score on training set: {my_random_forest.score(X_train_added_features, y_train):0.3f}")
    print(f"DT R^2 score on validation set: {my_random_forest.score(X_validation_added_features, y_validation):0.3f}")
    

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'max_features': [1,2,3,4,5],
    'min_samples_leaf': [1,2,3,4,5],
    'min_samples_split': [1,2,3,4,5,6,7,8,9,10],
    'n_estimators': [10,20,30,100,200]
}

my_random_forest = RandomForestRegressor()

clf = GridSearchCV(my_random_forest, param_grid)

my_random_forest.fit(X_remainder_added_features, y_remainder.values.ravel())

my_random_forest.score(X_test_added_features, y_test)
get_errors(X_test_added_features, y_test, my_random_forest)
param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'max_features': [180],
    'min_samples_leaf': [1,2,3,4,5],
    'min_samples_split': [1,2,3,4,5,6,7,8,9,10],
    'n_estimators': [10,20,30,100,200],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1]
}

gb_model = GradientBoostingRegressor()

clf = GridSearchCV(gb_model, param_grid)

gb_model.fit(X_remainder_added_features, y_remainder.values.ravel())
gb_model.score(X_test_added_features, y_test)
get_errors(X_test_added_features, y_test, gb_model)
#Scaling data for KNN (only scaling continuous data and not the one hot encoded data)
from sklearn.preprocessing import StandardScaler

scalerx = StandardScaler()
scalerx.fit(X_train_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])
X_train_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] = scalerx.transform(X_train_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])
X_validation_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] = scalerx.transform(X_validation_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])
X_test_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] = scalerx.transform(X_test_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])
X_remainder_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] = scalerx.transform(X_remainder_added_features[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']])


from sklearn.neighbors import KNeighborsRegressor

param_grid = {'n_neighbors':[2,3,4,5,6,7,8,9]}

KNN = KNeighborsRegressor()

KNN_model = GridSearchCV(KNN, param_grid, cv=5)
KNN_model.fit(X_remainder_added_features,y_remainder)
KNN_model.best_params_

KNN_model.score(X_test_added_features,y_test)
#Lets try KNN 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


KNNmodel = KNeighborsRegressor(weights='distance', n_neighbors=2)
KNNmodel.fit(X_train_added_features, scaled_y_remainder)

print(f"R^2 score on training set: {KNNmodel.score(X_train_added_features, scaled_y_remainder)}")
print(f"R^2 score on test set: {KNNmodel.score(X_test_added_features, scaled_y_test)}")
get_errors(X_test_added_features, y_test, KNN_model)

y_train_scaled=np.log10(y_train+1)
y_test_scaled=np.log10(y_test+1)
y_validation_scaled=np.log10(y_validation+1)
y_train_scaled.head()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.metrics import RootMeanSquaredError
model = Sequential([
    Dense(512, activation='relu', input_shape=(1227,)),
    Dense(512),
    Dense(1),
])


model.compile(
              loss='mse', optimizer='adam')

hist = model.fit(X_train_added_features, y_train_scaled,
          batch_size=32, epochs=2,
          validation_data=(X_validation_added_features, y_validation_scaled))


print(hist.history.keys())
# list all data in history
print(hist.history.keys())

# summarize history for loss
plt.plot(hist.history['loss'][1:])
plt.plot(hist.history['val_loss'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(X_val.shape)
print(y_validation.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
y_pred_log=model.predict(X_test_added_features)
y_pred_normal=(10**y_pred_log)-1
y_test_df=y_test.copy()
y_test_df["pred"]=y_pred_normal
y_test_df["abs_error"]=np.abs(y_test_df["pred"]-y_test_df["price"])
RMSE=((((y_test_df["price"]-y_test_df["pred"])**(2))/len(y_test_df)).sum())**(1/2)

print(f'RMSE: {RMSE}')
fig=px.histogram(x=y_test_df['abs_error'], nbins=30)


fig.show()

get_errors(X_test_added_features, y_test_scaled, model)
#Compare Models
RMSE=[37, 36 , 34, 40.47, 32.6, 40.19, 33.8]
RMSE=sorted(RMSE)

plt.figure()
plt.bar(x=["RFR", "GBR", "Ridge", "DT", "LR2", "LR1",  "NN"], height=RMSE)
plt.xlabel("Model")
plt.ylabel("RMSE on Test")
plt.title("Model Comparison")
plt.show()
RMSE
