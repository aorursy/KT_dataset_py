#import modules:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Open the data file and convert to a Data Frame

#Here I also separated randomly selected 20% of the data for later validation



data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

#randomize the data

data.sample(frac=1)

df=data.copy()

df.shape
df.head().iloc[:,0:8]
df.head().iloc[:,8:]
#fill missing values for last review and reviews per month with 0

df[["last_review", "reviews_per_month"]] = df[["last_review", "reviews_per_month"]].fillna(0)



#if there is no host name or listing name fill in None

df[["name", "host_name"]] = df[["name", "host_name"]].fillna("None")



#Drop rows were price of the listing is 0. We are not intersted in "free" 

#listings as they are most likely an error.

free = len(df[df.price == 0])

df = df[df.price != 0].copy()



#Print initial insights:

print("The initial dataset contained " + str(free)+ " listings with price of 0 USD, that had been removed")

print("There are " + str(len(df["id"].unique()))+" listings")

print("There are "+str(len(df.host_id.unique()))

      +" unique and indentifiable "+ "hosts.")

print("There are "+str(len(df[df["host_name"]=="None"]))

      +" unindentifiable "+ "hosts.")

print("Dataframe shape: "+str(df.shape))
(len(df[df["host_id"]==30985759]) == df[df["id"]==36485609]["calculated_host_listings_count"]).tolist()
df[(df["calculated_host_listings_count"]>1)][["host_id","calculated_host_listings_count"]].sort_values(by=['host_id']).head(10)
df_old=df.copy()

df = df[df["minimum_nights"] <=31].copy()

removed_listings = len(df_old)-len(df)



fig = plt.figure(figsize=(14,3))

ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)





ax1.hist(df_old.minimum_nights, bins=100, log=True)

ax1.set_ylabel("Frequency")

ax1.set_title("No limit on minimum nights")



ax2.hist(df.minimum_nights, bins=31, log=True)

ax2.set_ylabel("Frequency")

ax2.set_title("Maximum 31 minimum nights")



plt.show()



print("As a result of imposing minimum nights limit, " + str(removed_listings)+" listings were removed.")
df.isnull().sum()
df.describe().iloc[:,0:8]
df.describe().iloc[:,8:]
df.dtypes
#separate out numerical variables

a=pd.DataFrame(df.dtypes.copy())

b= a[a[0] != 'object'].reset_index()

#drop id and host id:

numeric_vars=b["index"].tolist()[2:]



fig = plt.figure(figsize=(14,14))

ax1 = fig.add_subplot(3, 3, 1)

ax2 = fig.add_subplot(3, 3, 2)

ax3 = fig.add_subplot(3, 3, 3)

ax4 = fig.add_subplot(3, 3, 4)

ax5 = fig.add_subplot(3, 3, 5)

ax6 = fig.add_subplot(3, 3, 6)

ax7 = fig.add_subplot(3, 3, 7)

ax8 = fig.add_subplot(3, 3, 8)



ax1.hist(df[numeric_vars[0]], bins=30)

ax1.set_ylabel("Frequency")

ax1.set_title(numeric_vars[0])



ax2.hist(df[numeric_vars[1]], bins=30)

ax2.set_ylabel("Frequency")

ax2.set_title(numeric_vars[1])



ax3.hist((df[numeric_vars[2]]), bins=30)

ax3.set_ylabel("Frequency")

ax3.set_title('price')



ax4.hist(df[numeric_vars[3]], bins=31)

ax4.set_ylabel("Frequency")

ax4.set_title(numeric_vars[3])



ax5.hist(df[numeric_vars[4]], bins=30)

ax5.set_ylabel("Frequency")

ax5.set_title("number of reviews")



ax6.hist(df[numeric_vars[5]], bins=30)

ax6.set_ylabel("Frequency")

ax6.set_title("last review")



ax7.hist(df[numeric_vars[6]], bins=30)

ax7.set_ylabel("Frequency")

ax7.set_title(numeric_vars[6])



ax8.hist(df[numeric_vars[7]])

ax8.set_ylabel("Frequency")

ax8.set_title(numeric_vars[7])

plt.show()
numeric_vars
fig = plt.figure(figsize=(14,14))

ax1 = fig.add_subplot(3, 3, 1)

ax2 = fig.add_subplot(3, 3, 2)

ax3 = fig.add_subplot(3, 3, 3)

ax4 = fig.add_subplot(3, 3, 4)

ax5 = fig.add_subplot(3, 3, 5)

ax6 = fig.add_subplot(3, 3, 6)

ax7 = fig.add_subplot(3, 3, 7)

ax8 = fig.add_subplot(3, 3, 8)



ax1.hist(df[numeric_vars[0]], bins=30)

ax1.set_ylabel("Frequency")

ax1.set_title(numeric_vars[0])



ax2.hist(df[numeric_vars[1]], bins=30)

ax2.set_ylabel("Frequency")

ax2.set_title(numeric_vars[1])



ax3.hist(np.log((df[numeric_vars[2]])), bins=30)

ax3.set_ylabel("Frequency")

ax3.set_title('log(price)')



ax4.hist(np.log((df[numeric_vars[3]])), bins=31)

ax4.set_ylabel("Frequency")

ax4.set_title("log(minimum nights + 1)")



ax5.hist(np.log(df[numeric_vars[4]]+1), bins=30)

ax5.set_ylabel("Frequency")

ax5.set_title("log(number of reviews + 1)")



ax6.hist(np.log(df[numeric_vars[5]]+1), bins=30)

ax6.set_ylabel("Frequency")

ax6.set_title("log(last review + 1)")



ax7.hist(np.log(df[numeric_vars[6]]+1), bins=30)

ax7.set_ylabel("Frequency")

ax7.set_title("log(calculated host listing count) + 1)")



ax8.hist(np.log(df[numeric_vars[7]]+1), bins=30)

ax8.set_ylabel("Frequency")

ax8.set_title("log(availability 365 + 1)")



plt.show()
for num in numeric_vars[3:]:

    df["log_("+num+" +1)"] = np.log(df[num]+1)

df["log_price"] = np.log(df.price)

df=df.drop(columns = numeric_vars[2:]).copy()
df.columns.tolist()
df.shape
numeric_vars = df.columns.tolist()[6:8]+df.columns.tolist()[10:]
numeric_vars
import seaborn as sns

x=df[numeric_vars].apply(lambda x: np.log(np.abs(x+1))).corr(method='pearson')

sns.heatmap(x, annot=True)

plt.show()
#separate out numerical variables

a=pd.DataFrame(df.dtypes.copy())

b= a[a[0] == 'object'].reset_index()

#drop id and host id:

non_num=b["index"].tolist()

print(non_num)
y = df.latitude

x = df.longitude

p = df.log_price

plt.figure(figsize=(16,9))

plt.scatter(x,y,c=p,cmap='viridis')

plt.colorbar()

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Distribution of listing prices")

plt.show()
grouped = df.groupby("neighbourhood")

price_grouped = grouped["log_price"]

price = price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")





fig = plt.figure(figsize=(14,4))

ax1 = fig.add_subplot(1, 3, 1)

ax2 = fig.add_subplot(1, 3, 2)

ax3 = fig.add_subplot(1, 3, 3)



ax1.barh(price.index,price["mean"])

ax1.set_yticklabels([])

ax1.set_ylabel("Neighborhood")

ax1.set_xlabel("Mean Price")

ax1.set_title("Mean Listing Price per Neighborhood, Sorted")

ax1.set_xlim(3,7)



ax2.barh(price.index,price["median"])

ax2.set_yticklabels([])

ax2.set_ylabel("Neighborhood")

ax2.set_xlabel("Median Price")

ax2.set_title("Median Listing Price per Neighborhood")

ax2.set_xlim(3,7)



ax3.barh(price.index,price["std"])

ax3.set_yticklabels([])

ax3.set_ylabel("Neighborhood")

ax3.set_xlabel("Standard Deviation of Price")

ax3.set_title("StDev of Listing Prices per Neighborhood")

plt.show()
#One hot encoding

df = pd.concat([df, pd.get_dummies(df["neighbourhood"], drop_first=False)], axis=1)

#save neighborhoods into a list for further analysis:

neighborhoods = df.neighbourhood.values.tolist()

boroughs = df.neighbourhood_group.unique().tolist()

#drop the neighbourhood column from the database

df.drop(['neighbourhood'],axis=1, inplace=True)
df.shape
grouped = df.groupby("room_type")

room_type_price_grouped = grouped["log_price"]

room_type_price = room_type_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")

room_type_price
sns.boxplot(x="room_type",y="log_price", data=df)

plt.show()
def removal_of_outliers(df,room_t, nhood, distance):

    '''Function removes outliers that are above 3rd quartile and below 1st quartile'''

    '''The exact cutoff distance above and below can be adjusted'''



    new_piece = df[(df["room_type"]==room_t)&(df["neighbourhood_group"]==nhood)]["log_price"]

    #defining quartiles and interquartile range

    q1 = new_piece.quantile(0.25)

    q3 = new_piece.quantile(0.75)

    IQR=q3-q1



    trimmed = df[(df.room_type==room_t)&(df["neighbourhood_group"]==nhood) &(df.log_price>(q1-distance*IQR))&(df.log_price<(q3+distance*IQR))]

    return trimmed



#apply the function

df_private = pd.DataFrame()

for neighborhood in boroughs:

    a = removal_of_outliers(df, "Private room",neighborhood,3)

    df_private = df_private.append(a)



df_shared = pd.DataFrame()

for neighborhood in boroughs:

    a = removal_of_outliers(df, "Shared room",neighborhood,3)

    df_shared = df_shared.append(a)

    

df_apt = pd.DataFrame()

for neighborhood in boroughs:

    a = removal_of_outliers(df, "Entire home/apt",neighborhood,3)

    df_apt = df_apt.append(a)

    

# Create new dataframe to absorb newly produced data    

df_old=df.copy()    

df = pd.DataFrame()

df = df.append([df_private,df_shared,df_apt])



#plot the results

fig = plt.figure(figsize=(14,4))

ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



ax1.hist(df_old.log_price)

ax1.set_xlim(2,7)

ax1.set_ylabel("Frequency")

ax1.set_xlabel("Log Price")

ax1.set_title("Original price distribution")



ax2.hist(df.log_price)

ax2.set_xlim(2,7)

ax2.set_ylabel("Frequency")

ax2.set_xlabel("Log Price")

ax2.set_title("Price distribution after removal of extreme outliers")

plt.show()



print("As a result of oulier removal " + str(df_old.shape[0]-df.shape[0]) + " rows of data were removed.")
df.shape
grouped = df.groupby("room_type")

room_type_price_grouped = grouped["log_price"]

room_type_price = room_type_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")

room_type_price
#convert room types to dummies

df = pd.concat([df, pd.get_dummies(df["room_type"], drop_first=False)], axis=1)

df.drop(['room_type'],axis=1, inplace=True)
df.shape
y = df[(df["SoHo"]==1) & (df["Private room"]==1)].latitude

x = df[(df["SoHo"]==1) & (df["Private room"]==1)].longitude

p = df[(df["SoHo"]==1) & (df["Private room"]==1)].log_price

plt.scatter(x,y,c=p,cmap='viridis')

plt.xlim(-74.01,-73.995)

plt.ylim(40.718,40.73)

plt.colorbar()

plt.show()
import datetime as dt

#convert object to datetime:

df["last_review"] = pd.to_datetime(df["last_review"])

#Check the latest review date in the datebase:

print(df["last_review"].max())
df.shape
df["last_review"]=df["last_review"].apply(lambda x: dt.datetime(2019,7,8)-x)

df["last_review"]=df["last_review"].dt.days.astype("int").replace(18085, 1900)

plt.hist(df["last_review"], bins=100)

plt.ylabel("Frequency")

plt.xlabel("Days since last review")

plt.ylabel("Frequency")

plt.title("Histogram of days since last review")

plt.show()
def date_replacement(date):

    if date <=3:

        return "Last_review_last_three_day"

    elif date <= 7:

        return "Last_review_last_week"

    elif date <= 30:

        return "Last_review_last_month"

    elif date <= 183:

        return "Last_review_last_half_year"

    elif date <= 365:

        return "Last_review_last year"

    elif date <= 1825:

        return "Last_review_last_5_years"

    else:

        return "Last_review_never" 



    

df["last_review"]=df["last_review"].apply(lambda x: date_replacement(x))

sns.boxplot(x="last_review", y=df.log_price, data=df)

plt.show()
grouped = df.groupby("last_review")

last_review_price_grouped = grouped["log_price"]

last_review_price = last_review_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")

last_review_price
#convert last review to dummies

df = pd.concat([df, pd.get_dummies(df["last_review"], drop_first=False)], axis=1)

df.drop(["last_review"],axis=1, inplace=True)
#import necessary libraries

import nltk

import os

import nltk.corpus

from nltk import ne_chunk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer

import string

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('maxent_ne_chunker')

nltk.download('words')
#initiate stopwords

a = set(stopwords.words('english'))

#obtain text

text = df["name"].iloc[10]

#tokenize text

text1 = word_tokenize(text.lower())

#create a list free of stopwords

no_stopwords = [x for x in text1 if x not in a]

#lemmatize the words

lemmatizer = WordNetLemmatizer() 

lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]
def unique_words1(dwelling):



    apt = df[df[dwelling]==1]["name"]

    a = set(stopwords.words('english'))

    words = []

    # append each to a list

    for lis in range(0, len(apt)):

        listing = apt.reset_index().iloc[lis,1]

        #tokenize text

        text1 = word_tokenize(listing.lower())

        #create a list free of stopwords

        no_stopwords = [x for x in text1 if x not in a]

        #lemmatize the words

        lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]

        no_punctuation = [x.translate(str.maketrans('','',string.punctuation)) for x in lemmatized]

        no_digits = [x.translate(str.maketrans('','',"0123456789")) for x in no_punctuation ]

        for item in no_digits:

            words.append(item)





    #create a dictionary

    unique={}

    for word in words:

        if word in unique:

            unique[word] +=1

        else:

            unique[word] = 1



    #sort the dictionary

    a=[]

    b=[]



    for key, value in unique.items():

        a.append(key)

        b.append(value)



    aa=pd.Series(a)

    bb=pd.Series(b)    



    comb=pd.concat([aa,bb],axis=1).sort_values(by=1, ascending=False).copy()



    return comb



#apply the function

private = unique_words1("Private room")

home = unique_words1("Entire home/apt")

shared = unique_words1("Shared room")



words_private = private.iloc[1:,1]

words_home = home.iloc[1:,1] 

words_shared = shared.iloc[1:,1] 



#plot the results

plt.plot(words_shared.reset_index()[1], label="shared")

plt.plot(words_private.reset_index()[1], label ="private")

plt.plot(words_home.reset_index()[1], label="Entire home/apt")

plt.xlim(0,200)

plt.ylabel("WordFrequency")

plt.xlabel("Word position on the list")

plt.legend()

plt.show()

home_new = home.reset_index().iloc[1:50,1:3].copy()

private_new = private.reset_index().iloc[1:50,1:3].copy()

shared_new = shared.reset_index().iloc[1:50,1:3].copy()



all_words = pd.concat([home_new, private_new, shared_new], axis=1, sort=False)

all_words
#see how many listing there are for each type of room:

print("Numer of shared room listings: "+str(len(df[df["Shared room"]==1])))

print("Numer of private room listings: "+str(len(df[df["Private room"]==1])))

print("Numer of entire home/apt listings: "+str(len(df[df["Entire home/apt"]==1])))
#Create a list of the most popular words common for all room types:

most_popular_words = home_new.iloc[:,0].tolist()+private_new.iloc[:,0].tolist()+shared_new.iloc[:,0].tolist()

most_popular = pd.Series(most_popular_words)

popular_descriptors=most_popular.unique().tolist()
def unique_words2(name, word):

    '''This function takes individual name and looks for a matching word in it'''

    a = set(stopwords.words('english'))

    #tokenize the name

    text1 = word_tokenize(str(name).lower())

    #create a list free of stopwords

    no_stopwords = [x for x in text1 if x not in a]

    #lemmatize the words

    lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]

    no_punctuation = [x.translate(str.maketrans('','',string.punctuation)) for x in lemmatized]

    no_digits = [x.translate(str.maketrans('','',"0123456789")) for x in no_punctuation ]

    counter = 0

    for item in no_digits:

        if str(item) == str(word):

            counter += 1

        else:

            continue



    if counter != 0:

        return 1

    else:

        return 0

    

#Apply the function 

for item in popular_descriptors:

    df[item]= df["name"].apply(lambda x: unique_words2(x,item))
#convert last review to dummies

df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'], drop_first=False)], axis=1)

df.drop(['neighbourhood_group'],axis=1, inplace=True)
#drop unnecessary columns

df = df.drop(['id','name','host_id','host_name'], axis=1).copy()

#copy for later

df2 = df.copy()

df.shape
len(popular_descriptors)
def plot_by_word(word):

    '''creates a plot of price for listings matching given word'''

    y = df[(df[word]==1)].latitude

    x = df[(df[word]==1)].longitude

    p = df[(df[word]==1)].log_price

    plt.figure(figsize=(16,9))

    plt.scatter(x,y,c=p,cmap='viridis')

    plt.xlabel

    plt.colorbar()

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    plt.title("Word 'Luxury' in the name of the listing\nColormap indicates price")

    plt.show()

    

plot_by_word("Manhattan")
#import modules:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb



from sklearn.tree            import DecisionTreeRegressor

from sklearn.neural_network  import MLPRegressor

from sklearn.linear_model    import LinearRegression

from sklearn.ensemble        import RandomForestRegressor

from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics         import mean_squared_error

from sklearn.metrics         import r2_score
target = df['log_price'].copy()

#drop unnecessary columns

df = df.drop(['log_price'], axis=1).copy()

#strip the target column from input columns and put it in front

df = pd.concat([target, df], axis=1).copy()

#select input variable columns

nums = df.iloc[:,1:]
#first few rows of the dataframe:

df.head()
#dataframe shape

print(df.shape)
#column names in the dataframe

df.columns.tolist()
y= target

x = nums

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)
rmse_dt=[]

dt = DecisionTreeRegressor()

kf = KFold(5, shuffle = True, random_state=1)

mse = cross_val_score(dt ,x,y, scoring = "neg_mean_squared_error", cv=kf) 

rmse = np.sqrt(np.absolute(mse))

avg_rmse = np.sum(rmse)/len(rmse)

rmse_dt.append(avg_rmse)

print("Root mean square error: " +str(round(rmse_dt[0],2)))
rmse_xg = []

data_dmatrix = xgb.DMatrix(data=x,label=y)

params = {

              'colsample_bytree': 0.9,

              'learning_rate': 0.1,

              'max_depth': 1, 

              'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5, num_boost_round=300,

                        early_stopping_rounds=10, metrics="rmse", as_pandas=True, 

                        seed=123)

    

rmse_xg.append(cv_results["test-rmse-mean"].tolist()[-1])

print("Root mean square error: " +str(round(rmse_xg[0],2)))
rmse_rf=[]

rf=RandomForestRegressor(n_estimators = 100, random_state=1,  min_samples_leaf=2)

kf = KFold(5, shuffle = True, random_state=1)

mse = cross_val_score(rf ,x,y, scoring = "neg_mean_squared_error", cv=kf) 

rmse = np.sqrt(np.absolute(mse))

avg_rmse = np.sum(rmse)/len(rmse)

rmse_rf.append(avg_rmse)

print(rmse_rf)
rmse_nndf=[]

mlp = MLPRegressor(activation='relu', max_iter=1000)

kf = KFold(5, shuffle = True, random_state=1)

mse = cross_val_score(mlp ,x,y, scoring = "neg_mean_squared_error", cv=kf) 

rmse = np.sqrt(np.absolute(mse))

avg_rmse = np.sum(rmse)/len(rmse)

rmse_nndf.append(avg_rmse)

print(rmse_nndf)
dt = pd.Series(rmse_dt, name ="Decision Tree")

rand = pd.Series(rmse_rf, name ="Random Forest")

xgb = pd.Series(rmse_xg, name ="XG Boost")

nn = pd.Series(rmse_nndf, name="Neural Network")

pd.concat([dt,rand,xgb,nn],axis=1)
#optimizing number of estimators

train_results = []

test_results = []

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

for estimator in n_estimators:

    rf = RandomForestRegressor(n_estimators=estimator, n_jobs=-1, random_state=1)

    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    rmse = round(np.sqrt(mean_squared_error(y_train, train_pred)),2)

    train_results.append(rmse)

    y_pred = rf.predict(X_test)

    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),2)

    test_results.append(rmse)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label='Train RMSE')

line2, = plt.plot(n_estimators, test_results, 'r', label='Test RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('RMSE')

plt.xlabel('n_estimators')

plt.show()
#optimizing max_features

train_results = []

test_results = []

max_features = ['auto','sqrt','log2']

for feature in max_features:

    rf = RandomForestRegressor(max_features=feature, n_estimators=100, n_jobs=-1, random_state=1)

    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    rmse = round(np.sqrt(mean_squared_error(y_train, train_pred)),2)

    train_results.append(rmse)

    y_pred = rf.predict(X_test)

    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),2)

    test_results.append(rmse)

    

plt.bar(max_features,test_results)

plt.bar(max_features,train_results)

plt.ylabel('RMSE')

plt.xlabel('max_features, test, train')

plt.show()
#optimizing min_sample_leaf

train_results = []

test_results = []

min_samples_leaf = [1,2,10,50,70,100]

for leaf in min_samples_leaf:

    rf = RandomForestRegressor(min_samples_leaf = leaf, max_features='auto', n_estimators=100, n_jobs=-1, random_state=1)

    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    rmse = round(np.sqrt(mean_squared_error(y_train, train_pred)),3)

    train_results.append(rmse)

    y_pred = rf.predict(X_test)

    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),3)

    test_results.append(rmse)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leaf, train_results, 'b', label='Train RMSE')

line2, = plt.plot(min_samples_leaf, test_results, 'r', label='Test RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('RMSE')

plt.xlabel('min_sample_leaf')

plt.show()
test_results 
#apply the hyperparameter optmized model:

rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=2, random_state=1)

rf.fit(X_train,y_train)

predicted = rf.predict(X_test)
#plot the results of the model:

plt.figure(figsize=(8,4.5))

plt.scatter(y_test,predicted, label="Model Results")

plt.plot([2,7],[2,7], color="red", label = "Equality Line")

plt.title("Predictions for test portion of the dataset")

plt.xlim(2,7)

plt.ylim(2,7)

plt.legend()

plt.ylabel("Predicted log_price")

plt.xlabel("Actual log_price")

plt.show()
print("Model accuracy measures for withheld data:\nR2: "+str(round(r2_score(y_test,predicted),2)))

print("Root mean square error: "+str(round(np.sqrt(mean_squared_error(y_test,predicted)),3)))
#derive important features

feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)



print("Number of important features: "+str(feature_importances[feature_importances["importance"]!=0].shape[0]))

print("\nTop fifteen features by importance:")

feature_importances[feature_importances["importance"]!=0].head(15)
predicted_ = rf.predict(X_test)

pred = pd.DataFrame({'Predicted log_price':predicted_,'log_price':y_test})

df_with_predictions = pd.concat([X_test, pred], axis=1).copy()

df_with_predictions["price"]=df_with_predictions["log_price"].apply(lambda x: np.exp(x))

df_with_predictions["predicted_price"]=df_with_predictions["Predicted log_price"].apply(lambda x: round(np.exp(x),1))
prices=df_with_predictions.sort_values(by="price").reset_index()

prices["error"]=np.abs(prices.price-prices.predicted_price)/prices.price*100
plt.figure(figsize=(15,4.5))

plt.plot(prices["predicted_price"], label="Predicted Price")

plt.plot(prices["price"], label = "Actual Price")

plt.title("Price prediction vs. actual price for listings in the test dataset sorted")

plt.xlim(0,10000)

plt.ylim(0,800)

plt.legend()

plt.ylabel("Price USD")

plt.xlabel("Lisitng")

plt.show()
plt.figure(figsize=(15,4.5))

plt.plot(prices["error"], label="Price Error")

plt.title("Absolute price error % sorted")

plt.xlim(0,10000)

plt.ylim(0,400)

plt.legend()

plt.ylabel("Price Error (%)")

plt.xlabel("Lisitng")

plt.show()
small_error = prices[prices.error<20].copy()

y = small_error["latitude"]

x = small_error["longitude"]

p = small_error.error

plt.figure(figsize=(16,9))

plt.scatter(x,y,c=p,cmap='viridis')

plt.colorbar()

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Distribution of errors")

plt.show()
print(prices[prices.Manhattan==1]["error"].mean())

print(prices[prices.Brooklyn==1]["error"].mean())

print(prices[prices.Bronx==1]["error"].mean())

print(prices[prices.Queens==1]["error"].mean())

print(prices[prices["Staten Island"]==1]["error"].mean())
#define random forest function with kfold cross-validation

def random_forest(df):

    target = df['log_price'].copy()

    #select input variable columns

    nums = df.iloc[:,1:]



    #split the data into test and train

    y= target

    x = nums

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)



    rmse_rf=[]

    rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=1, random_state=1)

    kf = KFold(5, shuffle = True, random_state=1)

    mse = cross_val_score(rf ,x,y, scoring = "neg_mean_squared_error", cv=kf) 

    rmse = np.sqrt(np.absolute(mse))

    avg_rmse = np.sum(rmse)/len(rmse)

    rmse_rf.append(avg_rmse)

    return rmse_rf
#separate datasets

private = df[df["Private room"]==1].copy()

shared = df[df["Shared room"]==1].copy()

homes = df[df["Entire home/apt"]==1].copy()



private_rmse = random_forest(private)

shared_rmse = random_forest(shared)

home_rmse = random_forest(homes)



print("\nShared RMSE: "+str(round(shared_rmse[0],3)))

print("Private RMSE: "+str(round(private_rmse[0],3)))

print("Home RMSE: "+str(round(home_rmse[0],3)))
print(private.log_price.std())

print(homes.log_price.std())
#separate datasets

manhattan = df[(df["Manhattan"]==1)].copy()

brooklyn = df[(df["Brooklyn"]==1)].copy()

queens = df[(df["Queens"]==1)].copy()

bronx = df[(df["Bronx"]==1)].copy()

staten_island = df[(df["Staten Island"]==1)].copy()



manhattan_rmse = random_forest(manhattan)

brooklyn_rmse = random_forest(brooklyn)

queens_rmse = random_forest(queens)

bronx_rmse = random_forest(bronx)

staten_island_rmse = random_forest(staten_island)



print("\nManhattan RMSE: "+str(round(manhattan_rmse[0],3)))

print("Brooklyn RMSE: "+str(round(brooklyn_rmse[0],3)))

print("Queens RMSE: "+str(round(queens_rmse[0],3)))

print("Bronx RMSE: "+str(round(bronx_rmse[0],3)))

print("Staten Island RMSE: "+str(round(staten_island_rmse[0],3)))
print("Number of listings in Manhattan: "+str(len(manhattan)))

print("Number of listings in Brooklyn: "+str(len(brooklyn)))

print("Number of listings in Queens: "+str(len(queens)))

print("Number of listings in Bronx: "+str(len(bronx)))

print("Number of listings in Staten Island: "+str(len(staten_island)))
plt.scatter(x=[0.384,0.369,0.369,0.419,0.458],y=[21192,19801,5592,1071,370])

plt.xlabel("RMSE")

plt.ylabel("Number of listings in a borough")

plt.title("Listing number vs. Model RMSE")

plt.show()
#separate datasets

manhattan = df[(df["Manhattan"]==1)&(df["Private room"]==1)].copy()

brooklyn = df[(df["Brooklyn"]==1)&(df["Private room"]==1)].copy()

queens = df[(df["Queens"]==1)&(df["Private room"]==1)].copy()

bronx = df[(df["Bronx"]==1)&(df["Private room"]==1)].copy()

staten_island = df[(df["Staten Island"]==1)&(df["Private room"]==1)].copy()



manhattan_rmse = random_forest(manhattan)

brooklyn_rmse = random_forest(brooklyn)

queens_rmse = random_forest(queens)

bronx_rmse = random_forest(bronx)

staten_island_rmse = random_forest(staten_island)



print("\nManhattan RMSE: "+str(round(manhattan_rmse[0],3)))

print("Brooklyn RMSE: "+str(round(brooklyn_rmse[0],3)))

print("Queens RMSE: "+str(round(queens_rmse[0],3)))

print("Bronx RMSE: "+str(round(bronx_rmse[0],3)))

print("Staten Island RMSE: "+str(round(staten_island_rmse[0],3)))


target = df2['log_price'].copy()

#drop unnecessary columns

df = df2.drop(['log_price'], axis=1).copy()

#strip the target column from input columns and put it in front

df = pd.concat([target, df], axis=1).copy()

#select input variable columns

nums = df.iloc[:,1:]
# RMSE for the global model considering Queens and Private room only

qns_priv_price=df_with_predictions[(df_with_predictions["Queens"]==1)&(df_with_predictions["Private room"]==1)].log_price

qns_priv_predprice=df_with_predictions[(df_with_predictions["Queens"]==1)&(df_with_predictions["Private room"]==1)]["Predicted log_price"]

rmse_global = round(np.sqrt(mean_squared_error(qns_priv_price, qns_priv_predprice)),3)

print(rmse_global)
#separating the data to make it specific to the borough of Queens and the listing type

df=df[(df["Queens"]==1)&(df["Private room"]==1)].copy()



target = df['log_price'].copy()

#drop unnecessary columns

df = df.drop(['log_price'], axis=1).copy()

#strip the target column from input columns and put it in front

df = pd.concat([target, df], axis=1).copy()

#select input variable columns

nums = df.iloc[:,1:]



y= target

x = nums

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)



rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=2, random_state=1)

rf.fit(X_train,y_train)

predicted = rf.predict(X_test)
round(np.sqrt(mean_squared_error(y_test, predicted)),3)
def final_model(borough, room_type):

    '''Build a function specifc to a borough and room_type'''

    #read the cleaned data from a file:

    df = df2.copy()

    #filter the data

    df=df[(df[borough]==1)&(df[room_type]==1)].copy()    

    target = df['log_price'].copy()

    #drop unnecessary columns

    df = df.drop(['log_price'], axis=1).copy()    

    #strip the target column from input columns and put it in front

    df = pd.concat([target, df], axis=1).copy()

    #select input variable columns

    nums = df.iloc[:,1:]

    

    y= target

    x = nums

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=1)



    rf=RandomForestRegressor(n_estimators = 300, max_features = 'auto', min_samples_leaf=2, random_state=1)

    rf.fit(X_train,y_train)

    predicted = rf.predict(X_test)

    y_test = y_test.values.tolist()

    predicted = predicted.tolist()

    return y_test, predicted 
boroughs = ["Manhattan", "Brooklyn", "Bronx","Queens","Staten Island"]

listings= ["Private room","Shared room","Entire home/apt"]



actual=[]

predicted=[]

for borough in boroughs:

    for listing in listings:

        a,b = final_model(borough, listing)

        actual +=a

        predicted +=b

        

round(np.sqrt(mean_squared_error(actual, predicted)),3)
#plot the results of the model:

plt.figure(figsize=(8,4.5))

plt.scatter(actual, predicted, label="Model Results")

plt.plot([2,7],[2,7], color="red", label = "Equality Line")

plt.title("Predictions for test portion of the dataset")

plt.xlim(2,7)

plt.ylim(2,7)

plt.legend()

plt.ylabel("Predicted log_price")

plt.xlabel("Actual log_price")

plt.show()
print("Model accuracy measures for withheld data:\nR2: "+str(round(r2_score(actual,predicted),3)))

print("Root mean square error: "+str(round(np.sqrt(mean_squared_error(actual,predicted)),3)))