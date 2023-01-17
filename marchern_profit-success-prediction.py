import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import metrics



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier



import warnings

with warnings.catch_warnings():

    warnings.filterwarnings("ignore",category=DeprecationWarning)
movies_dataframe = pd.read_csv('../input/movie_metadata.csv')

movies_dataframe.head(5)
movies_dataframe.shape
# PRE-PROCESSING



# Remove all movies made outside the USA

movies_dataframe = movies_dataframe.drop(movies_dataframe[movies_dataframe.country != 'USA'].index)

# Any block with missing information is dropped

movies_dataframe.dropna()





# Calculate profit to adjust for inflation

# Subtract budget from gross to calculate profits and add it as a new feature

movies_dataframe['profit'] = np.subtract(movies_dataframe['gross'].values, movies_dataframe['budget'].values)



movies_dataframe['profitpercent'] = np.subtract(movies_dataframe['gross'].values, movies_dataframe['budget'].values)

movies_dataframe['profitpercent'] = np.divide(movies_dataframe['profitpercent'].values, movies_dataframe['gross'].values)

movies_dataframe['profitpercent'] = np.multiply(movies_dataframe['profitpercent'].values, 100)



movies_dataframe = movies_dataframe.fillna(0)





movies_dataframe.head(5)
movies_dataframe.shape
# Graph for Profits vs Number of Movies

%matplotlib inline

plt.hist(movies_dataframe['profit'], bins=6, normed=False, range=(0, 550000000))



plt.xlabel('Profit (in hundred millions)')

plt.ylabel('Number of movies')

plt.show()



# We also want to see what the most profitable movie in our dataset made

max_profit = movies_dataframe[['profit']].max()

print(max_profit)
figure, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15, 5))

ax1.scatter(movies_dataframe['imdb_score'], movies_dataframe['profit'], alpha = 0.5)

ax2.scatter(movies_dataframe['budget'], movies_dataframe['profit'], alpha = 0.5)

ax3.scatter(movies_dataframe['duration'], movies_dataframe['profit'], alpha = 0.5)



ax1.set_ylabel('Profit (in hundred-millions)')

ax1.set_title('IMDB Score 1-10')

ax2.set_title('Budget (in millions)')

ax3.set_title('Duration (in minutes)')
# Profit vs. Title Year



x_axis = movies_dataframe[['title_year']]

y = movies_dataframe[['profit']]



plt.ylabel('Profit')

plt.xlabel('Title Year')



plt.ylim([-300000000, 525000000])   #Profit range of -$300M - $525M

plt.xlim([1920, 2020])              #Title Year range of 1920 - 2020

plt.scatter(x_axis, y, alpha=0.5)

plt.show()
# Drop samples with Title Year before 1980

movies_dataframe = movies_dataframe.drop(movies_dataframe[movies_dataframe.title_year < 1980].index)



# Checking new dataframe size for change.  OLD: (3793, 29)

movies_dataframe.shape
# Profit vs. Title Year (Revised)



x_axis = movies_dataframe[['title_year']]

y = movies_dataframe[['profit']]

%matplotlib inline



plt.ylabel('Profit')

plt.xlabel('Title Year')



plt.ylim([-300000000, 525000000])   #Profit range of -$300M - $525M

plt.xlim([1920, 2020])              #Title Year range of 1920 - 2020 to show change

plt.scatter(x_axis, y, alpha=0.5)

plt.show()
figure, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15, 5))

ax1.scatter(movies_dataframe['num_critic_for_reviews'], movies_dataframe['profit'], alpha = 0.5)

ax2.scatter(movies_dataframe['num_user_for_reviews'], movies_dataframe['profit'], alpha = 0.5)

ax3.scatter(movies_dataframe['num_voted_users'], movies_dataframe['profit'], alpha = 0.5)



ax1.set_ylabel('Profit (in hundred-millions)')

ax1.set_title('Number of Critic Reviews')

ax2.set_title('Number of User Reviews')

ax3.set_title('Number of Voted Users')
# Function to convert color values to binary values

def color_to_numeric(x):

    if x == 'Color':

        return 1;

    else:

        return 0;

    

# Creates a new feature named colorlabel using converted values

movies_dataframe['colorlabel'] = movies_dataframe['color'].apply(color_to_numeric)

movies_dataframe.head(5)
figure, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (15, 5))

ax1.scatter(movies_dataframe['facenumber_in_poster'], movies_dataframe['profit'], alpha = 0.5)

ax2.scatter(movies_dataframe['colorlabel'], movies_dataframe['profit'], alpha = 0.5)



ax1.set_ylabel('Profit (in hundred-millions)')

ax1.set_title('Face Number in Poster')

ax2.set_title('Color')
figure, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15, 5))

ax1.scatter(movies_dataframe['movie_facebook_likes'], movies_dataframe['profit'], alpha = 0.5)

ax2.scatter(movies_dataframe['cast_total_facebook_likes'], movies_dataframe['profit'], alpha = 0.5)

ax3.scatter(movies_dataframe['director_facebook_likes'], movies_dataframe['profit'], alpha = 0.5)



ax1.set_ylabel('Profit (in hundred-millions)')

ax1.set_title('Movie Facebook Likes')

ax2.set_title('Cast Total Facebook Likes')

ax3.set_title('Director Facebook Likes')
movies_dataframe.corr()
fig, axn = plt.subplots(figsize = (13, 8))

sns.heatmap(movies_dataframe.corr(),

            cmap = sns.diverging_palette(200, 220, 100, l = 45, n = 7, 

                                         as_cmap = True), 

            cbar_kws = {'shrink': 0.7},

            linewidths = 1,

            ax = axn)
# Function casts input into an integer and measures whether or not a movie made ANY profit

# New feature named 'profitlabel' is added to our dataset



def profit_to_numeric(x):

    int(x)

    if x == 0:      # If profit = 0

        return 0

    elif x < 0:     # If profit < 0

        return 1

    elif x > 0:     # If profit > 0

        return 2

    

movies_dataframe['profitlabel'] = movies_dataframe['profit'].apply(profit_to_numeric)

movies_dataframe.head(5)
print('Feature                        KNN Score')

print('---------------------------------------------')



y = movies_dataframe['profitlabel']

knn = KNeighborsClassifier(n_neighbors = 16)



X = movies_dataframe[['imdb_score']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('IMDB Score:                   ', score)







X = movies_dataframe[['colorlabel']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('ColorLabel:                   ', score)







X = movies_dataframe[['num_user_for_reviews']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Number of User Reviews:       ', score)







X = movies_dataframe[['budget']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Budget:                       ', score)







X = movies_dataframe[['facenumber_in_poster']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Face Number in Poster:        ', score)







X = movies_dataframe[['title_year']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Title Year:                   ', score)







X = movies_dataframe[['num_critic_for_reviews']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Number of Critic Reviews:     ', score)







X = movies_dataframe[['num_voted_users']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Number of Voted Users:        ', score)







X = movies_dataframe[['duration']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('Duration:                     ', score)





X = movies_dataframe[['imdb_score', 'budget', 'duration', 'title_year', 'num_critic_for_reviews', 'num_user_for_reviews', 'num_voted_users', 'facenumber_in_poster']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)

score = accuracy_score(y_test, y_predict)

print('')

print('KNN TOTAL:                    ', score)
X = movies_dataframe[['imdb_score', 'budget', 'duration', 'title_year', 'num_critic_for_reviews', 'num_user_for_reviews', 'num_voted_users', 'facenumber_in_poster']]

y = movies_dataframe['profitlabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



knn = KNeighborsClassifier(n_neighbors=16)

logreg = LogisticRegression()

decisiontree = DecisionTreeClassifier(random_state = 2)

adaboost = AdaBoostClassifier(n_estimators=19)

randomforest = RandomForestClassifier(n_estimators=19, bootstrap=True, random_state=2)



knn.fit(X_train, y_train)

logreg.fit(X_train, y_train)

decisiontree.fit(X_train, y_train)

adaboost.fit(X_train, y_train)

randomforest.fit(X_train, y_train)



knnPredict = knn.predict(X_test)

logregPredict = logreg.predict(X_test)

dectreePredict = decisiontree.predict(X_test)

adaPredict = adaboost.predict(X_test)

forestPredict = randomforest.predict(X_test)



knnScore = accuracy_score(y_test, knnPredict)

logregScore = accuracy_score(y_test, logregPredict)

dectreeScore = accuracy_score(y_test, dectreePredict)

adaScore = accuracy_score(y_test, adaPredict)

forestScore = accuracy_score(y_test, forestPredict)



print("KNN Score:            ", end="")

print(knnScore)

print("LogReg Score:         ", end="")

print(logregScore)

print("DecTree Score:        ", end="")

print(dectreeScore)

print("AdaBoost Score:       ", end="")

print(adaScore)

print("RandomForest Score:   ", end="")

print(forestScore)