# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import collections 

from sklearn.preprocessing import MinMaxScaler

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import warnings

from textblob import TextBlob 

from sklearn.ensemble import RandomForestClassifier

import folium

warnings.filterwarnings("ignore")



Hotels = pd.read_csv("/kaggle/input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv")



Uniq_ = Hotels.sort_values('Hotel_Name', ascending=False)

Uniq_hotels = Uniq_.drop_duplicates(subset='Hotel_Address', keep='first')

hotels = Uniq_.drop(["Review_Date","days_since_review","lat","lng","Tags","Total_Number_of_Reviews_Reviewer_Has_Given","Positive_Review","Negative_Review","Reviewer_Score"], axis=1)

 

nationality_counter = collections.Counter(Hotels["Reviewer_Nationality"].tolist())

hotel_counter = collections.Counter(Hotels["Hotel_Name"].tolist())



Uniq_hotels.tail()

uniq_hotels = Uniq_hotels.drop(["Review_Date","Reviewer_Nationality","Negative_Review","Review_Total_Negative_Word_Counts","Positive_Review","Review_Total_Positive_Word_Counts","Total_Number_of_Reviews_Reviewer_Has_Given","Reviewer_Score","Tags","days_since_review","lat","lng"],axis=1)



hoteladdress = uniq_hotels["Hotel_Address"]



uniq_hotels_lat = Uniq_hotels["lat"]

uniq_hotels_lng = Uniq_hotels["lng"]



Hotel_num = []



for c in hoteladdress:

    if "United Kingdom" in c:

        Hotel_num.append(0)

    elif "France" in c:

        Hotel_num.append(1)

    elif "Italy" in c:

        Hotel_num.append(2)

    elif "Spain" in c:

        Hotel_num.append(3)

    elif "Austria" in c:

        Hotel_num.append(4)

    elif "Netherlands" in c:

        Hotel_num.append(5)

        

uniq_hotels["hotel_loc"] = Hotel_num

hotel_add = hotels["Hotel_Address"].unique()



neg_rev_avg_words = []

pos_rev_avg_words = []



for i in range(len(hotel_add)):

    

    neg_rev_avg_words.append(sum(hotels[hotels.Hotel_Address == hotel_add[i]]["Review_Total_Negative_Word_Counts"])/len(hotels[hotels.Hotel_Address == hotel_add[i]]["Review_Total_Negative_Word_Counts"]))

    pos_rev_avg_words.append(sum(hotels[hotels.Hotel_Address == hotel_add[i]]["Review_Total_Positive_Word_Counts"])/len(hotels[hotels.Hotel_Address == hotel_add[i]]["Review_Total_Positive_Word_Counts"]))



uniq_hotels["positive_review_average_word_count"] = pos_rev_avg_words

uniq_hotels["negative_review_average_word_count"] = neg_rev_avg_words  



uniq_hotels.drop(['Hotel_Address'],axis=1, inplace=True)

total_reviews = uniq_hotels['Total_Number_of_Reviews']

additional_scoring = uniq_hotels['Additional_Number_of_Scoring']

average_score = uniq_hotels['Average_Score']

pos_rev_avg_word_count = uniq_hotels["positive_review_average_word_count"]

neg_rev_avg_word_count = uniq_hotels["negative_review_average_word_count"]



total_reviews = total_reviews.values.astype(float)

additional_scoring = additional_scoring.values.astype(float)

average_score = average_score.values.astype(float)

pos_rev_avg_word_count = pos_rev_avg_word_count.values.astype(float)

neg_rev_avg_word_count = neg_rev_avg_word_count.values.astype(float)



total_reviews = total_reviews .reshape(-1, 1)

additional_scoring = additional_scoring.reshape(-1,1)

average_score = average_score.reshape(-1,1)

pos_rev_avg_word_count = pos_rev_avg_word_count.reshape(-1,1)

neg_rev_avg_word_count = neg_rev_avg_word_count.reshape(-1,1)
min_max_scaler = MinMaxScaler()



uniq_hotels["total_reviews"] = min_max_scaler.fit_transform(total_reviews)

uniq_hotels["additional_scoring"] = min_max_scaler.fit_transform(additional_scoring)

uniq_hotels["average_score"] = min_max_scaler.fit_transform(average_score)

uniq_hotels["pos_rev_avg_word_count"] = min_max_scaler.fit_transform(pos_rev_avg_word_count)

uniq_hotels["neg_rev_avg_word_count"] = min_max_scaler.fit_transform(neg_rev_avg_word_count)



uniq_hotels.drop(['Additional_Number_of_Scoring','Total_Number_of_Reviews','Average_Score'], axis=1, inplace = True)

uniq_hotels.tail()

# Comparing how Additional and Average scoring compares to the number of reviews

t = uniq_hotels["total_reviews"]

y = uniq_hotels["average_score"]



plt.scatter(y, t, c='purple')

plt.xlabel('Scoring')

plt.ylabel('Number of Reviews')

plt.show()
# Comparing how Additional and Average scoring compares to the number of positive reviews



t = uniq_hotels["pos_rev_avg_word_count"]

y = uniq_hotels["average_score"]



plt.scatter(t, y, c='purple')

plt.xlabel('Positive review word count')

plt.ylabel('Score')

plt.show()
# Comparing how Additional and Average scoring compares to the number of negative reviews



t = uniq_hotels["neg_rev_avg_word_count"]

y = uniq_hotels["average_score"]



plt.scatter(t, y, c='purple')

plt.xlabel('Negative review word count')

plt.ylabel('Score')

plt.show()
top_hotels = dict(hotel_counter.most_common(5))



objects_hotels = list(top_hotels.keys())

performance_hotels = top_hotels.values()



plt.barh(objects_hotels, performance_hotels, alpha=1)

plt.xlabel("Number of reviews")

plt.show()
top_nationalities = dict(nationality_counter.most_common(5))



objects_nationalities = list(top_nationalities.keys())

performance_nationalities = top_nationalities.values()



plt.barh(objects_nationalities, performance_nationalities, alpha=1)

plt.xlabel("Number of reviews")

plt.show()


from sklearn.cluster import KMeans



# f1 = Hotels_['neg'].values

# f2 = Hotels_['pos'].values

# f3 = Hotels_['Total_revs'].values 

# f4 = Hotels_['Add_sc'].values

# f5 = Hotels_['Av_sc'].values



array = uniq_hotels.drop(["Hotel_Name","additional_scoring", "pos_rev_avg_word_count","neg_rev_avg_word_count","hotel_loc"],axis=1)

X = array.to_numpy()



kmeans = KMeans(n_clusters=3).fit(X)



labels = kmeans.predict(X)



C = kmeans.cluster_centers_



print(C)

print(array.columns)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

ax.scatter(C[:, 0], C[:, 1], C[:, 2],)

import folium



Europe_coordinates = (54.5260, 15.2551)



Uniq_hotels=Uniq_hotels.dropna(subset=['lng'])

Uniq_hotels=Uniq_hotels.dropna(subset=['lat'])





lat = list(Uniq_hotels["lat"])

long = list(Uniq_hotels["lng"])

hotel_name = list(Uniq_hotels["Hotel_Name"])

average_score = list(Uniq_hotels["Average_Score"])

    

def color(score): 

    for i in average_score:

        if score >= 9:

            col = "green"

        elif score < 9 and score > 7:

            col = "orange"

        elif score < 7 and score > 4.8:

            col = "red"

        else:

            col = "black"

    return col





hotel_map = folium.Map(location=Europe_coordinates, zoom_start=4)



for lt, ln, name, score in zip(lat, long, hotel_name, average_score):

    folium.Marker(location=[lt, ln], popup=str(name), icon= folium.Icon(color=color(score))).add_to(hotel_map)







hotel_map
from sklearn.model_selection import train_test_split

X = uniq_hotels.drop(["Hotel_Name", "hotel_loc", 'positive_review_average_word_count', 'negative_review_average_word_count'], axis=1)

y = uniq_hotels["hotel_loc"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Logistic Regression



logreg = LogisticRegression()

param_grid = {'C':np.arange(0.01,100,10)}

grid = GridSearchCV(logreg,param_grid)

grid.fit(X_train,y_train)



logR = grid.best_estimator_

logR.fit(X_train,y_train)



y_predicts = logR.predict(X_test)



print(classification_report(y_test, y_predicts))
df = pd.read_csv("Hotel_Reviews.csv")



reviewer_score = df["Reviewer_Score"]

negative_review = df["Negative_Review"]

positive_review = df["Positive_Review"]



reviewer_score_label = pd.qcut(reviewer_score, 2, labels = False)

negative_review_polarity = []

negative_review_subjectivity = []

positive_review_polarity = []

positive_review_subjectivity = []



for i in range(len(negative_review)):

    term_1 = TextBlob(negative_review[i]).sentiment

    term_2 = TextBlob(positive_review[i]).sentiment

    

    negative_review_polarity.append(term_1[0])

    negative_review_subjectivity.append(term_1[1])

    positive_review_polarity.append(term_2[0])

    positive_review_subjectivity.append(term_2[1])

    

 

X = df[['Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given']]



X['negative_review_polarity'] = negative_review_polarity

X['negative_review_subjectivity'] = negative_review_subjectivity

X['positive_review_polarity'] = positive_review_polarity

X['positive_review_subjectivity'] = positive_review_subjectivity 



y = pd.qcut(reviewer_score, 2, labels = False)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



tree = RandomForestClassifier(random_state=0)

tree.fit(X_train, y_train)



y_pred = tree.predict(X_test)

print(classification_report(y_pred, y_test))



feature_imp = pd.Series(tree.feature_importances_,index=X.columns).sort_values(ascending=False)

print(feature_imp)



log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(classification_report(y_pred, y_test))
