import pandas as pd

airbnb = pd.read_csv('../input/BristolAirbnbListings.csv')

airbnb.shape

airbnb.iloc[0:3,0:13]
airbnb.iloc[0:3,13:22]
airbnb.iloc[0:3,22:28]
airbnb = airbnb.drop(['id', 'name', 'host_name', 'host_id', 'postcode', 'minimum_nights', 'number_of_reviews'

                      , 'last_review', 'reviews_per_month', 'availability_365'], axis =1)
airbnb.iloc[0:3,0:12]
airbnb.iloc[0:3,12:18]
airbnb_region = airbnb[['neighbourhood','latitude','longitude']]

airbnb_region.head()
airbnb_region.isnull().sum()
airbnb_region.dtypes
print(airbnb_region['neighbourhood'].unique())

print(len(airbnb_region['neighbourhood'].unique()))
print(airbnb_region['latitude'].max())

print(airbnb_region['latitude'].min())

print(airbnb_region['longitude'].max())

print(airbnb_region['longitude'].min())
ii = 0

jj = 0

for i in airbnb["latitude"]:

    if i > 52 or i < 51:

        print (i)

        ii = +1



for j in airbnb["longitude"]:

    j = j*-1

    if j > 3 or j < 2: 

        print (j)

        jj = +1



if (ii + jj) == 0:

    print('All latitude values between 53 and 52,', 'and', 'all longitude values between -3 and -2')

    
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "http://www.bristolnhwnetwork.org.uk/uploads/1/2/6/4/12643669/6710274_orig.png")
import numpy as np

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.colors as colors

colors_list = list(colors._colors_full_map.values())

plt.rcParams["figure.figsize"] = (15,15)

airbnb_neighbourhood = airbnb['neighbourhood'].tolist()



labels_values = []

for k in range(1,35):

    labels_values.append(k)



neighbourhood_label = []

labels = airbnb['neighbourhood'].unique()

labels = airbnb['neighbourhood'].unique().tolist()

airbnb_neighbourhood = airbnb['neighbourhood'].tolist()



airbnb_labels = []

region=[]



for i in range(len(airbnb_neighbourhood)):

    region = airbnb_neighbourhood[i]

    for j in range(len(labels)):

        if labels[j] == region:

            airbnb_labels.append(labels_values[j])



xx = airbnb['latitude']

yy = airbnb['longitude']  



scatter_x = np.array(xx)

scatter_y = np.array(yy)

group = np.array(airbnb_labels)



color_dict = dict(zip(labels_values, colors_list[0:34])) 



label_list= airbnb['neighbourhood'].unique().tolist()



for g in range(len(airbnb['latitude'])):

    q = airbnb_labels[g]

    plt.scatter(scatter_y[g], scatter_x[g], c = color_dict[q], facecolor = 'grey', label = airbnb_neighbourhood[g] 

                if airbnb_neighbourhood[g] in label_list else "" ) 



    if airbnb_neighbourhood[g] in label_list:

        label_list.remove(airbnb_neighbourhood[g])

    

plt.legend()



plt.xlabel('Longitude', fontsize=25)

plt.ylabel('Latitude', fontsize=25)

plt.title('Estimated Visulized Map over Bristol', fontsize=25)

plt.show()
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

labels_values = []

for k in range(1,35):

    labels_values.append(k)





neighbourhood_label = []

labels = airbnb['neighbourhood'].unique()

labels = airbnb['neighbourhood'].unique().tolist()

airbnb_neighbourhood = airbnb['neighbourhood'].tolist()



airbnb_labels = []

region=[]



for i in range(len(airbnb_neighbourhood)):

    region = airbnb_neighbourhood[i]

    for j in range(len(labels)):

        if labels[j] == region:

            airbnb_labels.append(labels_values[j])



airbnb_kmeans = airbnb_region

airbnb_kmeans['labels'] = airbnb_labels

airbnb_kmeans = airbnb_kmeans.drop(['neighbourhood'], axis = 1)

airbnb_kmeans.head(10)
X = airbnb_kmeans[['latitude','longitude']]

y = airbnb_kmeans['labels']

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=42)

clf = KMeans(n_clusters = 34, random_state = 42, n_init=10)

clf.fit(X_train,y_train)
import matplotlib.colors as colors

longitude_latitude = X_train[['latitude', 'longitude']]

l_l_array = longitude_latitude.values

colors_list = list(colors._colors_full_map.values())

clf = KMeans(n_clusters=34, random_state = 42)

clf.fit(X_train, y_train)



centroids = clf.cluster_centers_

labels = clf.labels_



colors = colors_list[0:34]



plt.figure(figsize=(15,15), dpi=80)

for i in range (len(l_l_array-1)):

    plt.scatter(l_l_array[i][1], l_l_array[i][0], c = colors[labels[i]] )

    

plt.xlabel('Longitude', fontsize=25)

plt.ylabel('Latitude', fontsize=25)

plt.title('Estimated K-means Visulized Map over Bristol', fontsize=25)    

    

plt.show()
print("Training set:", clf.score(X_train,y_train))

print("Test set:    ", clf.score(X_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

knn_train = []

knn_test = []

for i in range(1,20):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    knn_train.append(knn.score(X_train,y_train))

    knn_test.append(knn.score(X_test,y_test))



knn_train_df = pd.DataFrame(knn_train)

knn_test_df = pd.DataFrame(knn_test)

plt.figure(figsize=(8,8))

plt.plot(knn_train_df)

plt.plot(knn_test_df)

plt.show()
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)
from sklearn import metrics

y_expect = y_test

y_pred = knn.predict(X_test)

print(metrics.classification_report(y_expect, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

tree_clf = DecisionTreeClassifier(random_state=42)

tree_clf.fit(X_train, y_train)



y_pred = tree_clf.predict(X_test)



print('Decision Tree accuracy:', accuracy_score(y_test, y_pred))
from sklearn.ensemble import BaggingClassifier



bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), random_state=42)

bag_clf.fit(X_train, y_train)



y_pred = bag_clf.predict(X_test)



print('Decision Tree with bootstrap aggregating accuracy:', accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



boost = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(boost, X_train, y_train, cv=5)

print(scores.mean())



gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42).fit(X_train, y_train)

print(gradient_boosting.score(X_test, y_test))

from sklearn.ensemble import RandomForestClassifier



rnd_clf = RandomForestClassifier(random_state = 42, n_estimators= 10)

rnd_clf.fit(X_train, y_train)



y_pred_rf = rnd_clf.predict(X_test)



print("Random forest accuracy", accuracy_score(y_test, y_pred_rf))
knn.score(X_test,y_test) == accuracy_score(y_test, y_pred_rf)
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression





log_clf = LogisticRegression(random_state=42)

rnd_clf = RandomForestClassifier(random_state=42,)

knn_clf=KNeighborsClassifier(n_neighbors=1)



maximum = 0

weights_list= []

for i in range(0,3):

    for j in range(0,3):

        for k in range(0,3):

            if (i + j + k) == 0: i = 1

            voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('knn', knn_clf)],

                              voting='soft', weights = [i,j,k])

            voting_clf.fit(X_train, y_train)

            print(i,j,k)

            for clf in (log_clf, rnd_clf, knn_clf, voting_clf):

                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

                

                if accuracy_score(y_test, y_pred) > accuracy_score(y_test, y_pred_rf):

                    print("  High accuracy with the weight", i, j,k)



                    if accuracy_score(y_test, y_pred) > maximum:

                        maximum = accuracy_score(y_test, y_pred)

                        weights_list = [i,j,k]

print("----------------------------------")                        

print("The top weights are:", weights_list)

print("with a accuracy of:", maximum)
import time

start_time = time.time()



from sklearn.model_selection import GridSearchCV



voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('knn', knn_clf)],

                              voting='soft', weights = [1,2,1])



eclf = voting_clf

params = {'lr__C': [1.0, 100.0], 'rf__max_leaf_nodes': [20, 500], 'rf__n_estimators': [1,500], 

          'rf__n_jobs':[-5,10], 'rf__max_features': ['auto', 'sqrt', 'log2'], 'knn__n_neighbors':[1,50]}



grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

grid = grid.fit(X_train, y_train)

print(grid.best_params_)

print("--- %s seconds ---" % (time.time() - start_time))
log_clf = LogisticRegression(random_state=42, C =1.0)

rnd_clf = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=500, n_jobs=-5, 

                                 max_features = 'auto')



knn_clf=KNeighborsClassifier(n_neighbors=1)



voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('knn', knn_clf)],

                              voting='soft', weights = [1,2,1])



for clf in (log_clf, rnd_clf, knn_clf, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
log_clf = LogisticRegression(random_state=42, C =1.0)

rnd_clf = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=500, n_jobs=-5, 

                                 max_features = 'auto')

knn_clf=KNeighborsClassifier(n_neighbors=1)



maximum = 0

weights_list= []

for i in range(0,3):

    for j in range(0,3):

        for k in range(0,3):

            if (i + j + k) == 0: j = 1

            voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('knn', knn_clf)],

                              voting='soft', weights = [i,j,k])

            voting_clf.fit(X_train, y_train)



            for clf in (log_clf, rnd_clf, knn_clf, voting_clf):

                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

       

                if accuracy_score(y_test, y_pred) > accuracy_score(y_test, y_pred_rf):

 

                    if accuracy_score(y_test, y_pred) > maximum:

                        maximum = accuracy_score(y_test, y_pred)

                        weights_list = [i,j,k]

                       

print("The top weights are:", weights_list)

print("with an accuracy of:", maximum)
log_clf = LogisticRegression(random_state=42, C =1.0)

rnd_clf = RandomForestClassifier(random_state=42, n_estimators=500, max_leaf_nodes=500, n_jobs=-5, 

                                 max_features = 'auto')



knn_clf=KNeighborsClassifier(n_neighbors=1)



voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('knn', knn_clf)],

                              voting='soft', weights = [0,1,0])







for clf in (log_clf, rnd_clf, knn_clf, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
airbnb2 = airbnb[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count']]

airbnb2.head(25)
airbnb2.info()
airbnb2.isnull().sum(axis = 0)
airbnb3 = airbnb2.dropna()

airbnb3.info()
airbnb3['review_scores_rating'].unique()
print(airbnb3['review_scores_accuracy'].unique())

print(airbnb3['review_scores_cleanliness'].unique())

print(airbnb3['review_scores_checkin'].unique())

print(airbnb3['review_scores_communication'].unique())

print(airbnb3['review_scores_location'].unique())

print(airbnb3['review_scores_value'].unique())

print(airbnb3['calculated_host_listings_count'].unique())
import re

p = re.compile('\D', re.IGNORECASE)

airbnb3 = airbnb3.replace(p, np.nan)

airbnb3 = airbnb3.dropna()
print(airbnb3['review_scores_rating'].unique())

print(airbnb3['review_scores_accuracy'].unique())

print(airbnb3['review_scores_cleanliness'].unique())

print(airbnb3['review_scores_checkin'].unique())

print(airbnb3['review_scores_communication'].unique())

print(airbnb3['review_scores_location'].unique())

print(airbnb3['review_scores_value'].unique())

print(airbnb3['calculated_host_listings_count'].unique())

airbnb3.describe()
airbnb3 = airbnb3.astype({"review_scores_rating": int, "review_scores_accuracy": int, 

                          "review_scores_cleanliness": int, "review_scores_checkin": int})

airbnb3.dtypes
airbnb3['host_label'] = airbnb3['calculated_host_listings_count'].apply(lambda x: 1 if x > 1 else 0)

print('Amatures:', airbnb3['calculated_host_listings_count'][airbnb.calculated_host_listings_count == 1 ].count(), 'listings')

print('Professionals:', airbnb3['calculated_host_listings_count'][airbnb.calculated_host_listings_count != 1 ].count(), 'listings')

airbnb3.iloc[:,4:].head(10)
airbnb4 = airbnb3

airbnb5 = airbnb4.drop(['calculated_host_listings_count'], axis = 1)

airbnb5 = airbnb5.groupby('host_label').mean()

airbnb5 = airbnb5.T

airbnb5
ax = airbnb5.plot.bar()
pro_review = []

am_review = []

for i in range(len(airbnb4)):

    if airbnb4.iloc[i,8] == 1:

        pro_review.append(airbnb4.iloc[i,0])

    else:

        am_review.append(airbnb4.iloc[i,0])

plt.hist(am_review)

plt.xlabel('Score', fontsize=25)

plt.ylabel('Quanitity', fontsize=25)

plt.show()
plt.hist(pro_review)

plt.xlabel('Score', fontsize=25)

plt.ylabel('Quanitity', fontsize=25)

plt.show()
pro_over_90 = [i for i in pro_review if i >= 90]

print("Professials with review score rating of 90 and above:", len(pro_over_90)/len(pro_review))



am_over_90 = [i for i in am_review if i >= 90]

print("Ammatures with review score rating of 90 and above:",len(pro_over_90)/len(am_review))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split



lr = LogisticRegression() 

rfc = RandomForestClassifier( n_estimators = 100)

svc = LinearSVC(C=1.0)



train, test = train_test_split(airbnb4, test_size = 0.2)

train_feat = train.iloc[:,0:7]

train_target = train['host_label']

test_feat = test.iloc[:,0:7]

test_target = test['host_label']
lr.fit(train_feat, train_target)

print(lr.score(train_feat, train_target))

print(lr.score(test_feat, test_target))
from sklearn.metrics import confusion_matrix

print('- Training -')

print(confusion_matrix(lr.predict(train_feat), train_target))

print('-   Test   -')

print(confusion_matrix(lr.predict(test_feat), test_target))
from sklearn.multiclass import OneVsOneClassifier



from sklearn.multiclass import OneVsOneClassifier

from sklearn.svm import LinearSVC



xx = train_feat.values

yy = train_target.values



classifier = OneVsRestClassifier(LinearSVC(random_state=42))

classifier.fit(xx, yy)

print('Accuracy', classifier.score(xx, yy))
from sklearn import datasets

iris = datasets.load_iris()

xxx, yyy = iris.data, iris.target



classifier = OneVsRestClassifier(LinearSVC(random_state=42))

classifier.fit(xxx, yyy)

print('Accuracy Iris Dataset', classifier.score(xxx, yyy))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(random_state=42, max_leaf_nodes= 500, n_jobs=-1, n_estimators=500)



rfc.fit(train_feat, train_target)

rfc.score(train_feat, train_target)
rfc.score(test_feat, test_target)
import tensorflow as tf

warnings.filterwarnings('ignore')



columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 

                'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 

                'review_scores_value']



feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]



def input_fn(df,labels):

    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}

    label = tf.constant(labels.values, shape = [labels.size,1])

    return feature_cols,label



classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20],n_classes = 2)



classifier.fit(input_fn=lambda: input_fn(train_feat, train_target),steps = 10000)
ev = classifier.evaluate(input_fn=lambda: input_fn(test_feat,test_target),steps=1)

print(ev)
print('Accuracy',ev['accuracy'])
def input_predict(df):

    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}

    return feature_cols



pred = classifier.predict_classes(input_fn=lambda: input_predict(test_feat))
print(list(pred))
print(test_target.values)

p = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 

     1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 

     0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 

     1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 

     0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

     1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 

     1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 

     1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 

     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 

     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 

     0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 

     1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 

     1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 

     0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 

     0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 

     0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 

     1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 

     0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
tt = test_target.values



correct = 0

false = 0

counter = 0

for i in p:

    if i == tt[counter]:

        correct = correct + 1

    else:

        false = false + 1    

    counter = counter +1

    

print(correct/(false+correct))

                               

round(correct/(false+correct), 5) == round(float(ev['accuracy']),5)                               