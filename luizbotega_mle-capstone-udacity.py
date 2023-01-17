import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

from sklearn import preprocessing #normalizing values

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, GridSearchCV #dividing into train and test for cross_validation

from sklearn.multiclass import OneVsRestClassifier #strategy for star multiclass classification

from sklearn.metrics import roc_curve, roc_auc_score, make_scorer, confusion_matrix, accuracy_score, r2_score #scorers

from sklearn.svm import LinearSVC, SVC #ML model

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

import time
df = pd.read_csv('../input/booking_com-travel_sample.csv') #import data

df.head()
df['hotel_facilities'].iloc[0]
#features considered unecessary

to_drop = ['address', 'city','country', 'crawl_date', 'hotel_brand', 'hotel_description',  'locality', 'pageurl',

           'property_id', 'property_name', 'property_type', 'province', 'qts', 'room_type', 'similar_hotel', 'sitename',

           'site_review_count', 'site_stay_review_rating', 'special_tag', 'uniq_id', 'zone']

#remove rows without name (unreliable) or explicit facilities (lack information)

df_reliable = df.dropna(subset = ['property_name', 'hotel_facilities'], thresh = 2, inplace = True)

#drop unecessary features and duplicates 

df_reduced = df.drop(to_drop, axis = 1)

df_reduced.drop_duplicates(inplace = True)

#fill NaN

df_reduced['image_count'].fillna(0, inplace = True)



#create dummies for states

df_dummies = pd.concat([df_reduced, pd.get_dummies(df_reduced['state'])], axis=1)

df_dummies.drop('state', axis = 1, inplace = True) #drop states (attribute was dummied)

df_dummies.reset_index(drop = True, inplace = True)

df_dummies.head()
#find hotel_facilities keys

columns_facilities = []

for row in df_dummies.hotel_facilities:

    splitten = row.split(sep = '•')

    columns_facilities.extend([a.split(':', 1)[0]for a in splitten])

columns_facilities = sorted(list(set(columns_facilities)))



#create datafame with such keys

hotel_facilities_preparation = pd.DataFrame(columns = columns_facilities)



#iterates over all df rows and input number of items in each facilities to facilities_dataframe 

for row in df_dummies.hotel_facilities:

    first = row.split(sep = '•')

    features_columns = [row.split(':', 1)[0] for row in first]

    second = [row.split(':', 1)[-1] for row in first]

    third = [row.split(sep = '|') for row in second]

    lenghts = [len(row) for row in third]

    

    to_special_cases = dict(zip(features_columns,third))

    to_df = dict(zip(features_columns,lenghts))

    

    if to_special_cases['Pets'] == ['Pets are not allowed.']:

        to_df['Pets'] = 0



    if to_special_cases['Internet'] == ['No internet access available.']:

        to_df['Internet'] = 0



    try:

        if to_special_cases['Parking'] == ['No parking available.']:

            to_df['Parking'] = 0

    except:

        None

    

    hotel_facilities_preparation = hotel_facilities_preparation.append(to_df, ignore_index=True)

    

    

hotel_facilities_preparation.fillna(0, inplace = True)



#concat facilities dataframe to df_dummies

df_final = pd.concat([df_dummies, hotel_facilities_preparation], axis = 1)

df_final.drop('hotel_facilities', axis = 1, inplace = True) #drop hotel_facilities

df_final.head()
hotel_facilities_preparation.columns.values
#create df hotel_star_rating and remove rows with NaN

df_star_rating = df_final.drop('site_review_rating', axis = 1)

df_star_rating.dropna(subset = ['hotel_star_rating'], inplace = True)

df_star_rating.reset_index(drop = True, inplace = True)



#create df site_review_rating and remove rows with NaN

df_review_rating = df_final.drop('hotel_star_rating', axis = 1)

df_review_rating.dropna(subset = ['site_review_rating'], inplace = True)

df_review_rating.reset_index(drop = True, inplace = True)
df_review_rating.shape
# organizing hotel_star_rating values

df_star_rating['hotel_star_rating'] = df_star_rating['hotel_star_rating'].map({'1-star hotel': 1,

                                                                               '2-star hotel': 2,

                                                                               '3-star hotel': 3,

                                                                               '4-star hotel': 4,

                                                                               '5-star hotel': 5,

                                                                               '1 stars': 1,

                                                                               '2 stars': 2,

                                                                               '3 stars': 3,

                                                                               '4 stars': 4,

                                                                               '5 stars': 5})

# scale all columns to go between 0 and 1

df_star_rating.iloc[:,1:] = preprocessing.MinMaxScaler().fit_transform(df_star_rating.iloc[:,1:])

df_review_rating.iloc[:,:] = preprocessing.MinMaxScaler().fit_transform(df_review_rating)



#reorder columns of df_review_rating so site_review_rating comes first

column_names_list = df_review_rating.columns.tolist()

column_names_list.remove('site_review_rating')

column_names_list.insert(0, 'site_review_rating')

df_review_rating = df_review_rating[column_names_list]
label=np.arange(0,10,1)

values=df_review_rating.groupby(pd.cut(df_review_rating['site_review_rating']*10, np.arange(0,11,1))).count().iloc[:,0].values

plt.bar(label,values)

plt.xticks(np.arange(0, 10, step=1), rotation=90)

plt.title('Number of hotels for each customer score')

plt.ylim([0, 590])

for i, v in enumerate(values):

    plt.text(i-0.4, v+10, str(round(v,3)), color='black', rotation = 0)
df_star_rating.groupby('hotel_star_rating').count().iloc[:,1].values



label=['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

values=df_star_rating.groupby('hotel_star_rating').count().iloc[:,1].values

plt.bar(label,values)

plt.xticks(rotation=90)

plt.title('Number of hotels for each star rating')

for i, v in enumerate(values):

    plt.text(i-0.2, v-50, str(round(v,3)), color='white', rotation = 0)
#df_star_rating heatmap with states



df_star_rating_heatmap = df_star_rating.copy()



fig, ax = plt.subplots(figsize=(20,20)) #figsize in inches

plt.rcParams.update({'font.size': 12}) #font size

plt.yticks(va="center")

sns.heatmap(df_star_rating_heatmap.iloc[:,:33].corr().iloc[0:1,:], square = True, ax=ax, annot = True, fmt = '.2f',

            vmin = -1, vmax = 1,

            cmap = sns.diverging_palette(220, 20, n = 10), cbar_kws = dict(use_gridspec=False,location="top", shrink = 0.5))
#df_star_rating heatmap with hotel_facilities



fig, ax = plt.subplots(figsize=(20,20)) #figsize in inches

plt.rcParams.update({'font.size': 12}) #font size

plt.yticks(va="center")

sns.heatmap(df_star_rating_heatmap.drop(df_star_rating_heatmap.columns[5:33], axis = 1).corr().iloc[0:1,:],

            square = True, ax=ax, annot = True, fmt = '.2f', vmin = -1, vmax = 1,

            cmap = sns.diverging_palette(220, 20, n = 10), cbar_kws = dict(use_gridspec=False,location="top", shrink = 0.5))
#df_review_rating heatmap with states



fig, ax = plt.subplots(figsize=(20,20)) #figsize in inches

plt.rcParams.update({'font.size': 12}) #font size

plt.yticks(va="center")

sns.heatmap(df_review_rating.iloc[:,:33].corr().iloc[0:1,:], square = True, ax=ax, annot = True, fmt = '.2f', vmin = -1, vmax = 1,

            cmap = sns.diverging_palette(220, 20, n = 10), cbar_kws = dict(use_gridspec=False,location="top", shrink = 0.5))
#df_review_rating heatmap with hotel_facilities



fig, ax = plt.subplots(figsize=(20,20)) #figsize in inches

plt.rcParams.update({'font.size': 12}) #font size

plt.yticks(va="center")

sns.heatmap(df_review_rating.drop(df_review_rating.columns[5:33], axis = 1).corr().iloc[0:1,:], square = True, ax=ax,

            annot = True, fmt = '.2f', vmin = -1, vmax = 1,

            cmap = sns.diverging_palette(220, 20, n = 10), cbar_kws = dict(use_gridspec=False,location="top", shrink = 0.5))
# creating training and testing sets



X_star = df_star_rating.iloc[:,1:]

y_star = df_star_rating['hotel_star_rating']



#dividing into train and test - sss to star rating

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)

sss.get_n_splits(X_star, y_star)



for train_index, test_index in sss.split(X_star, y_star):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_star_train, X_star_test = X_star.iloc[train_index], X_star.iloc[test_index]

    y_star_train, y_star_test = y_star[train_index], y_star[test_index]
#selecting random values for each hotel using weights 

from numpy.random import choice



acc = 0

iterations = 100

weight = np.array(df_star_rating.groupby(['hotel_star_rating'])['hotel_star_rating'].count()/df_star_rating['hotel_star_rating'].count())



for i in range(iterations):

 

    #create random weighted answers

    random_results = []

    for i in range(len(y_star_test)):

        random_results.append(choice(sorted(df_star_rating['hotel_star_rating'].unique()), 1, p = weight)[0])

    

    #check correct values

    acc += sum(list(random_results == y_star_test))/len(y_star_test)

    

acc/iterations
acc = 0

iterations = 100



for i in range(iterations):

    

    model = DecisionTreeClassifier(max_depth = 1, max_features = 1)

    model.fit(X_star_train, y_star_train)

    predictions = model.predict(X_star_test)

    acc += accuracy_score(y_star_test, predictions)



acc/iterations
#gridsearch Decision Trees - could do better -> ensamble methods may improve



estimator = DecisionTreeClassifier()

parameters = {'max_depth': [30, 50, 70, 100], 'min_samples_split': [2, 3, 4, 5]}

scorer = make_scorer(accuracy_score) #this metric will be used for GridSearch to find best models/parameters. Promissing results will be testes with other metrics



clf = GridSearchCV(estimator = estimator, scoring = scorer, param_grid  = parameters, cv = 10, return_train_score=True)

clf.fit(X = X_star_train,y = y_star_train)

clf.cv_results_ 
clf.best_score_
#gridsearch Random Forest - best



estimator = RandomForestClassifier()

parameters = {'max_depth': [10, 30, 50, 70], 'n_estimators': [20, 50, 100]}

scorer = make_scorer(accuracy_score) #this metric will be used for GridSearch to find best models/parameters. Promissing results will be testes with other metrics



clf = GridSearchCV(estimator = estimator, scoring = scorer, param_grid  = parameters, cv = 10, return_train_score=True)

clf.fit(X = X_star_train,y = y_star_train)

clf.cv_results_ 
clf.best_params_
#gridsearch AdaBoost - slower with same results as RandomForest



estimator = AdaBoostClassifier()

parameters = {'base_estimator': [DecisionTreeClassifier(max_depth=10)], 'learning_rate': [0.3, 0.5, 0.7, 1], 'n_estimators': [20, 50, 100]}

scorer = make_scorer(accuracy_score) #this metric will be used for GridSearch to find best models/parameters. Promissing results will be testes with other metrics



clf = GridSearchCV(estimator = estimator, scoring = scorer, param_grid  = parameters, cv = 10, return_train_score=True)

clf.fit(X = X_star_train,y = y_star_train)

clf.cv_results_ 
clf.best_params_ 
#consfusion matrix

iterations = 10

cm_train = [0] #creates confusion matrix instance for train

cm_test = [0] #creates confusion matrix instance for test



train_acc = 0

test_acc = 0



X_star = df_star_rating.iloc[:,1:]

y_star = df_star_rating['hotel_star_rating']



start = time.time() 

for n in range(iterations):



    #dividing into train and test - sss to star rating

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=n)

    sss.get_n_splits(X_star, y_star)



    for train_index, test_index in sss.split(X_star, y_star):

        #print("TRAIN:", train_index, "TEST:", test_index)

        X_star_train, X_star_test = X_star.iloc[train_index], X_star.iloc[test_index]

        y_star_train, y_star_test = y_star[train_index], y_star[test_index]

    

    model = RandomForestClassifier(max_depth = 30, n_estimators = 100)

    model.fit(X = X_star_train, y = y_star_train)

    

    predict_train = model.predict(X_star_train)    

    cm_train += confusion_matrix(y_star_train, predict_train)



    predict_test = model.predict(X_star_test)

    cm_test += confusion_matrix(y_star_test, predict_test)

    

    train_acc += accuracy_score(y_star_train, predict_train)

    test_acc += accuracy_score(y_star_test, predict_test)



cm_train = cm_train/iterations #takes the mean

total_train = [i.sum() for i in cm_train]

acc_train = [cm_train[i][i]/total_train[i] for i in range(5)]

score_train = pd.DataFrame(0, columns = np.append(np.sort(y_star.unique()),['Acc']), index = np.sort(y_star.unique()))

score_train.iloc[:,:5] = cm_train

score_train.iloc[:,5:] = acc_train



cm_test = cm_test/iterations # takes the mean

total_test = [i.sum() for i in cm_test]

acc_test = [cm_test[i][i]/total_test[i] for i in range(5)]

score_test = pd.DataFrame(0, columns = np.append(np.sort(y_star.unique()),['Acc']), index = np.sort(y_star.unique()))

score_test.iloc[:,:5] = cm_test

score_test.iloc[:,5:] = acc_test



train_acc = train_acc/iterations

test_acc = test_acc/iterations

(time.time() - start)/iterations
score_test
np.diag(score_test).sum()/score_test.iloc[:,:-1].sum().sum()
n_estimators = 100

acc_train = []

acc_test = []



#dividing into train and test - sss to star rating

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)

sss.get_n_splits(X_star, y_star)



for train_index, test_index in sss.split(X_star, y_star):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_star_train, X_star_test = X_star.iloc[train_index], X_star.iloc[test_index]

    y_star_train, y_star_test = y_star[train_index], y_star[test_index]

    

for n in range(1,n_estimators+1):

    model = RandomForestClassifier(n_estimators = n)

    model.fit(X = X_star_train, y = y_star_train)

    

    predict_train = model.predict(X_star_train)    

    predict_test = model.predict(X_star_test)

    

    acc_train.append(accuracy_score(y_star_train, predict_train))

    acc_test.append(accuracy_score(y_star_test, predict_test))



plt.plot(list(range(1,n_estimators+1)),acc_train, label = 'Train score')

plt.plot(list(range(1,n_estimators+1)),acc_test, label = 'Test score')

plt.grid(True)

plt.xlabel('Number of estimators')

plt.ylabel('Accuracy score')

plt.xlim([0, n_estimators])

plt.legend()
n_estimators = 40

max_depth = [10, 30, 50, 100]



#dividing into train and test - sss to star rating

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)

sss.get_n_splits(X_star, y_star)



for train_index, test_index in sss.split(X_star, y_star):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_star_train, X_star_test = X_star.iloc[train_index], X_star.iloc[test_index]

    y_star_train, y_star_test = y_star[train_index], y_star[test_index]



for md in max_depth:

    acc_train = []

    acc_test = []

    for n in range(1,n_estimators+1):



        model = RandomForestClassifier(n_estimators = n, max_depth = md)

        model.fit(X = X_star_train, y = y_star_train)



        predict_train = model.predict(X_star_train)    

        predict_test = model.predict(X_star_test)



        acc_train.append(accuracy_score(y_star_train, predict_train))

        acc_test.append(accuracy_score(y_star_test, predict_test))



    plt.plot(list(range(1,n_estimators+1)),acc_test, label = 'Test score (md = '+str(md)+')')



plt.plot(list(range(1,n_estimators+1)),acc_train, label = 'Train score')

plt.grid(True)

plt.xlabel('Number of estimators')

plt.ylabel('Accuracy score')

plt.xlim([0, n_estimators])

plt.legend(loc='lower right')
#consfusion matrix

iterations = 100

cm_train = [0] #creates confusion matrix instance for train

cm_test = [0] #creates confusion matrix instance for test

FI = [0]*63



X_star = df_star_rating.iloc[:,1:]

y_star = df_star_rating['hotel_star_rating']



start = time.time()

for n in range(iterations):



    #dividing into train and test - sss to star rating

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=n)

    sss.get_n_splits(X_star, y_star)



    for train_index, test_index in sss.split(X_star, y_star):

        #print("TRAIN:", train_index, "TEST:", test_index)

        X_star_train, X_star_test = X_star.iloc[train_index], X_star.iloc[test_index]

        y_star_train, y_star_test = y_star[train_index], y_star[test_index]

    

    model = RandomForestClassifier(max_depth = 30, n_estimators = 30)

    model.fit(X = X_star_train, y = y_star_train)

    

    predict_train = model.predict(X_star_train)    

    cm_train += confusion_matrix(y_star_train, predict_train)



    predict_test = model.predict(X_star_test)

    cm_test += confusion_matrix(y_star_test, predict_test)

    FI += model.feature_importances_

    

cm_train = cm_train/iterations #takes the mean

total_train = [i.sum() for i in cm_train]

acc_train = [cm_train[i][i]/total_train[i] for i in range(5)]

score_train = pd.DataFrame(0, columns = np.append(np.sort(y_star.unique()),['Acc']), index = np.sort(y_star.unique()))

score_train.iloc[:,:5] = cm_train

score_train.iloc[:,5:] = acc_train



cm_test = cm_test/iterations # takes the mean

total_test = [i.sum() for i in cm_test]

acc_test = [cm_test[i][i]/total_test[i] for i in range(5)]

score_test = pd.DataFrame(0, columns = np.append(np.sort(y_star.unique()),['Acc']), index = np.sort(y_star.unique()))

score_test.iloc[:,:5] = cm_test

score_test.iloc[:,5:] = acc_test



FI1 = FI/iterations



(time.time() - start)/iterations
np.array(sorted(zip(FI1, X_star_test.columns),reverse=True)[:15])[:,0]
fig, ax = plt.subplots(figsize=(15,5))



#using 15 most important features

label1=np.array(sorted(zip(FI1, X_star_test.columns),reverse=True)[:15])[:,1]

values1=np.array(sorted(zip(FI1, X_star_test.columns),reverse=True)[:15])[:,0]

values1 = [round(float(v),3) for v in values1]

plt.bar(label1,values1)

plt.xticks(rotation=90)

plt.ylim([0,0.12])



for i, v in enumerate(values1):

    plt.text(i-0.15, v+0.012, str(round(v,3)), color='black', rotation = 90)
roc = {label: [0] for label in y_star.unique()}

auc = {label: [0] for label in y_star.unique()}

estimator = DecisionTreeClassifier()

for label in y_star.unique():

    single_class_train = []

    single_class_test = []

    for item in y_star_train:

        if item == label:

            single_class_train.append(1)

        else:

            single_class_train.append(0)

    for item in y_star_test:

        if item == label:

            single_class_test.append(1)

        else:

            single_class_test.append(0)

    estimator.fit(X = X_star_train, y = single_class_train)

    predictions_proba = estimator.predict_proba(X_star_test)

    roc[label] += roc_curve(single_class_test, predictions_proba[:,1])

    auc[label] += roc_auc_score(single_class_test, predictions_proba[:,1])
plt.figure()

color = ['red', 'darkorange', 'yellow', 'green', 'purple']

i=0

for lbl in np.sort(y_star.unique()):

    plt.plot(roc[lbl][1], roc[lbl][2], color=color[i], lw=1, label='ROC curve (area = %0.2f)' % auc[label])

    i+=1



plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic ')

plt.legend(loc="lower right")

plt.show()
# creating training and testing sets



X_rev = df_review_rating.iloc[:,1:]

y_rev = df_review_rating['site_review_rating']



#dividing into train and test - train test split

X_rev_train, X_rev_test, y_rev_train, y_rev_test = train_test_split(X_rev, y_rev, test_size=0.25, random_state=0)
X_rev_train.shape
aver_predict = [np.mean(y_rev_train)]*len(y_rev_test)

r2_score(y_rev_test, aver_predict)
model = DecisionTreeRegressor(max_depth = 1, max_features = 1)

model.fit(X_rev_train, y_rev_train)

predict_train = model.predict(X_rev_train)

score_train = r2_score(y_rev_train, predict_train)

predict_test = model.predict(X_rev_test)

score_test = r2_score(y_rev_test, predict_test)

score_train, score_test
from sklearn.linear_model import Lasso, ElasticNet, Ridge

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor



model = RandomForestRegressor(n_estimators = 100)

model.fit(X_rev_train, y_rev_train)

predict_train = model.predict(X_rev_train)

score_train = r2_score(y_rev_train, predict_train)

predict_test = model.predict(X_rev_test)

score_test = r2_score(y_rev_test, predict_test)

score_train, score_test
n_estimators = 100

r2_train = []

r2_test = []



#dividing into train and test

X_rev_train, X_rev_test, y_rev_train, y_rev_test = train_test_split(X_rev, y_rev, test_size=0.25, random_state=0)

    

for n in range(1,n_estimators+1):

    model = BaggingRegressor(n_estimators = n)

    model.fit(X = X_rev_train, y = y_rev_train)

    

    predict_train = model.predict(X_rev_train)    

    predict_test = model.predict(X_rev_test)

    

    r2_train.append(r2_score(y_rev_train, predict_train))

    r2_test.append(r2_score(y_rev_test, predict_test))



plt.plot(list(range(1,n_estimators+1)),r2_train, label = 'Train score')

plt.plot(list(range(1,n_estimators+1)),r2_test, label = 'Test score')

plt.grid(True)

plt.xlabel('Number of estimators')

plt.ylabel('R2 score')

plt.xlim([0, n_estimators])

plt.legend(loc = 'lower right')
r2_test[-1]
n_estimators = 40

max_depth = [10, 30, 50, 100]



#dividing into train and test

X_rev_train, X_rev_test, y_rev_train, y_rev_test = train_test_split(X_rev, y_rev, test_size=0.25, random_state=0)



for md in max_depth:

    r2_train = []

    r2_test = []

    for n in range(1,n_estimators+1):



        model = ExtraTreesRegressor(n_estimators = n, max_depth = md)

        model.fit(X = X_rev_train, y = y_rev_train)



        predict_train = model.predict(X_rev_train)    

        predict_test = model.predict(X_rev_test)



        r2_train.append(r2_score(y_rev_train, predict_train))

        r2_test.append(r2_score(y_rev_test, predict_test))



    plt.plot(list(range(1,n_estimators+1)),r2_test, label = 'Test score (md = '+str(md)+')')



plt.plot(list(range(1,n_estimators+1)),r2_train, label = 'Train score')

plt.grid(True)

plt.xlabel('Number of estimators')

plt.ylabel('R2 score')

plt.xlim([0, n_estimators])

plt.legend(loc='lower right')
#consfusion matrix

iterations = 100

r2_train = 0 #creates instance for train

r2_test = 0 #creates instance for test

FI = [0]*63 



X_rev = df_review_rating.iloc[:,1:]

y_rev = df_review_rating['site_review_rating']



start = time.time()

for n in range(iterations):



    #dividing into train and test - sss to star rating

    X_rev_train, X_rev_test, y_rev_train, y_rev_test = train_test_split(X_rev, y_rev, test_size=0.25, random_state=n)

    

    model = ExtraTreesRegressor(n_estimators = 30, max_depth = 30)

    model.fit(X = X_rev_train, y = y_rev_train)



    predict_train = model.predict(X_rev_train)    

    predict_test = model.predict(X_rev_test)



    r2_train += r2_score(y_rev_train, predict_train)

    r2_test += r2_score(y_rev_test, predict_test)

    FI += model.feature_importances_



r2_train = r2_train/iterations #takes the mean

r2_test = r2_test/iterations #takes the mean



FI2 = FI/iterations



(time.time() - start)/iterations
fig, ax = plt.subplots(figsize=(15,5))



label2=np.array(sorted(zip(FI2, X_rev_test.columns),reverse=True)[:15])[:,1]

values2=np.array(sorted(zip(FI2, X_rev_test.columns),reverse=True)[:15])[:,0]

values2 = [round(float(v),3) for v in values2]

plt.bar(label2,values2)

plt.xticks(rotation=90)

plt.ylim([0,0.08])



for i, v in enumerate(values2):

    plt.text(i-0.15, v+0.008, str(round(v,3)), color='black', rotation = 90)
#comparing feature importances for both analysis



fig, ax = plt.subplots(figsize=(15,8))

plt.subplots_adjust(hspace = 1.2)

plt.figure(1)



plt.subplot(211)

label1=np.array(sorted(zip(FI1, X_star_test.columns),reverse=True)[:15])[:,1]

values1=np.array(sorted(zip(FI1, X_star_test.columns),reverse=True)[:15])[:,0]

values1 = [round(float(v),3) for v in values1]

plt.bar(label1,values1)

plt.xticks(rotation=90)

plt.ylim([0,0.11])

plt.title('Star rating feature importances')



for i, v in enumerate(values1):

    plt.text(i-0.15, v-0.01, str(round(v,3)), color='white', rotation = 90)





plt.subplot(212)

label2=np.array(sorted(zip(FI2, X_rev_test.columns),reverse=True)[:15])[:,1]

values2=np.array(sorted(zip(FI2, X_rev_test.columns),reverse=True)[:15])[:,0]

values2 = [round(float(v),3) for v in values2]

plt.bar(label2,values2)

plt.xticks(rotation=90)

plt.ylim([0,0.07])

plt.title('Customer review feature importances')



for i, v in enumerate(values2):

    plt.text(i-0.15, v-0.008, str(round(v,3)), color='white', rotation = 90)



plt.show()