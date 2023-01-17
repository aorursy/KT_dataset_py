import pandas as pd

import numpy as np

import seaborn as sns #plotting

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.plotly as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

%matplotlib inline

init_notebook_mode(connected=True)
df_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")

df_reviews.head()
df_apps = pd.read_csv("../input/googleplaystore.csv")

df_apps.head()
categories = list(df_apps["Category"].unique())

print("There are {0:.0f} categories! (Excluding/Removing Category 1.9)".format(len(categories)-1))

print(categories)

#Remove Category 1.9

categories.remove('1.9')
a = df_apps.loc[df_apps["Category"] == "1.9"]

print(a.head())

print("This mislabeled app category affects {} app at index {}.".format(len(a),int(a.index.values)))

df_apps = df_apps.drop(int(a.index.values),axis=0)
df_apps['Rating'].isnull().sum()
df_apps = df_apps.drop(df_apps[df_apps['Rating'].isnull()].index, axis=0)
df_apps.info()
df_apps["Rating"].describe()
layout = go.Layout(

    xaxis=dict(title='Ratings'),yaxis=dict(title='Number of Apps'))

data = [go.Histogram(x=df_apps["Rating"])]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic histogram')

#Show top 35 app genres

plt.figure(figsize=(16, 9.5))

genres = df_apps["Genres"].value_counts()[:35]

ax = sns.barplot(x=genres.values, y=genres.index, palette="PuBuGn_d")
sns.set(rc={'figure.figsize':(20,10)}, font_scale=1.5, style='whitegrid')

ax = sns.boxplot(x="Category",y="Rating",data=df_apps)

labels = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')

#Cut away rows which have < 4.0 ratings

highRating = df_apps.copy()

highRating = highRating.loc[highRating["Rating"] >= 4.0]

highRateNum = highRating.groupby('Category')['Rating'].nunique()

highRateNum
df_apps.dtypes

df_apps["Type"] = (df_apps["Type"] == "Paid").astype(int)

corr = df_apps.apply(lambda x: x.factorize()[0]).corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=True)
#Extract App, Installs, & Content Rating from df_apps

popApps = df_apps.copy()

popApps = popApps.drop_duplicates()

#Remove characters preventing values from being floats and integers

popApps["Installs"] = popApps["Installs"].str.replace("+","") 

popApps["Installs"] = popApps["Installs"].str.replace(",","")

popApps["Installs"] = popApps["Installs"].astype("int64")

popApps["Price"] = popApps["Price"].str.replace("$","")

popApps["Price"] = popApps["Price"].astype("float64")

popApps["Size"] = popApps["Size"].str.replace("Varies with device","0")

popApps["Size"] = (popApps["Size"].replace(r'[kM]+$', '', regex=True).astype(float) *\

        popApps["Size"].str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))

popApps["Reviews"] = popApps["Reviews"].astype("int64")



popApps = popApps.sort_values(by="Installs",ascending=False)

popApps.reset_index(inplace=True)

popApps.drop(["index"],axis=1,inplace=True)

popApps.loc[:40,['App','Installs','Content Rating']]
popAppsCopy = popApps.copy()

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'Category'. 

popAppsCopy['Category']= label_encoder.fit_transform(popAppsCopy['Category']) 

popAppsCopy['Content Rating']= label_encoder.fit_transform(popAppsCopy['Content Rating']) 

popAppsCopy['Genres']= label_encoder.fit_transform(popAppsCopy['Genres']) 

popAppsCopy.dtypes
popAppsCopy = popAppsCopy.drop(["App","Last Updated","Current Ver","Android Ver"],axis=1)

print("There are {} total rows.".format(popAppsCopy.shape[0]))

countPop = popAppsCopy[popAppsCopy["Installs"] > 100000].count()

print("{} Apps are Popular!".format(countPop[0]))

print("{} Apps are Unpopular!\n".format((popAppsCopy.shape[0]-countPop)[0]))

print("For an 80-20 training/test split, we need about {} apps for testing\n".format(popAppsCopy.shape[0]*.20))

popAppsCopy["Installs"] = (popAppsCopy["Installs"] > 100000)*1 #Installs Binarized

print("Cut {} apps off Popular df for a total of 3558 Popular training apps.".format(int(4568*.22132)))

print("Cut {} apps off Unpopular df for a total of 3558 Unpopular training apps.\n".format(int(4324*.17738)))



testPop1 = popAppsCopy[popAppsCopy["Installs"] == 1].sample(1010,random_state=0)

popAppsCopy = popAppsCopy.drop(testPop1.index)

print("Values were not dropped from training dataframe.",testPop1.index[0] in popAppsCopy.index)



testPop0 = popAppsCopy[popAppsCopy["Installs"] == 0].sample(766,random_state=0)

popAppsCopy = popAppsCopy.drop(testPop0.index)

print("Values were not dropped from training dataframe.",testPop0.index[0] in popAppsCopy.index)



testDf = testPop1.append(testPop0)

trainDf = popAppsCopy



#Shuffle rows in test & training data set

testDf = testDf.sample(frac=1,random_state=0).reset_index(drop=True)

trainDf = trainDf.sample(frac=1,random_state=0).reset_index(drop=True)



#Form training and test data split

y_train = trainDf.pop("Installs")

X_train = trainDf.copy()

y_test = testDf.pop("Installs")

X_test = testDf.copy()



X_train = X_train.drop(['Reviews', 'Rating'], axis=1) #REMOVE ROW TO INCLUDE REVIEWS & RATINGS IN ML MODEL ~93% accurate

X_test = X_test.drop(['Reviews', 'Rating'], axis=1)   #REMOVE ROW TO INCLUDE REVIEWS & RATINGS IN ML MODEL ~93% accurate
print("{} Apps are used for Training.".format(y_train.count()))

print("{} Apps are used for Testing.".format(y_test.count()))

X_test.head(3)
popularity_classifier = DecisionTreeClassifier(max_leaf_nodes=29, random_state=0)

popularity_classifier.fit(X_train, y_train)
predictions = popularity_classifier.predict(X_test)

print("Predicted: ",predictions[:30])

print("Actual:    ",np.array(y_test[:30]))
accuracy_score(y_true = y_test, y_pred = predictions)
X_testCopy = X_test.copy()

X_testCopy["Popular?"] = y_test

X_testCopy[X_test["Size"] == 3600000].head(10)