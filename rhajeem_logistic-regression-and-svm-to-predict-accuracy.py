import re
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem.porter import PorterStemmer
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVR
#from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.mode.chained_assignment = None 
df = pd.read_csv("../input/winemag-data_first150k.csv", nrows=50000,index_col=0)
df.head()
#drop columns not needed
df = df.drop(['designation','province','region_1','region_2','winery'], axis = 1)

#We will now test for duplictes in the dataset to ensure that we are using unique reviews
df[df.duplicated('description',keep=False)].head()

#We will now remove the dulicates based on the descripton column 
df = df.drop_duplicates('description')
df.head()
#Return a sum count of rows with missing data
df.apply(lambda x: sum(x.isnull()),axis=0) 
#fill in missing price with mean values
df['price'].fillna(df['price'].mean(), inplace=True)

#Return a sum count of rows with missing data
df.apply(lambda x: sum(x.isnull()),axis=0) 
 #Drop rows with missing/invalid data
df.dropna(axis='rows',inplace=True)

#Return a sum count of rows with missing data
df.apply(lambda x: sum(x.isnull()),axis=0) 
df['description'][0]
#Get rid of the less useful parts like symbols and digits
description =  re.sub('[^a-zA-Z]',' ',df['description'][0])
description
#All the words should be in same case so lowercase the words and remove trailing whitespaces
description = description.lower().strip()
description
#convert string to a list of words
description_words = description.split() 

#iterate over each word and include it if it is not stopword 
description_words = [word for word in description_words if not word in stopwords.words('english')]
description_words
ps = PorterStemmer()
description_words=[ps.stem(word) for word in description_words]
description_words
#Now the description is clean the cleaned list of words can be converted to string and pushed to the dataset
df['description'][0]=' '.join(description_words)
df['description'][0]
#Now to clean other rows too one can iterate over all rows of the dataset and clean each
stopword_list = stopwords.words('english')
ps = PorterStemmer()
for i in range(1,len(df['description'])):
    try:
        description = re.sub('[^a-zA-Z]',' ',df['description'][i])
        description = description.lower().strip()
        description_words = description.split()
        description_words = [word for word in description_words if not word in stopword_list]
        description_words = [ps.stem(word) for word in description_words]
        df['description'][i] = ' '.join(description_words)
    except:
        pass
#Displaying all the descriptions after cleaning
for i in range(len(df['description'])):
    try:
        print(str(i+1)+".",df['description'][i],"\n")
    except:
        pass
#We will test for a correlation between the price of wine and its rating
print("Pearson Correlation:", pearsonr(df.price, df.points))
print(sm.OLS(df.points, df.price).fit().summary())
sns.lmplot(y = 'price', x='points', data=df)
fig, ax = plt.subplots(figsize = (20,7))
chart = sns.boxplot(x='country',y='points', data=df, ax = ax)
plt.xticks(rotation = 90)
plt.show()
df.country.value_counts()
country=df.groupby('country').filter(lambda x: len(x) >100)
df1 = pd.DataFrame({col:vals['points'] for col,vals in country.groupby('country')})
meds = df1.median()
meds.sort_values(ascending=False, inplace=True)

fig, ax = plt.subplots(figsize = (20,7))
chart = sns.boxplot(x='country',y='points', data=country, order=meds.index, ax = ax)
plt.xticks(rotation = 90)

plt.show()
df2 = pd.DataFrame({col:vals['price'] for col,vals in country.groupby('country')})
meds2 = df2.median()
meds2 = meds2.sort_values(ascending=False)

plt.rcParams['figure.figsize']=15,8 
meds2.plot("bar")
plt.title('Bar Chart Showing Median Wine Prices from Highest to Lowest')
plt.xlabel('Country')
plt.ylabel('Median Wine Price')
plt.show()
#Medians for the above Barplot
print(meds2)
df = df.groupby('variety').filter(lambda x: len(x) >100)
list = df.variety.value_counts().index.tolist()
fig4, ax4 = plt.subplots(figsize = (20,7))
sns.countplot(x='variety', data=df, order = list, ax=ax4)
plt.xticks(rotation = 90)
plt.show()
df = df.groupby('variety').filter(lambda x: len(x) >200)

df3 = pd.DataFrame({col:vals['points'] for col,vals in df.groupby('variety')})
meds3 = df3.median()
meds3.sort_values(ascending=False, inplace=True)

fig3, ax3 = plt.subplots(figsize = (20,7))
chart = sns.boxplot(x='variety',y='points', data=df, order=meds3.index, ax = ax3)
plt.xticks(rotation = 90)
plt.show()
df4 = pd.DataFrame({col:vals['points'] for col,vals in df.groupby('variety')})
mean1 = df4.mean()
mean1 = mean1.sort_values(ascending=False)

plt.rcParams['figure.figsize']=15,8 
mean1.plot("bar")
plt.title('Bar Chart Showing Median Wine Prices from Highest to Lowest')
plt.xlabel('Variety')
plt.ylabel('Median Wine Price')
plt.show()
#Mean for the above Barplot
print(mean1)
#We will now test the variations in price
df5 = pd.DataFrame({col:vals['price'] for col,vals in df.groupby('variety')})
mean2 = df5.mean()
mean2.sort_values(ascending=False, inplace=True)

fig3, ax3 = plt.subplots(figsize = (20,7))
chart = sns.barplot(x='variety',y='price', data=df, order=mean2.index, ax = ax3)
plt.xticks(rotation = 90)
plt.show()
X = df.drop(['country','points', 'variety'], axis = 1)
y = df.variety

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
wine =df.variety.unique().tolist()
wine.sort()
wine
#Split wine varieties with space and make new list
output = set()
for x in df.variety:
    x = x.lower()
    x = x.split()
    for y in x:
        output.add(y)

variety_list =sorted(output)
variety_list
extras = ['',' ',""," ",'.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', 'cab',"%"]
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(variety_list)
stop.update(extras)
vect = CountVectorizer(stop_words = stop)
X_train_dtm = vect.fit_transform(X_train.description)
price = X_train.price.values[:,None]
X_train_dtm = hstack((X_train_dtm, price))

X_test_dtm = vect.transform(X_test.description)
price_test = X_test.price.values[:,None]
X_test_dtm = hstack((X_test_dtm, price_test))
# X_test_dtm
models = {}
for z in wine:
    model = LogisticRegression()
    y = y_train == z
    model.fit(X_train_dtm, y)
    models[z] = model

testing_probs = pd.DataFrame(columns = wine)
for variety in wine:
    testing_probs[variety] = models[variety].predict_proba(X_test_dtm)[:,1]
    
predicted_wine = testing_probs.idxmax(axis=1)

comparison = pd.DataFrame({'actual':y_test.values, 'predicted':predicted_wine.values})   
# comparison = pd.DataFrame({'actual':'Malbec', 'predicted':predicted_wine.values})   

print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
comparison
filtered = df.groupby('variety').filter(lambda x: len(x) >= 500) #taking only the highest occuring to reduce size and keeping distribution in mind.
#Making a new column that is encoded version of variety
filtered['variety_id'] = filtered['variety'].factorize()[0]
category_id_df = filtered[['variety', 'variety_id']].drop_duplicates().sort_values('variety_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['variety_id', 'variety']].values)

filtered.head()
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='UTF-8', ngram_range=(1, 2), stop_words='english') 

features = tfidf.fit_transform(filtered.description).toarray() #Removing Stop words from descriptions 
labels = filtered.variety_id #Varity Numberical values saved as labels

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, filtered.index, test_size=0.30, random_state=0) #70/30 Split
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_df.variety.values, yticklabels=category_id_df.variety.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
X = df.drop('country', axis=1)  
y = df['points']

X = category_id_df.drop('variety_id', axis=1)  
y = category_id_df['variety_id'] 
category_id_df.head()
#Droping the duplicates
df[df.duplicated('description',keep=False)].sort_values('description').head(5)

df.head()
#Dropping all duplicated based and description and missing prices

df = df.drop_duplicates('description')
df = df[pd.notnull(df.price)]
df.shape
from scipy.stats import pearsonr
import statsmodels.api as sm
print("Pearson Correlation:", pearsonr(df.price, df.points))
print(sm.OLS(df.points, df.price).fit().summary())
sns.lmplot(y = 'price', x='points', data=df)
fig, ax = plt.subplots(figsize = (10,7))
chart = sns.boxplot(x='country',y='points', data=df, ax = ax)
plt.xticks(rotation = 90)
plt.show()
df.country.value_counts()[:]
df6 = pd.DataFrame({col:vals['price'] for col,vals in country.groupby('country')})
meds2 = df6.median()
meds2.sort_values(ascending=False, inplace=True)

fig, ax = plt.subplots(figsize = (20,5))
chart = sns.barplot(x='country',y='price', data=country, order=meds2.index, ax = ax)
plt.xticks(rotation = 90)
plt.show()
df = df.reset_index()
X = df.drop(['country','description','variety'], axis = 1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X = X.as_matrix().astype(np.float)
y = y.as_matrix().astype(np.float)
df.apply(lambda x: sum(x.isnull()),axis=0)
classifier = svm.SVR(kernel='linear') # We set a SVM classifier, the default SVM Classifier (Kernel = Radial Basis Function)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))
classifier.fit(X_train, y_train) # Then we train our model, with our balanced data train.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60) 
from sklearn.svm import SVR  
svclassifier = SVR(kernel='linear')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test) 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
clf = RandomForestRegressor(n_estimators=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy Score:',clf.score(X_test, y_test) *100,"%")
y_test

