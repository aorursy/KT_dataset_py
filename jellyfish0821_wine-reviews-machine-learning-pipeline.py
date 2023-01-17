# Load CSV 
import csv
import numpy
import pandas
%pylab inline
filename = '../input/winemag-data_first150k.csv'
winedata = pandas.read_csv(filename)
print(winedata.shape)
winedata.head()
# Find out the duplicaet value
print("Total number of examples: ", winedata.shape[0])
print("Number of examples with the same description: ", winedata[winedata.duplicated(['description'])].shape[0])
Cleanwine = winedata.drop_duplicates('description', keep='first', inplace=False)
Cleanwine.dropna(subset=['description','points','price'], inplace=True)
Cleanwine.shape
## Finding missing values
total = Cleanwine.isnull().sum().sort_values(ascending = False)
total
CountryList = Cleanwine['points'].groupby(winedata['country']).mean()
CountryList = CountryList.to_frame().reset_index()
CountryList.columns = ['country', 'AvgPoints']
CountryList.sort_values(by='AvgPoints', ascending=False).head(10)
# Calculate average points for wines from each country
count = Cleanwine.country.value_counts()
count = count.to_frame('count').reset_index()
count.columns = ['country', 'count']
CountryList = CountryList.merge(count, on="country", how='left')
CountryList.sort_values(by='count',ascending=True).head(10)
CountryList.sort_values(by='count',ascending=False).head(10)
Cleanwine['country'].value_counts()[:10].plot(kind='bar',figsize=(12,8));
plt.xticks(rotation=45)
plt.xlabel('country')
plt.ylabel('Number of country count')
plt.show()
Cleanwine2 = Cleanwine.merge(CountryList, on="country", how='left')
Major = Cleanwine2[Cleanwine2['count'] >= 500]
Major.shape

MajorCountry = Major['points'].groupby(Major['country']).mean()
MajorCountry= MajorCountry.to_frame().reset_index()
MajorCountry.columns = ['country', 'AvgPoints']
MajorCountry.sort_values(by='AvgPoints', ascending=False)
count2 = Major.variety.value_counts()
count2 = count2.to_frame('count').reset_index()
count2.columns = ['variety', 'countgrape']
count2.shape
count2.sort_values(by='countgrape', ascending=False).head(10)
count2.sort_values(by='countgrape', ascending=True).head(10)
Cleanwine3 = Major.merge(count2, on="variety", how='left')
Major2 = Cleanwine3[Cleanwine3['countgrape'] >= 500]
Major2. head(10)
Major2.variety.nunique()

Major2.variety.value_counts()
import seaborn as sns
sns.countplot(x='points',data = Major2, palette='hls' )
plt.show()
CountryList2 = Major2['price'].groupby(Major2['country']).mean()
CountryList2 = CountryList2.to_frame().reset_index()
CountryList2.columns = ['country', 'price']
CountryList2.sort_values(by='price', ascending=False).head(10)

plt.figure(figsize=(20,25))
plt.subplot(2,1,1)
g = sns.boxplot(x='country', y='price',data=Major2)
g.set_title("Which country has the most expensive wines", fontsize=25)
g.set_xlabel("Country", fontsize=20)
g.set_ylabel("Price ($)", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
                
plt.show()
plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax = sns.distplot(Major2["price"])
# if we want to see better price distribution we have to scale our price or drop the tail.  

plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax = sns.distplot(Major2[Major2["price"]<200]['price'])

percent=Major2[Major2['price']<200].shape[0]/Major2.shape[0]*100
print("There are :", percent, "% wines less than 200 USD")
percent=Major2[Major2['price']>200].shape[0]/Major2.shape[0]*100
print("There are :", percent, "% wines more expensive than 200 USD")
print("Number of wines costs more than 200USD:", Major2[Major2['price']>200].shape[1])
plt.figure(figsize=(20,16))
plt.subplot(2,1,2)
g1 = sns.boxplot(x='country', y='points',data=Major2)
g1.set_title("Which country has the highest rated wines", fontsize=25)
g1.set_xlabel("Country's ", fontsize=20)
g1.set_ylabel("Points", fontsize=20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
plt.subplots_adjust(hspace = 0.6,top = 0.9)
plt.show()
variety = Major2.variety.unique().tolist()
variety[:5]
# Extract a list of grape varieties
lcvariety = [x.lower() for x in variety]
lcvariety[:5] #convert the grape varieties to lowercase
bkvariety = [i.replace('-', ' ').replace(',', ' ').split(' ') for i in lcvariety] #generated a list of list
bkvariety = [item for sublist in bkvariety for item in sublist] # convert list of list to a python list
bkvariety[:8] #break down the grape variety names into separate words
bkvariety.remove('red')
bkvariety.remove('white')
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(variety,lcvariety,bkvariety) #update stop words
print(stop)
import sklearn
sklearn.__version__
from sklearn.model_selection import train_test_split
X = Major2.description
y = Major2.variety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
Major2.info()
#Heuristic data exploration
print(type(X))
print(type(y))
print(len(X))
print(len(y))
set(y) #grape varieties as the target
#take a look at some sample data
print(X[17])
print(y[17])
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words = stop, ngram_range=(1, 3), max_features=3000) 
#vec = CountVectorizer(stop_words = stop, ngram_range=(1, 3), min_df=3, max_features=5000) 
X_train_vec = vec.fit_transform(X_train)
X_train_vec
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train_vec, y_train)
X_test_vec= vec.transform(X_test)
y_pred = logit.predict(X_test_vec)
list(zip(y_pred, y_test))[:10]
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
print(metrics.precision_score(y_test, y_pred, average='weighted'))
print(metrics.recall_score(y_test, y_pred, average='weighted'))
print(metrics.f1_score(y_test, y_pred, average='weighted'))
print(metrics.classification_report(y_test, y_pred))
Myreview =['This wine is clear pale lemon-green. The nose is crisp clean, full of lovely youthful aromas of grapfruit peel, lemon zest, and apricot. The signature herbaceous flavor is reminiscent of asparagus, green pepper, and a hint of crushed wet stones. The wine offers high acidity, balanced with pronounced fruitiness and decent length. Can drink now, but has potential for aging to further develop complexity and tertiary flavors. It pairs well with seafood - show off the umami in oysters or cut off the fattiness in salmon. It would also be a great choice to pair with Asian hotpot and sichuan cuisine with its refreshing acidity.']
X_test_vec= vec.transform(Myreview)
y_pred = logit.predict(X_test_vec)
print(y_pred)
from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer()
X_tf = tf.fit_transform(X_train_vec)
X_tf.shape
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_tf, y_train)
X_test_vec= vec.transform(X_test)
X_test_tf = tf.transform(X_test_vec)
y_pred = logit.predict(X_test_tf)
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
print(metrics.precision_score(y_test, y_pred, average='weighted'))
print(metrics.recall_score(y_test, y_pred, average='weighted'))
print(metrics.f1_score(y_test, y_pred, average='weighted'))
print(metrics.classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=13)
clf.fit(X=X_train_vec,y=y_train)
clf.score(X=X_test_vec, y=y_test)
df = Major2[['price','points']]
df.dropna(subset=['price','points'], inplace=True)
from scipy.stats import pearsonr
print("Pearson Correlation:", pearsonr(df.price, df.points))

import statsmodels.api as sm
print(sm.OLS(df.points, df.price).fit().summary())
sns.lmplot(y = 'price', x='points', data=df)
