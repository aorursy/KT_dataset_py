import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
path = "/kaggle/input/kindle-reviews/kindle_reviews.csv"
df = pd.read_csv(path)
df.drop(["Unnamed: 0"], axis = 1, inplace = True) 
df
#column_names = df.columns.tolist()
#column_names
df.info()
def convert_float(lst):
    lst = ast.literal_eval(lst) 
    return [float(i) for i in lst]

df['helpful'] = df['helpful'].apply(convert_float)
# Use apply(), faster than looping [coz interpretor!!]
helpful_rating = []
for i in tqdm(range(df.shape[0])):
  if df.loc[i].helpful[0] == 0 or df.loc[i].helpful[1] == 0:
    helpful_rating.append(0)
  else:
    rating = df.loc[i].helpful[0]*1.0 / df.loc[i].helpful[1]
    helpful_rating.append(rating)
df['helpful_percent'] = helpful_rating
df
df.overall.value_counts() 
plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
df.overall.value_counts().plot(kind='barh', color="blue", alpha=.65)
ax.set_ylim(-1, len(df.overall.value_counts())) 
plt.title("Reviews breakdown")
plt.show()
# Gives the central tendencies or idea about the approx range of values     
# Can run describe() per product to get per product central tendencies
df.describe() 
# isna() = to find for empty values in a column of a dataframe                  
# to get rid of all IDs (not useful)
df.isna().sum()
df.hist(bins = 25, figsize = (7,5))
plt.show()
correlation_matrix = df.corr(method = 'pearson')
correlation_matrix
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (5, 5))
hm = sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = "RdYlGn")
plt.show()
# moving average to fill up empty cells (incase of ordered data / time-series)
# df.rolling(window=2).mean()  
df['overall'].sort_values()
len(df.reviewerID.unique())
# gives general rating range of the reviewer -> to analyse the actual product rating
review_details = df.groupby('reviewerID')['overall'].mean().reset_index()
review_details.sort_values("overall", axis = 0, ascending = False, inplace = True, na_position ='last') 
review_details
# gives general helpful_percent range of the reviewer -> to analyse if he/she likes the products (one use-case)
review_details1 = df.groupby('reviewerID')['helpful_percent'].mean().reset_index()
review_details1.sort_values("helpful_percent", axis = 0, ascending = False, inplace = True, na_position ='last') 
review_details1
merge_review = pd.merge(review_details, review_details1, on = "reviewerID")
merge_review.sort_values(["overall", "helpful_percent"], axis = 0, ascending = (False, False), inplace = True, na_position ='last')
merge_review
merge_review_trunc = merge_review.iloc[np.random.permutation(len(merge_review))][:10]
### Visualization of the above combined dataframe

barWidth = 0.25
r1 = np.arange(len(merge_review_trunc.reviewerID))
r2 = [x + barWidth for x in r1]

# print(len(r1))
# print(len(r2))
# print(len(merge_review_trunc.reviewerID))

plt.bar(r1, merge_review_trunc.overall, color='#7f6d5f', width=barWidth, edgecolor='white', label='overall')
plt.bar(r2, merge_review_trunc.helpful_percent, color='#557f2d', width=barWidth, edgecolor='white', label='helpful_percent')

plt.xlabel('ReviewerID', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(merge_review_trunc.reviewerID))], merge_review_trunc.index.values)
 
plt.legend()
plt.show()
len(df.asin.unique())
# gives the most liked product wrt the ratings / get the product with the highest rating
product_details = df.groupby('asin')['overall'].mean().reset_index()
product_details.sort_values("overall", axis = 0, ascending = False, inplace = True, na_position ='last') 

# gives the most liked product wrt the helpful_percent / get the product with the highest helpful_percent
product_details
product_details1 = df.groupby('asin')['helpful_percent'].mean().reset_index()
product_details1.sort_values("helpful_percent", axis = 0, ascending = False, inplace = True, na_position ='last') 

product_details1
merge_product = pd.merge(product_details, product_details1, on = "asin")
merge_product.sort_values(["overall", "helpful_percent"], axis = 0, ascending = (False, False), inplace = True, na_position ='last')
merge_product
merge_product_trunc = merge_product.iloc[np.random.permutation(len(merge_product))][:25]
### Visualization of the above combined dataframe

fig= plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
xpos = list(range(len(merge_product_trunc.asin)))
ypos = xpos
zpos = xpos

ax.bar3d(xpos, ypos, zpos, merge_product_trunc.helpful_percent.to_list(), merge_product_trunc.overall.to_list(), merge_product_trunc.index.values.tolist())

ax.set_xlabel('$helpful\_percent$', rotation=150)
ax.set_ylabel('$overall$')
ax.set_zlabel(r'$asin\ index$', rotation=60)

plt.title("Trend of product overall rating with the helpful ratings\n", fontweight = "bold")
plt.show()
### Another use-case wrt numeric data: can find what months/weeks have highest
### sale and which products have highest sales in which seasons (either 
### use the topmost products above or make this independent and then combine
# nltk.download('stopwords')
# nltk.download('punkt')
import string
import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))
# Taking 50,000 random samples from the dataset with 9L+ entries for easier computation, analysis

new_df = df.iloc[np.random.permutation(len(merge_product))][:50000]
new_df 
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
stop
def preprocess(line):
    new_line = []
    # nopunc = [char for char in line if char not in string.punctuation]
    # nopunc = ''.join(nopunc)
    # nopunc.split()
    for i in line.split():
        if i.strip().lower() not in stop:
            word = i.strip().lower()
            new_line.append(word)
    
    return " ".join(new_line)
new_df['reviewText'] = new_df['reviewText'].fillna("").apply(preprocess)
new_df['summary'] = new_df['summary'].apply(preprocess)
new_df
# Heat-map correlation can be done after converting strings to numeric vectors
data = new_df[['overall' , 'reviewText' , 'summary']]
data
data['reviewText'] = data['reviewText'] + ' ' + data['summary']
del data['summary']

# Another idea: Find a weighted approach or some way to combine both overall and helpful_percent
data.head()
def overall_threshold(value):
    # taking threshold here as 3
    if(value == 1 or value == 2 or value == 3):
        return 0
    else:
        return 1

data.overall = data.overall.apply(overall_threshold)
data.head()
# default random state is 0
X_train, x_test, Y_train, y_test = train_test_split(data.reviewText, data.overall, test_size = 0.2, random_state = 0) 

# Try using a k-fold approach or mini-batch way to include all data points in a random fashion + cover all trends present
CV = CountVectorizer()
X_train_overall = CV.fit_transform(X_train)
x_test_overall = CV.transform(x_test)
print(X_train_overall.shape)
print(x_test_overall.shape)
# Multinomial Naive Bayes is used particularly for documents, which explicitly models the count of words and related 
# calculations rather than just stating the presence or absence of a particular word

mnb = MultinomialNB()
mnb.fit(X_train_overall, Y_train)
mnb_results = mnb.predict(x_test_overall)
accuracy_score(y_test, mnb_results)
classification_ans = classification_report(y_test, mnb_results)
print(classification_ans)
# Confusion matrix consists of true negatives, false positives, false negatives, true positives 
print(confusion_matrix(y_test, mnb_results))
# this code is very time-taking, optimization techniques can be used

clf = SVC()
clf.fit(X_train_overall, Y_train) 
svm = clf.predict(x_test_overall)
accuracy_score(y_test, svm)
classification_ans = classification_report(y_test, svm)
print(classification_ans)
# Confusion matrix consists of true negatives, false positives, false negatives, true positives 
print(confusion_matrix(y_test, mnb_results))