%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



plt.style.use('fivethirtyeight')



# Read in the data

df = pd.read_csv('../input/amazon-reviews-unlocked-mobile-phones/Amazon_Unlocked_Mobile.csv')

df.head()


mostreviewd = (df.set_index('Product Name').groupby(level=0)['Reviews']

    .agg(['count'])).sort_values(['count'], ascending=False)[:10]







plt.figure(figsize=(12, 8))

sns.barplot(mostreviewd.reset_index().index, y=mostreviewd['count'], hue=mostreviewd.index.str[:50] + '...', dodge=False)

plt.ylim(1000,)

plt.xticks([]);

plt.ylabel('Reviews count')

plt.title('Top 10 most reviewed products');
bestbrand = (df[df['Rating'] > 3].set_index('Brand Name').groupby(level=0)['Reviews'].

    agg(['count'])).sort_values(['count'], ascending=False)[:10]



plt.figure(figsize=(12, 8))

sns.barplot(bestbrand.index, y=bestbrand['count'], hue=bestbrand.index, dodge=False)

plt.legend([])

plt.ylabel('Positive reviews count')

plt.title('Top 10 best brand');
# Filter out rating above 3 and get the review count

worstproduct = (df[df['Rating'] < 3].set_index('Product Name').groupby(level=0)['Reviews'].

    agg(['count'])).sort_values(['count'], ascending=False)[:10]



plt.figure(figsize=(12, 8))

sns.barplot(worstproduct.reset_index().index, y=worstproduct['count'], hue=worstproduct.index.str[:50] + '...', dodge=False)

plt.ylim(250,)

plt.xticks([]);

plt.ylabel('Negative reviews count')

plt.title('Top 10 worst products');
## Best budget product

budget = (df[(df['Rating'] > 3) & (df['Price'] < 500)].set_index('Product Name').groupby(level=0)['Price'].

    agg(['count'])).sort_values(['count'], ascending=False)[:10]



grouped = df.set_index('Product Name').loc[budget.index].groupby(level=0)



price = pd.Series(index = budget.index)

for name, group in grouped:

    price.loc[name] = group.Price.iloc[0] 

    

budget['Price'] = price

budget.reset_index(inplace=True)



plt.figure(figsize=(12, 8))

sns.barplot(x='Price', y='count', dodge=False, hue='Product Name', data=budget, palette=sns.color_palette("cubehelix", 12))

plt.ylim(750,)

plt.ylabel('Positive reviews count')

plt.title('Best budget products under $500');

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
highend = (df[(df['Rating'] > 3) & (df['Price'] > 900)].set_index('Product Name').groupby(level=0)['Price'].

    agg(['count'])).sort_values(['count'], ascending=False)[:10]



grouped = df.set_index('Product Name').loc[highend.index].groupby(level=0)



price = pd.Series(index = budget.index)

for name, group in grouped:

    price.loc[name] = group.Price.iloc[0] 

    

highend['Price'] = price

highend.reset_index(inplace=True)



plt.figure(figsize=(8, 8))

sns.barplot(x='Price', y='count', dodge=False, hue='Product Name', data=highend, palette=sns.color_palette("cubehelix", 12))

plt.ylabel('Positive reviews count')

plt.title('Best high end products under $2000');

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
plt.figure(figsize=(10,8))

sns.violinplot(x="Rating", y="Price", data=df)

plt.title('Price vs Rating distribution');
# Drop missing values

df.dropna(inplace=True)



# Remove any 'neutral' ratings equal to 3

df = df[df['Rating'] != 3]



# Encode 4s and 5s as 1 (rated positively)

# Encode 1s and 2s as 0 (rated poorly)

df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

df.head(10)
df.describe()
# Get training and test data from dataset. 

from sklearn.model_selection import train_test_split



# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 

                                                    df['Positively Rated'], 

                                                    random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer



# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5

vect = TfidfVectorizer(min_df=5).fit(X_train)

len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)
feature_names = np.array(vect.get_feature_names())



sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()



print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))

print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



X_train_vectorized = vect.transform(X_train)



model = LogisticRegression(solver='saga')

model.fit(X_train_vectorized, y_train)



predictions = model.predict(vect.transform(X_test))



print('AUC: ', roc_auc_score(y_test, model.decision_function(vect.transform(X_test))))

sorted_coef_index = model.coef_[0].argsort()



print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



def PlotWordCloud(words, title):

    wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white' 

                ).generate(words) 

                                                           

    # plot the WordCloud image                        

    plt.figure(figsize = (10, 10), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 

    plt.title(title, fontsize=50)



    plt.show() 
negative = ''

for word in feature_names[sorted_coef_index[:100]]:

    negative += word + ' '

PlotWordCloud(negative, 'Most negative words')
positive = ''

for word in feature_names[sorted_coef_index[:-101:-1]]:

    positive += word + ' '    

PlotWordCloud(positive, 'Most positive words')
print(model.predict(vect.transform(['not an issue, phone is working',

                                    'an issue, phone is not working'])))
# extracting 1-grams and 2-grams

vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)



X_train_vectorized = vect.transform(X_train)



len(vect.get_feature_names())
model = LogisticRegression(solver='saga')

model.fit(X_train_vectorized, y_train)



predictions = model.predict(vect.transform(X_test))



print('AUC: ', roc_auc_score(y_test, model.decision_function(vect.transform(X_test))))
feature_names = np.array(vect.get_feature_names())



sorted_coef_index = model.coef_[0].argsort()



print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
print(model.predict(vect.transform(['not an issue, phone is working',

                                    'an issue, phone is not working'])))