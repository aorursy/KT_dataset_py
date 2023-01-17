import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')#

df
df.columns
df['uses_ad_boosts'] = df['uses_ad_boosts']

df['badge_local_product'] = df['badge_local_product']

df['badge_product_quality'] = df['badge_product_quality']

df['badge_fast_shipping'] = df['badge_fast_shipping']

df['shipping_is_express'] = df['shipping_is_express']

df['has_urgency_banner'] = df['has_urgency_banner']

df['merchant_has_profile_picture'] = df['merchant_has_profile_picture']



df.info()
df.isnull().sum()
round(df.describe())
def histograms_and_countplots(column, data, columns_to_exclude):

    if column not in columns_to_exclude:

        if data[column].dtype not in ['int64', 'float64']:

            f, axes = plt.subplots(1,1,figsize=(15,5))

            sns.countplot(x=column, data = data)

            plt.xticks(rotation=90)

            plt.suptitle(column,fontsize=20)

            plt.show()

        else:

            g = sns.FacetGrid(data, margin_titles=True, aspect=4, height=3)

            g.map(plt.hist,column,bins=100)

            plt.show()

        plt.show()
columns_to_exclude = ['title', 'title_orig', 'currency_buyer', 'tags', 'merchant_title', 'merchant_name', 

                'merchant_info_subtitle', 'merchant_id', 'merchant_profile_picture', 'product_url', 

                'product_picture', 'product_id', 'theme', 'crawl_month' ]





for column in df.columns:

    histograms_and_countplots(column, df, columns_to_exclude)
print('Median of units sold is',df['units_sold'].median())

print('Mean of units sold is',df['units_sold'].mean())

df['units_sold'].value_counts()
def below_ten(units_sold):

    if units_sold < 10:

        return 10

    else:

        return units_sold
df['units_sold'] = df['units_sold'].apply(below_ten)
df['units_sold'].value_counts()
df[df['units_sold'] == 100000]
def is_successful(units_sold):

    if units_sold > 1000:

        return 1

    else:

        return 0
df['is_successful'] = df['units_sold'].apply(is_successful)

#df['is_successful'] = df['units_sold'].apply(is_successful).astype('category')

print('Percent of successful products: ', df['is_successful'].value_counts()[1] / len(df['is_successful'])*100)

sns.countplot(data=df, x='is_successful')

plt.show()
print('Overall stats:')

print(df['price'].mean())

print(df['retail_price'].mean())

print('----------------------')

print('Stats for successful products:')

print(df[df['is_successful'] == 1]['price'].mean())

print(df[df['is_successful'] == 1]['retail_price'].mean())

print('----------------------')

print('Stats for unsuccessful products:')

print(df[df['is_successful'] == 0]['price'].mean())

print(df[df['is_successful'] == 0]['retail_price'].mean())
df['diff_in_price'] = round(df['price']/df['retail_price'],2)

df['diff_in_price']
sns.violinplot(data=df, y='diff_in_price', x='is_successful')
print('Percent of products using ad boosts: ', df['uses_ad_boosts'].value_counts()[1] / len(df['uses_ad_boosts'])*100)
sns.countplot(data=df, x='uses_ad_boosts', hue='is_successful')
pd.crosstab(df['uses_ad_boosts'], df['is_successful'])
pd.crosstab(df['uses_ad_boosts'], df['units_sold'])
df[df['rating_five_count'].isnull()==True][['rating', 'rating_count',

       'rating_five_count', 'rating_four_count', 'rating_three_count',

       'rating_two_count', 'rating_one_count']]
def ratings_to_zero(rating_and_count):

    rating = rating_and_count[0]

    count = rating_and_count[1]

    

    if count == 0:

        rating = 0

    else:

        rating = rating

        

    return rating
df['rating'] = df[['rating', 'rating_count']].apply(ratings_to_zero, axis=1)

df['rating_five_count'].fillna(0, inplace=True)

df['rating_four_count'].fillna(0, inplace=True)

df['rating_three_count'].fillna(0, inplace=True)

df['rating_two_count'].fillna(0, inplace=True)

df['rating_one_count'].fillna(0, inplace=True)





df[df['rating_five_count'].isnull()==True][['rating', 'rating_count',

       'rating_five_count', 'rating_four_count', 'rating_three_count',

       'rating_two_count', 'rating_one_count']]
df['rating']
ratings_column = ['rating', 'rating_count',

       'rating_five_count', 'rating_four_count', 'rating_three_count',

       'rating_two_count', 'rating_one_count']



for column in ratings_column:

    g = sns.FacetGrid(df, row='is_successful', margin_titles=True, aspect=4, height=3)

    g.map(plt.hist, column, bins=100)

    plt.title(column)

    plt.show()
df.groupby('is_successful').mean()[ratings_column]
df.groupby('units_sold').mean()[ratings_column]
badges_column = ['badges_count',

       'badge_local_product', 'badge_product_quality', 'badge_fast_shipping']
df[df['badges_count'] != 0][badges_column]
for column in badges_column:

    sns.countplot(data=df, x=column, hue='is_successful')

    plt.title(column)

    plt.show()
from wordcloud import WordCloud, STOPWORDS



df['tags']
def remove_stopwords(text):

    from nltk.tokenize import word_tokenize



    text_tokens = word_tokenize(text)



    tokens_without_sw = [word for word in text_tokens if not word in STOPWORDS]

    

    filtered_sentence = (" ").join(tokens_without_sw)



    return filtered_sentence
df['tags'] = df['tags'].apply(remove_stopwords)
df['tags']
from collections import Counter

results = Counter()

df['tags'].str.lower().str.split().apply(results.update)

counter_df = pd.DataFrame.from_dict(results, orient='index')

counter_df.sort_values(by=0, axis=0, ascending=False).head(15)
import matplotlib.pyplot as plt

word_string=" ".join(df['tags'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



plt.subplots(figsize=(15,15))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
word_string=" ".join(df[df['is_successful']==1]['tags'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



plt.subplots(figsize=(15,15))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
word_string=" ".join(df[df['is_successful']==0]['tags'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



plt.subplots(figsize=(15,15))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
X = df['tags']

y = df['is_successful']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn import metrics



text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=STOPWORDS)),

                         ('clf', LinearSVC(random_state=0)),

    ])



text_clf.fit(X_train, y_train)  



predictions = text_clf.predict(X_test)



print('-----------------------')

print(metrics.confusion_matrix(y_test,predictions))

print('-----------------------')

print(metrics.classification_report(y_test,predictions))
df['product_color'].fillna('missing', inplace=True)

word_string=" ".join(df['product_color'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



plt.subplots(figsize=(15,15))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df['product_variation_size_id'].fillna('missing', inplace=True)

word_string=" ".join(df['product_variation_size_id'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



plt.subplots(figsize=(15,15))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
shipping_column = ['shipping_option_name',

       'shipping_option_price', 'shipping_is_express', 'countries_shipped_to']



for column in shipping_column:

    sns.countplot(data=df, x=column, hue='is_successful')

    plt.title(column)

    plt.show()
sns.violinplot(data=df, y='countries_shipped_to', x='is_successful')
df['inventory_total'].value_counts()
df['product_variation_inventory'].value_counts()
sns.violinplot(data=df, y='product_variation_inventory', x='is_successful')
sns.violinplot(data=df, y='product_variation_inventory', x='units_sold')
df['has_urgency_banner'].fillna(0, inplace=True)

df['urgency_text'].fillna(0, inplace=True)
urgency_column = ['has_urgency_banner', 'urgency_text', 'origin_country']



for column in urgency_column:

    sns.countplot(data=df, x=column, hue='is_successful')

    plt.title(column)

    plt.show()
merchant_columns = ['merchant_rating_count', 'merchant_rating',

       'merchant_id', 'merchant_has_profile_picture']





sns.countplot(data=df, x='merchant_has_profile_picture', hue='is_successful')
sns.violinplot(data=df, y='merchant_rating', x='is_successful')
sns.violinplot(data=df, y='merchant_rating', x='units_sold')
sns.countplot(x=df['merchant_id'].value_counts())
df.drop(labels=['title', 'title_orig', 'currency_buyer', 'merchant_title', 'merchant_name', 

                'merchant_info_subtitle', 'merchant_id', 'merchant_profile_picture', 'product_url', 

                'product_picture', 'product_id', 'theme', 'crawl_month', 'rating', 'rating_count',

       'rating_five_count', 'rating_four_count', 'rating_three_count',

       'rating_two_count', 'rating_one_count', 'units_sold','badges_count',

       'badge_local_product','badge_fast_shipping', 'tags', 'product_color',

       'product_variation_size_id', 'shipping_option_name',

       'shipping_option_price', 'shipping_is_express', 'countries_shipped_to','has_urgency_banner', 'urgency_text',

       'origin_country'], axis=1, inplace=True)
df.columns
y = df['is_successful']

X = df[['price', 'retail_price', 'uses_ad_boosts', 'badge_product_quality',

       'product_variation_inventory',

       'inventory_total', 'merchant_rating_count', 'merchant_rating',

       'merchant_has_profile_picture','diff_in_price']]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)





models = [LogisticRegression(), DecisionTreeClassifier(), XGBClassifier(),  

          GradientBoostingClassifier(), KNeighborsClassifier(), RandomForestClassifier()]



for model in models:

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    print(model)

    print('---------------------------')

    print(metrics.classification_report(y_test,y_pred))

    print('')

    print('')
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



for k in range(1,11):

    

    print(k)

    print('---------------')

    y = df['is_successful']

    X = df[['price', 'retail_price', 'uses_ad_boosts', 'badge_product_quality',

       'product_variation_inventory',

       'inventory_total', 'merchant_rating_count', 'merchant_rating',

       'merchant_has_profile_picture','diff_in_price']]





    X = SelectKBest(chi2, k=k).fit_transform(X, y)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)





    models = [LogisticRegression(), DecisionTreeClassifier(), XGBClassifier(),  

          GradientBoostingClassifier(), KNeighborsClassifier(), RandomForestClassifier()]



    for model in models:

    

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    

        print(model)

        print('---------------------------')

        print(metrics.accuracy_score(y_test,y_pred))

        print('')

        print('')

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



y = df['is_successful']

X = df[['price', 'retail_price', 'uses_ad_boosts', 'badge_product_quality',

       'product_variation_inventory',

       'inventory_total', 'merchant_rating_count', 'merchant_rating',

       'merchant_has_profile_picture','diff_in_price']]





X = SelectKBest(chi2, k=8).fit_transform(X, y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)





models = RandomForestClassifier()



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(metrics.accuracy_score(y_test,y_pred))
