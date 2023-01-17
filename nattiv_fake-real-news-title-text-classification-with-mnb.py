import pandas as pd

import matplotlib.pyplot as plt
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv', usecols=['title', 'subject'])

fake['label'] = 'fake'



true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv', usecols=['title', 'subject'])

true['label'] = 'real'



data = pd.concat([fake, true])

data.reset_index(inplace=True)

data.drop(columns=['index'], inplace=True)



fake = None

true = None
data.head()
data.isna().sum()
data['title_word_count'] = data.title.apply(len)
_ = plt.bar([0, 1], [data[data.label == 'fake'].shape[0], data[data.label == 'real'].shape[0]])

plt.xticks([0,1], ['fake', 'real'])

plt.xlabel('Label')

plt.ylabel('Frequency')

plt.title('Observations Per Class')

plt.show()
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

ax0.hist(data[data.label=='fake'].title_word_count)

ax1.hist(data[data.label=='real'].title_word_count)



fig.suptitle('Distribution of Title Word Count Per Class')

ax0.set_xlabel('Fake Class')

ax1.set_xlabel('Real Class')



plt.show()
subjects = set(data.subject)



word_count_by_subject = {'fake': {}, 'real': {}}



for subject in subjects:

    word_count_by_subject['fake'][subject] = data[(data.label=='fake') & (data.subject == subject)].title_word_count.median()

    word_count_by_subject['real'][subject] = data[(data.label=='real') & (data.subject == subject)].title_word_count.median()



xs = range(0, len(subjects))





plt.bar(xs, [value for subject, value in word_count_by_subject['fake'].items()], width=.5, label='Fake')

plt.bar([x +.2 for x in xs], [value for subject, value in word_count_by_subject['real'].items()], width=0.5, color='orange', label='Real')



plt.legend(loc='lower right',framealpha=.75)



plt.xticks(xs, subjects, rotation=90)

plt.tick_params('both')

plt.xlabel('Subject')

plt.ylabel('Median Title Word Count')

plt.title('Median Title Word Count by Subject and Class')

plt.show()
import regex as re



import nltk

from nltk.corpus import stopwords



cached_stopwords = stopwords.words('english')
def replace_spec(text):

    regex = r'[^a-zA-z0-9/s]'

    text = re.sub(regex, ' ', text)

    return text



def process_title(title):

    title = title.lower()

    title = replace_spec(title)

        

    title_list = str.split(title)

    

    final_title =[]

    

    for item in title_list:

        if item not in cached_stopwords:

            final_title.append(item)

            

    return " ".join(final_title)
data['title_final'] = data.title.apply(process_title)
# previewing the before and after of text processing

print(data.at[0, 'title'], '\n', data.at[0, 'title_final'])
import time

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
# changing to a binary (0, 1) label

data.label = data.label.apply(lambda x: 1 if x == 'fake' else 0)
x_train, x_test, y_train, y_test = train_test_split(data[['title_final']], data.label, test_size=.1, random_state=42)
docs = x_train.title_final.tolist()



vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=3)



time_start = time.time()

vectorizer.fit(docs)

time_end = time.time()





print(f"vectorizer fit in {(time_end-time_start)/60} mins")
train_tfidf = vectorizer.transform(x_train.title_final.tolist())

test_tfidf = vectorizer.transform(x_test.title_final.tolist())
y_train.shape, train_tfidf.shape
mnb = MultinomialNB()



time_start = time.time()

mnb.fit(train_tfidf, y_train.values)

time_end = time.time()



print(f"mnb trained in {(time_end-time_start)/60} mins")
for k, v in {'TRAIN': [train_tfidf, y_train], 'TEST': [test_tfidf, y_test]}.items():

    preds = mnb.predict(v[0])

    

    print(f"{k} RESULTS\n\n{classification_report(v[1],preds)}\n\n")
# getting the indexes of the top features by class

fake_prob_sorted = mnb.feature_log_prob_[1, :].argsort()

real_prob_sorted = mnb.feature_log_prob_[0, :].argsort()



# getting the top feature names

fake_top_features = np.take(vectorizer.get_feature_names(), fake_prob_sorted[:20])

real_top_features = np.take(vectorizer.get_feature_names(), real_prob_sorted[:20])



# creating dictionaries for each class with the feature name and log probability

real_dict = {k:v for k, v in zip(

                                real_top_features, 

                                np.take(mnb.feature_log_prob_[0, :], real_prob_sorted[:20])

                                )

            }



fake_dict = {k:v for k, v in zip(

                                fake_top_features, 

                                np.take(mnb.feature_log_prob_[1, :], fake_prob_sorted[:20])

                                )

            }
# creating a dataframe using the dictionaries of top terms per class

top_terms = pd.DataFrame.from_dict(real_dict, orient='index')

top_terms.reset_index(inplace=True)

top_terms.rename(columns={'index': 'term', 0: 'log_prob'}, inplace=True)

top_terms['label'] = 'Real'



# add the top terms for the fake class

for term, log_prob in fake_dict.items():

    top_terms = top_terms.append({'term': term, 'log_prob': log_prob, 'label': 'Fake'}, ignore_index=True)



# convert log probability to odds ratio

top_terms['odds'] = top_terms.log_prob.apply(np.exp)



# sort alphabetically

top_terms = top_terms.sort_values('term', ascending=True)
# creating dictionary to be used for plotting

y_map = {term: y for y, term in zip(range(0, top_terms.shape[0]), top_terms.term)}



plot_map = {} 

for index, row in top_terms.iterrows():

    term = row['term']

    

    plot_map[term] = {

        'x': row['odds'],

        'y': y_map[term],

        'c': 'red' if row['label'] == 'Fake' else 'blue'

    } 
# plotting

fig = plt.figure(figsize=(8, 8))

_ = plt.scatter(

            [value['x'] for key, value in plot_map.items()], 

            [value['y'] + 5 for key, value in plot_map.items()],

            s=5,

            color=[value['c'] for key, value in plot_map.items()],

)



_ = plt.scatter(

    [0]* top_terms.shape[0],

    [r + 5 for r in range(0, top_terms.shape[0])],

    s=5,

    color='grey'

)



plt.legend()

plt.yticks(ticks=[r + 5 for r in range(0, top_terms.shape[0])], labels=y_map.keys())

plt.xlabel('Odds Ratio')

plt.ylabel('Term')

plt.title('Top Important Terms by Class')

plt.show()