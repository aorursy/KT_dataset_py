import pandas as pd
data = pd.read_csv('../input/turkish cyberbullying.csv')
data.head()
from nltk.corpus import stopwords
import re
# storing stopwords of Turkish (set structure for speed)
stops = set(stopwords.words('turkish'))
print(stops)
# pattern string in order to exclude non-Turkish-letter characters 
# such as punctuations and numbers
exc_letters_pattern = '[^a-zçğışöü]'
def text_to_wordlist(text, remove_stopwords=False, return_list=False):
    # 1. convert to lower
    text = text.lower()
    # 2. replace special letters
    special_letters = {'î':'i', 'â': 'a'}
    for sp_let, tr_let in special_letters.items():
        text = re.sub(sp_let, tr_let, text)
    # 3. remove non-letters
    text = re.sub(exc_letters_pattern, ' ', text)
    # 4. split
    wordlist = text.split()
    # 5. remove stopwords
    if remove_stopwords:
        wordlist = [w for w in wordlist if w not in stops]
    # 6.
    if return_list:
        return wordlist
    else:
        return ' '.join(wordlist)
clean_messages = []
for message in data['message']:
    clean_messages.append(text_to_wordlist(
        message, remove_stopwords=True, return_list=False))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    clean_messages, data['cyberbullying'], test_size=0.33, random_state=1)
from sklearn.feature_extraction.text import CountVectorizer

# limit vocabulary size as at most 5000; thus, words with 
# the least frequency are not included in the vocabulary
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)

# since we assume that we are not familiar with test set in advance,
# we use our training set in the construction of the vocabulary 
train_data_features = vectorizer.fit_transform(x_train)

# convert it to numpy array since it is easier to work with
train_data_features = train_data_features.toarray()
from sklearn.ensemble import RandomForestClassifier

# training Random Forest
forest = RandomForestClassifier(n_estimators=100)
_ = forest.fit(train_data_features, y_train)
# converting test data to BOW features
test_data_features = vectorizer.transform(x_test)
test_data_features = test_data_features.toarray()
y_pred = forest.predict(test_data_features)

predictions = pd.DataFrame(
    data={"message": x_test, "cyberbullying_true": y_test, "cyberbullying_pred": y_pred})
# correct_count = sum(y_pred == y_test)
correct_count = (predictions["cyberbullying_pred"] == predictions["cyberbullying_true"]).sum()
print("Accuracy is %{:.3f}".format(100 * correct_count / len(y_test)))

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print('\tPredictions')
print('\t{:>5}\t{:>5}'.format(0,1))
for row_id, real_row in enumerate(cf):
    print('{}\t{:>5}\t{:>5}'.format(row_id, real_row[0], real_row[1]))
wrong_predictions = predictions[predictions["cyberbullying_pred"] != predictions["cyberbullying_true"]]

wrong_predictions.to_csv("wrong_predictions.csv", index=False, quoting=3)
# wrong_predictions.head()
wrong_predictions.iloc[10:15]

