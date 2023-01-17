%matplotlib inline



import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import numpy as np

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



"""

    df: dataframe

    vec: Object that convert a collection of text documents to a matrix of token counts

    ascending: order creteria

    qty: first n words to return

    

    Return list with all words and their global frequency (all samples)

"""

def get_resume(df, vec, ascending = False, n = None):

    X = vec.fit_transform(df["text"].values)

    feature_names = vec.get_feature_names()



    resume = pd.DataFrame(columns = feature_names, data = X.toarray()).sum()



    if(n):

        return resume.sort_values(ascending = ascending)[:n]



    return resume
df = pd.read_csv("../input/spam.csv", encoding='latin-1')



df.columns = ["sms_type", "text", "2", "3", "4"]



df.drop("2", axis=1, inplace=True)

df.drop("3", axis=1, inplace=True)

df.drop("4", axis=1, inplace=True)



df["text"] = df["text"].str.lower() # Convert to lowercase
df.describe()
df["sms_type"].value_counts()
vec = CountVectorizer(decode_error = 'ignore', stop_words = "english")

spam_resume = get_resume(df[df["sms_type"] == "spam"], vec, n = 25)

spam_resume
spam_resume.plot(kind = 'bar', figsize = (8, 4), fontsize = 12)
df["sms_type"][df["text"].str.contains("\d{4,}")].value_counts()
WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""



df["sms_type"][df["text"].str.contains(WEB_URL_REGEX)].value_counts()
df["text"] = df["text"].str.replace(WEB_URL_REGEX," someurl ")

df["text"] = df["text"].str.replace("\d{4,}"," suspectnumber ")
vec = CountVectorizer(decode_error = 'ignore', stop_words = "english")

spam_resume = get_resume(df[df["sms_type"] == "spam"], vec, n = 25)

spam_resume
spam_resume.plot(kind = 'bar', figsize = (8, 4), fontsize = 12)
X = df["text"].values

y = df["sms_type"].values

    

folds = StratifiedKFold(n_splits = 5, shuffle = True)

test_scores, train_scores = np.array([]), np.array([])



vocabulary = list(set(list(spam_resume.keys())))



relevant_vec = CountVectorizer(decode_error = 'ignore',

                                stop_words = "english",  

                                vocabulary = vocabulary)



for train_idx, test_idx in folds.split(X, y):



    X_train, y_train = relevant_vec.fit_transform(X[train_idx]), y[train_idx]

    X_test, y_test = relevant_vec.transform(X[test_idx]), y[test_idx]



    log_model = LogisticRegression()

    log_model.fit(X_train, y_train)

    y_pred_test = log_model.predict(X_test)



    c = confusion_matrix(y_test, y_pred_test)

    print(c)

    print("------------------------------")



    train_scores = np.append(train_scores, log_model.score(X_train, y_train))

    test_scores = np.append(test_scores, log_model.score(X_test, y_test))



print(train_scores.mean(), test_scores.mean())