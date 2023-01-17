import numpy as np 

import pandas as pd

from fuzzywuzzy import fuzz



from subprocess import check_output

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from nltk.corpus import stopwords

stopwords = stopwords.words('english')



print(check_output(["ls", "../input"]).decode("utf8"))
data_frame = pd.read_csv('../input/train.csv')
def extract_features(df):

    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)

    df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

    df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    return df

df = extract_features(data_frame)
print (df[0:10])
feature_columns = df.columns.drop(['id','question1', 'question2', 'is_duplicate'])

X_normalized = normalize(df[feature_columns], norm='l2',axis=1, copy=True, return_norm=False)
print (X_normalized[0:10])
x_train, x_test, y_train, y_test = train_test_split(X_normalized, df['is_duplicate'], random_state = 1,test_size=0.2)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
result_cols = ["Classifier", "Accuracy"]

result_frame = pd.DataFrame(columns=result_cols)
classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GaussianNB()]
for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(x_train, y_train)

    predicted = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test,predicted)

    print (name+' accuracy = '+str(acc*100)+'%')

    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)

    result_frame = result_frame.append(acc_field)

#sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()