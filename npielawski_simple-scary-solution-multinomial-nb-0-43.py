import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import confusion_matrix, accuracy_score, log_loss



%matplotlib inline

matplotlib.style.use('ggplot')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



authors = df_train['author'].unique()

print('Authors are:', authors)
X_train, X_test, y_train, y_test = train_test_split(

    df_train['text'].values,

    df_train['author'].values,

    test_size=0.2,

    random_state=42

)
vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(X_train)



classifier = MultinomialNB()

classifier.fit(counts, y_train)
examples = ['How peculiar!', "That is a monster!", "the old man hadn't much time"]

example_counts = vectorizer.transform(examples)

pred = classifier.predict(example_counts)

pred
test_counts = vectorizer.transform(X_test)

y_pred = classifier.predict(test_counts)



accu = accuracy_score(y_test, y_pred)

print("Accuracy: %.02lf" % (100.*accu))
y_pred_proba = classifier.predict_proba(test_counts)

y_label = LabelBinarizer().fit_transform(y_test)

loss = log_loss(y_label, y_pred_proba)



print("Log-loss: %.04lf" % loss)
conf = confusion_matrix(y_test, y_pred)

conf = pd.DataFrame(

    conf.astype(np.float)/conf.sum(axis=1),

    index=authors,

    columns=authors

)



plt.figure()

plt.title('Confusion matrix of the predictions')

cmap = sns.cubehelix_palette(as_cmap=True)

sns.heatmap(conf, annot=True, cmap=cmap)

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.show()
final_counts = vectorizer.transform(df_test['text'])

result = pd.DataFrame(classifier.predict_proba(final_counts), columns=authors)

result.insert(0, 'id', df_test['id'])

result.to_csv('kaggle_solution.csv', index=False, float_format='%.15f')