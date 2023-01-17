import os

DATASET_PATH = "../input/20_newsgroups/20_newsgroups"

classes = os.listdir(DATASET_PATH)

y = []

x = []

no_utf8 = 0

utf8 = 0

for c in classes:

    class_path = os.path.join(DATASET_PATH, c)

    files = os.listdir(class_path)

    for f in files:

        file_path = os.path.join(class_path, f)

        try:

            with open(file_path, 'r', encoding="utf-8") as f:

                file_content = f.read()

            data = file_content.split('\n\n')

            x.append(data[1])

            y.append(c)

            utf8 += 1

        except:

            with open(file_path, 'r', encoding="ISO-8859-1") as f:

                file_content = f.read()

            data = file_content.split('\n\n')

            x.append(data[1])

            y.append(c)

            no_utf8 += 1

        print('utf-8 files: %d, ISO-8859-1 files: %d' % (utf8, no_utf8), end="\r", flush=True)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

count_vect = CountVectorizer()

dictionary = count_vect.fit(x)

# store the dictionary!

transformed_x = dictionary.transform(x)



x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.33)
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2, random_state=0)

clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

predictions = clf.predict(x_test)

print("accuracy score: ", accuracy_score(y_test, predictions))