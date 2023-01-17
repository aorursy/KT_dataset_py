import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/mushrooms.csv')

df.head(2)
le = LabelEncoder()



features = list(df.columns.values)



for i in features:

    df[i] = le.fit_transform(df[i])



features.remove('class')



X = df[features]

y = le.fit_transform(df['class'])



df.head(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)



print("X_train = {}, y_train = {}".format(X_train.shape, y_train.shape))

print("X_test = {}, y_test = {}".format(X_test.shape, y_test.shape))
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=X_train.shape[1])]



# Build 3 layer DNN with 10, 20, 10 units respectively.

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,

                                            hidden_units=[10, 20, 10],

                                            n_classes=2)
wrap = classifier.fit(X_train, y_train, batch_size=100, steps=2000)
score = metrics.accuracy_score(y_test, list(classifier.predict(X_test)))

print('Accuracy: {0:f}'.format(score))
df = df.drop('class', 1)



sns.set(font_scale=0.6)



correlation = df.corr()

plt.figure(figsize=(10,7))

sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='viridis', fmt='.1f')



wrap = plt.title('Correlation between different features')