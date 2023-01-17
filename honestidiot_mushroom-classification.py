import seaborn as sns

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/mushrooms.csv")
df.head()
df.describe()
le = LabelEncoder()

for col in df.columns:

    df[col] = le.fit_transform(df[col])
df.drop('veil-type', axis=1, inplace=True)
def heat_map(df):

    corr = df.corr()

    cmap = sns.diverging_palette(20, 250, as_cmap=True)

    __, ax = plt.subplots(figsize=(20, 20))

    __ = sns.heatmap(

        corr,

        ax=ax,

        cmap=cmap,

        cbar_kws={'shrink': 0.6},

        annot=True,

        annot_kws={'fontsize': 12},

        square=True

    )
heat_map(df)
x = df.drop('class', axis=1)

y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = SVC(kernel='linear', C=5)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(clf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
dup1 = df.drop('gill-attachment', axis=1)
heat_map(dup1)
x = df.drop('class', axis=1)

y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = SVC(kernel='linear', C=5)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(clf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
except_top5 = list(df.columns)

except_top5.remove('gill-size')

except_top5.remove('gill-color')

except_top5.remove('bruises')

except_top5.remove('ring-type')

except_top5.remove('stalk-root')
x = df.drop(except_top5, axis=1)

y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = SVC(kernel='linear', C=5)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(clf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
clf = SVC(kernel='rbf', C=3)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(clf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
except_top6 = list(df.columns)

except_top6.remove('gill-size')

except_top6.remove('gill-color')

except_top6.remove('bruises')

except_top6.remove('ring-type')

except_top6.remove('stalk-root')

except_top6.remove('gill-spacing')
x = df.drop(except_top6, axis=1)

y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = SVC(kernel='linear', C=1)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(clf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
clf = SVC(kernel='rbf', C=3)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(clf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))