import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline 
import warnings

warnings.filterwarnings("ignore")
sns.set_palette("bright")
data = pd.read_csv("../input/amazon_alexa.tsv", sep="\t")
data.head()
data.columns
data['rating'].unique()
type(data['date'][0]) , data['date'][0]
data['date'] = pd.to_datetime(data['date'])

data['date'][0]
dates = data['date']

only_dates = []

for date in dates:

    only_dates.append(date.date())



data['only_dates'] = only_dates

data['only_dates'][0]
only_year = []

for date in dates:

    only_year.append(date.year)

data['year'] = only_year





only_month = []

for date in dates:

    only_month.append(date.month)

data['month'] = only_month



# 1 -> monday

# 7 -> sunday

only_weekday = []

for date in dates:

    only_weekday.append(date.isoweekday())

data['day_of_week'] = only_weekday
reviews = data['verified_reviews']

len_review = []

for review in reviews:

    len_review.append(len(review))



data['len_of_reviews'] = len_review
data['len_of_reviews'][0], data['verified_reviews'][0]
data.columns
plt.figure(figsize=(15,7))

plt.bar(height = data.groupby('rating').count()['date'], x = sorted(data['rating'].unique(), reverse= False))

plt.xlabel("Ratings")

plt.ylabel("Count")

plt.title("Count of Ratings")

plt.show()
plt.figure(figsize=(15,7))

sns.countplot(x="rating", hue="feedback", data=data)

plt.show()
plt.figure(figsize=(15,7))

sns.barplot(x="rating", y="variation", hue="feedback", data=data, estimator= sum, ci = None)

plt.show()
plt.figure(figsize=(15,7))

sns.barplot(x="rating", y="variation", hue="feedback", data=data, ci = None)

plt.show()
plt.figure(figsize=(15,7))

sns.barplot(y="rating", x="month", hue="feedback", data=data, ci = None, estimator= sum)

plt.show()
plt.figure(figsize=(15,7))

sns.barplot(y="rating", x="month", hue="feedback", data=data, ci = None)

plt.show()
plt.figure(figsize=(15,7))

sns.countplot(x="day_of_week", hue="feedback", data=data)

plt.show()
plt.figure(figsize=(15,7))

sns.barplot(y="rating", x="day_of_week", hue="feedback", data=data, ci = None)

plt.show()
plt.figure(figsize=(15,7))

sns.countplot(x="feedback", data=data)

plt.show()
plt.figure(figsize=(15,7))

sns.distplot(data[data['feedback'] == 0]['len_of_reviews'], label = 'Feedback - 0')

sns.distplot(data[data['feedback'] == 1]['len_of_reviews'], label = 'Feedback - 1')

plt.legend()

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
tdf = TfidfVectorizer(stop_words='english')
pd.DataFrame(tdf.fit_transform(data['verified_reviews']).toarray())
tdf_data = pd.DataFrame(tdf.fit_transform(data['verified_reviews']).toarray())
pd.get_dummies(data['variation'], drop_first= True)
one_hot_data = pd.get_dummies(data['variation'])
X = pd.concat([one_hot_data, tdf_data, data['month'], data['day_of_week'], data['len_of_reviews']], axis=1)
X.head()
y = data['feedback']
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score



rf = RandomForestClassifier()



k_fold = KFold(n_splits=5)



cross_val_score(rf, X, y, cv=k_fold, scoring='accuracy')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
rf = RandomForestClassifier()

fit_model = rf.fit(X_train, y_train)
t = zip(fit_model.feature_importances_, X_train.columns)

t1 = reversed(sorted(t , key=lambda x: x[0]))

i = 0

for element in t1:

    if (i < 10):

        print(element)

        i = i + 1
y_pred = rf.predict(X_test)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)



train_scores = np.mean(train_scores, axis = 1)

test_scores = np.mean(test_scores, axis = 1)



plt.plot(train_sizes, train_scores, 'o-', label="Training score")

plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")

plt.legend();
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
print("==============================================")

print("For Random Forest Classifier:\n")

print("Accuracy Score: ",accuracy_score(y_test, y_pred))

print("Precision Score: ",precision_score(y_test, y_pred))

print("Recall Score: ",recall_score(y_test, y_pred))

print("F1 Score: ",f1_score(y_test, y_pred))

print("Confusion Matrix:\t \n",confusion_matrix(y_test, y_pred))



print("==============================================")
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print("==============================================")

print("For Gradient Boosting Classifier:\n")

print("Accuracy Score: ",accuracy_score(y_test, y_pred))

print("Precision Score: ",precision_score(y_test, y_pred))

print("Recall Score: ",recall_score(y_test, y_pred))

print("F1 Score: ",f1_score(y_test, y_pred))

print("Confusion Matrix:\t \n",confusion_matrix(y_test, y_pred))

print("==============================================")
results = pd.DataFrame(data = {'Y Test': y_test, 'Y Predictions': y_pred})
results.head()
results.to_csv('Results.csv')