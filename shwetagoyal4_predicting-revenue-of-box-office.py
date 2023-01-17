import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.isnull().sum()
train.describe()
train.describe(include="all")
print(train.shape, test.shape)
train.info()
train.dtypes
from wordcloud import WordCloud



wordcloud = WordCloud().generate(train.title.to_string())



sns.set(rc={'figure.figsize':(12, 8)})

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(train.tagline.to_string())



sns.set(rc={'figure.figsize':(12, 8)})

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(train.overview.to_string())



sns.set(rc={'figure.figsize':(12, 8)})

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
train.original_language.value_counts().plot.bar()
train['runtime'] = train['runtime'].fillna(method='ffill')

test['runtime'] = test['runtime'].fillna(method='ffill')
train['status'] = pd.get_dummies(train['status'])

test['status'] = pd.get_dummies(test['status'])
X = train[['runtime', 'budget','popularity','status']]

y = train.revenue



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state=1)
def rmsle(y, y1):

    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y1))))


from sklearn.ensemble import RandomForestRegressor



RF = RandomForestRegressor(n_estimators=300, n_jobs=-1)

model = RF.fit(X_train, y_train)

y_pred = RF.predict(X_val)

print(rmsle(y_val, y_pred))

# Applying model on test data



X_test = test[['runtime', 'budget', 'popularity', 'status']]

Pred = RF.predict(X_test)
Sub = pd.read_csv("../input/sample_submission.csv")

Sub['revenue'] = Pred

Sub.to_csv('submission.csv', index=False)