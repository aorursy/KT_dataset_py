import pandas as pd



# Data Viz

import seaborn as sns

import matplotlib.pyplot as plt



#ML

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_score, LeaveOneOut
data = pd.read_csv('/kaggle/input/winedataset/WineDataset.csv')

data.head()
# 3 categories

data['Wine'].unique()
data.shape
X = data.drop('Wine', axis=1).values #features
y = data['Wine'].values #labels
trainX, testX, trainy, testy = train_test_split(X, y, random_state=3)
model = GaussianNB()
model.fit(trainX, trainy)
p = model.predict(testX)
accuracy_score(testy, p)
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
#Perfect

scores.mean()
cmap = sns.cm.rocket_r

sns.heatmap(confusion_matrix(testy, p), cmap=cmap, annot=True)



plt.title('Confusion Matrix')



plt.xlabel('Predictions')

plt.ylabel('Real values')



plt.show()