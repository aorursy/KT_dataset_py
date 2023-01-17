import numpy as np

import pandas as pd

from sklearn.ensemble import  GradientBoostingClassifier
df = pd.read_csv("../input/spotifyclassification/data.csv")
df.describe()
df.head()
print(df.columns.values)
# 2. Features & Target --------------------------------------------------------

features = ['acousticness', 'danceability', 'duration_ms', 'energy',

       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',

       'speechiness', 'tempo', 'time_signature', 'valence',]



target = ['target']
print("Correlation with the target. \nThe stars(*) are indicates of song feature preference.")

for feat in features:

    correlation = round(df[feat].corr(df[target[0]]),3)

    rating = int(correlation/0.03)

    print("   ",feat," "*(20-len(feat)),correlation, "  ","*"*rating)    
# 2. Train & Test --------------------------------------------------------

train=df.sample(frac=0.8,random_state=10) #random state is a seed value

test=df.drop(train.index)



# train_X, train_y

train_X = train[features]

train_y = train[target]



# test_X, test_y

test_X = test[features]

test_y = test[target]
# 4. Create model -------------------------------------------------------

model = GradientBoostingClassifier(n_estimators = 15, max_features = None, min_samples_split = 2)

model.fit(train_X, train_y.values.ravel())



# evaluate the model on Training data

accuracy = model.score(train_X, train_y)

print('    Training Accuracy:    ' + str(round(accuracy,3)) + '%')



# evaluate the model on Test data

accuracy = model.score(test_X, test_y)

print('    Test Accuracy:  ' + str(round(accuracy,3)) + '%')