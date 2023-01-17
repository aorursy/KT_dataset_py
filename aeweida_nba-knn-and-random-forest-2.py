import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
#save file path
main_file_path = '../input/NBA 2017-2018 Data.csv'
#read data into DataFrame
data = pd.read_csv(main_file_path)
data.head()
data.tail()
data.columns = ['TEAM', 'DATE', 'HOMEADV', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PlusMinus']
data.head()
data.tail()
data.WL.replace("(W)", 1, regex=True, inplace=True)
data.WL.replace("(L)", 0, regex=True, inplace=True)

data.HOMEADV.replace("(@)", 0, regex=True, inplace=True)
data.HOMEADV.replace("(vs)", 1, regex=True, inplace=True)
data.tail()
data.isnull().sum()
data.shape
#summarize data
data.describe()
features = data[['HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']].values
target = data['WL'].values
# cross validation function with average score
def cross_validate(features, target, classifier, k_fold, r_state=None) :

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(features), n_folds=k_fold,
                           shuffle=True, random_state=r_state)

    k_score_total = 0
    
    # for each training and testing slices run the classifier, and score the results
    for train_indices, test_indices in k_fold_indices :

        model = classifier.fit(features[train_indices],
                           target[train_indices])

        k_score = model.score(features[test_indices],
                              target[test_indices])

        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold
cross_validate(features, target, KNeighborsClassifier(3), 10, 0)
for k in range(1,120,5):
    model = KNeighborsClassifier(k, weights='distance')
    print (cross_validate(features, target, model, 10, 0))
model = RandomForestClassifier(random_state=0)
cross_validate(features, target, model, 10, 0)
print (model.fit(features,target).feature_importances_)
confusion_matrix(y_true, y_pred)
conf = sklearn.metrics.confusion_matrix(y_true, y_pred)
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()
features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.20, random_state=0)
model = RandomForestClassifier(random_state=0).fit(features_train,target_train)
predictions =  model.predict(features_test)
predictions
confusion_matrix(target_test, predictions)
conf = confusion_matrix(target_test, predictions)
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()
