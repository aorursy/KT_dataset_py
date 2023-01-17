# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import numpy as np

train = pd.read_csv("../input/train.csv")
ytrain = train.Cover_Type
train.Id.value_counts().plot(kind='bar')

train.drop(['Cover_Type', 'Id'], inplace=True, axis=1)
print (train.head(5))
print (train.shape)
print (ytrain.head(5))
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=600, learning_rate=1)
classifier.fit(train, ytrain)
test = pd.read_csv("../input/test.csv")
Id = test.Id
test.drop(['Id'], inplace=True, axis=1)
print(test.head(3))
print(test.shape)
predictions = classifier.predict(test)
print(predictions)
submission = pd.DataFrame()
submission['Id'] = Id
submission['Cover_Type'] = predictions
submission.to_csv('submission.csv', index=False)
'''Plotting the important features'''

feature_importances = pd.DataFrame(classifier.feature_importances_,
                                   index = train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)


length = np.arange(len(feature_importances.query('importance > 0.01').index))
plt.barh(length, feature_importances.query('importance > 0.01').importance, align='center', alpha=0.5)
plt.yticks(length, feature_importances.query('importance > 0.01').index)
plt.ylabel('Feature name')
plt.xlabel('Feature importance')
plt.show()
