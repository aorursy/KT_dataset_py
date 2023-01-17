import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/competition" directory.
df = pd.read_csv("../input/predicting-traffic-on-a-bridge/commuters_clf_train.csv")

df.head()
X = df[['weekday', 'hour']]

y = df.crowded
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



clf = KNeighborsClassifier()



param_grid = [

    {'n_neighbors': range(1, 100, 20)}

]



search = GridSearchCV(clf, param_grid, cv=10)



search.fit(X, y)

print("Best parameter (CV score={:.2f}):{})".format(search.best_score_, search.best_params_))
df_test = pd.read_csv("../input/predicting-traffic-on-a-bridge/commuters_clf_test.csv").set_index('index')  # important - setting the index!

df_test.head()
prediction = search.predict(df_test[['weekday', 'hour']])

prediction
# set the prediction column to crowded (see sample-submission csv file)

df_test['crowded'] = prediction
df_test.head()
df_submission = df_test[['crowded']]  # important - only provide the target column

df_submission.reset_index(inplace=True) # important - the column "index" is required for submission



df_submission.to_csv('submission.csv', index=False)  # index=False is important - there shouldn't be another pandas index!

df_submission.head()