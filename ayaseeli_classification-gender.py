import numpy as np

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
voice = pd.read_csv("../input/voice.csv")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



X, y = voice.iloc[:, :-1].values, voice.iloc[:, -1].values
gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline



pipe_svc = Pipeline([('std_scl', StandardScaler()), 

                    ('pca', PCA(n_components=10)),

                    ('svc', SVC(random_state=1))])



pipe_svc.fit(X_train, y_train)



print('Test Accuracy: %.3f' % pipe_svc.score(X_test, y_test))
from sklearn.model_selection import cross_val_score



scores = cross_val_score(estimator=pipe_svc,

                        X=X_train,

                        y=y_train,

                        cv=10,

                        n_jobs=1)



print('Cross validation scores: %s' % scores)



import matplotlib.pyplot as plt

plt.title('Cross validation scores')

plt.scatter(np.arange(len(scores)), scores)

plt.axhline(y=np.mean(scores), color='g') # Mean value of cross validation scores

plt.show()