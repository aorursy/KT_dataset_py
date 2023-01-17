import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
# Make sure that unnecessary  warnings are avoided.
# Thanks to https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
# We use a pipeline to make things easire
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
svc_clf = Pipeline([('vect', TfidfVectorizer()),
                    ('transformer', TfidfTransformer()),
                    ('classify', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                               random_state=0, max_iter=5, tol=None))])
svc_clf.fit(train_data['Name'][:700], train_data['Survived'][:700])
predictions = svc_clf.predict(train_data['Name'][700:]) 
survived_or_not = train_data['Survived'][700:]
np.mean(predictions == survived_or_not)
parameters = {'vect__ngram_range' : [(1, 1), (2, 2), (3 , 3)],
              'transformer__use_idf' : (True, False),
              'classify__alpha' : (1e-2, 1e-3),
              }
gs_clf = GridSearchCV(svc_clf, parameters, n_jobs=-1)
gs_clf.fit(train_data['Name'], train_data['Survived'])
cv_result = pd.DataFrame(gs_clf.cv_results_)
gs_clf.best_params_
# We use a pipeline to make things easire
from sklearn.linear_model import SGDClassifier
best_model_svc = Pipeline([('vect', TfidfVectorizer()),
                           ('transformer', TfidfTransformer(use_idf=False)),
                           ('classify', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                                      random_state=0, max_iter=5, tol=None))])
best_model_svc.fit(train_data['Name'], train_data['Survived'])
# Generate out of sample predictions
predictions = best_model_svc.predict(test_data['Name'])
test_data['Predictions'] = predictions
kaggle_data = test_data[['PassengerId', 'Predictions']].copy()
kaggle_data.rename(columns={'Predictions' : 'Survived'}, inplace=True)
kaggle_data.sort_values(by=['PassengerId']).to_csv('kaggle_out_svc_names.csv', index=False)