import pandas as pd

# reading the data

data = pd.read_csv('../input/heart-disease-uci/heart.csv')



# getting the shape

data.shape
#Below code is taken from https://www.kaggle.com/roshansharma/heart-diseases-analysis

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']



data['sex'] = data['sex'].astype('object')

data['chest_pain_type'] = data['chest_pain_type'].astype('object')

data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')

data['rest_ecg'] = data['rest_ecg'].astype('object')

data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')

data['st_slope'] = data['st_slope'].astype('object')

data['thalassemia'] = data['thalassemia'].astype('object')
X = data.drop(['target'], axis = 1)

y = data.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
!pip install lazypredict
from lazypredict.Supervised import LazyClassifier



clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models.to_csv('models.csv')

predictions.to_csv('predictions.csv')
models