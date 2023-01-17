import pandas as pd

import numpy  as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score
csv_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter01/Dataset/dataset_44_spambase.csv'
features = pd.read_csv(csv_url)
features.head()
np.shape(features)
target = features.pop('class')
print(np.shape(features))
print(np.shape(target))
seed = 123
rf_model = RandomForestClassifier(random_state=seed)
rf_model.fit(features, target)
preds = rf_model.predict(features)
np.shape(preds)
accuracy_score(target, preds)