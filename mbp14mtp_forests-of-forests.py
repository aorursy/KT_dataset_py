import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
training_data = pd.read_csv('../input/train.csv', index_col='Id')

training_data_dummies = training_data.copy()

training_data_dummies['Cover_Type'] = training_data_dummies['Cover_Type'].astype(str)

training_data_dummies = pd.get_dummies(training_data_dummies)

training_data_dummies.head()
covercols = [x for x in training_data_dummies.columns.values if x.startswith('Cover_Type')]

showcols = ['Elevation', 'Aspect', 'Slope'] + covercols

sns.heatmap(training_data_dummies[showcols].corr(), annot=True)

plt.show()
showcols = ['Horizontal_Distance_To_Hydrology',

            'Vertical_Distance_To_Hydrology',

            'Horizontal_Distance_To_Roadways',

            'Horizontal_Distance_To_Fire_Points'] + covercols

sns.heatmap(training_data_dummies[showcols].corr(), annot=True)

plt.show()
showcols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'] + covercols

sns.heatmap(training_data_dummies[showcols].corr(), annot=True)

plt.show()
soilcols = [x for x in training_data_dummies.columns.values if x.startswith('Soil_Type')]

sns.heatmap(training_data_dummies[soilcols[:5] + covercols].corr(), annot=True)

plt.show()
soilcols = [x for x in training_data_dummies.columns.values if x.startswith('Soil_Type')]

sns.heatmap(training_data_dummies[soilcols[5:10] + covercols].corr(), annot=True)

plt.show()
training_data['Soil_Type7'].describe()
soilcols = [x for x in training_data_dummies.columns.values if x.startswith('Soil_Type')]

sns.heatmap(training_data_dummies[soilcols[10:15] + covercols].corr(), annot=True)

plt.show()
from sklearn.preprocessing import scale



training_data_scaled = training_data.copy()

for col in training_data_scaled.columns.values[:11]:

    training_data_scaled[col] = scale(training_data_scaled[col])

training_data_scaled.head()
from sklearn.model_selection import train_test_split



y = training_data_scaled.pop('Cover_Type').values

X = training_data_scaled.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix



et = ExtraTreesClassifier(n_estimators=200).fit(X_train, y_train)

print(et.score(X_test, y_test))

confusion_matrix(et.predict(X_test), y_test)
from collections import OrderedDict

X_test_proper = pd.read_csv('../input/test.csv')

X_test_ids = X_test_proper.pop('Id')

prediction = pd.DataFrame(OrderedDict((

    ('Id', X_test_ids),

    ('Cover_Type', et.predict(X_test_proper.values)

    ))))

prediction.to_csv('./submission.csv', index=False)

prediction.head()
sns.countplot(training_data['Cover_Type'])

plt.show()

sns.countplot(prediction['Cover_Type'])
prediction.to_csv('./predictions.csv', index=False)

!head ./submission.csv
