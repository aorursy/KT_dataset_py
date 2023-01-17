# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
processed_data = pd.read_csv('/kaggle/input/ufcdata/preprocessed_data.csv')
processed_data.head()
processed_data.B_draw.value_counts()
#Since B_draw and R_draw value is 0 for each row so  we can drop it .
processed_data.drop(columns = ['B_draw','R_draw'], inplace = True)
processed_data['Winner']=processed_data.Winner.astype('category').cat.codes
corr_matrix = processed_data.corr().abs()
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))
#as you can see correlation is not that much vary so we have to use other method for dimensionality reduction
processed_data.shape
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_data.drop(columns='Winner'), processed_data['Winner'], test_size=0.1, random_state=43)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=1, max_depth=10)

model.fit(X_train,y_train)
features = processed_data.columns

importances = model.feature_importances_

indices = np.argsort(importances)[-49:]  # top 30 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()

features
new_features = [features[i] for i in indices]

new_features
X_train = X_train[['B_avg_opp_LEG_att',

 'B_avg_LEG_att',

 'B_avg_opp_GROUND_att',

 'B_avg_opp_TD_pct',

 'R_avg_opp_TD_pct',

 'R_Height_cms',

 'R_avg_DISTANCE_att',

 'B_avg_DISTANCE_landed',

 'B_avg_opp_DISTANCE_landed',

 'B_avg_opp_REV',

 'R_avg_SIG_STR_att',

 'R_total_rounds_fought',

 'R_avg_opp_TOTAL_STR_att',

 'B_avg_opp_BODY_landed',

 'B_avg_opp_HEAD_att',

 'R_avg_opp_GROUND_landed',

 'B_Height_cms',

 'R_avg_opp_TOTAL_STR_landed',

 'B_current_win_streak',

 'B_avg_TD_pct',

 'R_avg_KD',

 'R_avg_opp_DISTANCE_landed',

 'R_avg_SIG_STR_landed',

 'R_avg_opp_SIG_STR_landed',

 'R_avg_opp_REV',

 'R_avg_GROUND_att',

 'B_avg_HEAD_att',

 'B_avg_opp_SIG_STR_landed',

 'R_avg_opp_GROUND_att',

 'R_current_win_streak',

 'R_avg_opp_CLINCH_landed',

 'B_avg_KD',

 'B_losses',

 'B_avg_opp_CLINCH_att',

 'B_avg_SIG_STR_landed',

 'R_avg_GROUND_landed',

 'R_avg_opp_KD',

 'B_avg_opp_GROUND_landed',

 'B_avg_SIG_STR_att',

 'B_avg_GROUND_landed',

 'R_avg_opp_LEG_landed',

 'B_avg_opp_DISTANCE_att',

 'R_avg_opp_DISTANCE_att',

 'R_Weight_lbs',

 'R_avg_opp_HEAD_att',

 'B_avg_DISTANCE_att',

 'B_age',

 'R_avg_opp_SIG_STR_att',

 'B_avg_CLINCH_landed']]
X_test = X_test[['B_avg_opp_LEG_att',

 'B_avg_LEG_att',

 'B_avg_opp_GROUND_att',

 'B_avg_opp_TD_pct',

 'R_avg_opp_TD_pct',

 'R_Height_cms',

 'R_avg_DISTANCE_att',

 'B_avg_DISTANCE_landed',

 'B_avg_opp_DISTANCE_landed',

 'B_avg_opp_REV',

 'R_avg_SIG_STR_att',

 'R_total_rounds_fought',

 'R_avg_opp_TOTAL_STR_att',

 'B_avg_opp_BODY_landed',

 'B_avg_opp_HEAD_att',

 'R_avg_opp_GROUND_landed',

 'B_Height_cms',

 'R_avg_opp_TOTAL_STR_landed',

 'B_current_win_streak',

 'B_avg_TD_pct',

 'R_avg_KD',

 'R_avg_opp_DISTANCE_landed',

 'R_avg_SIG_STR_landed',

 'R_avg_opp_SIG_STR_landed',

 'R_avg_opp_REV',

 'R_avg_GROUND_att',

 'B_avg_HEAD_att',

 'B_avg_opp_SIG_STR_landed',

 'R_avg_opp_GROUND_att',

 'R_current_win_streak',

 'R_avg_opp_CLINCH_landed',

 'B_avg_KD',

 'B_losses',

 'B_avg_opp_CLINCH_att',

 'B_avg_SIG_STR_landed',

 'R_avg_GROUND_landed',

 'R_avg_opp_KD',

 'B_avg_opp_GROUND_landed',

 'B_avg_SIG_STR_att',

 'B_avg_GROUND_landed',

 'R_avg_opp_LEG_landed',

 'B_avg_opp_DISTANCE_att',

 'R_avg_opp_DISTANCE_att',

 'R_Weight_lbs',

 'R_avg_opp_HEAD_att',

 'B_avg_DISTANCE_att',

 'B_age',

 'R_avg_opp_SIG_STR_att',

 'B_avg_CLINCH_landed']]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit (X_train, y_train)

pred = neigh.predict(X_test)

print(accuracy_score(pred, y_test))

print(classification_report(pred, y_test))

print(confusion_matrix(pred, y_test))