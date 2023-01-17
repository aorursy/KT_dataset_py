import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier

diabetes_df = pd.read_csv('../input/diabetes.csv')
diabetes_df.head()
diabetes_df.info()
percentiles = [0.15, 0.3, 0.75, 0.9]
diabetes_df.describe(percentiles = percentiles)
diabetes_df[['Pregnancies', 'Outcome']].groupby(['Pregnancies'], as_index = False).mean().sort_values(by = 'Pregnancies', ascending=False)
fig = plt.figure()
fig.set(alpha = 0.2)

Outcome_1 = diabetes_df.Pregnancies[diabetes_df.Outcome == 1].value_counts()
Outcome_0 = diabetes_df.Pregnancies[diabetes_df.Outcome == 0].value_counts()

df = pd.DataFrame({'ill':Outcome_1, 'normal':Outcome_0})
df.plot(kind = 'line', stacked = False)
plt.title('The relationship between the number of pregnancies and the illness.')
plt.xlabel('The number of pregnancies')
plt.ylabel('Number of samples')
plt.show()
data = diabetes_df.iloc[:, 0:-1]
target = diabetes_df['Outcome']

X_train_data, X_test_data, y_train_target, y_test_target = train_test_split(data, target, test_size = 0.3, random_state = 4)

randomForest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=0, verbose=0,
            warm_start=False)
# parameters = {
#     'n_estimators' : [10, 15, 20],
#     'criterion' : ["gini", "entropy"],
#     'min_sample_leaf' : [2, 4, 6],
# }

# grid = GridSearchCV(estimator = randomForest, param_grid = parameters, cv = 5)
randomForest.fit(X_train_data, y_train_target)
roc_auc = np.mean(cross_val_score(randomForest, X_test_data, y_test_target, cv=5, scoring='roc_auc'))
print('roc_auc:{}', np.around(roc_auc, decimals = 4))
