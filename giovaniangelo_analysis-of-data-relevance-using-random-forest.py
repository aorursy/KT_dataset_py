import warnings



warnings.filterwarnings('ignore')



import time

import numpy as np

import pandas as pd



import seaborn as sb

from sklearn import preprocessing

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split







plt.rcParams['figure.figsize'] = [12, 8]

plt.rcParams['figure.dpi'] = 100
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

print(df.columns)

print(df.info())
df.Class.hist()

df.Class.value_counts()
df.loc[:,'Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()



coef = []

for i in range(200):

	df_ones = df.loc[df.Class == 1]

	df_zeros = df.loc[df.Class == 0].sample(df_ones.shape[0])



	df_homogenous = pd.concat([df_ones, df_zeros])

	df_homogenous.reset_index(inplace=True)



	df_input_candidates = df_homogenous[df_homogenous.columns[df_homogenous.columns.map(lambda col: col not in ['Time', 'Class'])]]

	y_target = df_homogenous['Class']



	rng = np.random.RandomState(seed=42)

	random_values = rng.randn(df_input_candidates.shape[0])

	df_input_candidates.loc[:,'random_values'] = random_values



	clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

	clf.fit(df_input_candidates, y_target)

	coef.append(clf.feature_importances_)

df_coefs = pd.DataFrame(np.array(coef),columns=df_input_candidates.columns)

df_coefs.drop(columns=['index'], inplace=True)



tree_feature_importances = df_coefs.mean().values

sorted_idx = tree_feature_importances.argsort()



y_ticks = np.arange(0, len(df_coefs.columns))

fig, ax = plt.subplots()

ax.barh(y_ticks, tree_feature_importances[sorted_idx])

ax.set_yticklabels(df_coefs.columns[sorted_idx].to_list())

ax.set_yticks(y_ticks)

ax.set_title("Random Forest Feature Importances (MDI)")

fig.tight_layout()

plt.axvline(df_coefs.mean().mean(), 0, 1, linestyle='--', color='red')

plt.show()
df_input_candidates.drop(columns=['index'], inplace=True, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(df_input_candidates, y_target, test_size=0.33, random_state=42)



#USING ONLY THE BEST FEATURES

begin_time = time.time()



columns_selected = df_input_candidates.columns[tree_feature_importances > tree_feature_importances.mean()]



clf_with_filtered_data = LogisticRegression(solver='liblinear')

clf_with_filtered_data.fit(X_train[columns_selected], y_train)



elapse_time_with_rf = time.time() - begin_time



#USING ALL FEATURES



begin_time = time.time()



clf_full_data = LogisticRegression(solver='liblinear')

clf_full_data.fit(X_train, y_train)



elapse_time_without_rf = time.time() - begin_time
from sklearn.metrics import accuracy_score

df_Xtest = pd.DataFrame(X_test, columns = df_input_candidates.columns)

X_test_selected = df_Xtest[columns_selected].values



print(f'Accuracy: {round(100*accuracy_score(y_test, clf_with_filtered_data.predict(X_test_selected)),2)}% and time lapse: {elapse_time_with_rf}')

print(f'Accuracy: {round(100*accuracy_score(y_test, clf_full_data.predict(X_test)),2)}% and time lapse: {elapse_time_without_rf}')