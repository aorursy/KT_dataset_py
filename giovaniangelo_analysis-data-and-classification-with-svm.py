import warnings



warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

from tabulate import tabulate

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn import preprocessing

import seaborn as sb

from sklearn.metrics import accuracy_score

import time



from IPython.display import display, HTML



CSS = """

.output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))



def beauty_print(df):

    display(HTML(df.to_html()))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

plt.rcParams['figure.figsize'] = [12, 8]

plt.rcParams['figure.dpi'] = 100 

    
df = pd.read_csv('/kaggle/input/bank-customers-data/BankCustomerData.csv')

beauty_print(df.head(5))



print(df.term_deposit.value_counts())
df.info()
df_filtered = df.replace('unknown',np.nan)

df_filtered.dropna(inplace=True)

df_filtered.reset_index(drop=True, inplace=True)

beauty_print(df_filtered.head(5))

beauty_print(df_filtered.term_deposit.value_counts().to_frame())
df_filtered.loc[:,'balance'] = (df_filtered['balance'] - df_filtered['balance'].mean()) / df_filtered['balance'].std()

df_filtered.loc[:,'duration'] = (df_filtered['duration'] - df_filtered['duration'].mean()) / df_filtered['duration'].std()

df_filtered.loc[:,'pdays'] = (df_filtered['pdays'] - df_filtered['pdays'].mean()) / df_filtered['pdays'].std()
df_input_candidates = df_filtered[df_filtered.columns[df_filtered.columns.map(lambda col: col not in ['term_deposit'])]]

y_target = df_filtered['term_deposit']
le = LabelEncoder()

for col in df_input_candidates.columns[ [i == object for i in df_input_candidates.dtypes] ]:

    df_input_candidates.loc[:,col] = le.fit_transform(df_input_candidates[col])



beauty_print(df_input_candidates.head(5))
coefs = []



for i in range(100):

    df_filtered_yes = df_input_candidates.loc[y_target == 'yes']

    df_filtered_no = df_input_candidates.loc[y_target == 'no'].sample(df_filtered_yes.shape[0])

    

    df_homogenous = pd.concat([df_filtered_no, df_filtered_yes], ignore_index=True)

    

    rng = np.random.RandomState(seed=42)

    random_values = rng.randn(df_input_candidates.shape[0])

    df_input_candidates.loc[:,'random_values'] = random_values



    clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=42)

    clf.fit(df_input_candidates, y_target)

    coefs.append(clf.feature_importances_)
df_coefs = pd.DataFrame(np.array(coefs),columns=df_input_candidates.columns)



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



clf_with_filtered_data = SVC(kernel='rbf')

clf_with_filtered_data.fit(X_train[columns_selected], y_train)



elapse_time_with_rf = time.time() - begin_time



#USING ALL FEATURES



begin_time = time.time()



clf_full_data =  SVC(kernel='rbf')

clf_full_data.fit(X_train, y_train)



elapse_time_without_rf = time.time() - begin_time



df_Xtest = pd.DataFrame(X_test, columns = df_input_candidates.columns)

X_test_selected = df_Xtest[columns_selected].values



print(f'Accuracy: {round(100*accuracy_score(y_test, clf_with_filtered_data.predict(X_test_selected)),2)}% and time lapse: {elapse_time_with_rf}')

print(f'Accuracy: {round(100*accuracy_score(y_test, clf_full_data.predict(X_test)),2)}% and time lapse: {elapse_time_without_rf}')