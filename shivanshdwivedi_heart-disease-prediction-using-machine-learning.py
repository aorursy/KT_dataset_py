import pandas as pd # load and manipulate data and for One-Hot Encoding

import numpy as np # calculate the mean and standard deviation

import matplotlib.pyplot as plt # drawing graphs

from sklearn.tree import DecisionTreeClassifier # a classification tree

from sklearn.tree import plot_tree # draw a classification tree

from sklearn.model_selection import train_test_split # split  data into training and testing sets

from sklearn.model_selection import cross_val_score # cross validation

from sklearn.metrics import confusion_matrix # creates a confusion matrix

from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/Heart_disease_prediction.csv' , header = None)
dataset.head()
dataset.columns = ['age', 

              'sex', 

              'cp', 

              'restbp', 

              'chol', 

              'fbs', 

              'restecg', 

              'thalach', 

              'exang', 

              'oldpeak', 

              'slope', 

              'ca', 

              'thal', 

              'hd']

dataset.head()
dataset.dtypes
dataset['ca'].unique()
dataset['thal'].unique()
len(dataset.loc[(dataset['ca'] == '?')

                |

                (dataset['thal'] == '?')])

dataset.loc[(dataset['ca'] == '?')

                |

                (dataset['thal'] == '?')]

len(dataset)
df_no_missing = dataset.loc[(dataset['ca'] != '?')

                &

                (dataset['thal'] != '?')]
len(df_no_missing)
df_no_missing['ca'].unique()
df_no_missing['thal'].unique()
X = df_no_missing.drop('hd', axis=1).copy()

X.head()
y = df_no_missing['hd'].copy()

y.head()
X.dtypes
X['cp'].unique()
pd.get_dummies(X , columns=['cp']).head()
X_encoded = pd.get_dummies(X , columns=[ 'cp' , 'restecg', 'slope', 'thal' , 'oldpeak'])

X_encoded.head()
y.unique()
y_not_zero_index = y>0

y[y_not_zero_index] = 1

y.unique()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(random_state=42)

print(clf_dt)

clf_dt = clf_dt.fit(X_train, y_train)
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["Does not have HD", "Has HD"])
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

ccp_alphas = ccp_alphas[:-1]



clf_dts = []

for ccp_alpha in ccp_alphas:

    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

#     print(clf_dt)

    clf_dt.fit(X_train, y_train)

    clf_dts.append(clf_dt)
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]

test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]



fig, ax = plt.subplots()

ax.set_xlabel("alpha")

ax.set_ylabel("accuracy")

ax.set_title("Accuracy vs alpha for training and testing sets")

ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")

ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")

ax.legend()

plt.show()
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)

scores = cross_val_score(clf_dt, X_train, y_train, cv=5)

dataset = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})



dataset.plot(x='tree', y='accuracy', marker='o', linestyle='--')
alpha_loop_values = []

for ccp_alpha in ccp_alphas:

    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)

    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

    

alpha_results = pd.DataFrame(alpha_loop_values, 

                             columns=['alpha', 'mean_accuracy', 'std'])



alpha_results.plot(x='alpha', 

                   y='mean_accuracy', 

                   yerr='std', 

                   marker='o', 

                   linestyle='--')
alpha_results [(alpha_results['alpha'] > 0.014)

              &

              (alpha_results['alpha'] < 0.015)]
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)

                               &

                               (alpha_results['alpha'] < 0.015)]['alpha']

ideal_ccp_alpha
ideal_ccp_alpha = float(ideal_ccp_alpha)

ideal_ccp_alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42, 

                                       ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train) 
plot_confusion_matrix(clf_dt_pruned, 

                      X_test, 

                      y_test, 

                      display_labels=["Does not have HD", "Has HD"])
plt.figure(figsize=(15,7.5))

plot_tree(clf_dt_pruned, 

          filled=True, 

          rounded=True, 

          class_names=["No HD", "Yes HD"], 

          feature_names=X.columns) 



pred = clf_dt_pruned.predict([[ 63 , 0 , 1, 130, 235 , 0 , 0 , 150, 0 , 1 , 0 , 1 , 1]])

print(pred)