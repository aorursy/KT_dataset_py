!pip install mlens
import numpy as np

import pandas as pd



import os



from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from mlens.ensemble import SuperLearner

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from mlens.metrics import make_scorer



import matplotlib.pyplot as plt

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read in the data 

train_path = '/kaggle/input/learn-together/train.csv'

test_path = '/kaggle/input/learn-together/test.csv'



X = pd.read_csv(train_path, index_col="Id")

X_test = pd.read_csv(test_path, index_col="Id")



X["Euc_Dist_to_Hyd"] = np.sqrt((X["Horizontal_Distance_To_Hydrology"].pow(2)+X["Vertical_Distance_To_Hydrology"].pow(2)))

X_test["Euc_Dist_to_Hyd"] = np.sqrt((X_test["Horizontal_Distance_To_Hydrology"].pow(2)+X_test["Vertical_Distance_To_Hydrology"].pow(2)))



X_test.head()
def plot_pretty_corr(df):

    plt.figure(figsize=(12, 12))

    corr = df.corr()

    ax = sns.heatmap(corr,

                    vmin=-1,

                    vmax=1,

                    center=0,

                    cmap=sns.diverging_palette(20, 220, n=200),

        square=True

    )

    ax.set_xticklabels(

        ax.get_xticklabels(),

        rotation=45,

        horizontalalignment='right'

    );
non_binary_cols = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Cover_Type', 'Euc_Dist_to_Hyd']

plot_pretty_corr(X[non_binary_cols])
# There are no missing values in the training set, all categorical fields have already been transformed

# We can go ahead and separate target from predictors

y = X.Cover_Type

X.drop(['Cover_Type', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'], axis=1, inplace=True)

X_test.drop(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'], axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
y.value_counts()
# def get_score(n_estimators):

#     """Return the average MAE over 3 CV folds of random forest model.

#     """

#     my_pipeline = Pipeline(steps=[

#     ('model', XGBClassifier(n_estimators=n_estimators, random_state=0, learning_rate=0.05))])

#     scores = cross_val_score(my_pipeline, X, y,

#                               cv=3,

#                               scoring='accuracy')



#     print("Average Accuracy:", scores.mean())

#     return scores.mean()
# n_estimators_param = list(range(200, 1000, 200))

# scores = [get_score(n) for n in n_estimators_param]

# results = dict(zip(n_estimators_param, scores))

# print(results)
# XGBoost

# parameters

# n_estimators=1000

# learning_rate=0.05

my_model = XGBClassifier(random_state=0, n_estimators=200, learning_rate=0.05)

my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)

accuracy_score(y_valid, predictions)
# Train on entire dataset for submission

my_model = XGBClassifier(random_state=1, n_estimators=300, learning_rate=0.05)

my_model.fit(X, y)
# Train on entire dataset for submission

my_model = XGBClassifier(random_state=1, n_estimators=300, learning_rate=0.05)

my_model.fit(X, y)
# Train on entire dataset for submission

my_model = XGBClassifier(random_state=1, n_estimators=300, learning_rate=0.05)

my_model.fit(X, y)
# Save predictions to a csv for submission

preds_test = my_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv', index=False)
def feature_importances(my_model, X, y, figsize=(18, 6)):

    

    importances = pd.DataFrame({'Features': X.columns, 

                                'Importances': my_model.feature_importances_})

    

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



    fig = plt.figure(figsize=figsize)

    sns.barplot(x='Features', y='Importances', data=importances)

    plt.xticks(rotation='vertical')

    plt.show()

    

feature_importances(my_model, X, y)
seed = 2017

np.random.seed(seed)

# --- Build ---

# Passing a scoring function will create cv scores during fitting

# the scorer should be a simple function accepting to vectors and returning a scalar

ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)



# Build the first layer

ensemble.add([RandomForestClassifier(random_state=seed), LogisticRegression(), GaussianNB(), KNeighborsClassifier()])



# Build the second layer

ensemble.add([LogisticRegression(), SVC()])



# Attach the final meta estimator

ensemble.add_meta(SVC())



# --- Use ---



# Fit ensemble

ensemble.fit(X_train, y_train)



# Predict

preds = ensemble.predict(X_valid)
print("Fit data:\n%r" % ensemble.data)
accuracy_scorer = make_scorer(accuracy_score, greater_is_better=True)
# Fit ensemble

ensemble.fit(X, y)



# Predict

preds = ensemble.predict(X_test)



output = pd.DataFrame({'Id': X_test.index,

                       'Cover_Type': preds})

output['Cover_Type'] = output['Cover_Type'].astype(int)

output.to_csv('submission.csv', index=False)
print(output)