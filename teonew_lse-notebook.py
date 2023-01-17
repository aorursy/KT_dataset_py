import pandas as pd

import numpy as np



pd.set_option('max_columns', 200)

np.random.seed(42)

%config InlineBackend.figure_format = 'retina'



data = (pd

        .read_csv('../input/hair-salon-no-show-data-set/hair_salon_no_show_wrangled_df.csv')

        .fillna({'book_tod':'Undecided','last_category':'NotApplicable','last_staff':'NotApplicable','last_dow':'NotApplicable', 'last_tod':'NotApplicable'})

        .drop(columns=['Unnamed: 0']))



data.head()
data.head()
import plotly.express as px



px.pie(data, names='noshow', title='How many appointments are No-Shows?')
print(f'This dataset has {data.shape[0]} rows and {data.shape[1]-1} features')
import matplotlib.pyplot as plt

import seaborn as sns



def score_group(y_true, y_pred):

    '''Closer to 0 is better'''

    bookings = len(y_true)

    actual_no_shows = sum(y_true)

    expected_noshows = sum(y_pred)

    actual_arrivals = bookings - actual_no_shows

    predicted_arrivals = bookings - expected_noshows



    return float(predicted_arrivals - actual_arrivals)



def score_predictions(y_true, y_pred, store_size=10, n_samples=1000):

    scores = []

    y_true = pd.Series(y_true)

    y_pred = pd.Series(y_pred)

    

    for i in range(n_samples):

        sample_ids = np.random.randint(low=0, high=len(y_true), size=store_size)

        y_true_sample = np.take(y_true, sample_ids)

        y_pred_sample = np.take(y_pred, sample_ids)

        scores.append(score_group(y_true_sample, y_pred_sample))

        

    ax = sns.boxplot(scores)

    ax.set_title(f'Average: {np.mean(scores):.2f}')

    ax.set_xlabel("Number of extra employees")

    ax.set_xlim(-store_size,store_size)

    plt.show()

    return pd.Series(scores).describe().to_frame().T.round(3)
# Example

y_true = [0,0,0,1,0,0,1,0,0]

y_pred = [0,0,0,0,0,0,0,0,1]



score_predictions(y_true, y_pred)
data['noshow']
# Example

y_true = [0 ,  1,  0, 1, 1,  0,  1,  0,  1]

y_pred = [.4, .4, .2, 0, 0, .1, .7, .3, .9] # Probabilities of not showing up



score_predictions(y_true, y_pred)
features = data.drop(columns=['noshow'])

y_true = data['noshow']
y_pred = np.zeros_like(y_true)



score_predictions(y_true, y_pred)
data.head()
# https://plot.ly/python/parallel-categories-diagram/

px.parallel_categories(data, color='noshow')
# Sample Solution

def predict_on_row(row):

    if row['book_tod'] is 'Undecided':

        return 0.5

    elif row['book_tod'] in ['morning','evening']:

        return 0.05

    else:

        return 0.10

    

y_pred = features.apply(predict_on_row, axis=1)
score_predictions(y_true, y_pred)
# Try writing your own rules



def predict_on_row(row):

    if (row['last_noshow'] is 1) and (row['last_staff'] is 'NotApplicable'):

        return 0.90

    elif (row['last_noshow'] is 0) and (row['last_staff'] is not 'NotApplicable'):

        return 0.05

    else:

        return 0.20

    

y_pred = features.apply(predict_on_row, axis=1)

score_predictions(y_true, y_pred)
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(max_depth=4, random_state=2)



model
# Example

df = pd.DataFrame({

    'Col1':['A','A','B','A'],

    'Col2':['X','Y','Y','Y']

})



df
pd.get_dummies(df)
X = pd.get_dummies(features)

X.head()
from sklearn.model_selection import cross_val_predict



def get_predictions_from_model(model, X, y):

    y_proba = cross_val_predict(model, X, y_true, cv=5, method='predict_proba')

    y_pred = y_proba[:,1]

    return y_pred

    

y_pred = get_predictions_from_model(model, X, y_true)
score_predictions(y_true, y_pred)
from sklearn.tree import plot_tree



def visualize_model(model, X, y):

    model.fit(X, y)

    plt.figure(figsize=(50,10))

    

    plot_tree(model, filled=True, feature_names=X.columns, 

                  class_names=model.classes_.astype(str), 

                  proportion=True, impurity=False, rounded=True)

    

    plt.show()

    

visualize_model(model, X, y_true)


help(DecisionTreeClassifier)
model = DecisionTreeClassifier(max_depth=1, random_state=1)



y_pred = get_predictions_from_model(model, X, y_true)



score_predictions(y_true, y_pred)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



model_lr = LogisticRegression()

model_rfc = RandomForestClassifier(n_estimators=100)

model_knn = KNeighborsClassifier()
# Experiment

import shap

shap.initjs()
model = DecisionTreeClassifier(max_depth=5)

model.fit(X,y_true)



explainer = shap.TreeExplainer(model, data=X)

shap_values =  explainer.shap_values(X=X)[1]



def visualize_row_predictions(ix, shap_values=shap_values, features=X):

    return shap.force_plot(np.mean(shap_values), shap_values[ix,:], feature_names=X.columns, features=features.iloc[ix,:], link='logit')
visualize_row_predictions(2)
visualize_row_predictions(190)