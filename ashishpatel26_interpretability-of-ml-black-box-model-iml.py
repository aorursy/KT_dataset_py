import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')

y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary



feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]

X = data[feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
import eli5

from eli5.sklearn import PermutationImportance
perm = PermutationImportance(my_model, random_state = 1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
!conda install -c conda-forge Skater -y
%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import pandas as pd

# Reference for customizing matplotlib: https://matplotlib.org/users/style_sheets.html

plt.style.use('ggplot')
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, VotingClassifier



from skater.core.explanations import Interpretation

from skater.model import InMemoryModel
data = load_breast_cancer()

# Description of the data

print(data.DESCR)

pd.DataFrame(data.target_names)
X = data.data

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def model_training(X_train, y_train):

    clf1 = LogisticRegression(random_state=1)

    clf2 = RandomForestClassifier(random_state=1)

    clf3 = GaussianNB()

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

    eclf = eclf.fit(X_train, y_train)

    return (clf1,clf2,clf3, eclf)



clf1,clf2,clf3, eclf = model_training(X_train,y_train)
def train_all_model(clf1,clf2,clf3, X_train,y_train):

    clf1 = clf1.fit(X_train, y_train)

    clf2 = clf2.fit(X_train, y_train)

    clf3 = clf3.fit(X_train, y_train)

    models = {'lr':clf1, 'rf':clf2, 'gnb':clf3, 'ensemble':eclf}

    return (clf1,clf2,clf3, models)



clf1,clf2,clf3, models = train_all_model(clf1,clf2,clf3,X_train,y_train)
# Ensemble Classifier does not have feature importance enabled by default

f, axes = plt.subplots(2, 2, figsize = (26, 18))



ax_dict = {'lr':axes[0][0],'rf':axes[1][0],'gnb':axes[0][1],'ensemble':axes[1][1]}

interpreter = Interpretation(X_test, feature_names=data.feature_names)



for model_key in models:

    pyint_model = InMemoryModel(models[model_key].predict_proba, examples=X_test)

    ax = ax_dict[model_key]

    interpreter.feature_importance.plot_feature_importance(pyint_model, ascending=True, ax=ax)

    ax.set_title(model_key)
# Before interpreting, lets check on the accuracy of all the models

from sklearn.metrics import f1_score

for model_key in models:

        print("Model Type: {0} -> F1 Score: {1}".

              format(model_key, f1_score(y_test, models[model_key].predict(X_test))))
%matplotlib inline

X_train = pd.DataFrame(X_train)

X_train.head()
def understanding_interaction():

    pyint_model = InMemoryModel(eclf.predict_proba, examples=X_test, target_names=data.target_names)

    # ['worst area', 'mean perimeter'] --> list(feature_selection.value)

    interpreter.partial_dependence.plot_partial_dependence(list(feature_selection.value),

                                                                    pyint_model, 

                                                                    grid_resolution=grid_resolution.value, 

                                                                    with_variance=True)

        

    # Lets understand interaction using 2-way interaction using the same covariates

    # feature_selection.value --> ('worst area', 'mean perimeter')

    axes_list = interpreter.partial_dependence.plot_partial_dependence([feature_selection.value],

                                                                       pyint_model, 

                                                                       grid_resolution=grid_resolution.value, 

                                                                       with_variance=True)
!conda install ipywidgets --yes
# One could further improve this by setting up an event callback using

# asynchronous widgets

import ipywidgets as widgets

from ipywidgets import Layout

from IPython.display import display

from IPython.display import clear_output

grid_resolution = widgets.IntSlider(description="GR", 

                                    value=10, min=10, max=100)

display(grid_resolution)



# dropdown to select relevant features from the dataset

feature_selection = widgets.SelectMultiple(

    options=tuple(data.feature_names),

    value=['worst area', 'mean perimeter'],

    description='Features',

    layout=widgets.Layout(display="flex", flex_flow='column', align_items = 'stretch'),

    disabled=False,

    multiple=True

)

display(feature_selection)
# Reference: http://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html

button = widgets.Button(description="Generate Interactions")

display(button)



def on_button_clicked(button_func_ref):

    clear_output()

    understanding_interaction()



button.on_click(on_button_clicked)
from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer

from IPython.display import display, HTML, clear_output

int_range = widgets.IntSlider(description="Index Selector", value=9, min=0, max=100)

display(int_range)



def on_value_change(change):

    index = change['new']

    exp = LimeTabularExplainer(X_test, 

                           feature_names=data.feature_names, 

                           discretize_continuous=False, 

                           class_names=['p(Cancer)-malignant', 'p(No Cancer)-benign'])

    print("Model behavior at row: {}".format(index))

    # Lets evaluate the prediction from the model and actual target label

    print("prediction from the model:{}".format(eclf.predict(X_test[index].reshape(1, -1))))

    print("Target Label on the row: {}".format(y_test.reshape(1,-1)[0][index]))

    clear_output()

    display(HTML(exp.explain_instance(X_test[index], models['ensemble'].predict_proba).as_html()))

    

int_range.observe(on_value_change, names='value')
from IPython.display import Image

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import xgboost

import matplotlib.pyplot as plt

from IPython.display import Image
data = pd.read_csv("../input/FIFA 2018 Statistics.csv")

y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=9487)
params = {'base_score': 0.5,

         'booster': 'gbtree',

         'colsample_bylevel': 1,

         'colsample_bytree': 1,

         'gamma': 0,

         'learning_rate': 0.05,

         'max_delta_step': 0,

         'max_depth': 3,

         'min_child_weight': 1,

         'missing': None,

         'n_estimators': 400,

         'n_jobs': 1,

         'objective': 'binary:logistic',

         'random_state': 0,

         'reg_alpha': 0,

         'reg_lambda': 1,

         'scale_pos_weight': 1,

         'seed': 0,

         'silent': True,

         'subsample': 1}



params['eval_metric'] = 'auc'
d_train = xgboost.DMatrix(X_train, y_train)

d_val = xgboost.DMatrix(X_val, y_val)

watchlist = [(d_train, "train"), (d_val, "valid")]



#train model



model = xgboost.train(params, d_train, num_boost_round=2000, evals=watchlist, early_stopping_rounds=100, verbose_eval=10)
# Simple check what model ran out of things

data_for_prediction = xgboost.DMatrix(X_train.iloc[[83],:])  # use 1 row of data here. Could use multiple rows if desired

model.predict(data_for_prediction)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(model)

# Calculate Shap values

shap_values = explainer.shap_values(X_train)

shap.initjs()
shap.summary_plot(shap_values, X_train)
data_for_prediction = xgboost.DMatrix(X_train.iloc[[10],:])  # use 1 row of data here. Could use multiple rows if desired

print(f"The 85th data is predicted to be True's probability: {model.predict(data_for_prediction)}")

shap.force_plot(explainer.expected_value, shap_values[10,:], X_train.iloc[10,:])
data_for_prediction = xgboost.DMatrix(X_train.iloc[[83],:])  # use 1 row of data here. Could use multiple rows if desired

print(f"The 83rd data is predicted to be True's probability: {model.predict(data_for_prediction)}")

shap.force_plot(explainer.expected_value, shap_values[83,:], X_train.iloc[83,:])
plt.figure(figsize=(20,8))

xs = np.linspace(-5,5,100)

plt.xlabel("Log odds of winning")

plt.ylabel("Probability of winning")

plt.title("Log odds & prob of winning convert")

plt.plot(xs, 1/(1+np.exp(-xs)))



new_ticks = np.linspace(-5, 5, 11)

plt.xticks(new_ticks)

plt.show()
shap.force_plot(explainer.expected_value, shap_values, X_train)
shap.dependence_plot('Ball Possession %', shap_values, X_train, interaction_index="Goal Scored")