#Kaggle's scikit-opt is old, has bugs

!pip install scikit-optimize -U
import numpy as np

import pandas as pd

import seaborn as sns

import shap

import eli5



from matplotlib import pyplot as plt

from tqdm.notebook import tqdm as notebook

from tqdm import tqdm



plt.style.use('ggplot')
shap.initjs();
PATH_TRAIN_CSV = '../input/santander-customer-transaction-prediction/train.csv'

PATH_TEST_CSV = '../input/santander-customer-transaction-prediction/test.csv'



train_df = pd.read_csv(PATH_TRAIN_CSV)

test_df = pd.read_csv(PATH_TEST_CSV)
train_df.head(3)
num_samples = train_df.groupby('target')['target'].count()

num_samples.plot.pie(title='Number of samples per category', autopct="%.0f%%");
correlation_matrix = train_df.iloc[:,2:].corr()

fig, ax = plt.subplots(figsize=(7,7))

sns.heatmap(correlation_matrix, ax=ax);
# Commented for kernel commit speedup

#fig, ax = plt.subplots(2, figsize=(30,10))

#for i in tqdm(range(200)):

#    var_name = f"var_{i}"

#    sns.distplot(train_df[var_name], ax=ax[0], label=var_name)

#    sns.distplot(test_df[var_name], ax=ax[1], label=var_name)

#ax[0].set_title("KDEs of TRAIN variables.")

#ax[1].set_title("KDEs of TEST variables.")
# for example, this ones

variables = ['var_0', 'var_23', 'var_89', 'var_112', 'var_152', 'var_199']

fig, ax = plt.subplots(2,3, figsize=(16,8))

for i in range(len(variables)):

    var_name = variables[i]

    sns.distplot(train_df[var_name], ax=ax[i//3, i%3], label='train')

    sns.distplot(test_df[var_name], ax=ax[i//3, i%3], label='test')

    ax[i//3, i%3].set_title(f"Distribution of variable {var_name}")

    ax[i//3, i%3].legend()
#from IPython.html.widgets import *

from ipywidgets import *
def plot_kde(var_index):

    sns.distplot(train_df.loc[train_df['target'] == 0, 'var_'+str(var_index)], label='0')

    sns.distplot(train_df.loc[train_df['target'] == 1, 'var_'+str(var_index)], label='1')

    plt.legend()

    plt.show()



interact(plot_kde, var_index=np.arange(200));
nuniq_train_true = train_df[train_df['target'] == 1].iloc[:, 2:].nunique(axis=0)

nuniq_train_false = train_df[train_df['target'] == 0].iloc[:, 2:].nunique(axis=0)



nuniq_train_true /= len(train_df[train_df['target'] == 1])

nuniq_train_false /= len(train_df[train_df['target'] == 0])



fig, ax = plt.subplots(figsize=(30,5))

ax.bar(np.arange(start=0, stop=400, step=2), height=nuniq_train_true, color='red', label='1')

ax.bar(np.arange(stop=400, start=1, step=2), height=nuniq_train_false, color='blue', label='0', alpha=.7)

ax.set_title("Number of different values per varibale on each category (normalized)")

ax.legend();
len_true_samples = len(train_df[train_df['target']==1])

unique_ratios = train_df[train_df['target']==1].iloc[:,2:].apply(lambda c: len(c.unique()) / len_true_samples, axis=0)

fig, ax = plt.subplots(figsize=(30,30))

sns.barplot(y=unique_ratios.index.tolist(), x=1-unique_ratios.values, ax=ax)

ax.set_title('Percentage of repeated values for each variable FOR TRUE SAMPLES');
len_false_samples = len(train_df[train_df['target']==0])

unique_ratios = train_df[train_df['target']==0].iloc[:,2:].apply(lambda c: len(c.unique()) / len_false_samples, axis=0)

fig, ax = plt.subplots(figsize=(30,30))

sns.barplot(y=unique_ratios.index.tolist(), x=1-unique_ratios.values, ax=ax)

ax.set_title('Percentage of repeated values for each variable FOR FALSE SAMPLES');
input_names = [f'var_{i}' for i in range(200)]
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train_df, test_size=0.4, random_state=196)
print(f"Train shape: {X_train.shape}")

print(f"Test shape: {X_test.shape}")
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



def metrics_summary(y_real, y_pred):

    """ Returns a  figure with the ROC curve, the Accuracy and AUC metrics (in the figure title)"""

    fpr, tpr, thresholds = roc_curve(y_real, y_pred)

    

    fig, ax = plt.subplots(1, figsize=(5,5))

    ax.plot(fpr, tpr, color='red')

    ax.plot([0,1], [0,1], color='black', linestyle='--')

    ax.set_title('ROC Curve')

    plt.close()

    

    acc = accuracy_score(np.array(y_pred>.5, dtype='int'), y_real)

    auc = roc_auc_score(np.array(y_pred>.5, dtype='int'), y_real)

    

    print(f"- ACC {acc}\n- AUC {auc}")

    

    return fig





def predict_submission(model):

    """ Takes a model and predicts on the test split. Returns the submission DataFrame"""

    preds = model.predict_proba(test_df.drop("ID_code", axis=1))[:,1]

    return pd.DataFrame({'ID_code':test_df['ID_code'], 'target':preds})
import lightgbm as lgbm
bst = lgbm.LGBMClassifier()
%%time

bst.fit(X_train[input_names], X_train['target']);
preds = bst.predict_proba(X_test[input_names])[:,1]

metrics_summary(X_test['target'], preds)
importances = pd.DataFrame({'Feature': input_names, 'Importance':bst.feature_importances_}).sort_values('Importance', ascending=False)



fig, ax = plt.subplots(figsize=(30,30))

sns.barplot(y=importances.Feature, x=importances.Importance, ax=ax)

ax.set_title('Feature importances');
vals = X_train[input_names].sample(1000)

explainer = shap.TreeExplainer(bst, data=vals, model_output='probability', feature_perturbation='interventional')

shap_values = explainer.shap_values(vals)
shap.summary_plot(shap_values, vals)
interact(lambda var: shap.dependence_plot(f"var_{var}", shap_values, vals), var=np.arange(200));
# For example, estimating which features are more important at this particular example

shap.force_plot(explainer.expected_value, shap_values[110,:], vals.iloc[110,:])
var_names = [f'var_{i}' for i in range(200)] 

var_enc_names = [f'var_{i}_freq' for i in range(200)] 
def get_hist_frequencies(dataframe):

    hist_vars = {}

    for v in var_names:

        hist_vars[v] = dataframe[v].value_counts()

    return hist_vars



def encode_freqs(dataframe, return_calc_hist = False):

    """Adds 200 more feature with the frequency encodings of the variables"""

    # Build histogram of frequencies of each variable

    hist_vars = {}

    for v in var_names:

        hist_vars[v] = dataframe[v].value_counts()

        dataframe[v+"_freq"] = dataframe[v].map(hist_vars[v])

    

    if return_calc_hist:

        return dataframe, hist_vars

    return dataframe



def encode_freqs_with_hist(dataframe, histogram_data):

    dataframe = dataframe.copy()

    for v in var_names:

        dataframe[v+"_freq"] = dataframe[v].map(histogram_data[v])

    return dataframe
histogram_vars = get_hist_frequencies(pd.concat([train_df[var_names], test_df[var_names]]))
train_df_encoded = encode_freqs_with_hist(train_df, histogram_vars)
X_train, X_test = train_test_split(train_df_encoded, test_size=0.4, random_state=196)
bst = lgbm.LGBMClassifier()
%%time

bst.fit(X_train[var_names+var_enc_names], X_train['target']);
preds = bst.predict_proba(X_test[var_names+var_enc_names])[:,1]

metrics_summary(X_test['target'], preds)
importances = pd.DataFrame({'Feature': var_names+var_enc_names, 'Importance':bst.feature_importances_}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(30,80))

sns.barplot(y=importances.Feature, x=importances.Importance, ax=ax)

ax.set_title('Feature importances');
from skopt import BayesSearchCV

from skopt.space import Real, Categorical, Integer
search_space = {

    "feature_fraction": Real(0.001, 0.4),

    "max_depth": Integer(10, 25)

}



base_params = {

    "boost_from_average": "false",

    "metric" : "auc",

    "tree_learner": "serial",

}



optimizer_args = {

    'acq_func': "EI",

    'n_initial_points': 15,

    'acq_optimizer': 'sampling'

}





model = lgbm.LGBMClassifier(**base_params)



bayes_search = BayesSearchCV(

    estimator=model,

    search_spaces=search_space,

    n_iter=32,

    cv=3,

    n_jobs=-1,

    scoring='roc_auc',

    optimizer_kwargs=optimizer_args,

    random_state=2343

)



def on_step(optim_result):

    score = bayes_search.best_score_

    print("Best score: %s " % (score,))

    if score >= 0.98:

        print('Interrupting!')

        return True
# disabled for time execution reasons

#%%time

#bayes_search.fit(X_train[var_names+var_enc_names], X_train['target'], callback=on_step);
"""

Best score: 0.8819925481391506 

Best score: 0.8819925481391506 

Best score: 0.8819925481391506 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8824897736221471 

Best score: 0.8826816117256729 

Best score: 0.8826816117256729 

Best score: 0.8826816117256729 

CPU times: user 1min 18s, sys: 26.5 s, total: 1min 45s

Wall time: 19min 5s

"""
#params_model.update(bayes_search.best_params_)

#params_model



# These are the best hyperparams obtained using bayesian optimization

params_model = {

 'boost_from_average': 'false',

 'metric': 'auc',

 'tree_learner': 'serial',

 'feature_fraction': 0.10927610745498884,

 'max_depth': 10,

 'n_estimators': 100

}



model = lgbm.LGBMClassifier(**params_model)
%%time

model.fit(X_train[var_names+var_enc_names], X_train['target']);
preds = model.predict_proba(X_test[var_names+var_enc_names])[:,1]

metrics_summary(X_test['target'], preds)
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator



from scipy.special import logit



class MetaModel(BaseEstimator):

    """ Estimator which contains 200 models that are fitted with each var and var_encoded"""

    def __init__(self):

        super(MetaModel)

        self.models = []

        self.merger = LogisticRegression(solver='lbfgs')

    

    def fit(self, X, y, var_names, var_enc_names):

        #Train boosting models

        for v, venc in notebook(zip(var_names, var_enc_names)):

            model = lgbm.LGBMClassifier(**{

                     'boost_from_average': 'false',

                     'metric': 'auc',

                     'tree_learner': 'serial',

                     'max_depth': 10,

                     'n_estimators': 100

                }

            )

            model.fit(X[[v, venc]], y)

            self.models.append(model)

        

        # Train merger

        preds = self._ensemble_predict(X, var_names, var_enc_names)

        self.merger.fit(preds, y)

        



        return self

    

    def predict(self, X, var_names, var_enc_names):

        predictions = self._ensemble_predict(X, var_names, var_enc_names)

        preds = self.merger.predict_proba(predictions)[:, 1]

    

        return preds

    

    def _ensemble_predict(self, X, var_names, var_enc_names):

        index = 0

        predictions = np.zeros((len(X), len(self.models)))

        for v, venc in notebook(zip(var_names, var_enc_names)):

            model = self.models[index]

            predictions[:, index] = model.predict_proba(X[[v, venc]])[:,1]    

            index+=1

        return predictions
var_names = [f"var_{i}" for i in range(200)]

var_enc_names = [i+"_freq" for i in var_names]



mmodel = MetaModel()
mmodel.fit(X_train, X_train['target'], var_names, var_enc_names)
preds = mmodel.predict(X_test, var_names, var_enc_names)
metrics_summary(X_test['target'], preds)