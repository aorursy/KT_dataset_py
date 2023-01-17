import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import ExtraTreesClassifier

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from scipy import stats

import warnings



warnings.filterwarnings('ignore')

%matplotlib inline
bcell_data = pd.read_csv(r'/kaggle/input/epitope-prediction/input_bcell.csv')

sars_data = pd.read_csv(r'/kaggle/input/epitope-prediction/input_sars.csv')
bcell_data.head()
bcell_data['peptide_length'] = bcell_data.end_position - bcell_data.start_position 

sars_data['peptide_length'] = sars_data.end_position - sars_data.start_position



bcell_data['protein_length'] = bcell_data['protein_seq'].astype(str).map(len)

sars_data['protein_length'] = sars_data['protein_seq'].astype(str).map(len)



bcell_data['parent_protein_length'] = bcell_data['parent_protein_id'].astype(str).map(len)

sars_data['parent_protein_length'] = sars_data['parent_protein_id'].astype(str).map(len)



bcell_data['peptide_position_inprotein'] = bcell_data['start_position'] / bcell_data['protein_seq'].astype(str).map(len)

sars_data['peptide_position_inprotein'] = sars_data['start_position'] / sars_data['protein_seq'].astype(str).map(len)
bcell_data_train, bcell_data_test = train_test_split(bcell_data, test_size=0.2, random_state=None)
features = ['chou_fasman', 'kolaskar_tongaonkar', 'parker',

       'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability',

       'peptide_length']
X_train = bcell_data_train[features]

y_train = bcell_data_train.target

X_val = bcell_data_test[features]

y_val = bcell_data_test.target



sars_X = sars_data[features]

sars_y = sars_data.target
params = {

    "n_estimators" : 1000,

    "n_jobs" : 4,

    "verbose" : 1,

    "criterion" : "entropy",

    "random_state" : None,

    "min_samples_split" : 8,

    "min_weight_fraction_leaf" : 0.0,

    "max_features" : "sqrt",

    "bootstrap" : True,

    "oob_score" : True,

    "class_weight" : "balanced"

}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)
bcell_predictions = model.predict(X_val)

sars_predictions = model.predict(sars_data[features])
#confusion matrix

plt.figure(figsize = (10,10))

cm = confusion_matrix(y_val, bcell_predictions)

sns.heatmap(cm,cmap= "Blues", linecolor = 'black', linewidth = 1, annot = True, fmt='', 

            xticklabels = ['False', 'True'], yticklabels = ['False', 'True'])

plt.xlabel("Predicted")

plt.ylabel("Actual")
#confusion matrix

plt.figure(figsize = (10,10))

cm = confusion_matrix(sars_data.target, sars_predictions)

sns.heatmap(cm,cmap= "Blues", linecolor = 'black', linewidth = 1, annot = True, fmt='', 

            xticklabels = ['False', 'True'], yticklabels = ['False', 'True'])

plt.xlabel("Predicted")

plt.ylabel("Actual")
sars_data['status'] = 'NoStatus'

sars_data['status'][(sars_data.target == sars_predictions) & (sars_data.target == 1)] = 'True_Positive'

sars_data['status'][(sars_data.target != sars_predictions) & (sars_data.target == 1)] = 'False_Positive' 

sars_data['status'][(sars_data.target == sars_predictions) & (sars_data.target == 0)] = 'True_Negative'

sars_data['status'][(sars_data.target != sars_predictions) & (sars_data.target == 0)] = 'False_Negative'



confusionMatrixCols = ['True_Positive', 'False_Negative', 'True_Negative', 'False_Positive']
#Some features are excluded here because all values are the same.



for feature in features:

    if sars_data[feature].min() != sars_data[feature].max():

        try: 

            plt.figure(figsize=(16,5))



            for cm in confusionMatrixCols:

                

                subset = sars_data[sars_data['status'] == cm]



                sns.distplot(subset[feature], hist = False, kde = True,

                             kde_kws = {'linewidth': 2},

                             label = cm)

                

            plt.legend(prop={'size': 16}, title = 'status')

            plt.xlabel(feature)

            plt.ylabel('Density')

        except:

            pass
print(f"Bcell prediction accuracy score: {accuracy_score(bcell_predictions, y_val)}")

print(f"SARS prediction accuracy score: {accuracy_score(sars_predictions, sars_data.target)}")