%load_ext autoreload

%autoreload 2



%matplotlib inline
# Run to sort out dependency issues with Tensorflow

!pip install tensorflow

!pip install imblearn

# Test dependencies

from imblearn.over_sampling import SMOTENC
from pathlib import Path



import pandas as pd

import numpy as np



import seaborn as sns

from pandas_summary import DataFrameSummary



import matplotlib.pyplot as plt

plt.style.use('bmh')
from pathlib import Path

import pandas as pd

import numpy as np
data_path = Path('/kaggle/input/odfpresenceabsenceinvasivespecies/kaggle-dnn-invasive-species/'); data_path
# data

absence_data = pd.read_csv(Path(data_path, "Absence.csv"),

                           sep=';', index_col=0)

presence_data = pd.read_csv(Path(data_path,

                                "Presence.csv"), sep=';', index_col=0)



absence_data = absence_data[presence_data.columns]

assert(np.sum(absence_data.columns == presence_data.columns) == len(presence_data.columns))
# label the data

absence_data['obs'] = 0; presence_data['obs'] = 1
data = absence_data.append(presence_data).reset_index()

data = data[~data.eq(-9999.0).any(1)]
data.head(10)
df_summary = DataFrameSummary(data)
# Check for missing data and column types

df_summary.columns_stats
# Plot the distribution of feature by observation value. Try another feature to see how it works!

fig = plt.figure(figsize=(15,10))

sns.boxplot(x='obs', y=data.columns[2], data=data)
from sklearn.model_selection import train_test_split
X = data[[i for i in data.columns if i != 'obs' and i != 'pointid']]; y = data['obs']
y.value_counts()
from imblearn.under_sampling import ClusterCentroids

from imblearn.over_sampling import SMOTENC, SMOTE

from imblearn.ensemble import BalancedRandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rus = SMOTENC(categorical_features=[2], random_state=42)

X_train, y_train = rus.fit_resample(X_train, y_train)
y_train.value_counts()
train_ids = X_train.index

test_ids = X_test.reset_index().index

print(len(train_ids), len(test_ids))
## Tree Classifiers
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, auc, roc_curve, f1_score, confusion_matrix



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
models = [tree.DecisionTreeClassifier(max_depth=5), RandomForestClassifier(100),

         BalancedRandomForestClassifier(100), GradientBoostingClassifier(n_estimators=100)]
def fit_eval(model, X_train, y_train, X_test, y_test):

    m = model

    m.fit(X_train, y_train)

    pred = m.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)

    cm = confusion_matrix(y_test, pred)

    plot_confusion_matrix(cm, title=f'Model: {type(m).__name__}', target_names=['Absent','Present'], normalize=False)

    return {type(m).__name__: {'model': m, 'AUROC': auc(fpr, tpr), 'F1': f1_score(y_test, pred),  

            'ACCURACY': accuracy_score(y_test, pred)}}
%time results = [fit_eval(_, X_train, y_train, X_test, y_test) for _ in models]; results
#xgboost



import shap



# load JS visualization code to notebook

shap.initjs()



# train model

model = results[1]['RandomForestClassifier']['model']



# explain the model's predictions using SHAP

explainer = shap.TreeExplainer(model, model_output='margin')

shap_values = explainer.shap_values(X_test)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value[1], shap_values[1][1,:], X_test.iloc[1,:])
shap.summary_plot(shap_values, X_train)
# fastai library has a set of utility classes for processing tabular data, such as the csv file we are working with

from fastai.tabular import *

import torch
# specify the independent variable (target)

dep_var = 'obs'

# specify any categorical and continuous variables

cat_names = ['Absence_Substrate']

cont_names = list(set(data.columns) - set(cat_names) - set([dep_var]) - set(['pointid']))

# preprocessing steps removes any missing values, ensures categoricals are not read as continuous

# and normalises the data

procs = [FillMissing, Categorify, Normalize]
# Set up training and test datasets using pre-defined data structure TabularList from fastai
test = TabularList.from_df(X_test.copy(), path=data_path, cat_names=cat_names, cont_names=cont_names)



df = (TabularList.from_df(pd.concat([X_train,y_train],1).reset_index().set_index('index').sample(frac=1),

                          path=data_path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(range(int(0.8*len(X_train)), int(len(X_train))))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch(num_workers=0))
df
# Initialise the metrics that we want to use for evaluation again

f1 = FBeta() # same as f1-score

auroc = AUROC()
# Create a deep neural network with 2 layers, in this case the default of 200 hidden units in the first layer and

# 100 in the second layer. 
learn = tabular_learner(df, layers=[200,100], metrics=[accuracy, f1, auroc], callback_fns=ShowGraph)
# Since this is a classification problem, we use the Cross-Entropy loss function from PyTorch to evaluate the

# model predictions.

learn.loss_func = torch.nn.CrossEntropyLoss()
# View the model architecture

learn.model
#Fastai provides a utility to find an appropriate learning rate to accelerate training
learn.model_dir = '/kaggle/working/models'

learn.lr_find()
g = learn.recorder.plot(suggestion=True, return_fig=True)
# Fit the model based on selected learning rate

learn.fit_one_cycle(5, max_lr=slice(1e-03))



# Analyse our model

learn.recorder.plot_losses()
# Predict our target value

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)
accuracy_score(y_test, labels)
interp = ClassificationInterpretation.from_learner(learn, ds_type=DatasetType.Test)
y_test.value_counts()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, labels)
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
plot_confusion_matrix(cm, title='Confusion matrix: FF Neural Network', 

                      target_names=['Absent','Present'], normalize=False)
from sklearn.metrics import f1_score, roc_auc_score
f1_score(y_test, labels)
roc_auc_score(y_test, labels)