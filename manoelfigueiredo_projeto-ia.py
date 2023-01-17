import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns #for plotting

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123) #ensure reproducibility



pd.options.mode.chained_assignment = None  #hide any pandas warnings


dt = pd.read_csv("../input/heart.csv")
dt.head(5)
X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10) #split the data

model = RandomForestClassifier(max_depth=5)

model.fit(X_train, y_train)
estimator = model.estimators_[1]

feature_names = [i for i in X_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values
#code from https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c



export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
y_predict = model.predict(X_test)

y_pred_quant = model.predict_proba(X_test)[:, 1]

y_pred_bin = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred_bin)

confusion_matrix
total=sum(sum(confusion_matrix))



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC - Classificador de doença cardíaca')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr, tpr)
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test, plot_type="bar")