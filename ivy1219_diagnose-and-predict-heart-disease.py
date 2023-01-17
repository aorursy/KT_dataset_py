## It is cleaned data ,which is normally not gonna happen in real world. 

## But thanks to UCI, we can skip the data-cleaning job in this report.



# Basic Part

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



# Modeling Part

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz 

from sklearn.metrics import roc_curve, auc 

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

import eli5 

from eli5.sklearn import PermutationImportance

import shap 

from pdpbox import pdp, info_plots 



# Set a seed for tracing back and reproducing

np.random.seed(101) #ensure reproducibility



df = pd.read_csv('../input/heart.csv')

df.head()
# or, a fast and easy way

df.describe()
# Age Distribution

sns.violinplot(df.age,palette = 'Set2',bw = .1, cut =1)

plt.title('Age Distribution')
# Chest Pain Type Distribution

sns.countplot(x = 'cp', data = df)

plt.title('Chest Pain Type Distribution')
# if you want to quickly take a glance of regression relationship, it is a fast way

## sns.pairplot(hue = 'target',data = df)
df = pd.get_dummies(df,drop_first = True)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), y, test_size = .2, random_state=101)

model = RandomForestClassifier(max_depth=5)

model.fit(X_train, y_train)
estimator = model.estimators_[1]

feature_names = [i for i in X_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'Neg'

y_train_str[y_train_str == '1'] = 'Pos'

y_train_str = y_train_str.values
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
# The very manually way: (but it is good to understand the concept and the function)

def plot_feature_importances(n):

    n_features = X_train.shape[1]

    plt.figure(figsize = (10,10))

    plt.barh(range(n_features), n.feature_importances_, align='center',color = 'm',alpha =0.6)

    plt.yticks(np.arange(n_features), X_train.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)
plot_feature_importances(model)
# The faster way:

perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
y_predict = model.predict(X_test)

y_pred_quant = model.predict_proba(X_test)[:, 1]

y_pred_bin = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred_bin)



total=sum(sum(confusion_matrix))



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='r')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for RandomForest classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr, tpr)

# See, it works quite well.
sns.countplot(x = 'target',data = df)
features = df.columns.values.tolist()

features.remove('target')

feat_name = 'ca'

pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features = features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
# how about the maximum heart beat?

feat_name = 'thalach'

pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features = features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test)
def heart_disease_risk_predict(model, patient_Id):



    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(patient_Id)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_Id)
p1 = X_test.iloc[1,:].astype(float)

heart_disease_risk_predict(model, p1)
p5 = X_test.iloc[5,:].astype(float)

heart_disease_risk_predict(model, p5)
p10 = X_test.iloc[10,:].astype(float)

heart_disease_risk_predict(model, p10)


shap_values = explainer.shap_values(X_train.iloc[20:50])

shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[20:50])