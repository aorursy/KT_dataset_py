from sklearn.tree import DecisionTreeClassifier



import pandas as pd

import numpy as np
model = DecisionTreeClassifier()

model
#bank = pd.read_csv('/datasets/bank-full.csv', sep=';')

bank = pd.read_csv('../input/bank-full.csv', sep=';')

X = pd.get_dummies(bank.drop('y', axis=1), drop_first=True)

y = bank['y']



from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, 

                                                    test_size=0.3,

                                                   random_state=100)

train_x.shape, test_x.shape, train_y.shape, test_y.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score



dt_model = DecisionTreeClassifier(random_state=100, max_depth=4)

dt_model.fit(train_x, train_y)

test_pred_dt = dt_model.predict(test_x)



print(accuracy_score(test_y, test_pred_dt))

print(f1_score(test_y, test_pred_dt,  pos_label='yes'))
rf_model = RandomForestClassifier(random_state=100,

                                 n_estimators=300,

                                 max_features=8)

rf_model.fit(train_x, train_y)

test_pred_rf = rf_model.predict(test_x)



print(accuracy_score(test_y, test_pred_rf))

print(f1_score(test_y, test_pred_rf,  pos_label='yes'))
np.sqrt(len(train_x.columns))
def draw_tree(model, columns):

    import pydotplus

    from sklearn.externals.six import StringIO

    from IPython.display import Image

    import os

    from sklearn import tree

    

    #graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'

    #os.environ["PATH"] += os.pathsep + graphviz_path



    dot_data = StringIO()

    tree.export_graphviz(model,

                         out_file=dot_data,

                         feature_names=columns)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    return Image(graph.create_png())
draw_tree(rf_model.estimators_[0], train_x.columns)
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(n_estimators=800,

                               random_state=100)

ada_model.fit(train_x, train_y)



test_pred_ada = ada_model.predict(test_x)



print(accuracy_score(test_y, test_pred_ada))

print(f1_score(test_y, test_pred_ada,  pos_label='yes'))
draw_tree(ada_model.estimators_[298], train_x.columns)
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=800,

                               random_state=100)

gb_model.fit(train_x, train_y)



test_pred_gb = gb_model.predict(test_x)



print(accuracy_score(test_y, test_pred_gb))

print(f1_score(test_y, test_pred_gb,  pos_label='yes'))
from sklearn.metrics import auc, roc_curve

test_probs_dt = pd.DataFrame(dt_model.predict_proba(test_x),

                          columns=['Prob_no', 'Prob_yes'])

fpr_dt, tpr_dt, threshs_dt = roc_curve(test_y, test_probs_dt['Prob_yes'],

                                      pos_label='yes')

auc_dt = auc(fpr_dt, tpr_dt)
test_probs_rf = pd.DataFrame(rf_model.predict_proba(test_x),

                          columns=['Prob_no', 'Prob_yes'])

fpr_rf, tpr_rf, threshs_rf = roc_curve(test_y, test_probs_rf['Prob_yes'],

                                      pos_label='yes')

auc_rf = auc(fpr_rf, tpr_rf)
test_probs_ada = pd.DataFrame(ada_model.predict_proba(test_x),

                          columns=['Prob_no', 'Prob_yes'])

fpr_ada, tpr_ada, threshs_ada = roc_curve(test_y, test_probs_ada['Prob_yes'],

                                      pos_label='yes')

auc_ada = auc(fpr_ada, tpr_ada)
test_probs_gb = pd.DataFrame(gb_model.predict_proba(test_x),

                          columns=['Prob_no', 'Prob_yes'])

fpr_gb, tpr_gb, threshs_gb = roc_curve(test_y, test_probs_gb['Prob_yes'],

                                      pos_label='yes')

auc_gb = auc(fpr_gb, tpr_gb)
%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(fpr_dt, tpr_dt)

plt.plot(fpr_rf, tpr_rf)

plt.plot(fpr_ada, tpr_ada)

plt.plot(fpr_gb, tpr_gb)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(['Decision Tree; AUC: %.3f' % auc_dt,

            'Random Forest; AUC: %.3f' % auc_rf,

            'Adaboost; AUC: %.3f' % auc_ada,

            'Gradient Boosting; AUC: %.3f' % auc_gb])