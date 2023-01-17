import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df.info()
df.hist(figsize=(12, 10))

plt.show()
# import machine larning models metrics and model_selection

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


y = df['diagnosis']

df.drop(columns=['diagnosis'], inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.80, random_state=0)
lr = LogisticRegression(solver='liblinear')

svc = SVC(random_state=0, probability=True)

decision_tree = DecisionTreeClassifier(max_leaf_nodes=20, max_depth=6, random_state=0)

mlp = MLPClassifier(random_state=0)

xgboost_classifier = XGBClassifier(random_state=0)

random_forest = RandomForestClassifier(max_leaf_nodes=20, max_depth=5, n_estimators=150, random_state=0)

ada_boost = AdaBoostClassifier(random_state=0, base_estimator=lr, n_estimators=150)
models = [('Logistic Regression', lr), ('SVM', svc), ('Decision Tree', decision_tree),

          ('xgboost', xgboost_classifier), ('Random Forest', random_forest),

          ('Ada Boost', ada_boost)]
def run_models():

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink']

    for index, run_model in enumerate(models):

        model_name = run_model[0]

        model = run_model[1]

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        y_probably = model.predict_proba(x_test)[::, 1]

        print(model_name)

        print(f'acc: {accuracy_score(y_test, y_predict)}\n'

              f'f1_score: {f1_score(y_test, y_predict)}\n'

              f'recall: {recall_score(y_test, y_predict)}\n'

              f'precision: {precision_score(y_test, y_predict)}\n')

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_probably)

        auc = metrics.roc_auc_score(y_test, y_probably)

        plt.title('Roc Curve')

        plt.plot(fpr, tpr, colors[index], label=f'{model_name} auc: {auc}')

        plt.plot([0, 1], [0, 1], 'k--')

        plt.legend(loc=4)





run_models()

plt.show()