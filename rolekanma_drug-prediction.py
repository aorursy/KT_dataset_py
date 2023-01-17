import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib as mpl

from scipy import stats



import seaborn as sns





# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_excel("/kaggle/input/horizon/HZNP_DataScience_Exercise.xlsx")



xls = pd.ExcelFile("/kaggle/input/horizon/HZNP_DataScience_Exercise.xlsx")

df1 = pd.read_excel(xls, 'Instructions')

df2 = pd.read_excel(xls, 'chart_review_data')
df2.shape
df2.head(10)

#Nan values will have to be dealt with imputation
df2.describe(include="all")
age_mean = df2.age_at_rf_r1_dx.mean()



df2.age_at_rf_r1_dx.fillna(age_mean, axis=0, inplace=True)

horizon = df2.fillna(df2.mode().iloc[0])

horizon.head()
men = horizon[horizon['patient_gender'] == 'M']

women = horizon[horizon['patient_gender'] == 'F']

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



ax = sns.distplot(men[men['recommend_Product_X'] ==0].age_current, 

                 bins =20, label = 'not recommended', ax = axes[0], kde =False)

ax = sns.distplot(men[men['recommend_Product_X'] ==1].age_current, 

                  bins = 20, label = 'recommended', ax = axes[0], kde =False)

ax.legend()

ax.set_title('Men')



ax = sns.distplot(women[women['recommend_Product_X'] ==0].age_current, 

                 bins =20, label = 'not recommended',ax = axes[1], kde =False)

ax = sns.distplot(women[women['recommend_Product_X'] ==1].age_current, 

                  bins = 20, label = 'recommended',ax = axes[1], kde =False)

_ = ax.set_title('Women')

ax.legend()
horizon = horizon.reindex(np.random.permutation(horizon.index))

horizon.head(10)
horizon_target = horizon.recommend_Product_X
horizontable = pd.get_dummies(horizon[['patient_gender','current_severity','severity_at_HZD_dx','age_current', 

        'HZD_dx_age', 'has_had_surgery',

       'rf_r1_present', 'rf_r2_present', 'rf_r3_present', 'rf _r4_present',

       'rf_r5_present', 'rf_r6_present', 'had_dx1_at_HZD_dx',

       'had_dx2_at_HZD_dx', 'had_dx3_at_HZD_dx', 'had_dx4_at_HZD_dx',

       'had_dx5_at_HZD_dx', 'had_dx6_at_HZD_dx', 'had_dx7_at_HZD_dx',

       'had_dx8_at_HZD_dx', 'had_dx9_at_HZD_dx', 'had_dx10_at_HZD_dx',

       'had_dx11_at_HZD_dx', 'had_dx12_at_HZD_dx', 'had_dx13_at_HZD_dx',

       'had_dx14_at_HZD_dx', 'had_dx15_at_HZD_dx', 'had_dx16_at_HZD_dx',

       'had_dx17_at_HZD_dx', 'had_dx18_at_HZD_dx', 'had_dx19_at_HZD_dx',

       'had_dx20_at_HZD_dx', 'had_dx21_at_HZD_dx', 'currently_has_dx1',

       'currently_has_dx2', 'currently_has_dx3', 'currently_has_dx4',

       'currently_has_dx5', 'currently_has_dx6', 'currently_has_dx7',

       'currently_has_dx8', 'currently_has_dx9', 'currently_has_dx10',

       'currently_has_dx11', 'currently_has_dx12', 'currently_has_dx13',

       'currently_has_dx14', 'currently_has_dx15', 'currently_has_dx16',

       'currently_has_dx17', 'currently_has_dx18', 'currently_has_dx19',

       'currently_has_dx20', 'currently_has_dx21', 'has_ever_had_dx22',

       'has_ever_had_dx23', 'px_count_for_HZD_tried', 'currently_using_px1',

       'currently_using_px2', 'currently_using_px3', 'currently_using_px4',

       'currently_using_px5', 'currently_using_px6', 'currently_using_px7',

       'currently_using_px8', 'currently_using_px9', 'currently_using_px10',

       'currently_using_px11', 'currently_using_px12', 'previously_tried_px1',

       'previously_tried_px2', 'previously_tried_px3', 'previously_tried_px4',

       'previously_tried_px5', 'previously_tried_px6', 'previously_tried_px7',

       'previously_tried_px8', 'previously_tried_px9', 'previously_tried_px10',

       'previously_tried_px11', 'previously_tried_px12']])

horizontable.head()
from sklearn.model_selection import train_test_split



horizon_train, horizon_test, horizon_target_train, horizon_target_test = train_test_split(horizontable, horizon_target, test_size=0.2, random_state=42)

print(horizon_train.shape)

print(horizon_test.shape)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

min_max_scaler.fit(horizon_train)

horizon_train_norm = min_max_scaler.fit_transform(horizon_train)

horizon_test_norm = min_max_scaler.fit_transform(horizon_test)

horizon_target_train = np.array(horizon_target_train)

horizon_target_test = np.array(horizon_target_test)
#nearest neighbor classifier

from sklearn import neighbors, tree, naive_bayes

from sklearn.metrics import confusion_matrix



n_neighbors = 5



knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

knnclf.fit(horizon_train_norm, horizon_target_train)

knnpreds_test = knnclf.predict(horizon_test_norm)

#print(knnpreds_test)

knncm = confusion_matrix(horizon_target_test, knnpreds_test)



print('Accuracy =  %0.2f' % knnclf.score(horizon_test_norm, horizon_target_test))

import pylab as plt

%matplotlib inline

plt.figure(figsize=(10, 10))

plt.matshow(knncm)

plt.title('Confusion matrix')

plt.colorbar()

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

print(knncm)

# stochastic gradient descent (SGD) learning

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(horizon_train_norm, horizon_target_train)

Y_pred = sgd.predict(horizon_test_norm)



sgd.score(horizon_train_norm, horizon_target_train)



acc_sgd = round(sgd.score(horizon_train_norm, horizon_target_train) * 100, 2)





print(round(acc_sgd,2,), "%")
# Decision Tree

from sklearn import tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(horizon_train, horizon_target_train)



Y_pred = decision_tree.predict(horizon_test)



acc_decision_tree = round(decision_tree.score(horizon_train, horizon_target_train) * 100, 2)

print(round(acc_decision_tree,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(horizon_train_norm, horizon_target_train)



Y_prediction = random_forest.predict(horizon_test_norm)



random_forest.score(horizon_train_norm, horizon_target_train)

acc_random_forest = round(random_forest.score(horizon_train_norm, horizon_target_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(horizon_train_norm, horizon_target_train)



Y_pred = logreg.predict(horizon_test_norm)



acc_log = round(logreg.score(horizon_train_norm, horizon_target_train) * 100, 2)

print(round(acc_log,2,), "%")
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(horizon_train_norm, horizon_target_train)



Y_pred = linear_svc.predict(horizon_test_norm)



acc_linear_svc = round(linear_svc.score(horizon_train_norm, horizon_target_train) * 100, 2)

print(round(acc_linear_svc,2,), "%")
# Perceptron

perceptron = Perceptron(max_iter=5)

perceptron.fit(horizon_train_norm, horizon_target_train)



Y_pred = perceptron.predict(horizon_test_norm)



acc_perceptron = round(perceptron.score(horizon_train_norm, horizon_target_train) * 100, 2)

print(round(acc_perceptron,2,), "%")
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(horizon_train_norm, horizon_target_train)



Y_pred = gaussian.predict(horizon_test_norm)



acc_gaussian = round(gaussian.score(horizon_train_norm, horizon_target_train) * 100, 2)

print(round(acc_gaussian,2,), "%")
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, horizon_train, horizon_target_train, cv=10, scoring = "accuracy")
print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':horizontable.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head()
def plot_feature_importances(model, n_features, feature_names):

    plt.rcParams["figure.figsize"] = (20,15)



    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), feature_names)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)

    plt.rcParams["figure.figsize"] = (20,7)



    

horizon_names = horizontable.columns.values

horizon_names



features = horizon_names[1:]

plot_feature_importances(random_forest, len(horizon_names), horizon_names)
#Confusion matrix

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, horizon_train, horizon_target_train, cv=3)

confusion_matrix(horizon_target_train, predictions)
from sklearn.metrics import precision_score, recall_score



print("Precision:", precision_score(horizon_target_train, predictions))

print("Recall:",recall_score(horizon_target_train, predictions))
from sklearn.metrics import precision_recall_curve



# getting the probabilities of our predictions

y_scores = random_forest.predict_proba(horizon_train)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(horizon_target_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
from sklearn.metrics import roc_curve

# compute true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(horizon_target_train, y_scores)
# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(horizon_target_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
from sklearn.tree import export_graphviz

from IPython.display import SVG

from graphviz import Source

from IPython.display import display





tree = export_graphviz(decision_tree, out_file=None, feature_names=horizon_names, class_names=['No','Yes'], filled = True, rotate = True)

graph = Source(tree)

graph
#Using Adaboost to improve model accuracy with 100 weak trees

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier



clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,   random_state=0)

scores = cross_val_score(clf, horizon_train, horizon_target_train, cv=5)

print('Original decision tree accuracy', scores.mean())



ada = AdaBoostClassifier(n_estimators=100)

scores1 = cross_val_score(ada, horizon_train, horizon_target_train, cv=5)

print('Boosted decision tree accuracy', scores1.mean())



from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(horizon_train, horizon_target_train)

print('Gradient Boosted score', gbc.score(horizon_test, horizon_target_test))



from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier



bagging = BaggingClassifier(RandomForestClassifier(n_estimators=100), max_samples=0.5, max_features=0.5,

                   n_estimators=25, random_state=5).fit(horizon_train, horizon_target_train)

print('Bagging score', bagging.score(horizon_train, horizon_target_train))


