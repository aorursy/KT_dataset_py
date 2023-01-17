import pandas as pd



df = pd.read_csv('../input/bank.csv')
df.head()
df.describe()
df.info()
missing_data = df.isnull().sum().sort_values(ascending=False)

missing_per = (missing_data/len(df)) * 100

pd.concat([missing_data, missing_per], keys=["Missing Data", "Missing%"], axis=1)
import matplotlib.pyplot as plt

import seaborn as sns



fig, ax = plt.subplots(1, 2, figsize=(16, 8))

df['deposit'].value_counts().plot.pie(explode=[0,0.25],

                                      autopct='%1.2f%%', 

                                      shadow=True, ax=ax[0], 

                                      fontsize=12, startangle=25)

ax[0].set_ylabel('% of condition loans')



sns.barplot(x='education', y='balance', hue='deposit',data=df,

           estimator=lambda x: len(x)/len(df) * 100)

ax[1].set_ylabel("%")

plt.show()
sns.countplot(x="month", hue="deposit", data=df)
dict_yes_or_no = {

    "yes": 1,

    "no": 0

}



# Binary deposits 0 or 1.

bin_deposits = df["deposit"].map(dict_yes_or_no)

sns.lineplot(x="age", y=bin_deposits, data=df)
sns.countplot(x="loan", data=df)
sns.countplot(x='loan', data=df, hue="deposit")
df.hist(bins=50, figsize=(12, 8))
sns.countplot(df['marital'], data=df)
sns.countplot(df['marital'], hue='deposit', data=df)
sns.pairplot(df, hue='deposit')
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

fig = plt.figure(figsize=(12, 8))

df['deposit'] = LabelEncoder().fit_transform(df['deposit'])





numerical_data = df.select_dtypes(exclude='object')

corr_matrix = numerical_data.corr()



sns.heatmap(corr_matrix, cbar=True)

plt.title("Correlation Matrix", fontsize=16)
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit





def loan_proportions(data):

    return data["loan"].value_counts() / len(data)



stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_set, test_set in stratified.split(df, df["loan"]):

    stratified_train_set = df.loc[train_set]

    stratified_test_set = df.loc[test_set]

random_train, random_test = train_test_split(df)
compare_props = pd.DataFrame({

    "Overall": loan_proportions(df),

    "Stratified": loan_proportions(stratified_test_set),

    "Random": loan_proportions(random_test),

}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100

compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props
features = stratified_train_set.drop("deposit", axis=1)

labels = stratified_train_set['deposit'].copy()
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X[self.attribute_names].values
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_cols = list(features.select_dtypes(include=['int64']))



num_pipeline = Pipeline([

    ('select_data', DataFrameSelector(num_cols)),

    ('std_scaler', StandardScaler())

])
from sklearn.preprocessing import OneHotEncoder



cat_cols = list(features.select_dtypes(include=['object']))



cat_pipeline = Pipeline([

    ('select_data', DataFrameSelector(cat_cols)),

    ("cat_encoder", OneHotEncoder(sparse=False))

])
from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

    ("num_pipeline", num_pipeline, num_cols),

    ("cat_pipeline", cat_pipeline, cat_cols)

])
prepared_data_for_algos = full_pipeline.fit_transform(features)
prepared_data_for_algos
labels.shape
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

labels_train = encoder.fit_transform(labels)

labels_test = encoder.fit_transform(stratified_test_set['deposit'])
# Time for Classification Models

import time





from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder

 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
dict_classifiers = {

    "Logistic Regression": LogisticRegression(),

    "Nearest Neighbors": KNeighborsClassifier(),

    "Linear SVM": SVC(),

    "Gradient Boosting Classifier": GradientBoostingClassifier(),

    "Decision Tree": tree.DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(n_estimators=18),

    "Neural Net": MLPClassifier(alpha=1),

    "Naive Bayes": GaussianNB()

}
#  Thanks to Ahspinar for the function. 

no_classifiers = len(dict_classifiers.keys())



def batch_classify(X_train, Y_train, verbose = True):

    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers,3)), columns = ['classifier', 'train_score', 'training_time'])

    count = 0

    for key, classifier in dict_classifiers.items():

        t_start = time.clock()

        classifier.fit(X_train, Y_train)

        t_end = time.clock()

        t_diff = t_end - t_start

        train_score = classifier.score(X_train, Y_train)

        df_results.loc[count,'classifier'] = key

        df_results.loc[count,'train_score'] = train_score

        df_results.loc[count,'training_time'] = t_diff

        if verbose:

            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))

        count+=1

    return df_results
import numpy as np



df_results = batch_classify(prepared_data_for_algos, labels_train)

print(df_results.sort_values(by='train_score', ascending=False))
from sklearn.model_selection import cross_val_score



mean_scores = []

for model_name, model in dict_classifiers.items():

    scores = cross_val_score(model, prepared_data_for_algos, labels_train,cv=3)

    mean_score = scores.mean()

    mean_scores.append(mean_score)
index = list(dict_classifiers.keys())

pd.DataFrame(mean_scores, columns=["Mean Scores"], index=list(index)).sort_values(by=["Mean Scores"],

                                                                                 ascending=False)
from sklearn.model_selection import cross_val_predict



gradient_clf = GradientBoostingClassifier()

y_train_pred = cross_val_predict(gradient_clf, prepared_data_for_algos, labels_train, cv=3)
from sklearn.metrics import accuracy_score



gradient_clf.fit(prepared_data_for_algos, labels_train)

print("{} Accuracy : {}".format(gradient_clf.__class__.__name__, accuracy_score(labels_train, y_train_pred)))
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt





conf_matrix = confusion_matrix(labels_train, y_train_pred)

f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=0.7, ax=ax)

plt.title("Confusion Matrix", fontsize=20)

plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)

ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)

ax.set_xticklabels("")

ax.set_yticklabels(['Declined Deposits', 'Accepted Deposits'], fontsize=16, rotation=360)

plt.show()
from sklearn.metrics import precision_score, recall_score



print("Precision Score :{}\nRecall Score :{}".format(precision_score(labels_train, y_train_pred),

                                                    recall_score(labels_train, y_train_pred)))
from sklearn.metrics import f1_score

f1_score(labels_train, y_train_pred)
neural_clf = MLPClassifier(alpha=1)

svm_clf = SVC(gamma='auto', probability=True)



y_scores = cross_val_predict(gradient_clf, prepared_data_for_algos, labels_train, cv=3, method="decision_function")

neural_y_scores = cross_val_predict(neural_clf, prepared_data_for_algos, labels_train, cv=3, method="predict_proba")

svm_y_scores = cross_val_predict(svm_clf, prepared_data_for_algos, labels_train, cv=3, method="decision_function")
if y_scores.ndim == 2:

    y_scores = y_scores[:, 1]



if neural_y_scores.ndim == 2:

    neural_y_scores = neural_y_scores[:, 1]

    

if svm_y_scores.ndim == 2:

    naives_y_scores = naives_y_scores[:, 1]
from sklearn.metrics import precision_recall_curve



precisions, recalls, threshold = precision_recall_curve(labels_train, y_scores)
from sklearn.metrics import precision_recall_curve



precisions, recalls, threshold = precision_recall_curve(labels_train, y_scores)



def precision_recall_curve(precisions, recalls, thresholds):

    fig, ax = plt.subplots(figsize=(12,8))

    plt.plot(thresholds, precisions[:-1], "r--", label="Precisions")

    plt.plot(thresholds, recalls[:-1], "#424242", label="Recalls")

    plt.title("Precision and Recall \n Tradeoff", fontsize=18)

    plt.ylabel("Level of Precision and Recall", fontsize=16)

    plt.xlabel("Thresholds", fontsize=16)

    plt.legend(loc="best", fontsize=14)

    plt.xlim([-2, 4.7])

    plt.ylim([0, 1])

    plt.axvline(x=0.13, linewidth=3, color="#0B3861")

    plt.annotate('Best Precision and \n Recall Balance \n is at 0.13 \n threshold ', xy=(0.13, 0.83), xytext=(55, -40),

             textcoords="offset points",

            arrowprops=dict(facecolor='black', shrink=0.05),

                fontsize=12, 

                color='k')

    

precision_recall_curve(precisions, recalls, threshold)

plt.show()
def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])



plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions, recalls)
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(labels_train, y_scores)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)



plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr)

plt.show()
from sklearn.metrics import roc_auc_score

roc_auc_score(labels_train, y_scores)
from sklearn.metrics import roc_auc_score



print('Gradient Boost Classifier Score: ', roc_auc_score(labels_train, y_scores))

print('Neural Classifier Score: ', roc_auc_score(labels_train, neural_y_scores))

print('SVM Classifier: ', roc_auc_score(labels_train, svm_y_scores))
nn_fpr, nn_tpr, nn_thresholds = roc_curve(labels_train, neural_y_scores)

svm_fpr, svm_tpr, svm_thresholds = roc_curve(labels_train, svm_y_scores)
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, "b:", linewidth=2, label="Gradient Boosting Classifier")

plot_roc_curve(nn_fpr, nn_tpr, "Neural Network")

plot_roc_curve(svm_fpr, svm_tpr, "SVC")

plt.legend(loc="lower right", fontsize=16)

plt.show()
from sklearn.ensemble import RandomForestClassifier



for col in df.select_dtypes(exclude='int64').columns:

    df[col] = df[col].astype('category').cat.codes





X = df.drop('deposit', axis=1)

y = df['deposit']



rnd_clf = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=123)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



rnd_clf.fit(X_train, y_train)
importances = rnd_clf.feature_importances_

feature_names = df.drop('deposit', axis=1).columns

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

def feature_importance_graph(model, indices, importances, feature_names):

    plt.figure(figsize=(12,6))

    plt.title("Determining Feature importances \n with {}".format(model.__class__.__name__), fontsize=18)

    plt.barh(range(len(indices)), importances[indices], color='#31B173',  align="center")

    plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal',fontsize=14)

    plt.ylim([-1, len(indices)])

    plt.axhline(y=1.85, xmin=0.21, xmax=0.952, color='k', linewidth=3, linestyle='--')

    plt.text(0.30, 2.8, '46% Difference between \n duration and contacts', color='k', fontsize=15)

    

feature_importance_graph(rnd_clf, indices, importances, feature_names)

plt.show()
from sklearn.ensemble import VotingClassifier



voting_clf = VotingClassifier(

    estimators=[('gbc', gradient_clf), ('svm', svm_clf), ('neural_net', neural_clf)],

    voting='soft'

)

voting_clf.fit(X_train, y_train)
X_train = prepared_data_for_algos

y_train = labels_train



X_test = stratified_test_set.drop('deposit', axis=1)

X_test_prepapred = full_pipeline.transform(X_test)



y_test = labels_test
for clf in (gradient_clf, svm_clf, neural_clf, voting_clf):

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test_prepapred)

    print(clf.__class__.__name__, accuracy_score(y_test, predictions))