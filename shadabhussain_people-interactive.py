import os

os.listdir('../input/people-interactive-assignment')
! pip install --upgrade pip

# ! pip install autoplotter

! pip install git+git://github.com/thelittlebug/autoplotter

! pip install sweetviz dataprep nbconvert featuretools matplotlib
import numpy as np

import pandas as pd

# from autoplotter import run_app

import sweetviz as sv

from dataprep.eda import plot, plot_correlation, plot_missing, create_report

from sklearn.preprocessing import OneHotEncoder

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/people-interactive-assignment/Dataset.txt', sep="\t", header=0, index_col='Index')

df.head()
test_df = pd.read_csv('../input/people-interactive-assignment/Dataset_test.txt', sep="\t", header=0, index_col='Index')

test_df.head()
# Stats for the given data

df.describe()
# Checking for NULL Values in train data

df.info()
# Checking for NULL Values in train data

test_df.info()
# Converting column F15 and F16 to date column since they are present as string type

df[["F15", "F16"]] = df[["F15", "F16"]].apply(pd.to_datetime)

test_df[["F15", "F16"]] = test_df[["F15", "F16"]].apply(pd.to_datetime)
df.head()
# Number of Unique Elements in each column for train data

df.T.apply(lambda x: x.nunique(), axis=1)
# Number of Unique Elements in each column for test data

test_df.T.apply(lambda x: x.nunique(), axis=1)
# Converting F17, F18, F21, and F22 to string type to get counts of it and for treating them as categorical column

df[["F17", "F18", "F21", "F22"]] = df[["F17", "F18", "F21", "F22"]].astype('category')

test_df[["F17", "F18", "F21", "F22"]] = test_df[["F17", "F18", "F21", "F22"]].astype('category')
# Summary Report for train data

train_report = sv.analyze(df, target_feat='C')

train_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"
# Summary Report for train-test data

train_test_report = sv.compare([df.iloc[:,:-1], "Training Data"], [test_df, "Test Data"], feat_cfg= sv.FeatureConfig(skip=['F15','F16']))

train_test_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"
# Dataset stats and distribution

plot(df)
column_list = list(df.columns)
plot(df, column_list[0])
plot(df, column_list[1])
plot(df, column_list[2])
plot(df, column_list[3])
plot(df, column_list[4])
plot(df, column_list[5])
plot(df, column_list[6])
plot(df, column_list[7])
plot(df, column_list[8])
plot(df, column_list[9])
plot(df, column_list[10])
plot(df, column_list[11])
plot(df, column_list[12])
plot(df, column_list[13])
plot(df, column_list[16])
plot(df, column_list[17])
plot(df, column_list[18])
plot(df, column_list[19])
plot(df, column_list[20])
plot(df, column_list[21])
# Calculating Correlation for the given dataset

df.corr()
plot_correlation(df)
create_report(df)
numerical_columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F19', 'F20']

df[numerical_columns].head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[numerical_columns]=scaler.fit_transform(df[numerical_columns])

test_df[numerical_columns]=scaler.transform(test_df[numerical_columns])

df[numerical_columns].head()
df.info()
# Converting 'F17', 'F18', 'F21', 'F22' to one-hot encoded form

processed_df = pd.get_dummies(df, columns=['F17', 'F18', 'F21', 'F22'], drop_first=True)

processed_test_df = pd.get_dummies(test_df, columns=['F17', 'F18', 'F21', 'F22'], drop_first=True)
processed_df.head()
# Drop Date Columns

processed_df=processed_df.drop(['F15','F16'], axis=1)

processed_test_df=processed_test_df.drop(['F15','F16'], axis=1)
from sklearn.model_selection import StratifiedShuffleSplit

# Here we split the data into training and test sets and implement a stratified shuffle split.

stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=40)



for train_set, validation_set in stratified.split(processed_df, processed_df["C"]):

    stratified_train = processed_df.reindex(index = train_set)

    stratified_validation = processed_df.reindex(index = validation_set)

    

stratified_train["C"].value_counts()/len(stratified_train)
stratified_validation["C"].value_counts()/len(stratified_validation)
train_data = stratified_train.copy() # Make a copy of the stratified training set.

test_data = stratified_validation.copy()

print(train_data.shape)

print(test_data.shape)
train_data['C'].value_counts()
train_data[:] = np.nan_to_num(train_data)

test_data[:] = np.nan_to_num(test_data)
X_train=train_data.drop(['C'], axis=1)

y_train=train_data.C

X_test=test_data.drop('C', axis=1)

y_test=test_data.C
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

baselog_model = LogisticRegression()

baselog_model.fit(X_train,y_train)

y_pred = baselog_model.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.neural_network import MLPClassifier



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



dict_of_algos={'LR':LogisticRegression(),'svc':SVC(),'KNC':KNeighborsClassifier(),'DT':tree.DecisionTreeClassifier(),'MLPc':MLPClassifier(),

               'GRBC':GradientBoostingClassifier(),'RFC':RandomForestClassifier(),'GNB':GaussianNB()}
def accuracy_of_algos(dict_of_algos):

    dict_of_accuracy={}

    for k,v in dict_of_algos.items():

        v.fit(X_train,y_train)

        y_pred = v.predict(X_test)

        dict_of_accuracy[k] = accuracy_score(y_pred,y_test)

        y=v.score(X_train,y_train)

    return dict_of_accuracy



print(accuracy_of_algos(dict_of_algos)) 
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

    "Naive Bayes": GaussianNB()

}
no_classifiers = len(dict_classifiers.keys())



def batch_classify(X_train, Y_train,X_test, verbose = True):

    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers,4)), columns = ['classifier', 'train_score', 'training_time','test_score'])

    count = 0

    for key, classifier in dict_classifiers.items():

        t_start = time.clock()

        classifier.fit(X_train, Y_train)

        t_end = time.clock()

        t_diff = t_end - t_start

        train_score = classifier.score(X_train, Y_train)

        

        y_pred=classifier.predict(X_test)

        test_score=accuracy_score(y_test,y_pred)

        

        df_results.loc[count,'classifier'] = key

        df_results.loc[count,'train_score'] = train_score

        df_results.loc[count,'training_time'] = t_diff

        df_results.loc[count,'test_score']=test_score

        if verbose:

            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))

        count+=1

    return df_results
df_results = batch_classify(X_train, y_train,X_test)

print(df_results.sort_values(by='train_score', ascending=False))
# Use Cross-validation

from sklearn.model_selection import cross_val_score



# Logistic Regression

log_reg = LogisticRegression()

log_scores = cross_val_score(log_reg, X_train, y_train, cv=3)

log_reg_mean = log_scores.mean()
# KNearestNeighbors

knn_clf = KNeighborsClassifier()

knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=3)

knn_mean = knn_scores.mean()
# Decision Tree

tree_clf = tree.DecisionTreeClassifier()

tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=3)

tree_mean = tree_scores.mean()
# Gradient Boosting Classifier

grad_clf = GradientBoostingClassifier()

grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=3)

grad_mean = grad_scores.mean()
# Random Forest Classifier

rand_clf = RandomForestClassifier(n_estimators=18)

rand_scores = cross_val_score(rand_clf, X_train, y_train, cv=3)

rand_mean = rand_scores.mean()
# Naives Bayes

nav_clf = GaussianNB()

nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=3)

nav_mean = nav_scores.mean()
# Create a Dataframe with the results.

d = {'Classifiers': ['Logistic Reg.', 'KNN', 'Dec Tree', 'Grad B CLF', 'Rand FC', 'Naives Bayes'], 

    'Crossval Mean Scores': [log_reg_mean, knn_mean, tree_mean, grad_mean, rand_mean, nav_mean]}



result_df = pd.DataFrame(data=d)
# All our models perform well but I will go with GradientBoosting.

result_df = result_df.sort_values(by=['Crossval Mean Scores'], ascending=False)

result_df
from sklearn.metrics import accuracy_score



grad_clf = GradientBoostingClassifier()

grad_clf.fit(X_train, y_train)
y_pred=grad_clf.predict(X_test)

print ("Gradient Boost Classifier Train accuracy is %2.2f" % accuracy_score(y_test, y_pred))
y_test.value_counts()
from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

# 4697: no's, 4232: yes

conf_matrix = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(conf_matrix, annot=True, fmt='d', linewidths=.5, ax=ax)

plt.title("Confusion Matrix", fontsize=20)

plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)

ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(["Predicted False",'Predicted True'])

ax.set_yticklabels(['Actual False', 'Actual True'], fontsize=16, rotation=360)

plt.show()
from sklearn.metrics import precision_score, recall_score



# The model is only retaining 60% of clients that agree to suscribe a term deposit.

print('The model is {c} % sure that the potential client will buy'.format(c=np.round(precision_score(y_test, y_pred),2)*100))

print('The model is retaining {c} % of clients that agree to buy'.format(c=np.round(recall_score(y_test, y_pred),2)*100))
from sklearn.metrics import f1_score



f1_score(y_test, y_pred)*100
from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(14,8))

y_prob=grad_clf.predict_proba(X_test)[:,1]

precisions, recalls, threshold = precision_recall_curve(y_test, y_prob)

plt.plot(threshold,recalls[:-1],marker='.',label='recall')

plt.plot(threshold,precisions[:-1],marker='.',label='precision')

plt.legend(frameon=True,fontsize=20)

plt.axvline(x=0.354,c='black')



plt.show()
from sklearn.metrics import roc_curve

grd_fpr, grd_tpr, threshold = roc_curve(y_test, y_prob)

def graph_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.figure(figsize=(10,6))

    plt.title('ROC Curve \n Gradient Boosting Classifier', fontsize=18)

    plt.plot(false_positive_rate, true_positive_rate, label=label)

    plt.plot([0, 1], [0, 1], '#0C8EE0')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('ROC Score of 71.54% \n ', xy=(0.43, 0.74), xytext=(0.50, 0.67),

            arrowprops=dict(facecolor='#F75118', shrink=0.05),

            )

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#F75118', shrink=0.05),

                )

    

    

graph_roc_curve(grd_fpr, grd_tpr, threshold)

plt.show()
from sklearn.metrics import roc_auc_score



print('Gradient Boost Classifier Score: ', roc_auc_score(y_test, y_prob))
from sklearn.model_selection import GridSearchCV



param_test3 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1000,2000), 'min_samples_leaf':range(30,71,10),'n_estimators':range(20,81,10)}

gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,

                                                               max_features='sqrt', subsample=0.8, random_state=10),

                        param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)



gsearch.fit(X_train,y_train)

gsearch.best_params_, gsearch.best_score_
processed_df['predicted_C'] = gsearch.best_estimator_.predict(processed_df.drop(['C'], axis=1))
accuracy_score(processed_df['predicted_C'], processed_df['C'])
processed_test_df['predicted_C'] = gsearch.best_estimator_.predict(processed_test_df)
processed_test_df.head()
processed_df.to_csv("train_dataset_pred.csv", sep="\t")

processed_test_df.to_csv("test_dataset_pred.csv", sep="\t")