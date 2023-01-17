import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

#classifies
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

#other libraries
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
df.head()
df.info()
#categorical features
categories = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
df1 = pd.get_dummies(df,columns = categories, drop_first=True)

df1.head()
df1.isnull().sum().max()
#columns
df1.columns
#check out the output variable
print('Not Interested', round(df1['Response'].value_counts()[0]/len(df1) * 100, 2), '% of the dataset')
print('Interested', round(df1['Response'].value_counts()[1]/len(df1) * 100, 2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]

sns.countplot('Response', data=df1, palette=colors)
plt.title('Class Distributions \n (0: Not Interested || 1: Interested)', fontsize=14)
df1.drop('id', axis = 1, inplace = True)
df1.head(3)
X = df1.drop('Response', axis = 1)
y = df1['Response']
from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

scaled_values = rob_scaler.fit_transform(X)

X = pd.DataFrame(scaled_values, columns = X.columns)
df1 = pd.concat([X,y], axis = 1)
df1.head()
#Splitting the dataset
print('Not Interested', round(df1['Response'].value_counts()[0]/len(df1) * 100, 2), '% of the dataset')
print('Interested', round(df1['Response'].value_counts()[1]/len(df1) * 100, 2), '% of the dataset')

X = df1.drop('Response', axis = 1)
y = df1['Response']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle = False)

for train_index, test_index in sss.split(X, y):
    print("Train: ", train_index, "Test: ", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    

#turning them into arrays
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

#see if both the train and test label distribution are similarly distributed
train_unique_label, train_count_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_count_label = np.unique(original_ytest, return_counts=True)
print('-' * 50)

print('Label Distributions: \n')
print(train_count_label / len(original_ytrain))
print(test_count_label / len(original_ytest))
#Random Under-sampling
df1 = df1.sample(frac = 1)

#amount of Interested response values : 46710
interest_df = df1.loc[df1['Response'] == 1]
not_interest_df = df1.loc[df1['Response'] == 0][:46710]

normal_distributed_df = pd.concat([interest_df, not_interest_df])
#shuffle 
new_df = normal_distributed_df.sample(frac = 1, random_state = 420)
new_df.head()
#Equally distributed now
print("Distribution:\n")
print(new_df['Response'].value_counts()/len(new_df))

sns.countplot('Response', data = new_df, palette=colors)
plt.title("Equally Distributed Responses", fontsize = 14)
plt.show()
#Correlation Matrix
correlation = new_df.corr()
plt.figure(figsize = (20,13))
sns.heatmap(correlation, cmap = 'coolwarm_r', annot_kws={'size : 20'})
plt.title("Correlation matrix after doing undesampling", fontsize = 14)
plt.show()
#For negative correlation
f, axes = plt.subplots(ncols=3, figsize = (20,4))

#negative correlation with our Response.
sns.boxplot(x = 'Response', y = 'Previously_Insured', data = new_df, palette=colors, ax = axes[0])
axes[0].set_title("Previously Insured Vs Class Negative correlation")

sns.boxplot(x = 'Response', y = 'Policy_Sales_Channel', data = new_df, palette=colors, ax = axes[1])
axes[1].set_title("Policy Sales Channel Vs Class Negative correlation")

sns.boxplot(x = 'Response', y = 'Vehicle_Age_< 1 Year', data = new_df, palette=colors, ax = axes[2])
axes[2].set_title("Vehicle_Age_< 1 Years Vs Class Negative correlation")
plt.show()
#For positive correlation
f, axes = plt.subplots(ncols=2, figsize = (20,4))

#positive correlation with our Response.
#sns.boxplot(x = 'Response', y = 'Vehicle_Age_> 2 Years', data = new_df, palette=colors, ax = axes[0])
#axes[0].set_title("Vehicle_Age_> 2 Years Vs Class Positive correlation")

#sns.boxplot(x = 'Response', y = 'Region_Code_28.0', data = new_df, palette=colors, ax = axes[1])
#axes[1].set_title("Region Code 28.0 Vs Class Positive correlation")

sns.boxplot(x = 'Response', y = 'Age', data = new_df, palette=colors, ax = axes[0])
axes[0].set_title("Age Vs Class Positive correlation")

sns.boxplot(x = 'Response', y = 'Vehicle_Damage_Yes', data = new_df, palette=colors, ax = axes[1])
axes[1].set_title("Vehicle Damage Yes Vs Class Positive correlation")

plt.show()
#Removing Outliers From Age
Age_interested = new_df['Age'].loc[new_df['Response'] == 1].values
q25, q75 = np.percentile(Age_interested, 25), np.percentile(Age_interested, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
Age_Iqr = q75 - q25
print('iqr: {}'.format(Age_Iqr))

Age_cut_off = Age_Iqr * 1.5
Age_lower, Age_Upper = q25 - Age_cut_off, q75 + Age_cut_off
print('Cut_off: {}'.format(Age_cut_off))
print('Age_lower: {}'.format(Age_lower))
print('Age_Upper: {}'.format(Age_Upper))

outliers = [x for x in Age_interested if x < Age_lower or x > Age_Upper]
print('Feature Age Outliers for Interested cases: {}'.format(len(outliers)))
print('Age Outliers: {}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['Age'] > Age_Upper) | (new_df['Age'] < Age_lower)].index)
print('--' * 25)


#Vehicle_Damage_Yes Outliers
Vehicle_Damage_interested = new_df['Vehicle_Damage_Yes'].loc[new_df['Response'] == 1].values
q25, q75 = np.percentile(Vehicle_Damage_interested, 25), np.percentile(Vehicle_Damage_interested, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
Vehicle_Damage_Iqr = q75 - q25
print('iqr: {}'.format(Vehicle_Damage_Iqr))

Vehicle_Damage_cut_off = Vehicle_Damage_Iqr * 1.5
Vehicle_Damage_lower, Vehicle_Damage_Upper = q25 - Vehicle_Damage_cut_off, q75 + Vehicle_Damage_cut_off
print('Vehicle_Damage_cut_off: {}'.format(Vehicle_Damage_cut_off))
print('Vehicle_Damage_lower: {}'.format(Vehicle_Damage_lower))
print('Vehicle_Damage_Upper: {}'.format(Vehicle_Damage_Upper))

outliers = [x for x in Vehicle_Damage_interested if x < Vehicle_Damage_lower or x > Vehicle_Damage_Upper]
print('Feature Damage Outliers for Interested cases: {}'.format(len(outliers)))
print('Vehicle Damage Outliers: {}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['Vehicle_Damage_Yes'] > Vehicle_Damage_Upper) | (new_df['Vehicle_Damage_Yes'] < Vehicle_Damage_lower)].index)
print('--' * 25)
#Dimensional Reduction & Clustering
#T-SNE Algorithm and PCA

X = new_df.drop('Response', axis = 1)
y = new_df['Response']

#T-sne
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=420).fit_transform(X.values)
t1 = time.time()
print("T-SNE took: {:.2} seconds".format(t1 - t0))

#PCA
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=420).fit_transform(X.values)
t1 = time.time()
print("PCA took: {:.2} seconds".format(t1-t0))
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24,6))

f.suptitle('Clusters using Dimensionality Reduction', fontsize = 14)
blue_patch = mpatches.Patch(color = "#0A0AFF", label = "Not Interested")
red_patch = mpatches.Patch(color = "#AF0000", label = "Interested")

#t-sne scatterplot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1],
            c = (y == 0), cmap = "coolwarm",
            label = "Not Interested", linewidths = 2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1],
            c = (y == 1), cmap = "coolwarm",
            label = "Interested", linewidths = 2)
ax1.set_title('t-SNE', fontsize = 14)
ax1.grid(True)
ax1.legend(handles = [blue_patch, red_patch])

#PCA Scatterplot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1],
            c = (y == 0), cmap = "coolwarm",
            label = "Interested", linewidths = 2)

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1],
            c = (y == 1), cmap = "coolwarm",
            label = "Interested", linewidths = 2)
ax2.set_title('PCA', fontsize = 14)
ax2.grid(True)
ax2.legend(handles = [blue_patch, red_patch])
plt.show()
#Classifiers
X = new_df.drop('Response', axis = 1)
y = new_df['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

#Let's implement Classifiers
classifiers = {"LogisticRegression": LogisticRegression(),
               "KNearest": KNeighborsClassifier(),
               "Decision Tree Classifier": DecisionTreeClassifier(),
               "Random Forest Classifier": RandomForestClassifier()
              }

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of ", round(training_score.mean(), 2) * 100, "% accuracy score")
#We will choose 2 best models and apply hyper parameters.
#logistic, Random forest

from sklearn.model_selection import GridSearchCV

#logistic
log_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log = GridSearchCV(LogisticRegression(), log_params)
grid_log.fit(X_train, y_train)

#logistic best estimators
log_reg = grid_log.best_estimator_

random_params = {'n_estimators': [100, 200, 500], 'criterion': ['gini', 'entropy'],
                 'max_depth': [None, 10, 15, 20]}

grid_random = GridSearchCV(RandomForestClassifier(), random_params)
grid_random.fit(X_train, y_train)

#Random Forest best estimators
random_reg = grid_random.best_estimator_
# Overfitting Case

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(random_reg, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
undersample_X = new_df.drop('Response', axis=1)
undersample_y = new_df['Response']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# Implementing NearMiss Technique 
# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)
X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)
print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# Cross Validating the right way

for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
# Let's Plot LogisticRegression Learning Curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1, estimator2, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Random Forest Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

random_pred = cross_val_predict(random_reg, X_train, y_train, cv=5)

from sklearn.metrics import roc_auc_score

print("Logistic Regression: ", roc_auc_score(y_train, log_reg_pred))
print("Random Forest: ", roc_auc_score(y_train, random_pred))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, random_pred)

def graph_roc_curve_multiple(log_fpr, log_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 2 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred))) 
    plt.plot(tree_fpr, tree_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, random_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, tree_fpr, tree_tpr)
plt.show()
test_df = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
test_df.head()
test_df.drop('id', axis = 1, inplace = True)

categories = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
test_df = pd.get_dummies(test_df,columns = categories, drop_first=True)

scaled_test_values = rob_scaler.fit_transform(test_df)

test_df = pd.DataFrame(scaled_test_values, columns = test_df.columns)
test_df.head()
predictions = log_reg.predict(test_df)
test_df['Predictions'] = predictions
test_df.to_csv("submission.csv")
print("Submission file created")
test_df.head(5)