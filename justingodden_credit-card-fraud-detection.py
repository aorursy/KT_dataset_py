import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
# setting up default plotting parameters
%matplotlib inline

plt.rcParams['figure.figsize'] = [20.0, 7.0]
plt.rcParams.update({'font.size': 22,})

sns.set_palette('viridis')
sns.set_style('white')
sns.set_context('talk', font_scale=0.8)
# read in data
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(df.shape)
df.head()
print(df["Class"].value_counts())
# using seaborns countplot to show distribution of questions in dataset
fig, ax = plt.subplots()
g = sns.countplot(df["Class"], palette='viridis')
g.set_xticklabels(['Not Fraud', 'Fraud'])
g.set_yticklabels([])

# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Distribution of Transactions', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
plt.show()
# print percentage of questions where target == 0
print(f"Non Fraud count: {len(df[df['Class']==0]) / len(df) * 100:.2f}%")

# print percentage of questions where target == 1
print(f"Fraud count: {len(df[df['Class']==1]) / len(df) * 100:.2f}%")
# Prepare data for modeling
# Separate input features and target
y = df["Class"]
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# T-SNE Implementation
# X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

# PCA Implementation
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
import matplotlib.patches as mpatches

f, (ax2, ax3) = plt.subplots(1, 2, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax1.set_title('t-SNE', fontsize=14)

# ax1.grid(True)

# ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()
# DummyClassifier to predict only target 0
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# checking unique labels
print('Unique predicted labels: ', (np.unique(dummy_pred)))

# checking accuracy
print(f'Dummy model accuracy: {accuracy_score(y_test, dummy_pred) * 100:.2f}%')
# Modeling the data as is
# Train model
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
 
# Predict on training set
lr_pred = lr.predict(X_test)
# Checking accuracy
print(f"Linear model accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
# Checking unique values
predictions = pd.DataFrame(lr_pred)
predictions[0].value_counts()
# f1 score

print(f"Linear Model f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
# confusion matrix

pd.DataFrame(confusion_matrix(y_test, lr_pred))
# recall

print(f"Linear Model recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
# precision

print(f"Linear Model precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
from sklearn.model_selection import cross_val_score
from numpy import mean
# ROC AUC

lr_score = cross_val_score(lr, X_train, y_train, scoring='roc_auc', cv=5)
print(f"Mean ROC AUC: {round(mean(lr_score)*100,2)}")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "GradientBoosingClassifier": GradientBoostingClassifier()
}
import warnings
warnings.filterwarnings("ignore")
scores_dict = {}

for key, classifier in classifiers.items():
    print(f"Fitting {key} model")
    clf = classifier.fit(X_train, y_train)
    
    print(f"{key} model fit. Generating predictions")
    clf_pred = clf.predict(X_test)
    
    print(f"Generating ROC AUC")
    training_score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=5)
    
    print(f"Generating scores")
    scores_dict[key] = [round(accuracy_score(y_test, clf_pred)*100,2), round(f1_score(y_test, clf_pred)*100,2), round(recall_score(y_test, clf_pred)*100,2), round(precision_score(y_test, clf_pred)*100,2), round(mean(training_score)*100,2)]
    
    print()
    cm = pd.DataFrame(confusion_matrix(y_test, clf_pred))
    print(cm)
    
    print()
scores = pd.DataFrame.from_dict(scores_dict, orient='index', columns=["Accuracy","f1","Recall","Precision", "ROC AUC"])

scores
rfc = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

# predict on test set
rfc_pred = rfc.predict(X_test)
# ROC AUC

print(f"Baseline LR Mean ROC AUC: {round(mean(lr_score)*100,2)}")

rfc_score = cross_val_score(rfc, X_train, y_train, scoring='roc_auc', cv=5)
print(f"Random Forest Mean ROC AUC: {round(mean(rfc_score)*100,2)}")
# accuracy

print(f"Baseline LR Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rfc_pred) * 100:.2f}%")
# f1

print(f"Baseline LR f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
print(f"Random Forest f1: {f1_score(y_test, rfc_pred) * 100:.2f}%")
# confusion matrix
pd.DataFrame(confusion_matrix(y_test, rfc_pred))
# recall score

print(f"Baseline LR recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
print(f"Random Forest recall: {recall_score(y_test, rfc_pred) * 100:.2f}%")
# precision score

print(f"Baseline LR precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
print(f"Random Forest precision: {precision_score(y_test, rfc_pred) * 100:.2f}%")
import lightgbm as lgb

# num_leaves=31, learning_rate=0.05, n_estimators=20, is_unbalanced=True
gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20).fit(X_train, y_train, eval_metric='auc')

gbm_pred = gbm.predict(X_test)
gbm_score = cross_val_score(gbm, X_train, y_train, scoring='roc_auc', cv=5)
print(f"LightGBM Mean ROC AUC: {round(mean(gbm_score)*100,2)}")
# accuracy

print(f"Baseline LR Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"LightGBM Accuracy: {accuracy_score(y_test, gbm_pred) * 100:.2f}%")
# f1

print(f"Baseline LR f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
print(f"LightGBM f1: {f1_score(y_test, gbm_pred) * 100:.2f}%")
# confusion matrix
pd.DataFrame(confusion_matrix(y_test, gbm_pred))
# recall score

print(f"Baseline LR recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
print(f"LightGBM recall: {recall_score(y_test, gbm_pred) * 100:.2f}%")
# precision score

print(f"Baseline LR precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
print(f"LightGBM precision: {precision_score(y_test, gbm_pred) * 100:.2f}%")
from sklearn.utils import resample
# Separate input features and target
y = df["Class"]
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)
X.head()
# separate minority and majority classes
not_fraud = X[X["Class"]==0]
fraud = X[X["Class"]==1]

print(len(not_fraud))
print(len(fraud))
# downsample majority
not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = len(fraud), # match minority n
                                random_state = 42) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])

# checking counts
print(downsampled["Class"].value_counts())

sns.countplot('Class', data=downsampled)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
y_train = downsampled["Class"]
X_train = downsampled.drop('Class', axis=1)
from sklearn.manifold import TSNE

# T-SNE Implementation
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_train.values)

# PCA Implementation
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_train.values)

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X_train.values)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()
# trying logistic regression again with the undersampled dataset

undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)
# accuracy

print(f"Baseline LR Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling Accuracy: {accuracy_score(y_test, undersampled_pred) * 100:.2f}%")
# f1

print(f"Baseline LR f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling f1: {f1_score(y_test, undersampled_pred) * 100:.2f}%")

# confusion matrix
print(pd.DataFrame(confusion_matrix(y_test, lr_pred)))
print()
print(pd.DataFrame(confusion_matrix(y_test, undersampled_pred)))
# recall

print(f"Baseline LR recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling recall: {recall_score(y_test, undersampled_pred) * 100:.2f}%")
# precision

print(f"Baseline LR precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling precision: {precision_score(y_test, undersampled_pred) * 100:.2f}%")
# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=42) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
print(upsampled.Class.value_counts())

sns.countplot('Class', data=upsampled)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
y_train = upsampled["Class"]
X_train = upsampled.drop('Class', axis=1)
# T-SNE Implementation
# X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_train.values)

# PCA Implementation
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_train.values)

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X_train.values)
f, (ax2, ax3) = plt.subplots(1, 2, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax1.set_title('t-SNE', fontsize=14)

# ax1.grid(True)

# ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()
# trying logistic regression again with the balanced dataset

upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)
# accuracy

print(f"Baseline LR Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"Oversampling Accuracy: {accuracy_score(y_test, upsampled_pred) * 100:.2f}%")
# f1 score
f1_score(y_test, upsampled_pred)

print(f"Baseline LR f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
print(f"Oversampling f1: {f1_score(y_test, upsampled_pred) * 100:.2f}%")
# confusion matrix
pd.DataFrame(confusion_matrix(y_test, upsampled_pred))
# recall

print(f"Baseline LR recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
print(f"Oversampling recall: {recall_score(y_test, upsampled_pred) * 100:.2f}%")
# precision

print(f"Baseline LR precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
print(f"Oversampling precision: {precision_score(y_test, upsampled_pred) * 100:.2f}%")
from imblearn.over_sampling import SMOTE
# Separate input features and target
y = df["Class"]
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)
# check new class counts
print(y_train.value_counts())
# T-SNE Implementation
# X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_train.values)

# PCA Implementation
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_train.values)

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X_train.values)
f, (ax2, ax3) = plt.subplots(1, 2, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax1.set_title('t-SNE', fontsize=14)

# ax1.grid(True)

# ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()
smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = smote.predict(X_test)
# ROC AUC

print(f"Baseline LR Mean ROC AUC: {round(mean(lr_score)*100,2)}")

smote_score = cross_val_score(smote, X_train, y_train, scoring='roc_auc', cv=5)
print(f"SMOTE ROC AUC: {round(mean(smote_score)*100,2)}")
# accuracy

print(f"Baseline LR Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"SMOTE Accuracy: {accuracy_score(y_test, smote_pred) * 100:.2f}%")
# f1

print(f"Baseline LR f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
print(f"SMOTE f1: {f1_score(y_test, smote_pred) * 100:.2f}%")
# confusion matrix
pd.DataFrame(confusion_matrix(y_test, smote_pred))
# recall

print(f"Baseline LR recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
print(f"SMOTE recall: {recall_score(y_test, smote_pred) * 100:.2f}%")
# precision

print(f"Baseline LR precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
print(f"SMOTE precision: {precision_score(y_test, smote_pred) * 100:.2f}%")
from imblearn.under_sampling import RandomUnderSampler
# Separate input features and target
y = df["Class"]
X = df.drop('Class', axis=1)

# setting up testing and training sets
print("Original y_train class counts")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(y_train.value_counts())
print()

# Under-Sample to 10%
print("Under-Sampled class counts")
rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
X_train, y_train = rus.fit_sample(X_train, y_train)
print(y_train.value_counts())
print()

# Generate 10 times synthetic examples using SMOTE
print("SMOTE used class counts")
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)
print(y_train.value_counts())
print()
smote_downsample = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_downsample_pred = smote_downsample.predict(X_test)
# ROC AUC

print(f"Baseline LR Mean ROC AUC: {round(mean(lr_score)*100,2)}")

smote_downsample_score = cross_val_score(smote_downsample, X_train, y_train, scoring='roc_auc', cv=5)
print(f"Undersampling and SMOTE ROC AUC: {round(mean(smote_downsample_score)*100,2)}")
# accuracy

print(f"Baseline LR Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling and SMOTE Accuracy: {accuracy_score(y_test, smote_downsample_pred) * 100:.2f}%")
# f1

print(f"Baseline LR f1: {f1_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling and SMOTE f1: {f1_score(y_test, smote_downsample_pred) * 100:.2f}%")
# confusion matrix
print(pd.DataFrame(confusion_matrix(y_test, lr_pred)))
print()
print(pd.DataFrame(confusion_matrix(y_test, smote_downsample_pred)))
# recall

print(f"Baseline LR recall: {recall_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling and SMOTE recall: {recall_score(y_test, smote_downsample_pred) * 100:.2f}%")
# precision

print(f"Baseline LR precision: {precision_score(y_test, lr_pred) * 100:.2f}%")
print(f"Undersampling and SMOTE precision: {precision_score(y_test, smote_downsample_pred) * 100:.2f}%")
# Prepare data for modeling
# Separate input features and target
y = df["Class"]
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# predict on test set
rfc_pred = rfc.predict(X_test)
# Class counts
num_neg = y_test.value_counts()[0]
num_pos = y_test.value_counts()[1]
print(f"{num_neg} negative records in y_test")
print(f"{num_pos} positive records in y_test")
rfc_cw = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0:num_neg,1:num_pos}).fit(X_train, y_train)

# predict on test set
rfc_cw_pred = rfc_cw.predict(X_test)
rfc_bl = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)

# predict on test set
rfc_bl_pred = rfc_bl.predict(X_test)
# accuracy

print(f"Baseline RF Accuracy: {accuracy_score(y_test, rfc_pred) * 100:.2f}%")
print(f"Class-weighted RF Accuracy: {accuracy_score(y_test, rfc_cw_pred) * 100:.2f}%")
print(f"Balanced RF Accuracy: {accuracy_score(y_test, rfc_bl_pred) * 100:.2f}%")
# f1
# lr_pred
print(f"Baseline RF f1: {f1_score(y_test, rfc_pred) * 100:.2f}%")
print(f"Class-weighted RF f1: {f1_score(y_test, rfc_cw_pred) * 100:.2f}%")
print(f"Balanced RF f1: {f1_score(y_test, rfc_bl_pred) * 100:.2f}%")
# confusion matrix
print("Standard RFC")
print(pd.DataFrame(confusion_matrix(y_test, rfc_pred)))
print()
print("Class-weighted RFC")
print(pd.DataFrame(confusion_matrix(y_test, rfc_cw_pred)))
print()
print("Balanced RFC")
print(pd.DataFrame(confusion_matrix(y_test, rfc_bl_pred)))
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# predict on test set
brf_pred = brf.predict(X_test)
# accuracy

print(f"Baseline RF Accuracy: {accuracy_score(y_test, rfc_pred) * 100:.2f}%")
print(f"imblearn Balanced RF Accuracy: {accuracy_score(y_test, brf_pred) * 100:.2f}%")
# f1

print(f"Baseline RF f1: {f1_score(y_test, rfc_pred) * 100:.2f}%")
print(f"imblearn Balanced RF f1: {f1_score(y_test, brf_pred) * 100:.2f}%")
# confusion matrix
print("Standard RFC")
print(pd.DataFrame(confusion_matrix(y_test, rfc_pred)))
print()
print("imblearn Balanced RFC")
print(pd.DataFrame(confusion_matrix(y_test, brf_pred)))
# setting up testing and training sets
X_train_equal, X_test_equal, y_train_equal, y_test_equal = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(round(y_test_equal.value_counts()[0]/len(y_test_equal),7))
print(round(y_train_equal.value_counts()[0]/len(y_train_equal),7))
rfc_equal = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_equal, y_train_equal)

# predict on test set
rfc_equal_pred = rfc_equal.predict(X_test_equal)
# accuracy

print(f"Baseline RF accuracy: {accuracy_score(y_test, rfc_pred) * 100:.2f}%")
print(f"Stratified data RF accuracy: {accuracy_score(y_test_equal, rfc_equal_pred) * 100:.2f}%")
# f1

print(f"Baseline RF f1: {f1_score(y_test, rfc_pred) * 100:.2f}%")
print(f"Stratified data RF f1: {f1_score(y_test_equal, rfc_equal_pred) * 100:.2f}%")
# confusion matrix
print("Standard RFC")
print(pd.DataFrame(confusion_matrix(y_test, rfc_pred)))
print()
print("Stratified data RFC")
print(pd.DataFrame(confusion_matrix(y_test_equal, rfc_equal_pred)))