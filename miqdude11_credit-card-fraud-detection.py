# !pip install mglearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import mglearn
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
class_count = data['Class'].value_counts()
print("Valid {:.3f}%".format(class_count[0] / data.shape[0] * 100))
print("fraud {:.3f}%".format(class_count[1] / data.shape[0] * 100))
sb.barplot([0,1], data['Class'].value_counts())
plt.xticks([0,1], ['Valid', 'Fraud'])
# visualizing how each class behave in respect of time
fig, axes = plt.subplots(15, 2, figsize=(20, 24))

valid = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]
ax = axes.ravel()
features = data.drop(['Class', 'Time'], axis=1).columns.to_list()

for i in range(29):
    sb.scatterplot(valid['Time'], valid[features[i]], c=['red'], ax=ax[i])
    sb.scatterplot(fraud['Time'], fraud[features[i]], ax=ax[i])
# correlation matrix
matrix_corr = data.corr()
plt.figure(figsize=(20, 8))
sb.heatmap(matrix_corr, annot=True, cmap='viridis')
corr_df = matrix_corr.loc['Class']
corr_df = corr_df.drop('Class').sort_values()
plt.figure(figsize=(15,8))
sb.barplot(corr_df, corr_df.index)
corr_df.describe()
selected_features = corr_df[np.abs(corr_df) > 0.018].index.to_list()
selected_features
fig, axes = plt.subplots(ncols=5, nrows=6, figsize=(20, 8))
for i, feature, ax in zip(np.arange(30), data.columns.to_list(), axes.flat):
    sb.distplot(data[feature], ax=ax)
    ax.set_title(feature)
# Some algorithms may took hours to train
# to cover that let's take some sample
# of the data
sample = data.sample(frac = 0.1, random_state=42)
print("sample size at 10% from original ", sample.shape[0])
print("Number of valid transactions ", sample[sample['Class'] == 0].shape[0])
print("Number of fraud transactions ", sample[sample['Class'] == 1].shape[0])
# using PCA for visualization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mglearn # help visualize better

X = data[selected_features].values
y = data['Class'].values

# scaling X
X_scaled = StandardScaler().fit_transform(X)

# get 2 components for 2D visualization
pca = PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X)
print("Original shape {} reduced shape {}".format(X.shape, X_pca.shape))

mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], y)
# using TSNE for visualization
from sklearn.manifold import TSNE

# run the sample because it
# could took a long time if using
# whole dataset
X = sample[selected_features]
y = sample['Class']

X_scaled = StandardScaler().fit_transform(X)

# get 2 components for 2D visualization
tsne = TSNE()
tsne.fit(X_scaled)

X_tsne = pca.transform(X)
print("Original shape {} reduced shape {}".format(X.shape, X_tsne.shape))

mglearn.discrete_scatter(X_tsne[:, 0], X_tsne[:, 1], y)
# baseline model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

X, y = data[selected_features], data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gboost = GradientBoostingClassifier(learning_rate=0.01)
gboost.fit(X_train, y_train)

y_decision_gboost = gboost.decision_function(X_test)
score_auc = roc_auc_score(y_test, y_decision_gboost)
print("AUC score ", score_auc)
test_fraud = y_test[y_test == 1].count()
test_valid = y_test[y_test == 0].count()
print("Test valid ", test_valid)
print("Test fraud ", test_fraud)
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

y_pred_gboost = gboost.predict(X_test)

def show_performace(y_predicted, y_test):
    print("number of errors %d" % (y_predicted != y_test).sum())
    print("accuracy score %f" % accuracy_score(y_test, y_predicted))
    print("f1 score : %.3f" % f1_score(y_test, y_predicted))
    print(classification_report(y_test, y_predicted, labels=[0,1]))
    print(confusion_matrix(y_test, y_predicted))
    
show_performace(y_pred_gboost, y_test)
from sklearn.metrics import precision_recall_curve

def show_precision_recall(y_decision, y_test):
    precision, recall, thresholds = precision_recall_curve(
                                    y_test, y_decision)

    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
    label="threshold zero", fillstyle="none", c='k', mew=2)
    plt.plot(precision, recall, label="precision recall curve")
    plt.ylabel("Recall")
    plt.xlabel("Precision")
    
show_precision_recall(y_decision_gboost, y_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=41)
rf.fit(X_train, y_train)

y_decision_rf = rf.predict_proba(X_test)[:, 1]
score_auc = roc_auc_score(y_test, y_decision_rf)
print("AUC score ", score_auc)
y_pred_rf = rf.predict(X_test)

show_performace(y_pred_rf, y_test)
show_precision_recall(y_decision_rf - 0.5, y_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_decision_logreg = logreg.decision_function(X_test)
score_auc = roc_auc_score(y_test, y_decision_logreg)
print("AUC score ", score_auc)
y_pred_logreg = logreg.predict(X_test)
show_performace(y_pred_logreg, y_test)
show_precision_recall(y_decision_logreg, y_test)
correct_ans = y_test[(y_test == y_pred_rf) & (y_test == 1)]
fraud_test = y_test[y_test == 1]

total_amount_fraud_detected = data.iloc[correct_ans]['Amount'].sum()
total_amount_fraud = data.iloc[fraud_test]['Amount'].sum()

saved_loss_percentage = total_amount_fraud_detected/total_amount_fraud * 100

print("Total amount of fraud detected {:.2f}".format(total_amount_fraud_detected))
print("Total amount of fraud          {:.2f}".format(total_amount_fraud))
print("Saved loss percentage          {:.2F}%".format(saved_loss_percentage))