import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
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


X = data[selected_features].values
y = data['Class'].values

# get 2 components for 2D visualization
pca = PCA(n_components=2)
pca.fit(X)

X_pca = pca.transform(X)
print("Original shape {} reduced shape {}".format(X.shape, X_pca.shape))


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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
X, y = data[selected_features], data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_knb_model=roc_auc_score(y_test, y_pred)*100
acc_knb_model
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
acc_log_reg=roc_auc_score(y_test, y_pred1)*100
acc_log_reg
clf2 = GaussianNB().fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
acc_nb=roc_auc_score(y_test, y_pred2)*100
acc_nb
clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
acc_rmf_model=roc_auc_score(y_test, y_pred4)*100
acc_rmf_model
clf5 = SVC(gamma='auto').fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
acc_svm_model=roc_auc_score(y_test, y_pred5)*100
acc_svm_model
sgd_model=SGDClassifier()
sgd_model.fit(X_train,y_train)
sgd_pred=sgd_model.predict(X_test)
acc_sgd=round(sgd_model.score(X_train,y_train)*100,10)
acc_sgd
xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
acc_xgb=round(xgb_model.score(X_train,y_train)*100,10)
acc_xgb
lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)
lgbm_pred=lgbm.predict(X_test)
acc_lgbm=round(lgbm.score(X_train,y_train)*100,10)
acc_lgbm
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
regr_pred=regr.predict(X_test)
acc_regr=round(regr.score(X_train,y_train)*100,10)
acc_regr
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_regr,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
correct_ans = y_test[(y_test == y_pred_rf) & (y_test == 1)]
fraud_test = y_test[y_test == 1]

total_amount_fraud_detected = data.iloc[correct_ans]['Amount'].sum()
total_amount_fraud = data.iloc[fraud_test]['Amount'].sum()

saved_loss_percentage = total_amount_fraud_detected/total_amount_fraud * 100

print("Total amount of fraud detected {:.2f}".format(total_amount_fraud_detected))
print("Total amount of fraud          {:.2f}".format(total_amount_fraud))
print("Saved loss percentage          {:.2F}%".format(saved_loss_percentage))