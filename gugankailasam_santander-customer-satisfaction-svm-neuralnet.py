import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

plt.style.use("seaborn-dark")
np.random.seed(42)
data = pd.read_csv('../input/santander-customer-satisfaction/train.csv').drop('ID', axis=1)
data.head()
data.shape
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
labels = data['TARGET'].unique()
target = data['TARGET'].value_counts()
ax.pie(target, labels = labels,autopct='%1.2f%%')
plt.show()
X = data.loc[:,data.columns != 'TARGET']
y = data['TARGET']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   stratify = y,
                                                   test_size = 0.10)
corr_matrix = X_train.corr()
#Init
tol = 0.3

# correlation diagram creation
def corr_tol(x):
    return x.apply(lambda x : True if (x >= tol or x <= -tol) else False)
bool_corr_matrix = corr_matrix.apply(lambda x : corr_tol(x))

for i in range(0,len(bool_corr_matrix)):
    bool_corr_matrix.iloc[i,i] = False

bool_corr_matrix = pd.DataFrame(np.tril(bool_corr_matrix, k=0), 
                                columns=bool_corr_matrix.columns, index=bool_corr_matrix.index)

plt.figure(figsize=(10,5))
plt.grid(True)
sns_plot = sns.heatmap(bool_corr_matrix)
# sns_plot.set_ylim(len(bool_corr_matrix)-1, -1)
plt.show()
# Automatic field selection

columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[0]):
        if corr_matrix.iloc[i,j] >= 0.8:
            if columns[j]:
                columns[j] = False

selected_columns = X_train.columns[columns]
selected_columns.shape
X_train = X_train[selected_columns]
X_test = X_test[selected_columns]
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# from sklearn.decomposition import PCA

# pca = PCA(n_components=0.99)
# pca.fit(X_train)
# print("---Explained Variance Ratio---")
# print(pca.explained_variance_ratio_.sum()*100)
# X_train = pca.transform(X_train)
# from sklearn.svm import SVC
# clf = SVC()

# model_for_cv = clf

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='accuracy')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# clf = SVC(probability=True)
# clf.fit(X_train, y_train)
# X_test = pca.transform(scaler.transform(X_test))
# y_pred = clf.predict(X_test)
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print("----Classification Report----")
# print(classification_report(y_test, y_pred))
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve

# logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# # plt.savefig('Log_ROC')
# plt.show()
# from imblearn.under_sampling import NearMiss
# undersample = NearMiss(sampling_strategy=0.2, version=1, n_neighbors=3)
# X_train_under, y_train_under = undersample.fit_sample(X_train, y_train)
# sns.countplot(y_train_under)
# from imblearn.over_sampling import SMOTE

# os = SMOTE(sampling_strategy=1)
# X_train_smote, y_train_smote = os.fit_sample(X_train, y_train)
# # X_train_smote, y_train_smote = os.fit_sample(X_train_under, y_train_under)
from imblearn.over_sampling import BorderlineSMOTE

os = BorderlineSMOTE()
X_train_smote, y_train_smote = os.fit_sample(X_train, y_train)
# X_train_smote, y_train_smote = os.fit_sample(X_train_under, y_train_under)
# sns.countplot(y_train_smote)
from xgboost import XGBClassifier
model = XGBClassifier()
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

# weights = [1, 10, 25, 50, 75, 99, 100, 1000]
weights = [1, 96]
param_grid = dict(scale_pos_weight=weights)

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X_train_smote, y_train_smote)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
model = XGBClassifier(scale_pos_weight=1)
model.fit(X_train_smote, y_train_smote)
# X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("----Classification Report----")
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred)
logit_roc_auc
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# plt.figure()      
# plt.plot(fpr, tpr, label='Classification (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# # plt.savefig('Log_ROC')
# plt.show()
test_submit = pd.read_csv('../input/santander-customer-satisfaction/test.csv').drop('ID', axis=1)
test_submit.head()
y_pred = model.predict(test_submit[selected_columns])
y_pred
submit = pd.read_csv('../input/santander-customer-satisfaction/sample_submission.csv')
submit.loc[:,'TARGET'] = y_pred
submit.head()
submit.to_csv('Submission.csv', index=False)
import tensorflow as tf
# model = tf.keras.models.Sequential([tf.keras.layers.Dense(140,input_dim=369, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(70, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(35, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(5, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
def fit_model(trainX, trainy, testX, testy, lrate, batch_size, epochs):
    # define model
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1000,input_dim=369, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(500, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(500, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    # compile model
    model.compile(optimizer = tf.optimizers.Adam(learning_rate=lrate),
              loss = 'binary_crossentropy',
              metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, batch_size=batch_size,
                       use_multiprocessing=True)
    # plot learning curves
    plt.figure(1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('lrate='+str(lrate), pad=-50)
    plt.figure(2)
    recall = [val for key, val in history.history.items() if 'recall' in key.lower()]
    plt.plot(recall[0], label='train')
    plt.plot(recall[1], label='test')
    plt.title("Recall")
    plt.figure(3)
    precision = [val for key, val in history.history.items() if 'precision' in key.lower()]
    plt.plot(precision[0], label='train')
    plt.plot(precision[1], label='test')
    plt.title("Precision")
    
    return model
model = fit_model(X_train, y_train, X_test_scaled, y_test, lrate=0.001, batch_size=512, epochs=500)
model.evaluate(X_test_scaled, y_test)
y_pred = model.predict_classes(X_test_scaled)
y_pred_prob = model.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("----Classification Report----")
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label='Classification (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
plt.show()