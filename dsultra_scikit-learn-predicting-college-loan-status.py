import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
raw_data = pd.read_csv('loan_train.csv')
raw_data.head()
raw_data = raw_data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
raw_data.head()
data = raw_data.drop(['effective_date', 'due_date'], axis = 1)
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
data['loan_status'] = data['loan_status'].map({'COLLECTION': 0, 'PAIDOFF': 1})
data.count()
dummies = pd.get_dummies(data['education'], prefix='edu').astype('float')
print(type(dummies))
dummies.head()
dummies.columns = ['edu_bachelor', 'edu_hs_or_lower', 'edu_master_or_higher', 'edu_college']
dummies = dummies.drop('edu_master_or_higher', axis = 1)
dummies.count()
data.count()
x = data.drop('education', axis = 1)
x = x.join(dummies)
print(type(x), x.count())
# for i in range(len(x.index)):
#     x['edu_bachelor'][i] = dummies['edu_bachelor'][i]
#     x['edu_hs_or_lower'][i] = dummies['edu_hs_or_lower'][i]
#     x['edu_college'][i] = dummies['edu_college'][i]
x.head()
y = x['loan_status']
y = y.values
x = x.drop('loan_status', axis = 1)
min_max_scaler = preprocessing.MinMaxScaler()
x_np = x.values
x_scaled = min_max_scaler.fit_transform(x_np)
x_scaled
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2)
from sklearn.metrics import jaccard_score, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier

def find_best_knn(k):
    knn_predictions = []
    for k_ in range(1, k):
        knn = KNeighborsClassifier(n_neighbors = k_, n_jobs = -1)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        y_pred_proba = knn.predict_proba(x_test)
        score = knn.score(x_test, y_test)
        jaccard = jaccard_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        knn_predictions.append([y_pred, y_pred_proba, score, jaccard, f1])
    return knn_predictions
knn_predictions = find_best_knn(20)
knn_predictions[0][4]
plt.plot([i + 2 for i in range(len(knn_predictions))],
         [knn_predictions[i][3] for i in range(len(knn_predictions))], label = 'Jaccard')
plt.plot([i + 2 for i in range(len(knn_predictions))],
         [knn_predictions[i][4] for i in range(len(knn_predictions))], label = 'F1')
plt.title('K-nearest neighbors Jaccard & F1')
plt.xlabel('K')
plt.ylabel('Jaccard score / F1 Score')
plt.ylim(0,1)
plt.xticks([i for i in range(2, 21)])
plt.legend()
plt.show()
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
dec_tree_y_predict = dec_tree.predict(x_test)
dec_tree_y_predict_proba = dec_tree.predict_proba(x_test)
dec_tree_score = dec_tree.score(x_test, y_test)
dec_tree_f1_score = f1_score(y_test, dec_tree_y_predict)
dec_tree_jaccard_score = jaccard_score(y_test, dec_tree_y_predict)
dec_tree_f1_score, dec_tree_jaccard_score
from sklearn.svm import SVC

def test_poly_svm(poly_count):
    poly = []
    for p in range(poly_count):
        svm = SVC(kernel = 'poly', degree = p)
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        score = svm.score(x_test, y_test)
        f1 = f1_score(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred)
        poly.append([y_pred, score, f1, jaccard])
    return poly

def svm_kernel_test():
    f1_ = []
    jaccard_ = []
    for kernel in ['linear', 'rbf', 'sigmoid']:
        svm = SVC(kernel = kernel)
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        score = svm.score(x_test, y_test)
        f1 = f1_score(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred)
        f1_.append(f1)
        jaccard_.append(jaccard)
    return [f1_, jaccard_]
poly = 10
test_poly = test_poly_svm(poly)
kernel_comp = svm_kernel_test()
plt.plot([i for i in range(poly)],
         [test_poly[i][2] for i in range(len(test_poly))], label = 'F1')
plt.plot([i for i in range(poly)],
         [test_poly[i][3] for i in range(len(test_poly))], label = 'Jaccard')
plt.title('SVM F1 score over poly degree')
plt.xlabel('Poly degree')
plt.ylabel('F1 score / Jaccard score')
plt.ylim(0, 1)
plt.legend()
plt.show()
kernel_comp
# plt.bar(0 - 2, [f for f in kernel_comp[0]], label = 'Jaccard')
# plt.bar(0, [j for j in kernel_comp[1]], label = 'F1')
# plt.title('Linear, RBF, Sigmoid SVMs')
# plt.ylabel('Score')
# plt.legend()
# plt.xticks(ticks = ['Linear', 'RBF', 'Sigmoid'])
# plt.show()

plt.plot([i for i in range(3)],
         [f for f in kernel_comp[1]], label = 'F1')
plt.plot([i for i in range(3)],
         [f for f in kernel_comp[0]], label = 'Jaccard')
plt.title('SVM F1 score over poly degree')
plt.xlabel('Linear, RBF, Sigmoid')
plt.ylabel('F1 score / Jaccard score')
plt.ylim(0, 1)
plt.legend()
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

log_reg_solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

def test_log_reg():
    f1_ = []
    jaccard_ = []
    log_loss_ = []
    for solver in log_reg_solvers:
        log_reg = LogisticRegression(solver = solver, n_jobs = -1)
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred)
        l_loss = log_loss(y_test, y_pred)
        f1_.append(f1)
        jaccard_.append(jaccard)
        log_loss_.append(l_loss)
    return [f1_, jaccard_, log_loss_]
log_reg_scores = test_log_reg()
log_reg_scores
plt.plot([i for i in range(5)],
         [f for f in log_reg_scores[0]], label = 'F1')
plt.plot([i for i in range(5)],
         [j for j in log_reg_scores[1]], label = 'Jaccard')
plt.plot([i for i in range(5)],
         [l for l in log_reg_scores[2]], label = 'Log loss')
plt.title('Solvers')
plt.xlabel('')
# plt.ylim(0, 1)
plt.legend()
plt.show()
knn_f1 = max([knn_predictions[i][4] for i in range(len(knn_predictions))])
knn_jaccard = max([knn_predictions[i][3] for i in range(len(knn_predictions))])
svm_f1 = max(kernel_comp[1])
svm_jaccard = max(kernel_comp[0])
logreg_f1 = max(log_reg_scores[0])
logreg_jaccard = max(log_reg_scores[1])
knn_ = [knn_f1, knn_jaccard]
dec_ = [dec_tree_f1_score, dec_tree_jaccard_score]
svm_ = [svm_f1, svm_jaccard]
log_ = [logreg_f1, logreg_jaccard]

score_df = pd.DataFrame({'KNN': knn_, 'Descision tree': dec_, 'SVM': svm_, 'Logistic regression': log_}, 
                        index = ['F1', 'Jaccard'])
score_df
import seaborn as sns
ax = score_df.plot.bar(ylim = (0, 1), rot=0, figsize = (14, 10))