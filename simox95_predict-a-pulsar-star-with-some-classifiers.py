import os
import pandas as pd

DATA_PATH = '../input/predicting-a-pulsar-star/'
FILE_NAME = 'pulsar_stars.csv'
def load_data(data_path=DATA_PATH, file_name=FILE_NAME):
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)

dataset = load_data()
dataset.head()
dataset.info()
not_pulsar, pulsar = dataset['target_class'].value_counts()

print("Pulsar Star:\t ", pulsar,"\nNot Pulsar Star: ", not_pulsar)
import matplotlib.pyplot as plt

custom_color = '#ff7400'
dataset.hist(bins=50, figsize=(20,15), color=custom_color)
plt.show()
X, y =  dataset.drop('target_class', axis=1), dataset['target_class'].copy()

print("X:",X.shape,"\ny:",y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("80% - X_train size:", X_train.shape[0], " y_train size:", y_train.shape[0])
print("20% - X_test size:  ", X_test.shape[0], " y_test size:\t ", y_test.shape[0])
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_attr_pipeline = Pipeline([
                        ('std_scaler', StandardScaler())
                    ])
cols = list(X)
pipeline = ColumnTransformer([
                ('num_attr_pipeline', num_attr_pipeline, cols)
            ])

X_train_prepared = pipeline.fit_transform(X_train)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train_prepared, y_train)
# trying a prediction
pulsar_star = 42
single_prediction_sgd = sgd_clf.predict([X.iloc[pulsar_star]])

print("Expected value: ",[y[pulsar_star]],"\nPredicted value:", single_prediction_sgd)
from sklearn.model_selection import cross_val_score

K = 3
scores_sgd = cross_val_score(sgd_clf, X_train_prepared, y_train, cv=K, scoring="accuracy")

print("Accuracy:", round(scores_sgd.mean(),4))
import seaborn as sns

#Function for drawing confusion matrix
def draw_confusion_matrix(cm, title = 'Confusion Matrix', color = custom_color):
    palette = sns.light_palette(color, as_cmap=True)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d', cmap=palette)
    # labels
    ax.set_title('\n' + title + '\n', fontweight='bold')
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold');
    ax.xaxis.set_ticklabels(['Not Pulsar', 'Pulsar Star'], ha = 'center')
    ax.yaxis.set_ticklabels(['Not Pulsar', 'Pulsar Star'], va = 'center')
    
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred_sgd = cross_val_predict(sgd_clf, X_train_prepared, y_train, cv=K)

cm_sgd = confusion_matrix(y_train, y_train_pred_sgd)

draw_confusion_matrix(cm_sgd, 'SDG - Confusion Matrix')
from sklearn.metrics import precision_score, recall_score

precision_sgd = precision_score(y_train, y_train_pred_sgd)
recall_sgd = recall_score(y_train, y_train_pred_sgd)

print('Precision:', round(precision_sgd,4), '\nRecall:   ', round(recall_sgd,4))
from sklearn.metrics import f1_score

f1_score_sgd = f1_score(y_train, y_train_pred_sgd)

print('F1-Score:', round(f1_score_sgd,4))
#Function for plotting precision and recall
def plot_precison_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, custom_color)
    plt.title('Precision vs Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
from sklearn.metrics import precision_recall_curve

# decision scores
y_scores_sgd = cross_val_predict(sgd_clf, X_train_prepared, y_train, cv=K, method="decision_function")

precisions_sgd, recalls_sgd, _ = precision_recall_curve(y_train, y_scores_sgd)
  
plot_precison_vs_recall(precisions_sgd, recalls_sgd)
#Function for plotting the ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, custom_color, label='Area: %0.3f' %roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate - Recall')
    plt.legend(loc='lower right')
    plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

fpr_sgd, tpr_sgd, _ = roc_curve(y_train, y_scores_sgd)
roc_auc_sgd = roc_auc_score(y_train, y_scores_sgd)

plot_roc_curve(fpr_sgd, tpr_sgd, roc_auc_sgd)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_prepared, y_train)
# trying a prediction
single_prediction_knn = knn_clf.predict([X.iloc[pulsar_star]])

print("Expected value: ",[y[pulsar_star]],"\nPredicted value:", single_prediction_knn)
scores_knn = cross_val_score(knn_clf, X_train_prepared, y_train, cv=K, scoring="accuracy")

print("Accuracy:", round(scores_knn.mean(),4))
y_train_pred_knn = cross_val_predict(knn_clf, X_train_prepared, y_train, cv=K)

cm_knn = confusion_matrix(y_train, y_train_pred_knn)

draw_confusion_matrix(cm_knn, 'KNN - Confusion Matrix')
precision_knn = precision_score(y_train, y_train_pred_knn)
recall_knn = recall_score(y_train, y_train_pred_knn)

print('Precision:', round(precision_knn,4), '\nRecall:   ', round(recall_knn,4))
f1_score_knn = f1_score(y_train, y_train_pred_knn)

print('F1-Score:', round(f1_score_knn,4))
# decision scores
y_probas_knn = cross_val_predict(knn_clf, X_train_prepared, y_train, cv=K, method="predict_proba")
y_scores_knn = y_probas_knn[:,1]

precisions_knn, recalls_knn, _ = precision_recall_curve(y_train, y_scores_knn)
    
plot_precison_vs_recall(precisions_knn, recalls_knn)
fpr_knn, tpr_knn, _ = roc_curve(y_train, y_scores_knn)
roc_auc_knn = roc_auc_score(y_train, y_scores_knn)

plot_roc_curve(fpr_knn, tpr_knn, roc_auc_knn)
from sklearn.svm import SVC

svm_poly_clf = SVC(kernel="poly", degree=3, coef0=10, C=5, random_state=42)
svm_poly_clf.fit(X_train_prepared, y_train)
# trying a prediction
single_prediction_svm_poly = svm_poly_clf.predict([X.iloc[pulsar_star]])

print("Expected value: ",[y[pulsar_star]],"\nPredicted value:", single_prediction_svm_poly)
scores_svm_poly = cross_val_score(svm_poly_clf, X_train_prepared, y_train, cv=K, scoring="accuracy")

print("Accuracy:", round(scores_svm_poly.mean(),4))
y_train_pred_svm_poly = cross_val_predict(svm_poly_clf, X_train_prepared, y_train, cv=K)

cm_svm = confusion_matrix(y_train, y_train_pred_svm_poly)

draw_confusion_matrix(cm_svm, 'SVM - Confusion Matrix')
precision_svm_poly = precision_score(y_train, y_train_pred_svm_poly)
recall_svm_poly = recall_score(y_train, y_train_pred_svm_poly)

print('Precision:', round(precision_svm_poly,4), '\nRecall:   ', round(recall_svm_poly,4))
f1_score_svm_poly = f1_score(y_train, y_train_pred_svm_poly)

print('F1-Score:', round(f1_score_svm_poly,4))
# decision scores
y_scores_svm_poly = cross_val_predict(svm_poly_clf, X_train_prepared, y_train, cv=K, method="decision_function")

precisions_svm_poly, recalls_svm_poly, _ = precision_recall_curve(y_train, y_scores_svm_poly)
    
plot_precison_vs_recall(precisions_svm_poly, recalls_svm_poly)
fpr_svm_poly, tpr_svm_poly, _ = roc_curve(y_train, y_scores_svm_poly)
roc_auc_svm_poly = roc_auc_score(y_train, y_scores_svm_poly)

plot_roc_curve(fpr_svm_poly, tpr_svm_poly, roc_auc_svm_poly)
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=42)

# Decision Trees don't require feature scaling
dt_clf.fit(X_train, y_train)
# trying a prediction
single_prediction_dt = dt_clf.predict([X.iloc[pulsar_star]])

print("Expected value: ",[y[pulsar_star]],"\nPredicted value:", single_prediction_dt)
scores_dt = cross_val_score(dt_clf, X_train, y_train, cv=K, scoring="accuracy")

print("Accuracy:", round(scores_dt.mean(),4))
y_train_pred_dt = cross_val_predict(dt_clf, X_train, y_train, cv=K)

cm_dt = confusion_matrix(y_train, y_train_pred_dt)

draw_confusion_matrix(cm_dt, 'DT - Confusion Matrix')
precision_dt = precision_score(y_train, y_train_pred_dt)
recall_dt = recall_score(y_train, y_train_pred_dt)

print('Precision:', round(precision_dt,4), '\nRecall:   ', round(recall_dt,4))
f1_score_dt = f1_score(y_train, y_train_pred_dt)

print('F1-Score:', round(f1_score_dt,4))
# predict probabilities
y_probas_dt = cross_val_predict(dt_clf, X_train, y_train, cv=K, method="predict_proba")
# decision scores
y_scores_dt = y_probas_dt[:,1]

precisions_dt, recalls_dt, _ = precision_recall_curve(y_train, y_scores_dt)
    
plot_precison_vs_recall(precisions_dt, recalls_dt)
fpr_dt, tpr_dt, _ = roc_curve(y_train, y_scores_dt)
roc_auc_dt = roc_auc_score(y_train, y_scores_dt)

plot_roc_curve(fpr_dt, tpr_dt, roc_auc_dt)
from sklearn.model_selection import GridSearchCV
sgd_param_grid = [{
    'loss':['hinge', 'squared_hinge', 'perceptron'],
    'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty':['l2','l1','elasticnet'],
    'l1_ratio':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'early_stopping':[True],
    'shuffle':[True,False],
    'random_state': [42]
}]
knn_param_grid = [{
    'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'weights':['uniform', 'distance'],
    'metric':['minkowski'],
    'p':[1.0, 1.5, 2.0, 2.5, 3.0, 3.5] 
}]
svm_param_grid = [{
    'C': [10, 15, 20],
    'coef0':[0, 10, 100],
    'kernel': ['poly'],
    'random_state': [42]
}]
dt_param_grid = [{
    'criterion':['gini', 'entropy'],
    'splitter':['best','random'],
    'max_depth':range(2,20,2),
    'max_leaf_nodes': list(range(2, 100, 10)),
    'max_features': [None, 'sqrt', 'log2'],
    'random_state': [42]
}]
models = {
    'sgd': SGDClassifier(),
    'knn': KNeighborsClassifier(),
    'svm': SVC(),
    'dt' : DecisionTreeClassifier()
}

params = {
    'sgd': sgd_param_grid,
    'knn': knn_param_grid,
    'svm': svm_param_grid,
    'dt' : dt_param_grid
}

grid_searches = {}
for key in models:
    model = models[key]
    param_grid = params[key]
    
    grid_search = GridSearchCV(model, param_grid, cv=K, scoring='f1')
    if key != 'dt':
        grid_search.fit(X_train_prepared, y_train)
    else:
        grid_search.fit(X_train, y_train)
    grid_searches[key] = grid_search
model_scores = {}
for key in grid_searches:
    cv_res = grid_searches[key].cv_results_
    score = max(cv_res['mean_test_score'])
    model_scores[key] = score

    print(key + ' f1-score:', round(model_scores[key],4))
key_final_model = max(model_scores, key=model_scores.get)

final_model = grid_searches[key_final_model].best_estimator_

final_model
import joblib

model = final_model
model_name = key_final_model + '_model.pkl'

# Save Model
joblib.dump(model, model_name)

# Load Model
# final_model = joblib.load(model_name)

X_test_prepared = pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_recall = recall_score(y_test, final_predictions)
final_precision = precision_score(y_test, final_predictions)
final_f1_score = f1_score(y_test, final_predictions)

print('Precision:',   round(final_precision,4),
      '\nRecall:   ', round(final_recall,4))
print('F1-Score: ',   round(final_f1_score,4))
precisions, recalls, _ = precision_recall_curve(y_test, final_predictions)
 
plot_precison_vs_recall(precisions, recalls)
final_cm = confusion_matrix(y_test, final_predictions)

draw_confusion_matrix(final_cm, 'Final Confusion Matrix')
# trying a prediction
single_prediction = final_model.predict([X.iloc[pulsar_star]])

print("Expected value: ",[y[pulsar_star]],"\nPredicted value:", single_prediction)
