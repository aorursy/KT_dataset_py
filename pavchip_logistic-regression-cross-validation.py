%matplotlib inline
from sklearn import linear_model, grid_search, cross_validation, metrics, preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def enumerate_not_nan_feature_values(data, feature, inplace=False):
    not_nan_feature_values = data[pd.notnull(data[feature])][feature]
    feature_values = list(set(not_nan_feature_values.values))
    if inplace:
        data.loc[not_nan_feature_values.index, feature] = not_nan_feature_values.map(lambda el: feature_values.index(el))
    else:
        return not_nan_feature_values.map(lambda el: feature_values.index(el))
train_data = pd.read_csv("../input/train.csv", index_col="PassengerId")
train_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

imputer = preprocessing.Imputer()

# fill nan "Age" feature
train_data["Age"] = imputer.fit_transform(train_data["Age"].values.reshape(-1, 1)).astype(int)

# enumerate and fill nan "Embarked" feature
enumerate_not_nan_feature_values(train_data, "Embarked", True)
train_data["Embarked"] = imputer.fit_transform(train_data["Embarked"].values.reshape(-1, 1)).astype(int)

# enumerate "Sex" feature
enumerate_not_nan_feature_values(train_data, "Sex", True)

train_data, train_labels = train_data[train_data.columns[1:]].values, train_data.Survived.values

trainX, testX, trainY, testY = cross_validation.train_test_split(train_data, train_labels, test_size=0.3)
scaler = preprocessing.StandardScaler()
scaler.fit(trainX, trainY)
scaled_trainX, scaled_testX = scaler.transform(trainX), scaler.transform(testX)
%%time
params_grid = {
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag"],
    "C": np.linspace(0.25, 2, 15),
    "tol": np.linspace(0.0001, 0.5, 20)
}

lrc1 = linear_model.LogisticRegression()
lrc2 = linear_model.LogisticRegression()
lrc3 = linear_model.LogisticRegression(max_iter=10000)

gs1 = grid_search.GridSearchCV(lrc3, params_grid, cv=4)
gs2 = grid_search.GridSearchCV(lrc3, params_grid, cv=4)

scoresY = lrc1.fit(trainX, trainY).decision_function(testX)
scaled_scoresY = lrc2.fit(scaled_trainX, trainY).decision_function(scaled_testX)
gs_not_scaled_scoresY = gs1.fit(trainX, trainY).decision_function(testX)
gs_scaled_scoresY = gs2.fit(scaled_trainX, trainY).decision_function(scaled_testX)
predictedY = lrc1.predict(testX)
scaled_features_predictedY = lrc2.predict(scaled_testX)
gs_not_scaled_predictedY = gs1.predict(testX)
gs_scaled_predictedY = gs2.predict(scaled_testX)

target_names = ["not survived", "survived"]

print("Not scaled data:")
print(metrics.classification_report(testY, predictedY, target_names=target_names))

print("Scaled data:")
print(metrics.classification_report(testY, scaled_features_predictedY, target_names=target_names))

print("Grid search with not scaled data:")
print(metrics.classification_report(testY, gs_not_scaled_predictedY, target_names=target_names))

print("Grid search with scaled data:")
print(metrics.classification_report(testY, gs_scaled_predictedY, target_names=target_names))
fpr, tpr, _ = metrics.roc_curve(testY, scoresY)
fpr_scaled, tpr_scaled, _ = metrics.roc_curve(testY, scaled_scoresY)
fpr_gs_not_scaled, tpr_gs_not_scaled, _ = metrics.roc_curve(testY, gs_not_scaled_scoresY)
fpr_gs_scaled, tpr_gs_scaled, _ = metrics.roc_curve(testY, gs_scaled_scoresY)

plt.figure(figsize=(14, 6))
plt.plot(fpr, tpr, label="not scaled\nAUC - ROC score: %.4f" % metrics.roc_auc_score(testY, scoresY))
plt.plot(fpr_scaled, tpr_scaled,
         label="scaled\nAUC - ROC score: %.4f" % metrics.roc_auc_score(testY, scaled_scoresY))
plt.plot(fpr_gs_not_scaled, tpr_gs_not_scaled,
         label="grid search not scaled\nAUC - ROC score: %.4f" % metrics.roc_auc_score(testY, gs_not_scaled_scoresY))
plt.plot(fpr_gs_scaled, tpr_gs_scaled,
         label="grid search scaled\nAUC - ROC score: %.4f" % metrics.roc_auc_score(testY, gs_scaled_scoresY))
plt.title("ROC curve")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right")
plt.show()
precision, recall, _ = metrics.precision_recall_curve(testY, scoresY)
precision_scaled, recall_scaled, _ = metrics.precision_recall_curve(testY, scaled_scoresY)
precision_gs_not_scaled, recall_gs_not_scaled, _ = metrics.precision_recall_curve(testY, gs_not_scaled_scoresY)
precision_gs_scaled, recall_gs_scaled, _ = metrics.precision_recall_curve(testY, gs_scaled_scoresY)

plt.figure(figsize=(14, 6))
plt.plot(recall, precision, label="not scaled\nAUC - PR Score: %.4f" % metrics.auc(fpr, tpr))
plt.plot(recall_scaled, precision_scaled, label="scaled\nAUC - PR score: %.4f" % metrics.auc(fpr_scaled, tpr_scaled))
plt.plot(recall_gs_not_scaled, precision_gs_not_scaled,
         label="grid search not scaled\nAUC - PR score: %.4f" % metrics.auc(fpr_gs_not_scaled, tpr_gs_not_scaled))
plt.plot(recall_gs_scaled, precision_gs_scaled,
         label="grid search scaled\nAUC - PR score: %.4f" % metrics.auc(fpr_gs_scaled, tpr_gs_scaled))
plt.title("PR curve")
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.legend(loc="lower left")
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plt.subplots_adjust(wspace=0.7)

labels = ["not survived", "survived"]

cm = metrics.confusion_matrix(testY, predictedY)
cm_scaled = metrics.confusion_matrix(testY, scaled_features_predictedY)
cm_grid_search_not_scaled = metrics.confusion_matrix(testY, gs_not_scaled_predictedY)
cm_grid_search_scaled = metrics.confusion_matrix(testY, gs_scaled_predictedY)

colobars = []
cax = axes[0, 0].imshow(cm, interpolation="nearest")
cax_scaled = axes[0, 1].imshow(cm_scaled, interpolation="nearest")
cax_grid_search_not_scaled = axes[1, 0].imshow(cm_grid_search_not_scaled, interpolation="nearest")
cax_grid_search_scaled = axes[1, 1].imshow(cm_grid_search_scaled, interpolation="nearest")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0, 0].annotate(cm[i, j], xy=(j, i),
                      horizontalalignment="center",
                      verticalalignment="center",
                      fontsize=16, color="w")
        
for i in range(cm_scaled.shape[0]):
    for j in range(cm_scaled.shape[1]):
        axes[0, 1].annotate(cm_scaled[i, j], xy=(j, i),
                      horizontalalignment="center",
                      verticalalignment="center",
                      fontsize=16, color="w")
        
for i in range(cm_grid_search_not_scaled.shape[0]):
    for j in range(cm_grid_search_not_scaled.shape[1]):
        axes[1, 0].annotate(cm_grid_search_not_scaled[i, j], xy=(j, i),
                      horizontalalignment="center",
                      verticalalignment="center",
                      fontsize=16, color="w")

for i in range(cm_grid_search_scaled.shape[0]):
    for j in range(cm_grid_search_scaled.shape[1]):
        axes[1, 1].annotate(cm_grid_search_scaled[i, j], xy=(j, i),
                      horizontalalignment="center",
                      verticalalignment="center",
                      fontsize=16, color="w")

fig.suptitle("Confusion matrices", fontsize=15)
        
axes[0, 0].set_title("Not scaled data", y=1.1)
axes[0, 0].xaxis.tick_top()
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(labels)
axes[0, 0].set_yticklabels(labels)
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("True")

axes[0, 1].set_title("Scaled data", y=1.1)
axes[0, 1].xaxis.tick_top()
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(labels)
axes[0, 1].set_yticklabels(labels)
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("True")

axes[1, 0].set_title("Grid search with not scaled data", y=1.1)
axes[1, 0].xaxis.tick_top()
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(labels)
axes[1, 0].set_yticklabels(labels)
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("True")

axes[1, 1].set_title("Grid search with scaled data", y=1.1)
axes[1, 1].xaxis.tick_top()
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_yticklabels(labels)
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("True")

plt.show()