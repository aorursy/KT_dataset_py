import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
df=pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv",delimiter=";",index_col=None)
df.head()
df=df.drop(columns=["id"])
df["age"]=df["age"].div(365)
print("duplicate {}".format(df.duplicated().sum()))
print(df.isnull().sum())
corr= df.corr()
plt.figure(figsize=(16, 6))
sns.heatmap(corr,annot=True)
df.drop_duplicates(inplace=True)
print("duplicate {}".format(df.duplicated().sum()))
df.describe()
outlier_height=((df["height"]>200) | (df["height"]<140))
df=df[~outlier_height]
outlier_weight=((df["weight"]>150) | (df["weight"]<40))
df=df[~outlier_weight]
sns.lmplot(x='weight', y='height', hue='gender', data=df, fit_reg=False, height=6)
df["bmi"] = df["weight"]/ (df["height"]/100)**2
df=df.drop(columns=["height","weight"])
blood_pressure = df.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
print("Diastolic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))
outlier_bp1= ((df["ap_hi"]>250) | (df["ap_lo"]>160))
outlier_bp2 = ((df["ap_hi"] < 80) | (df["ap_lo"] < 30))
df = df[~outlier_bp1]
df= df[~outlier_bp2]
print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True)
print("Diastolic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))
blood_pressure = df.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
df.count()
corr= df.corr()
plt.figure(figsize=(16, 6))
sns.heatmap(corr,annot=True)
fig, ax =plt.subplots(1,2,figsize=(14,6))
sns.countplot(x=df["alco"],hue=df["cholesterol"],ax=ax[0])
sns.countplot(x=df["smoke"],hue=df["cholesterol"],ax=ax[1])
ax[0].set_title("Alcoholic vs Cholesterol")
ax[1].set_title("Smoker vs Cholesterol")
ax[0].set_xlabel("Acoholic")
ax[1].set_xlabel("Smoker")
ax[0].set_xticklabels(["No","Yes"])
ax[1].set_xticklabels(["No","Yes"])
ax[0].legend(["Low CHolesterol","High Cholesterol","Very High Cholesterol"],loc="center right")
ax[1].legend(["Low CHolesterol","High Cholesterol","Very High Cholesterol"],loc="center right")
fig.show()
fig, ax =plt.subplots(1,2,figsize=(14,6))
sns.countplot(x=df["alco"],hue=df["cardio"],ax=ax[0])
sns.countplot(x=df["smoke"],hue=df["cardio"],ax=ax[1])
ax[0].set_title("Alcoholic vs Cardio")
ax[1].set_title("Smoker vs Cardio")
ax[0].set_xlabel("Acoholic")
ax[1].set_xlabel("Smoker")
ax[0].set_xticklabels(["No","Yes"])
ax[1].set_xticklabels(["No","Yes"])
ax[0].legend(["Low CHolesterol","High Cholesterol","Very High Cholesterol"],loc="center right")
ax[1].legend(["Low CHolesterol","High Cholesterol","Very High Cholesterol"],loc="center right")
fig.show()
fig, ax = plt.subplots(figsize=(10,6))
ax=sns.boxplot(x=df["cholesterol"],y=df["bmi"])
ax.set_title("Boxplot of Cholesterol Level against bmi")
ax.set_xticklabels(["Low Cholesterol","High Cholesterol","Very High Cholesterol"])
ax.set_xlabel("Cholesterol")
ax.set_ylabel("bmi")
df['cardio'].value_counts()
X=df.drop(columns=["cardio"])
y=df["cardio"]
X_train,X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=0)
from sklearn.preprocessing import normalize
X = normalize(X)
X_train = normalize(X_train)
X_test = normalize(X_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

dec = DecisionTreeClassifier()
ran = RandomForestClassifier()
knn = KNeighborsClassifier()
svm = SVC(random_state=0)
naive = GaussianNB()
log=LogisticRegression()

models = {"Decision tree" : dec,"Random forest" : ran,"KNN" : knn,"SVM" : svm,"Naive bayes" : naive,"Logistic regression": log}
scores= { }

for key, value in models.items():    
    model = value
    model.fit(X_train, y_train)
    scores[key] = model.score(X_test, y_test)
scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
scores_frame.sort_values(by=["Accuracy Score"], axis=0,inplace=True)
scores_frame
from sklearn.model_selection import cross_val_score
acc_random_forest = cross_val_score(estimator=ran,X= X_train,y= y_train, cv=10)
acc_decission_tree=cross_val_score(estimator=dec, X=X_train, y=y_train, cv=10)
acc_knn = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=10)
acc_svm =cross_val_score(estimator=svm ,X=X_train, y=y_train, cv=10)
print("Random Forest Accuracy: ", acc_random_forest.mean())
print("Random Forest Standard Deviation: ", acc_random_forest.std())
print("Decission Tree Accuracy: ",acc_decission_tree.mean())
print("Decission Tree Standard Deviation: ", acc_decission_tree.std())
print("KNN Average Accuracy: ", acc_knn.mean())
print("KNN Standard Deviation: ", acc_knn.std())
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
DT_pred = DecisionTreeClassifier()#max_depth=3, min_samples_split=50, min_samples_leaf=50, random_state=0
DT_pred=DT_pred.fit(X_train, y_train)
y_pred = DT_pred.predict(X_test)
y_pred
print("Confusion Matrix \n",confusion_matrix(y_test,y_pred))
print("Clasification Accuracies\n",classification_report(y_test,y_pred))
DT_acc = round(accuracy_score(y_test, y_pred), 2)
print("Overall accuracy score: {} ".format(DT_acc))
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
max_depths = np.linspace(1, 10,10, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results,"b", label="Train AUC")
line2, = plt.plot(max_depths, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.grid(True)
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_splits, train_results,"b", label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Min split")
plt.xticks(np.linspace(0.1, 1.0, 10))
plt.grid(True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results,"b", label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Min leaf")
plt.grid(True)
max_features = list(range(1,X.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
    dt = DecisionTreeClassifier(max_features=max_feature)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results,"b", label="Train AUC")
line2, = plt.plot(max_features, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Max feature")
plt.grid(True)
from sklearn import tree
feature=["age","gender","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","bmi"]
fig, ax=plt.subplots(figsize=(16,10))
clf = DecisionTreeClassifier(criterion="gini", min_samples_split=0.1,max_depth=4,min_samples_leaf= 0.1)
pred=clf.fit(X,y)
ax=tree.plot_tree(pred.fit(X_train,y_train),feature_names=feature)
clf = DecisionTreeClassifier(criterion="gini", max_depth=4,min_samples_split=0.1,min_samples_leaf= 0.1)
pred=clf.fit(X,y)
y_pred=pred.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Accuracy=(TP+TN)/(TP+TN+FN+FP)
Error=(FP+FN)/(TP+TN+FN+FP)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
pd.DataFrame([[Accuracy,Error,Precision, Recall, F1_Score]],columns=["Accuracy","Error","Precision", "Recall", "F1 Score"], index=["Results"])
DT_pred = DecisionTreeClassifier(max_depth=4, min_samples_split=0.1, min_samples_leaf=0.1, random_state=0)
DT_pred=DT_pred.fit(X_train, y_train)
y_pred = DT_pred.predict(X_test)
y_pred
print("Clasification Accuracies\n",classification_report(y_test,y_pred))
DT_acc = round(accuracy_score(y_test, y_pred), 2)
print("Overall accuracy score: {} ".format(DT_acc))
from sklearn.neighbors import KNeighborsClassifier
KNN_pred = KNeighborsClassifier()
KNN=KNN_pred.fit(X_train, y_train)
y_pred_knn = KNN.predict(X_test)
y_pred_knn
print("Confusion Matrix \n",confusion_matrix(y_test,y_pred_knn))
n_neighbors = np.linspace(1, 30,30, endpoint=True).astype(int)
train_results = []
test_results = []
for n_neighbor in n_neighbors:
    dt = KNeighborsClassifier(n_neighbors=n_neighbor)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_neighbors, train_results,"b", label="Train AUC")
line2, = plt.plot(n_neighbors, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("N neighbor")
plt.grid(True)
distances = np.linspace(1,5,5,endpoint=True).astype(int)
train_results = []
test_results = []
fig = plt.figure()
ax = fig.add_subplot(111)
for distance in distances:
    dt = KNeighborsClassifier(p=distance)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = ax.plot(distances, train_results,"b", label="Train AUC")
line2, = ax.plot(distances, test_results, "r", label="Test AUC")
ymax_train = max(train_results)
xpos_train = train_results.index(ymax_train)
xmax_train = distances[xpos_train]
#ax.annotate('local max', xy=(xmax_train, ymax_train), xytext=(xmax_train, ymax_train+1),
           # arrowprops=dict(facecolor='black', shrink=0.05),
            #)
ax.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
ax.set_ylabel("AUC score")
ax.set_xlabel("Distance")
ax.grid(True)
num=np.linspace(1,10,10).astype(int)
param_dist={"n_neighbors":num,
           "weights":["uniform","distance"],
           "algorithm":["ball_tree", "kd_tree", "brute"],
           "p":np.linspace(1,2,2)}
KNN = KNeighborsClassifier()
KNN_cv=GridSearchCV(KNN,param_dist,cv=5)
KNN_cv.fit(X_train,y_train)
print("Tuned Parameter: {}".format(KNN_cv.best_params_))
print("Best Score: {}".format(KNN_cv.best_score_))
clf = KNeighborsClassifier(algorithm="ball_tree", n_neighbors= 9, p= 1.0, weights= 'uniform')
pred=clf.fit(X,y)
y_pred=pred.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Accuracy=(TP+TN)/(TP+TN+FN+FP)
Error=(FP+FN)/(TP+TN+FN+FP)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
pd.DataFrame([[Accuracy,Error,Precision, Recall, F1_Score]],columns=["Accuracy","Error","Precision", "Recall", "F1 Score"], index=["Results"])
random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]},cv=5).fit(X_train, y_train)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_random_forest,random_forest.best_params_)
acc_test_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
acc_test_random_forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred
print("Confusion Matrix \n",confusion_matrix(y_test,y_pred))
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, "b", label="Train AUC")
line2, = plt.plot(n_estimators, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("n_estimators")
plt.show()
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, "b", label="Train AUC")
line2, = plt.plot(max_depths, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.show()
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   rf = RandomForestClassifier(min_samples_split=min_samples_split)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, "b", label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("min samples split")
plt.show()
max_features = list(range(1,X_train.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   rf = RandomForestClassifier(max_features=max_feature)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, "b", label="Train AUC")
line2, = plt.plot(max_features, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("max features")
plt.show()
print("Confusion Matrix \n",confusion_matrix(y_test,y_pred))
clf = RandomForestClassifier( bootstrap = True,max_leaf_nodes=33,n_estimators= 188,max_depth=12,min_samples_split=0.6,max_features="sqrt")
 
pred=clf.fit(X,y)
y_pred=pred.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Accuracy=(TP+TN)/(TP+TN+FN+FP)
Error=(FP+FN)/(TP+TN+FN+FP)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
pd.DataFrame([[Accuracy,Error,Precision, Recall, F1_Score]],columns=["Accuracy","Error","Precision", "Recall", "F1 Score"], index=["Results"])
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter grid
param_grid = {
    'n_estimators': np.linspace(10, 200).astype(int),
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# Estimator for use in random search
estimator = RandomForestClassifier(random_state = 0)

# Create the random search model
rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                        scoring = 'roc_auc', cv = 3, 
                        n_iter = 10, verbose = 1, random_state=0)

# Fit 
rs.fit(X_train, y_train)
rs.best_params_
best_model = rs.best_estimator_
n_nodes = []
max_depths = []

for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')
from sklearn.metrics import roc_auc_score

sample_leaf_options = [1,2,3,4,5,10,20]
#X_train=X_train.reshape(1,-1)
# for loop to iterate for each leaf size
for leaf_size in sample_leaf_options :
    model = RandomForestClassifier(n_estimators = 200, n_jobs = -1,random_state =0, min_samples_leaf = leaf_size)
    model.fit(X_train,y_train)
    print("\n Leaf size :", leaf_size)
    print ("AUC - ROC : ", roc_auc_score(y_train,model.predict(X_train)))
clf=RandomForestClassifier(n_estimators = 1000, n_jobs = -1,random_state =0)
clf.fit(X_train,y_train)
feature_lbl=["age","gender","ap_hi" ,"ap_low","cholesterol","gluc","smoke","alco","active","bmi"]
for feature in zip(feature_lbl, clf.feature_importances_):
    print(feature)
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(clf, threshold=0.1)
sfm.fit(X_train, y_train)
for feature_list_index in sfm.get_support(indices=True):
    print(feature_lbl[feature_list_index])
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)
clf_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf_important.fit(X_important_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
y_important_pred = clf_important.predict(X_important_test)

accuracy_score(y_test, y_important_pred)
clf = RandomForestClassifier(n_estimators=188,min_samples_leaf=1,max_leaf_nodes=33,max_features=4,max_depth=12, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
clf_important = RandomForestClassifier(n_estimators=188,max_leaf_nodes=33,max_features=0.799,max_depth=12, random_state=0, n_jobs=-1)
clf_important.fit(X_important_train, y_train)
y_important_pred = clf_important.predict(X_important_test)
accuracy_score(y_test, y_important_pred)
cm=confusion_matrix(y_test, y_important_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Accuracy=(TP+TN)/(TP+TN+FN+FP)
Error=(FP+FN)/(TP+TN+FN+FP)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
pd.DataFrame([[Accuracy,Error,Precision, Recall, F1_Score]],columns=["Accuracy","Error","Precision", "Recall", "F1 Score"], index=["Results"])
clf_important = RandomForestClassifier(n_estimators=188,max_leaf_nodes=33,max_depth=12, min_samples_split=0.6,max_features="sqrt",random_state=0, n_jobs=-1)
clf_important.fit(X_important_train, y_train)
y_important_pred = clf_important.predict(X_important_test)
accuracy_score(y_test, y_important_pred)
#max_features=0.799
cm=confusion_matrix(y_test, y_important_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Accuracy=(TP+TN)/(TP+TN+FN+FP)
Error=(FP+FN)/(TP+TN+FN+FP)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
pd.DataFrame([[Accuracy,Error,Precision, Recall, F1_Score]],columns=["Accuracy","Error","Precision", "Recall", "F1 Score"], index=["Results"])