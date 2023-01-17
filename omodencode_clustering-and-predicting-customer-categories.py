import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df = pd.read_csv('../input/Mall_Customers.csv')
df.head(5)
df.describe()
sns.heatmap(df.drop(['CustomerID'],axis=1).corr(),annot=True,cmap='RdYlGn') 

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show
f, axes = plt.subplots(1, 3, figsize=(20, 5)) #sharex=True)

sns.distplot(df['Annual Income (k$)'],color="red", label="Annual Income (k$)",ax=axes[0])

sns.distplot(df['Age'],color="green", label="Age",ax=axes[1])

sns.distplot(df['Spending Score (1-100)'],color="skyblue", label="Spending Score",ax=axes[2])
f, axes = plt.subplots(1, 3, figsize=(20, 5)) #sharex=True)

sns.distplot(df['Annual Income (k$)'][df['Gender']=="Male"],color="salmon", label="Annual Income (k$)",ax=axes[0])

sns.distplot(df['Annual Income (k$)'][df['Gender']=="Female"],color="skyblue", label="Annual Income (k$)",ax=axes[0])

    

sns.distplot(df['Age'][df['Gender']=="Male"],color="salmon", label="Age",ax=axes[1])

sns.distplot(df['Age'][df['Gender']=="Female"],color="skyblue", label="Age",ax=axes[1])



sns.distplot(df['Spending Score (1-100)'][df['Gender']=="Male"],color="salmon", label="Spending Score",ax=axes[2])

sns.distplot(df['Spending Score (1-100)'][df['Gender']=="Female"],color="skyblue", label="Spending Score",ax=axes[2])



plt.show()
f, axes = plt.subplots(1, 2, figsize=(30, 10)) #sharex=True)

sns.swarmplot(x="Gender", y="Spending Score (1-100)",data=df, palette="Set2", dodge=True, ax=axes[0])

sns.swarmplot(x="Gender", y="Annual Income (k$)",data=df, palette="Set2", dodge=True, ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(20, 10)) #sharex=True)

sns.scatterplot(x="Age", y="Spending Score (1-100)",hue="Gender", data=df, ax=axes[0])

sns.scatterplot(x="Age", y="Annual Income (k$)",hue="Gender", data=df, ax=axes[1])
pd.scatter_matrix(df, figsize=(20,10))

plt.show()
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)",hue="Gender", data=df)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



xs = df['Age']

ys = df['Spending Score (1-100)']

zs = df['Annual Income (k$)']

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('Age')

ax.set_ylabel('Spending Score')

ax.set_zlabel('Annual Income')



plt.show()
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
X_k = df['Annual Income (k$)'].values

y_k = df['Spending Score (1-100)'].values
X_k1 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
model = KMeans(n_clusters=5)

model.fit(X_k1)
y_kmeans = model.fit_predict(X_k1)
model.labels_
unique_labels = set(model.labels_)

for c in unique_labels:  

    plt.scatter(X_k1[model.labels_ == c, 0],

                X_k1[model.labels_ == c, 1],

                label='Cluster {}'.format(c))

plt.xlabel('Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()
df_target = pd.DataFrame({'target':model.labels_})

df_new = pd.concat([df, df_target], axis=1, sort=False)

df_new = df_new.drop(['CustomerID'], axis=1)

df_new.head(5)
X_new = df_new.drop(['target'],axis=1)

y_new = df_new['target']



df_gender = pd.get_dummies(X_new['Gender'])

X_new = X_new.drop(['Gender'],axis=1)

X_new = pd.concat([X_new,df_gender],axis=1, sort=False)



X_train, X_test, y_train, y_test = train_test_split(X_new,y_new,stratify=y_new,test_size=0.25,random_state=42)
y_train_bin = pd.get_dummies(y_train)

y_test_bin = pd.get_dummies(y_test)

X_new.head(5)
dt = DecisionTreeClassifier()

rf = RandomForestClassifier()
model_dt = dt.fit(X_train,y_train_bin)

model_rf = rf.fit(X_train,y_train_bin)
y_pred_dt = model_dt.predict(X_test)

y_pred_rf = model_rf.predict(X_test)
y_test_pred = y_test_bin.idxmax(axis=1).tolist()
dfrf_results_pred = pd.DataFrame({'0':y_pred_rf[:,0],

                             '1':y_pred_rf[:,1],

                             '2':y_pred_rf[:,2],

                             '3':y_pred_rf[:,3],

                             '4':y_pred_rf[:,4]})



rf_pred = dfrf_results_pred.idxmax(axis=1).tolist()



final_df_rf =  pd.DataFrame({'predicted':rf_pred,

                         'actual':y_test_pred})

final_df_rf.head(5)
dfdt_results_pred = pd.DataFrame({'0':y_pred_dt[:,0],

                             '1':y_pred_dt[:,1],

                             '2':y_pred_dt[:,2],

                             '3':y_pred_dt[:,3],

                             '4':y_pred_dt[:,4]})



dt_pred = dfdt_results_pred.idxmax(axis=1).tolist()



final_df =  pd.DataFrame({'predicted':dt_pred,

                         'actual':y_test_pred})

final_df.head(5)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
X_feature_impt = X_train.keys().tolist()

y_feature_impt = model_dt.feature_importances_



plt.barh(X_feature_impt, y_feature_impt, align='center')

plt.xlabel('Relative Importance')

plt.title('Feature Importance')

plt.show()
print("Decision Tree Scores")

print("Accuracy: {}".format(accuracy_score(y_test_bin,y_pred_dt)))

print("MAE (test): {}".format(mean_absolute_error(y_test_bin, y_pred_dt)))

print("MSE (test): {}".format(mean_squared_error(y_test_bin, y_pred_dt)))



print("Random Forest Scores")

print("Accuracy: {}".format(accuracy_score(y_test_bin,y_pred_rf)))

print("MAE (test): {}".format(mean_absolute_error(y_test_bin, y_pred_rf)))

print("MSE (test): {}".format(mean_squared_error(y_test_bin, y_pred_rf)))
class_labels = list(y_train_bin.columns.values)

class_labels
import itertools

import sklearn.metrics



def plot_confusion_matrix(y_test, y_pred, labels,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    

    See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=labels)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels, rotation=45)

    plt.yticks(tick_marks, labels)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
pred1 = []

for yp in rf_pred:

    pred1.append(int(yp))
main_test = []

pred = []

for yt in y_test_pred:

    main_test.append(int(yt))

    

for yp in dt_pred:

    pred.append(int(yp))
print(main_test[1:5])

print(pred[1:5])
plt.figure()

plot_confusion_matrix(main_test, pred1, labels=class_labels,

                      title='[Decision Tree] Confusion matrix, without normalization')

plot_confusion_matrix(main_test, pred1, labels=class_labels, normalize=True,

                      title='[Decision Tree] Confusion matrix')
plt.figure()

plot_confusion_matrix(main_test, pred, labels=class_labels,

                      title='[Decision Tree] Confusion matrix, without normalization')

plot_confusion_matrix(main_test, pred, labels=class_labels, normalize=True,

                      title='[Decision Tree] Confusion matrix')