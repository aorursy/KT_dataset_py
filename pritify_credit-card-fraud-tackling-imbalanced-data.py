#Importing libraries ##



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import random

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.feature_selection import RFECV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score,GridSearchCV,cross_val_predict,train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.metrics import recall_score, auc, average_precision_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve



# Any results you write to the current directory are saved as output.
random.seed(1234)

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

df.head()
### Helper functions ###



### scaling ###



def feat_scale(df,col):

    sc = StandardScaler()

    df[col+"_scaled"] = sc.fit_transform(df[col].values.reshape(-1,1))

    df.drop(col, axis =1 , inplace = True)

    return df





### PCA with 2 components  ###



def pcanalysis(df, cols, n_comp):

            pca = PCA(n_components = n_comp)

            principalComponents = pca.fit_transform(df[cols])

            pca_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

            pca_df = pd.concat([pca_df, df[['Class']]], axis = 1)

            var = np.round((pca.explained_variance_ratio_*100),2)

            ax = sns.scatterplot(x = pca_df['PC1'], y = pca_df['PC2'], hue = pca_df['Class'])

            ax.set(xlabel="PC1", ylabel = "PC2")

            plt.show()

            print("%variance explained by PC1:{}".format(np.round((pca.explained_variance_ratio_[0]*100),2)))

            print("%variance explained by PC2:{}".format(np.round((pca.explained_variance_ratio_[1]*100),2)))

            return pca_df
df[['Time','Amount']].describe().transpose()
df.shape
df.isnull().sum().max()
### checking dtypes in the dataset ###



df.info()
### checking the counts of fraud and non-fraud records ###



df.Class.value_counts()
ax = sns.countplot(x="Class", data= df)

plt.title("Counts of fraud and non-fraud records in the dataset")

plt.show()
summary = df.Class.value_counts()

summary
print("% of non-fraud transactions = ",(summary[0]*(100/df.shape[0])) )

print("% of fraud transactions = ", (summary[1]*(100/df.shape[0])))
fig, axes = plt.subplots(1, 2, figsize=(18, 4))



sns.distplot(

    df['Time'],

    ax=axes[0])

            

sns.distplot(

    df['Amount'],

    ax=axes[1], color = 'red')





plt.show()
### Distribution of fraud transactions Amount with time ###



fraud = df.loc[df.Class == 1]

fraud.Timemin = (fraud.Time)/60

sns.scatterplot(x = 'Time', y='Amount', data = fraud, alpha = 0.5,color = 'navy')

plt.xlabel('Time (minutes)')

plt.title("Fraudulent transactions over time")
fraud.Amount.describe().transpose()
non_fraud = df.loc[df.Class == 0]

non_fraud.Amount.describe().transpose()
fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.boxplot(x="Class", y="Amount", ax = ax1,data = df, showfliers = True, hue = 'Class')

sns.boxplot(x="Class", y="Amount", ax = ax2, data = df, showfliers = False, hue = 'Class')

plt.show()
feat = df.drop(['Time', 'Amount','Class'], axis = 1).columns.tolist()

fig = plt.figure(figsize = (16,28))

for n, col in enumerate(feat):

        ax1 = fig.add_subplot(7,4,n+1)

        sns.kdeplot(fraud[col], label ='Fraud')

        sns.kdeplot(non_fraud[col], label = 'Not fraud')

        plt.title(col)

        

plt.show()  

    

### scaling Time and Amount features ###



df1 = df.copy()

feat_scale(df1,'Time')

feat_scale(df1,'Amount')
features = df1.drop(['Class'], axis = 1).columns.tolist()

features

pca_2D = pcanalysis(df1, features, 2)
### Train/test split on original dataset ###



X_train_origin, X_test_origin, y_train_origin, y_test_origin = train_test_split(df.drop(['Class'],axis =1), df['Class'], test_size = 0.2, random_state = 42)

#### Undersampling on train set ###



rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_sample(X_train_origin,y_train_origin)

under_df = pd.concat([X_rus,y_rus], axis = 1)

under_df.shape
X_under = under_df.drop(['Class'], axis =1)

y_under = under_df['Class']
ax = sns.countplot(x="Class", data= under_df)

plt.title("Counts of fraud and normal transactions after undersampling")

plt.show()
X_tsne = X_under.copy()

X_tsne = feat_scale(X_tsne,"Time")

X_tsne = feat_scale(X_tsne, "Amount")

tsne = TSNE(n_components=2, perplexity=50.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, random_state=0)

tsne_out = tsne.fit_transform(X_tsne)

td = pd.DataFrame(data = tsne_out, columns = ['TSNE1', 'TSNE2'])

td = pd.concat([td, under_df[['Class']]], axis = 1)

ax1 = sns.scatterplot(x = td['TSNE1'], y = td['TSNE2'], hue = td['Class'])

ax1.set(xlabel="TSNE1", ylabel = "TSNE2")

ax1.set_title('TSNE plot of Randomly undersampled data')

plt.show()
train = pd.concat([X_train_origin, y_train_origin], axis =1)
## correlations ##



fig, ax = plt.subplots(figsize=(15,8))      

sns.heatmap(train.corr(), xticklabels = True, yticklabels = True, square=True,  ax = ax, cmap = "Greens")

plt.show()
### Correlations of all features with the target ###

print(train.corr().Class)
X_test_origin.columns
### cross validation ##



def crossval(X_train_origin, y_train_origin):

    dictli = [{"name": "KNeighborsClassifier", "estimator":KNeighborsClassifier()}, 

          {"name": "LogisticRegression", "estimator":LogisticRegression(random_state = 42)}, 

          {"name": "RandomForestClassifier", "estimator":RandomForestClassifier(random_state = 42)}]

    for i in dictli:

        score = cross_val_score(i['estimator'], X_train_origin, y_train_origin, cv= 10)

        print(i['name'],"training score of:",  round(score.mean(), 2) * 100, "% accuracy score")

    return dictli

accuracy = crossval(X_train_origin, y_train_origin)
## Pipeline and GridSearchCV ##



model = Pipeline([

        ('sampling', RandomUnderSampler(random_state = 0)),

        ('scaling', StandardScaler()),

        ('classification', LogisticRegression())

    ])





#lr = LogisticRegression()

#pipe = make_pipeline(StandardScaler(), lr) 

param = {'classification__C': [0.001, 0.01, 0.1,  10, 100]}

grid_search_log = GridSearchCV(model, param, scoring='roc_auc',refit=True,  cv= 5)

#grid_search_log = GridSearchCV(pipe, param_grid=param, scoring='roc_auc',refit=True,  cv= 5)

grid_search_log.fit(X_train_origin, y_train_origin)

print(grid_search_log.best_estimator_)

log_best = grid_search_log.best_estimator_

resultsdf = pd.DataFrame(grid_search_log.cv_results_)

resultsdf
model = Pipeline([

        ('sampling', RandomUnderSampler(random_state = 0)),

        ('scaling', StandardScaler()),

        ('classification', KNeighborsClassifier())

    ])





#pipe = make_pipeline(StandardScaler(), KNeighborsClassifier()) 

param = {'classification__n_neighbors': range(1,20,2), 'classification__weights': ["distance", "uniform"],

         'classification__algorithm': ["ball_tree", "kd_tree", "brute"]}

grid_search_knn = GridSearchCV(model, param_grid=param, scoring='roc_auc',refit=True,  cv= 5)

grid_search_knn.fit(X_train_origin, y_train_origin)

print(grid_search_knn.best_estimator_)

knn_best = grid_search_knn.best_estimator_

resultsdf = pd.DataFrame(grid_search_knn.cv_results_)

resultsdf
model = Pipeline([

        ('sampling', RandomUnderSampler(random_state = 0)),

        ('scaling', StandardScaler()),

        ('classification', RandomForestClassifier())

    ])



#pipe = make_pipeline(StandardScaler(), RandomForestClassifier()) 

param = {'classification__max_depth': [2, 5, 10], 'classification__min_samples_leaf':[1, 5, 8],

         "classification__criterion": ["entropy", "gini"], "classification__min_samples_split": [2, 3, 5]}

grid_search_rf = GridSearchCV(model, param_grid=param, scoring='roc_auc',refit=True,  cv= 5)

grid_search_rf.fit(X_train_origin, y_train_origin)

print(grid_search_rf.best_estimator_)

rf_best = grid_search_rf.best_estimator_

resultsdf = pd.DataFrame(grid_search_rf.cv_results_)

resultsdf
### ROC curve using test set ###



log_fpr, log_tpr, log_thresh = roc_curve(y_test_origin,log_best.predict(X_test_origin))

knn_fpr, knn_tpr, knn_thresh = roc_curve(y_test_origin, knn_best.predict(X_test_origin))

rf_fpr, rf_tpr, rf_thresh = roc_curve(y_test_origin, rf_best.predict(X_test_origin))

def roc_compare_plot(log_fpr, log_tpr,knn_fpr, knn_tpr,rf_fpr, rf_tpr ):

    roc_auc = auc(log_fpr,log_tpr)

    plt.plot(log_fpr, log_tpr, color ='red', label = 'Logistic Regression, AUC = %0.2f'% roc_auc)

    roc_auc_knn = auc(knn_fpr,knn_tpr)

    plt.plot(knn_fpr, knn_tpr, color ='green', label = 'KNeighborsClassifier, AUC = %0.2f'% roc_auc_knn)

    roc_auc_rf = auc(rf_fpr,rf_tpr)

    plt.plot(rf_fpr, rf_tpr, color ='blue', label = 'RandomForestClassifier, AUC = %0.2f'% roc_auc_rf)

    plt.ylabel('True positive rate')

    plt.xlabel('False positive rate')

    plt.plot([0,1],[0,1],'r--')

    plt.legend()

roc_compare_plot(log_fpr, log_tpr,knn_fpr, knn_tpr,rf_fpr, rf_tpr)

    
### Precision recall curve using test set using logistic regression###



y_pred_prob = log_best.predict_proba(X_test_origin)[:,1]



# Generate precision recall curve values: precision, recall, thresholds

precision, recall, thresholds = precision_recall_curve(y_test_origin, y_pred_prob)



# Plot ROC curve

plt.plot(precision, recall)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Recall Curve')

plt.show()
### checking on  test set using logistic regression ###





y_pred = log_best.predict(X_test_origin)

print("\nRecall:",recall_score(y_test_origin,y_pred))

print(classification_report(y_test_origin, y_pred))

conf_matrix = confusion_matrix(y_test_origin,y_pred)

print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)

plt.xlabel("Predicted label")

plt.ylabel("True label")

plt.show()
### checking on  test set using RandomForest ###





y_pred = rf_best.predict(X_test_origin)

print("\nRecall:",recall_score(y_test_origin,y_pred))

print(classification_report(y_test_origin, y_pred))

conf_matrix = confusion_matrix(y_test_origin,y_pred)

print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)

plt.xlabel("Predicted label")

plt.ylabel("True label")

plt.show()
random.seed(1234)

data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

X_data = data.drop(["Class"], axis = 1)

y_data = data[['Class']]
X_tr_origin, X_holdout, y_tr_origin, y_holdout = train_test_split(X_data, y_data, test_size = 0.2, random_state = 42)
colnames = X_tr_origin.columns

colnames
print("Length of training dataset:", len(X_tr_origin))

print("Length of test dataset:", len(X_holdout))
### GridsearchCV and pipeline ###



model = Pipeline([

        ('sampling', SMOTE(random_state = 0)),

        ('scaling', StandardScaler()),

        ('classification', RandomForestClassifier(n_estimators = 10))

    ])





#pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 10)) 

param = {'classification__max_depth': [2, 5, 10], 'classification__min_samples_leaf':[1, 5, 8],

        "classification__criterion": ["entropy", "gini"], "classification__min_samples_split": [2, 3, 5]}

grid_search_rf = GridSearchCV(model, param, scoring='roc_auc',refit=True,  cv= 5)

#grid.fit(X, y)

#grid_search_rf = GridSearchCV(pipe, param_grid=param, scoring='roc_auc',refit=True,  cv= 5)

grid_search_rf.fit(X_tr_origin, y_tr_origin.values.ravel())

print(grid_search_rf.best_estimator_)

rf_best = grid_search_rf.best_estimator_

resultsdf = pd.DataFrame(grid_search_rf.cv_results_)

resultsdf
### checking on holdout set ###



y_pred = rf_best.predict(X_holdout)

print("\nRecall:",recall_score(y_holdout,y_pred))

print(classification_report(y_holdout, y_pred))

conf_matrix = confusion_matrix(y_holdout,y_pred)

print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)

plt.xlabel("Predicted label")

plt.ylabel("True label")

plt.show()
### GridsearchCV and pipeline ###



model = Pipeline([

        ('sampling', SMOTE(random_state = 0)),

        ('scaling', StandardScaler()),

        ('classification', SVC(random_state = 42))

    ])





#pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 10)) 

param = {'classification__C': [1]}

grid_search_sv = GridSearchCV(model,param,  scoring='roc_auc',refit=True,  cv= 5)

#grid.fit(X, y)

#grid_search_rf = GridSearchCV(pipe, param_grid=param, scoring='roc_auc',refit=True,  cv= 5)

grid_search_sv.fit(X_tr_origin, y_tr_origin.values.ravel())

print(grid_search_sv.best_estimator_)

sv_best = grid_search_sv.best_estimator_

resultsdf = pd.DataFrame(grid_search_sv.cv_results_)

resultsdf
### checking on holdout set ###



y_pred = sv_best.predict(X_holdout)

print("\nRecall:",recall_score(y_holdout,y_pred))

print(classification_report(y_holdout, y_pred))

conf_matrix = confusion_matrix(y_holdout,y_pred)

print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)

plt.xlabel("Predicted label")

plt.ylabel("True label")

plt.show()