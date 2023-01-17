import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, scale

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns; sns.set()

%matplotlib inline
#Files

!ls
#Eplore the files

cancer_targets = pd.read_csv("../input/gene-expression/actual.csv") #targets

cancer_targets['patient'] = cancer_targets['patient'].astype('int')



print(cancer_targets["cancer"].value_counts())

print("\nNumber of samples;", cancer_targets.shape)
cancer_train = pd.read_csv("../input/gene-expression/data_set_ALL_AML_train.csv")

cancer_test = pd.read_csv("../input/gene-expression/data_set_ALL_AML_independent.csv")



print("Train shape:",cancer_train.shape)

print("Test shape:",cancer_test.shape)



#There are 72 patients according to the target dataframe, but shapes are 78 and 70 in train and test?

cancer_train.head(4)
def rename_columns(df):

    """Get's the correct patient ID for the call columns"""

    for col in df.columns:

        if "call" in col:

            loc = df.columns.get_loc(col)

            patient = df.columns[loc-1]

            df.rename(columns={col: f'Call_{patient}'}, inplace=True)

            

            

rename_columns(df=cancer_train)

rename_columns(df=cancer_test)



#check for duplicate columns

#print(cancer_test.groupby(["Gene Description"]).size().value_counts(),

#      cancer_train.groupby(["Gene Description"]).size().value_counts())



#Gene description and Gene accesion should be kept together, otherwise there will be duplicates.

cancer_train["Gene"] = cancer_train["Gene Description"] + '_' + cancer_train["Gene Accession Number"]

cancer_test["Gene"] = cancer_test["Gene Description"] + '_' +  cancer_test["Gene Accession Number"]



#Transpose the dataset and fix the columns + label train and test set with new column

cancer_train = cancer_train.T

cancer_train.columns = cancer_train.iloc[-1]

cancer_train = cancer_train[2:-1]

cancer_train['dataset'] = 'train'



cancer_test = cancer_test.T

cancer_test.columns = cancer_test.iloc[-1]

cancer_test = cancer_test[2:-1]

cancer_test['dataset'] = 'test'





df = pd.concat([cancer_train, cancer_test], axis=0,join='inner', sort=False)

df.shape
#Genes with only A calls are not of any use:

call_rows = [row for row in df.index if "Call" in row]

conditional = df.filter(call_rows, axis=0).apply(lambda x: x == 'A', axis=1).all()

print(conditional.value_counts()) #True will be dropped.

df = df.loc[:, ~conditional]





#Next we can delete the Call rows and add the cancer labels

#del df.columns.name

df.drop(call_rows, axis = 0, inplace=True)

df['patient'] = df.index

df['patient'] = df['patient'].astype('int') #Have same dtype for the columns to merge on

df.reset_index(drop=True)

#df = pd.concat([df, cancer_targets], axis=1, join='inner')



#Merge



df = pd.merge(left=df, right=cancer_targets, left_on='patient', right_on='patient')



print(df.shape)



df.head(5)
#Assign train and test sets

train = df[df['dataset'] == 'train'].iloc[:,1:-3]

train_target = df[df['dataset'] == 'train'].iloc[:,-1]

test = df[df['dataset'] == 'test'].iloc[:,1:-3]

test_target = df[df['dataset'] == 'test'].iloc[:,-1]



print(train.shape, train_target.shape)

print(test.shape, test_target.shape)
#Scaling

#Initialize a scaler later to be used on test set as well.

scaler = StandardScaler().fit(train)

train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns)

test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)



#Awkward distribution!

fig, ax = plt.subplots(ncols=2, figsize=(15,5))

sns.distplot(np.concatenate(train.values), ax=ax[0])

sns.distplot(np.concatenate(train_scaled.values), ax=ax[1])

plt.tight_layout

plt.show()
#Feature selection using SelectFromModel and Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



print("Original Shape:", train_scaled.shape)

logistic_regression = LogisticRegression(penalty="l1", solver='saga').fit(train_scaled, train_target) #l1 for sparsity

log_coefficients = logistic_regression.coef_

selector_log = SelectFromModel(logistic_regression, prefit=True)

train_scaled_logreg = selector_log.transform(train_scaled)

print("Features after selection using Logistic Regression:", train_scaled_logreg.shape) 





#Alternatively

from sklearn.ensemble import ExtraTreesClassifier



clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train_scaled, train_target) 

selector_tree = SelectFromModel(clf, prefit=True)

train_scaled_tree = selector_tree.transform(train_scaled)

test_scaled_tree = selector_tree.transform(test_scaled)

print("Features after selection using ExtraTreesClassifier:", train_scaled_tree.shape)







#Absolute values for coefficients in Logistic Regression "=~importance"

log_coefficients_abs = abs(log_coefficients)

log_coefficients_abs_sort = np.sort(log_coefficients_abs).flatten()

sortedidx = log_coefficients_abs.argsort()

log_labels = train_scaled.columns[sortedidx].flatten()

sns.barplot(x=log_coefficients_abs_sort[-20:], y=log_labels[-20:]);
extratree_feature_importances_sort = np.sort(clf.feature_importances_ ).flatten()

sortedidx = clf.feature_importances_.argsort()

tree_labels = train_scaled.columns[sortedidx].values.flatten()

sns.barplot(x=extratree_feature_importances_sort[-20:], y=tree_labels[-20:]);
[gene for gene in tree_labels[-20:] if gene in log_labels[-20:]]
#Let's do a pca first 

pca = PCA(n_components=3)

pca.fit_transform(train_scaled_tree)

print(pca.explained_variance_ratio_) # Small variance explained



PCA_df = pd.DataFrame(data = pca.fit_transform(train_scaled_tree), 

                           columns = ['pc1', 'pc2', 'pc3'])
PCA_df = pd.concat([PCA_df, train_target], axis = 1)



fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')

colors = {'ALL':'red', 'AML':'blue'}

ax.scatter(PCA_df.pc1, PCA_df.pc2, PCA_df.pc3, 

           c=train_target.apply(lambda x: colors[x]))

plt.title('First 3 Principal Components after PCA')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

ax.set_zlabel('PC3')

ax.view_init(20, 80)

plt.tight_layout

plt.show()
# Model Selection

classifiers = [LogisticRegression(),

               KNeighborsClassifier(n_neighbors=5),

               SVC(probability=True),

               RandomForestClassifier(n_estimators=100, min_samples_leaf=4),

               MLPClassifier(hidden_layer_sizes=250),

               GaussianNB()

              ]

names = ['Logistic Regression',

         'K Nearest Neighbours', 

         'Support Vector Machine', 

         'Random Forest', 

         'Multi Layer Perceptron', 

         'Gaussian Naive Bayes'

        ]



ROC_results = dict()

PR_results = dict()



for name, classifier in zip(names,classifiers):

    

    #fit

    clf = classifier.fit(train_scaled_tree, train_target)

    

    #predictions on test set

    pred_targets = clf.predict(test_scaled_tree)

    probas = clf.predict_proba(test_scaled_tree)[:,1]

    

    #ROC and AUC

    tpr, fpr, _ = roc_curve(test_target, probas, pos_label='AML')

    ROCAUC = roc_auc_score(test_target, probas)

    ROC_results[name] = (tpr, fpr, ROCAUC)

    

    #PR

    precision, recall, _ = precision_recall_curve(test_target, probas, pos_label='AML')

    AUCPR = auc(recall, precision)

    PR_results[name] = (precision, recall, AUCPR)

    

    #print results

    print(f'\nResults for {name}:\n')

    print(classification_report(y_true=test_target, y_pred=pred_targets))

    print(confusion_matrix(y_true=test_target, y_pred=pred_targets))
fig, (roc,pr) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(15, 7))



#ROC

for clf in ROC_results.keys():

    roc.plot(ROC_results[clf][0], ROC_results[clf][1], label=f'{clf} (ROCAUC = {round(ROC_results[clf][2],3)})');



roc.set_xlabel('False Positive Rate')

roc.set_ylabel('True Positive Rate')

roc.set_title('ROC curve')

roc.legend();



#PR

for clf in PR_results.keys():

    pr.plot(PR_results[clf][1], PR_results[clf][0], label=f'{clf} (AUC = {round(PR_results[clf][2],3)})')



pr.set_xlabel('Recall')

pr.set_ylabel('Precision')

pr.set_title('PR curve')

pr.legend();


