# load packages



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl



from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

from scipy.stats import skew, norm, probplot, boxcox, f_oneway

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve

from matplotlib import pyplot as plt



%matplotlib inline

np.random.seed(42)



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier

from sklearn import svm 

from sklearn.neighbors import KNeighborsClassifier 
# read the data

wine_raw = pd.read_csv('../input/winequality.csv').rename(columns=lambda x: x.replace(" ","_"))
print("Shape of Wine data:\nrows:", wine_raw.shape[0], '\ncolumns:', wine_raw.shape[1])
wine_raw.head()
wine_raw.describe().T
# check missing data

total = wine_raw.isnull().sum().sort_values(ascending = False)

percent = (wine_raw.isnull().sum()/wine_raw.isnull().count()*100).sort_values(ascending = False)

pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose().T
wine_raw = wine_raw.dropna(how = 'any')

wine_raw.isnull().any()
features_wine = wine_raw.columns

red_wine = round(wine_raw.loc[wine_raw.color == 'red', features_wine].describe(),2)

white_wine = round(wine_raw.loc[wine_raw.color == 'white', features_wine].describe(),2)

pd.concat([red_wine, white_wine], axis=0, keys=['Red Wine', 'White Wine']).T
# check data unbalance (The data has not a large unbalance with respect of the target value.)

# explore the target variable: quality

qualitydata = wine_raw.quality.value_counts().sort_index()

qualitydata_df = pd.DataFrame({'Quality Level': qualitydata.index,'Frequency': qualitydata.values})

qualitydata_df
# visualize target variable

plt.figure(figsize=(7,5))

sns.barplot(x = 'Quality Level', y ="Frequency", data = qualitydata_df,palette="Blues_d")

plt.title('Quality Level Distribution (level 3 - level 9)',fontsize=16)

plt.show()
fig = plt.figure(figsize = (15, 5))

title = fig.suptitle("Wine Type Vs Quality (Original Dataset)", fontsize=16)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax1 = fig.add_subplot(1,2, 1)

ax1.set_title("Red Wine")

ax1.set_xlabel("Quality")

ax1.set_ylabel("Frequency") 

rw_q = wine_raw.quality[wine_raw.color == 'red'].value_counts()

rw_q = (list(rw_q.index), list(rw_q.values))

ax1.set_ylim([0, 2500])

ax1.tick_params(axis='both', which='major', labelsize=8.5)

bar1 = ax1.bar(rw_q[0], rw_q[1])





ax2 = fig.add_subplot(1,2, 2)

ax2.set_title("White Wine")

ax2.set_xlabel("Quality")

ax2.set_ylabel("Frequency") 

ww_q = wine_raw.quality[wine_raw.color == 'white'].value_counts()

ww_q = (list(ww_q.index), list(ww_q.values))

ax2.set_ylim([0, 2500])

ax2.tick_params(axis='both', which='major', labelsize=8.5)

bar2 = ax2.bar(ww_q[0], ww_q[1])
corr = wine_raw.corr()

top_corr_cols = corr.quality.sort_values(ascending=False).keys() 

top_corr = corr.loc[top_corr_cols, top_corr_cols]

dropSelf = np.zeros_like(top_corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

plt.figure(figsize=(12, 8))

sns.heatmap(top_corr, cmap=sns.diverging_palette(600, 600, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)

# sns.set(font_scale=1.0)

cols = wine_raw.columns

cols = cols.drop('quality')



plt.show()
wine2 = wine_raw

wine2['quality_label'] = (wine2['quality'] > 5.5)*1



wine2.head()
wine_raw.head()
features_wine2 = wine2.columns

low_wine = round(wine2.loc[wine2.quality_label == 0, features_wine2].describe(),2)

high_wine = round(wine2.loc[wine2.quality_label == 1, features_wine2].describe(),2)

pd.concat([low_wine, high_wine], axis=0, keys=['Low Quality', 'High Quality']).T
# explore the binary target variable: quality_label

qualitydata2 = wine2.quality_label.value_counts().sort_index()

qualitydata2_df = pd.DataFrame({'Quality Label': qualitydata2.index,'Frequency': qualitydata2.values})

qualitydata2_df
plt.figure(figsize=(7,5))

sns.barplot(x = 'Quality Label', y ="Frequency", data = qualitydata2_df,palette="Blues_d")

plt.title('Quality Label Distribution (High quality & Low quality)',fontsize=16)

plt.show()
wine3 = wine_raw

wine3['quality_level'] = wine3.quality.apply(lambda q: 'Level C' if q <= 4 

                                             else 'Level B' if q <= 6 

                                             else 'Level A')

wine3.head()
qualitydata3 = wine3.quality_level.value_counts().sort_index()

qualitydata3_df = pd.DataFrame({'Quality Level': qualitydata3.index,'Frequency': qualitydata3.values})

qualitydata3_df
plt.figure(figsize=(7,5))

sns.barplot(x = 'Quality Level', y ="Frequency", data = qualitydata3_df,palette="Blues_d")

plt.title('Quality Level Distribution (Level A, Level B & Level C)',fontsize=16)

plt.show()
wine_raw.head()
# converting categorical variables into dummy variables

def categorize(l):

    uniques = sorted(list(set(l)))

    return [uniques.index(x) + 1 for x in l]



wine_raw['color'] = categorize(wine_raw['color'])

wine2['color'] = categorize(wine2['color'])
numeric_features = list(wine_raw.dtypes[(wine_raw.dtypes != "str") & (wine_raw.dtypes !='object')].index)

skewed_features = wine_raw[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)



#compute skewness

skewness = pd.DataFrame({'Skewness' :skewed_features})   



# Get only higest skewed features

skewness = skewness[abs(skewness) > 0.7]

skewness = skewness.dropna()

print ("{} higest skewed numerical features need to be transformed".format(skewness.shape[0]))



l_opt = {}



for feat in skewness.index:

    wine_raw[feat], l_opt[feat] = boxcox((wine_raw[feat]+1))



skewed_features2 = wine_raw[skewness.index].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)



#compute skewness

skewness2 = pd.DataFrame({'Skewness After Transformation' :skewed_features2})   

display(pd.concat([skewness, skewness2], axis=1).sort_values(by=['Skewness'], ascending=False))
pca = PCA(n_components='mle')

features = wine_raw.drop(['color','quality_level'],axis=1)

features = scale(features);features

x_pca = pca.fit_transform(features)

print (pca.explained_variance_ratio_)

print ('\n')

sum_variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

print (sum_variance)

# print (pca.explained_variance_)

print ('\n')

print (pca.n_components_)



plt.ylabel('% Variance Explained')

plt.xlabel('Number of Features')

plt.title('PCA Analysis')

# plt.ylim(0,100.5)

plt.plot(sum_variance)

plt.show()
# Selecting the input and output features for multi-classification tasks



features = ['color',

            'fixed_acidity',

            'volatile_acidity',

            'citric_acid',

            'residual_sugar',

            'chlorides',

            'free_sulfur_dioxide',

            'total_sulfur_dioxide',

            'density',

            'pH',

            'sulphates',

            'alcohol']



target = ['quality_level']
# Split dataset into training set & test set

x = wine_raw[features]

y = wine_raw[target].values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=200)
# Selecting the input and output features for binary classification tasks

features2 = ['color',

             'fixed_acidity',

             'volatile_acidity',

             'citric_acid',

             'residual_sugar',

             'chlorides',

             'free_sulfur_dioxide',

             'total_sulfur_dioxide',

             'density',

             'pH',

             'sulphates',

             'alcohol']



target2 = ['quality_label']



# x = wine_raw[features_all]

x2 = wine2[features2]

y2 = wine2[target2].values.ravel()

# Visualize the combined table (which should looks the same as the original dataset)

# pd.concat([X, y], axis=1, sort=False).head()
# Split dataset|

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=200)

# x2_train_pca = pca.fit(x2_train)
def print_predict_vs_test_multi(clf,x_test,y_test):

    prediction = clf.predict(x_test)

    print('Predict values:')

    print(prediction[:10])

    print('-'*10)

    print('True values in test set:')

    print(y_test[:10])
def print_predict_vs_test_binary(clf,x_test,y_test):

    prediction = clf.predict(x_test)

    print('Predict values:')

    print(prediction[:10])

    print('-'*10)

    print('True values in test set:')

    print(y_test[:10])
def get_results_multi(clf,clf_name,x_train,y_train,x_test,y_test):

    y_pred = clf.predict(x_test)

    print('Training Accuracy('+clf_name+'): {:2.2%}'.format(accuracy_score(y_train, clf.predict(x_train))))    

    print('Test Accuracy('+clf_name+'): {:2.2%}\n'.format(accuracy_score(y_test, clf.predict(x_test))))

    print('Classification Report('+clf_name+'): \n' + classification_report(y_test, y_pred))

    

    probs = clf.predict_proba(x_test) # Predict class probabilities of the input samples 

    preds = probs[:,1]

    

    

    print('5 fold Cross Validation('+clf_name+'):')

    cv_accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')

    print('Accuracy: {:2.2%}'.format(np.mean(cv_accuracy)))

    

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,8))

    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True,cmap='Blues')

    plt.ylabel('Actual Label')

    plt.xlabel('Predicted Label')

    plt.title('Confusion Matrix ('+clf_name+')',fontsize=16)

    

    tmp = pd.DataFrame({'Feature': features, 'Feature Importance': clf.feature_importances_})

    tmp = tmp.sort_values(by='Feature Importance',ascending=False)

    plt.figure(figsize = (10,6))

    plt.title('Features Importance',fontsize=16)

    s = sns.barplot(x='Feature',y='Feature Importance',data=tmp)

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.show() 
def get_results_binary(clf,clf_name,x_train,y_train,x_test,y_test):

    y_pred = clf.predict(x_test)

    print('Training Accuracy('+clf_name+'): {:2.2%}'.format(accuracy_score(y_train, clf.predict(x_train))))    

    print('Test Accuracy('+clf_name+'): {:2.2%}\n'.format(accuracy_score(y_test, clf.predict(x_test))))

    print('Classification Report('+clf_name+'): \n' + classification_report(y_test, y_pred))

    

    probs = clf.predict_proba(x_test) # Predict class probabilities of the input samples 

    preds = probs[:,1]

    fpr,tpr,threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr,tpr)

    print('ROC AUC Score('+clf_name+'): {:2.2%}\n'.format(roc_auc))

    

    

    print('5 fold Cross Validation('+clf_name+'):')

    cv_accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')

    print('Accuracy: {:2.2%}'.format(np.mean(cv_accuracy)))

    cv_recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall')

    print('Recall: {:2.2%}'.format(np.mean(cv_recall)))

    cv_precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision')

    print('Precision: {:2.2%}'.format(np.mean(cv_precision)))

    cv_f1 = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1')

    print('F1-score: {:2.2%}'.format(np.mean(cv_f1)))

    cv_roc_auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')

    print('ROC AUC Score: {:2.2%}'.format(np.mean(cv_roc_auc)))



    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,8))

    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True,cmap='Blues')

    plt.ylabel('Actual Label')

    plt.xlabel('Predicted Label')

    plt.title('Confusion Matrix ('+clf_name+')',fontsize=16)

    

    

    # calculate the Optimum Threshold

    for i in range(len(fpr)):

        if fpr[i] + tpr[i] >= 1:

            i = i -1

            break



    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, lw=1,label='ROC Curve(area = %0.2f)    Optimum Threshold = %0.2f' % (roc_auc, threshold[i]))

    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve ('+clf_name+')',fontsize=16)

    plt.legend(loc="lower right")

    plt.show()

    

    tmp = pd.DataFrame({'Feature': features, 'Feature Importance': clf.feature_importances_})

    tmp = tmp.sort_values(by='Feature Importance',ascending=False)

    plt.figure(figsize = (10,6))

    plt.title('Features Importance',fontsize=16)

    s = sns.barplot(x='Feature',y='Feature Importance',data=tmp)

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.show() 
def get_results_multi_withoutFeatureImportance(clf,clf_name,x_train,y_train,x_test,y_test):

    y_pred = clf.predict(x_test)

    print('Training Accuracy('+clf_name+'): {:2.2%}'.format(accuracy_score(y_train, clf.predict(x_train))))    

    print('Test Accuracy('+clf_name+'): {:2.2%}\n'.format(accuracy_score(y_test, clf.predict(x_test))))

    print('Classification Report('+clf_name+'): \n' + classification_report(y_test, y_pred))

    

    probs = clf.predict_proba(x_test) # Predict class probabilities of the input samples 

    preds = probs[:,1]

    

    

    print('5 fold Cross Validation('+clf_name+'):')

    cv_accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')

    print('Accuracy: {:2.2%}'.format(np.mean(cv_accuracy)))

    

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,8))

    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True,cmap='Blues')

    plt.ylabel('Actual Label')

    plt.xlabel('Predicted Label')

    plt.title('Confusion Matrix ('+clf_name+')',fontsize=16)
def get_results_binary_withoutFeatureImportance(clf,clf_name,x_train,y_train,x_test,y_test):

    y_pred = clf.predict(x_test)

    print('Training Accuracy('+clf_name+'): {:2.2%}'.format(accuracy_score(y_train, clf.predict(x_train))))    

    print('Test Accuracy('+clf_name+'): {:2.2%}\n'.format(accuracy_score(y_test, clf.predict(x_test))))

    print('Classification Report('+clf_name+'): \n' + classification_report(y_test, y_pred))

    

    probs = clf.predict_proba(x_test) # Predict class probabilities of the input samples 

    preds = probs[:,1]

    fpr,tpr,threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr,tpr)

    print('ROC AUC Score('+clf_name+'): {:2.2%}\n'.format(roc_auc))

    

    

    print('5 fold Cross Validation('+clf_name+'):')

    cv_accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')

    print('Accuracy: {:2.2%}'.format(np.mean(cv_accuracy)))

    cv_recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall')

    print('Recall: {:2.2%}'.format(np.mean(cv_recall)))

    cv_precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision')

    print('Precision: {:2.2%}'.format(np.mean(cv_precision)))

    cv_f1 = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1')

    print('F1-score: {:2.2%}'.format(np.mean(cv_f1)))

    cv_roc_auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')

    print('ROC AUC Score: {:2.2%}'.format(np.mean(cv_roc_auc)))



    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,8))

    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True,cmap='Blues')

    plt.ylabel('Actual Label')

    plt.xlabel('Predicted Label')

    plt.title('Confusion Matrix ('+clf_name+')',fontsize=16)

    

    

    # calculate the Optimum Threshold

    for i in range(len(fpr)):

        if fpr[i] + tpr[i] >= 1:

            i = i -1

            break



    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, lw=1,label='ROC Curve(area = %0.2f)    Optimum Threshold = %0.2f' % (roc_auc, threshold[i]))

    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve ('+clf_name+')',fontsize=16)

    plt.legend(loc="lower right")

    plt.show()
def compare_accuracy(clf,clf_name):

    print('5 fold Cross Validation('+clf_name+'):')

    cv_accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')

    print('Accuracy: {:2.2%}'.format(np.mean(cv_accuracy)))
# Fit on train set

dt_clf = DecisionTreeClassifier(random_state = 42, 

                             criterion = 'entropy',

                             max_depth = 5,

                             min_samples_leaf = 2,                                                   

                            )

dt_clf.fit(x_train, y_train)
print_predict_vs_test_multi(dt_clf,x_test,y_test)
get_results_multi(dt_clf,'Decision Tree - Multiclass Classification',x_train,y_train,x_test,y_test)
# Fit on train set

# wine_clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, random_state=200)

dt_clf2 = DecisionTreeClassifier(

                              max_depth = 5,

                              random_state = 42,

                              criterion = 'entropy',                             

                              min_samples_leaf = 2

                             )

dt_clf2.fit(x2_train, y2_train)
print_predict_vs_test_binary(dt_clf2,x2_test,y2_test)
get_results_binary(dt_clf2,'Decision Tree - Binary Classification',x2_train,y2_train,x2_test,y2_test)
# import visualization libraries

from IPython.display import Image  

import pydotplus

from sklearn.externals.six import StringIO

from sklearn import tree
# Multi-class

feature_names = np.array(features)

target_names = ['Level 1','Level 2','Level 3']



dot_data = tree.export_graphviz(dt_clf, out_file=None,

                         feature_names=feature_names,

                         class_names=target_names,  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = pydotplus.graph_from_dot_data(dot_data)  

graph.write_png("wine_dt_1.png")

Image(graph.create_png()) 
# Binary

feature_names2 = np.array(features)

target_names2 = ['Bad','Good']



dot_data = tree.export_graphviz(dt_clf2, out_file=None,

                         feature_names=feature_names2,

                         class_names=target_names2,  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = pydotplus.graph_from_dot_data(dot_data)  

graph.write_png("wine_dt_2.png")

Image(graph.create_png()) 
rf_clf = RandomForestClassifier(random_state = 42,

                                criterion = 'entropy', 

                                max_depth=6,

                                min_samples_leaf = 2,

                                n_estimators = 150)

rf_clf.fit(x_train, y_train)
print_predict_vs_test_multi(rf_clf,x_test,y_test)
get_results_multi(rf_clf,'Random Forest - Multiclass Classification',x_train,y_train,x_test,y_test)
rf_clf2 = RandomForestClassifier(random_state = 42,

                                 criterion = 'entropy',                             

                                 min_samples_leaf = 2,

                                 max_depth=7,

                                 n_estimators = 175)

rf_clf2.fit(x2_train, y2_train)
print_predict_vs_test_binary(rf_clf2,x2_test,y2_test)
get_results_binary(rf_clf2,'Random Forest - Binary Classification',x2_train,y2_train,x2_test,y2_test)
xgb_clf = XGBClassifier(random_state = 42, learning_rate = 0.08)

xgb_clf.fit(x_train, y_train)
print_predict_vs_test_multi(xgb_clf,x_test,y_test)
get_results_multi(xgb_clf,'XGBoost - Multiclass Classification',x_train,y_train,x_test,y_test)
xgb_clf2 = XGBClassifier(random_state = 42, 

                        learning_rate = 0.06,                                

                                  )

xgb_clf2.fit(x2_train, y2_train)
print_predict_vs_test_binary(xgb_clf2,x2_test,y2_test)
get_results_binary(xgb_clf2,'XGBoost - Binary Classification',x2_train,y2_train,x2_test,y2_test)
svm_clf = svm.SVC(random_state = 42,gamma='scale',probability=True)

svm_clf.fit(x_train, y_train)
print_predict_vs_test_multi(svm_clf,x_test,y_test)
get_results_multi_withoutFeatureImportance(svm_clf,'SVM - Multiclass Classification',x_train,y_train,x_test,y_test)
svm_clf2 = svm.SVC(C = 1,random_state = 42, probability=True, gamma='scale')

svm_clf2.fit(x2_train, y2_train)
print_predict_vs_test_binary(svm_clf2,x2_test,y2_test)
get_results_binary_withoutFeatureImportance(svm_clf2,'SVM - Binary Classification',x2_train,y2_train,x2_test,y2_test)
knn_clf = KNeighborsClassifier(n_neighbors=10)

knn_clf.fit(x_train, y_train)
print_predict_vs_test_multi(knn_clf,x_test,y_test)
get_results_multi_withoutFeatureImportance(knn_clf,'KNN - Multiclass Classification',x_train,y_train,x_test,y_test)
knn_clf2 = KNeighborsClassifier(n_neighbors=5)

knn_clf2.fit(x2_train, y2_train)
print_predict_vs_test_binary(knn_clf2,x2_test,y2_test)
get_results_binary_withoutFeatureImportance(knn_clf2,'KNN - Binary Classification',x2_train,y2_train,x2_test,y2_test)
# Best models for Multiclass and Binary classification: Random Forest and XGBoost



compare_accuracy(dt_clf,'Decision Tree - Multiclass')

compare_accuracy(dt_clf2,'Decision Tree - Binary')

compare_accuracy(rf_clf,'Random Forest - Multiclass')

compare_accuracy(rf_clf2,'Random Forest - Binary')

compare_accuracy(xgb_clf,'XGBoost - Multiclass')

compare_accuracy(xgb_clf2,'XGBoost - Binary')

compare_accuracy(svm_clf,'SVM - Multiclass')

compare_accuracy(svm_clf2,'SVM - Binary')

compare_accuracy(knn_clf,'KNN - Multiclass')

compare_accuracy(knn_clf2,'KNN - Binary')