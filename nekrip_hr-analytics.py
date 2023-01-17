%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats



# use seaborn plotting defaults

import seaborn as sns; sns.set()



hr = pd.read_csv("../input/HR_comma_sep.csv")
hr.head(3)
hr.shape
hr.info()
list(hr.columns)
print(hr.describe())
hr.sales.value_counts()
hr.salary.value_counts()
# Generate a custom diverging colormap



corr = hr.loc[:,'satisfaction_level':'promotion_last_5years']

correlation = corr.corr()



cmap = sns.diverging_palette(255, 0, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(correlation, cmap=cmap,

            square=True, xticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .8})

plt.title("Variable Correlation Heatmap", fontsize = 18)
# Correlation tells relation between two attributes.

# Correlation requires continous data. Hence, ignore categorical data



# Calculates pearson co-efficient for all combinations

correlation = corr.corr()



# Set the threshold to select only highly correlated attributes

threshold = 0.1



# List of pairs along with correlation above threshold

corr_list = []



num_cont = 8



#Search for the highly correlated pairs

for i in range(0,num_cont): #for 'size' features

    for j in range(i+1,num_cont): #avoid repetition

        if (correlation.iloc[i,j] >= threshold and correlation.iloc[i,j] < 1) or (correlation.iloc[i,j] < 0 and correlation.iloc[i,j] <= -threshold):

            corr_list.append([correlation.iloc[i,j],i,j]) #store correlation and columns index



#Sort to show higher ones first            

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))



cols = corr.columns





#Print correlations and column names

for v,i,j in s_corr_list:

    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# Strong correlation is observed between the following pairs

# This represents an opportunity to reduce the feature set through transformations such as PCA
n_rows = 2

n_cols = 4



hr_except_left = hr.drop(['sales', 'salary'], axis = 1)

hr_except_left.hist(figsize = (15,12), ls = 'solid', edgecolor='k')

plt.subplots_adjust(hspace = 0.4)
category = ['sales', 'salary']



hr_expanded = pd.get_dummies(hr, prefix = category)
hr_expanded.shape
list(hr_expanded.columns)
hr_expanded.head()
X, y = hr_expanded.drop('left', axis = 1), hr_expanded.left



from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

pca.fit(X)

X_reduced = pca.transform(X)

print("Reduced dataset shape:", X_reduced.shape)



import pylab as pl

pl.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y,

           cmap = 'RdYlBu')
# Kmeans divides but it does not represent truth as seen above



from sklearn.cluster import KMeans

k_means = KMeans(n_clusters = 2, random_state = 0)

k_means.fit(X)

y_pred = k_means.predict(X)



pl.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y_pred,

           cmap = 'RdYlBu')
%%time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split

from sklearn import metrics



X, y = hr_expanded.drop('left', axis = 1), hr_expanded.left

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = KNeighborsClassifier(n_neighbors = 1)

clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)





print("Accuracy of KNeighborsClassifier is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

y_pred_prob = clf.predict_proba(X_test)[:,1]

print("AUC for KNeighborsClassifier is", format(metrics.roc_auc_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))
%%time

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.cross_validation import train_test_split





X = hr_expanded.drop('left', axis = 1)

y = hr_expanded.left



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

MODEL = RandomForestClassifier(n_estimators = 700)

MODEL.fit(X_train, y_train)

y_pred_class = MODEL.predict(X_test)



accuracy = metrics.accuracy_score(y_test, y_pred_class)

print("Accuracy of RandomForestClassifier is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

y_pred_prob = MODEL.predict_proba(X_test)[:,1]

print("AUC for RandomForestClassifier is", format(metrics.roc_auc_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))
data = {'Features': list(X.columns),

                      '% Important': list(MODEL.feature_importances_)}

important_features = pd.DataFrame(data)

important_features.sort_index(by = '% Important', ascending = False)
# Created a new Datafram with the top 5 important features

# Hooyah dimension reduction



data1 = {'satisfaction_level': list(hr_expanded.satisfaction_level),

        'time_spend_company': list(hr_expanded.time_spend_company),

        'average_monthly_hours': list(hr_expanded.average_montly_hours),

        'last_evaluation': list(hr_expanded.last_evaluation),

        'number_project': list(hr_expanded.number_project)}

reduced_hr = pd.DataFrame(data)
%%time

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.cross_validation import train_test_split





X = hr_expanded.drop('left', axis = 1)

y = hr_expanded.left



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

MODEL = RandomForestClassifier(n_estimators = 750)

MODEL.fit(X_train, y_train)

y_pred_class = MODEL.predict(X_test)



from sklearn import metrics



accuracy = metrics.accuracy_score(y_test, y_pred_class)

print("Accuracy of RandomForestClassifier is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

y_pred_prob = MODEL.predict_proba(X_test)[:,1]

print("AUC for RandomForestClassifier is", format(metrics.roc_auc_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))
%%time

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.cross_validation import train_test_split



X = hr_expanded.drop('left', axis = 1)

y = hr.left



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

MODEL = SVC()

MODEL.fit(X_train, y_train)

y_pred_class = MODEL.predict(X_test)



accuracy = metrics.accuracy_score(y_test, y_pred_class)

print("Accuracy of SVC is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))
%%time

from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics

from sklearn.cross_validation import train_test_split



X = hr_expanded.drop('left', axis = 1)

y = hr.left



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

MODEL = AdaBoostClassifier(n_estimators = 600)

MODEL.fit(X_train, y_train)

y_pred_class = MODEL.predict(X_test)



from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred_class)

print("Accuracy of AdaBoostClassifier is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

y_pred_prob = MODEL.predict_proba(X_test)[:,1]

print("AUC for AdaBoostClassifier is", format(metrics.roc_auc_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))
%%time

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics

from sklearn.cross_validation import train_test_split



X = hr_expanded.drop('left', axis = 1)

y = hr.left



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

MODEL = GradientBoostingClassifier(n_estimators=200)

MODEL.fit(X_train, y_train)

y_pred_class = MODEL.predict(X_test)



accuracy = metrics.accuracy_score(y_test, y_pred_class)

print("Accuracy of GradientBoostingClassifier is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

y_pred_prob = MODEL.predict_proba(X_test)[:,1]

print("AUC for GradientBoostingClassifier is", format(metrics.roc_auc_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))
%%time

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.cross_validation import train_test_split



X = hr_expanded.drop('left', axis = 1)

y = hr.left



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

MODEL = LogisticRegression()

MODEL.fit(X_train, y_train)

y_pred_class = MODEL.predict(X_test)



accuracy = metrics.accuracy_score(y_test, y_pred_class)

print("Accuracy of LogisticRegression is", format(metrics.accuracy_score(y_test, y_pred_class), '.3%'))

y_pred_prob = MODEL.predict_proba(X_test)[:,1]

print("AUC for LogisticRegression is", format(metrics.roc_auc_score(y_test, y_pred_class), '.3%'))

print()

print('Classification Matrix is:')

print(metrics.confusion_matrix(y_test, y_pred_class))