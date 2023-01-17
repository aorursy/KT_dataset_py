# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import eli5



#statistic

from scipy import stats

from pandas import DataFrame

from scipy.stats import chisquare

from scipy.stats import chi2_contingency

from statsmodels.graphics.mosaicplot import mosaic



# machine learning structures

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn import metrics

import statsmodels.api as sm

from IPython import display

import graphviz

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from eli5.sklearn import PermutationImportance



pd.options.mode.chained_assignment = None  #hide any pandas warnings



# get the data

hd_data_path = '../input/heart-disease-uci/heart.csv'

hd_data = pd.read_csv(hd_data_path)

hd_data = hd_data.apply(pd.to_numeric)



# preview the data

hd_data.head()
# check whether data has null value

hd_data.info()
# missing values

hd_data.isnull().sum()
# see basic values about the data

hd_data.describe()
# check correlation between columns

plt.figure(figsize=(16, 12))

corr = hd_data.corr()

ax = sns.heatmap(

    corr, square=True, annot=True, fmt='.2f'

)

plt.show()
hd_data.groupby('target').mean()
colors=['cadetblue', 'gold']

sns.countplot(x="target", data=hd_data, palette=colors, alpha=0.5, edgecolor='black', linewidth=2)

plt.show()
hd_data_target_1=hd_data[hd_data.target==1]

hd_data_target_0=hd_data[hd_data.target==0]
f,ax=plt.subplots(3,2,figsize=(16,12))

f.delaxes(ax[2,1])



for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak','age']):

    colors = ['cadetblue', 'gold']

    sns.countplot(x=feature,data=hd_data,hue='target',ax=ax[i//2,i%2], palette = colors, alpha=0.7, linewidth=2)

    

    #sns.despine(ax[i//2,i%2]=ax[i//2,i%2], left=True)

    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)

    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)



plt.tight_layout()

plt.show()
# the code from https://www.kaggle.com/vincentlugat/heart-disease-analysis-and-prediction



f,ax=plt.subplots(3,2,figsize=(12,12))

f.delaxes(ax[2,1])



for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak','age']):

    sns.distplot(hd_data[hd_data['target']==0][(feature)], ax=ax[i//2,i%2], kde_kws={"color":"cadetblue"}, hist=False )

    sns.distplot(hd_data[hd_data['target']==1][(feature)], ax=ax[i//2,i%2], kde_kws={"color":"gold"}, hist=False )



    # Get the two lines from the ax[i//2,i%2]es to generate shading

    l1 = ax[i//2,i%2].lines[0]

    l2 = ax[i//2,i%2].lines[1]



    # Get the xy data from the lines so that we can shade

    x1 = l1.get_xydata()[:,0]

    y1 = l1.get_xydata()[:,1]

    x2 = l2.get_xydata()[:,0]

    y2 = l2.get_xydata()[:,1]

    ax[i//2,i%2].fill_between(x2,y2, color="gold", alpha=0.6)

    ax[i//2,i%2].fill_between(x1,y1, color="cadetblue", alpha=0.6)



     #grid

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.3)

    

    ax[i//2,i%2].set_title('{} - target'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)



    #sns.despine(ax[i//2,i%2]=ax[i//2,i%2], left=True)

    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)

    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)



plt.tight_layout()

plt.show()
f,ax=plt.subplots(3,2,figsize=(12,12))

f.delaxes(ax[2,1])



for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak','age']):

    colors = ['cadetblue', 'gold']

    sns.boxplot(x="target", y=feature , data=hd_data, ax=ax[i//2,i%2], palette=colors, boxprops=dict(alpha=0.8))



    # Get the two lines from the ax[i//2,i%2]es to generate shading

    l1 = ax[i//2,i%2].lines[0]

    l2 = ax[i//2,i%2].lines[1]



    # Get the xy data from the lines so that we can shade

    x1 = l1.get_xydata()[:,0]

    y1 = l1.get_xydata()[:,1]

    x2 = l2.get_xydata()[:,0]

    y2 = l2.get_xydata()[:,1]

    ax[i//2,i%2].fill_between(x2,y2, color="gold", alpha=0.6)

    ax[i//2,i%2].fill_between(x1,y1, color="cadetblue", alpha=0.6)



     #grid

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.3)

    

    ax[i//2,i%2].set_title('{} - target'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)



    #sns.despine(ax[i//2,i%2]=ax[i//2,i%2], left=True)

    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)

    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)



plt.tight_layout()

plt.show()
# t-test



# age and target

tTestResult = stats.ttest_ind(hd_data_target_1['age'], hd_data_target_0['age'])

tTestResultDiffVar = stats.ttest_ind(hd_data_target_1['age'], hd_data_target_0['age'], equal_var=False)

print('* age & target')

print(tTestResultDiffVar)

print("")



# trestbps and target

tTestResult = stats.ttest_ind(hd_data_target_1['trestbps'], hd_data_target_0['trestbps'])

tTestResultDiffVar = stats.ttest_ind(hd_data_target_1['trestbps'], hd_data_target_0['trestbps'], equal_var=False)

print('* trestbps & target')

print(tTestResultDiffVar)

print("")



# chol and target

tTestResult = stats.ttest_ind(hd_data_target_1['chol'], hd_data_target_0['chol'])

tTestResultDiffVar = stats.ttest_ind(hd_data_target_1['chol'], hd_data_target_0['chol'], equal_var=False)

print('* chol & target')

print(tTestResultDiffVar)

print("")



# thalach and target

tTestResult = stats.ttest_ind(hd_data_target_1['thalach'], hd_data_target_0['thalach'])

tTestResultDiffVar = stats.ttest_ind(hd_data_target_1['thalach'], hd_data_target_0['thalach'], equal_var=False)

print('* thalach & target')

print(tTestResultDiffVar)

print("")



# oldpeak and target

tTestResult = stats.ttest_ind(hd_data_target_1['oldpeak'], hd_data_target_0['oldpeak'])

tTestResultDiffVar = stats.ttest_ind(hd_data_target_1['oldpeak'], hd_data_target_0['oldpeak'], equal_var=False)

print('* oldpeak & target')

print(tTestResultDiffVar)

print("")
# scatter plot between numeric variables

var=['age', 'thalach', 'oldpeak', 'target']

sns.pairplot(hd_data[var], kind='scatter', diag_kind='hist')

plt.show()
# check regression plot



# age and thalach

f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="age", y="thalach", data=hd_data)

plt.show()



# age and oldpeak

f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="age", y="oldpeak", data=hd_data)

plt.show()



# thalach and oldpeak

f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="thalach", y="oldpeak", data=hd_data)

plt.show()
# check correlation between columns

plt.figure(figsize=(16, 12))

corr = hd_data[var].corr()

ax = sns.heatmap(

    corr, square=True, annot=True, fmt='.2f')

plt.show()
# code from https://www.kaggle.com/vincentlugat/heart-disease-analysis-and-prediction



f,ax=plt.subplots(4,2,figsize=(12,12))



for i,feature in enumerate(['sex','cp','fbs','restecg','exang','slope','ca','thal']):

    colors = ['cadetblue', 'gold']

    sns.countplot(x=feature,data=hd_data,hue='target',ax=ax[i//2,i%2], palette = colors, alpha=0.7, edgecolor=('black'), linewidth=2)

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.4)

    ax[i//2,i%2].set_title('Count of {} vs target'.format(feature), fontsize=18)

    ax[i//2,i%2].legend(loc='best')

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('modality', fontsize=12)



plt.tight_layout()

plt.show()
mosaic(hd_data.sort_values('sex'), ['target', 'sex'],

      title='Mosaic chart of sex and target')

plt.show()



mosaic(hd_data.sort_values('cp'), ['target', 'cp'],

      title='Mosaic chart of cp and target')

plt.show()



mosaic(hd_data.sort_values('fbs'), ['target', 'fbs'],

      title='Mosaic chart of fbs and target')

plt.show()



mosaic(hd_data.sort_values('restecg'), ['target', 'restecg'],

      title='Mosaic chart of restecg and target')

plt.show()



mosaic(hd_data.sort_values('exang'), ['target', 'exang'],

      title='Mosaic chart of exang and target')

plt.show()



mosaic(hd_data.sort_values('slope'), ['target', 'slope'],

      title='Mosaic chart of slope and target')

plt.show()



mosaic(hd_data.sort_values('ca'), ['target', 'ca'],

      title='Mosaic chart of ca and target')

plt.show()



mosaic(hd_data.sort_values('thal'), ['target', 'thal'],

      title='Mosaic chart of thal and target')

plt.show()
print(hd_data.groupby('sex')['target'].value_counts())

print(hd_data.groupby('cp')['target'].value_counts())

print(hd_data.groupby('fbs')['target'].value_counts())

print(hd_data.groupby('restecg')['target'].value_counts())

print(hd_data.groupby('slope')['target'].value_counts())

print(hd_data.groupby('exang')['target'].value_counts())

print(hd_data.groupby('ca')['target'].value_counts())

print(hd_data.groupby('thal')['target'].value_counts())
sex_target0, sex_target1 = [24, 114], [72, 93]

sex_target = DataFrame([sex_target0, sex_target1], columns=['sex=0(F)', 'sex=1(M)'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



cp_target0, cp_target1 = [104, 9, 18, 7], [39, 41, 69, 16]

cp_target = DataFrame([cp_target0, cp_target1], columns=['cp=0', 'cp=1', 'cp=2', 'cp=3'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



fbs_target0, fbs_target1 = [116, 22], [142, 23]

fbs_target = DataFrame([fbs_target0, fbs_target1], columns=['fbs=0(<=120mg/dl)', 'fbs=1(>120mg/dl)'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



rest_target0, rest_target1 = [79, 56, 3], [68, 96, 1]

rest_target = DataFrame([rest_target0, rest_target1], columns=['restecg=0(normal)', 'restecg=1(ST-T)', 'restecg=2(Probable)'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



slp_target0, slp_target1 = [12, 91, 35], [9, 49, 107]

slp_target = DataFrame([slp_target0, slp_target1], columns=['slp=0', 'slp=1', 'slp=2'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



ex_target0, ex_target1 = [62, 76], [142, 23]

ex_target = DataFrame([ex_target0, ex_target1], columns=['ex=0(not induced)', 'ex=1(induced)'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



ca_target0, ca_target1 = [45, 44, 31, 17, 1], [130, 21, 7, 3, 4]

ca_target = DataFrame([ca_target0,ca_target1], columns=['ca=0', 'ca=1', 'ca=2', 'ca=3', 'ca=4'], index=['target=0(No Disease)', 'target=1(Have Disease)'])



thal_target0, thal_target1 = [1, 12, 36, 89], [1, 6, 130, 28]

thal_target = DataFrame([thal_target0,thal_target1], columns=['thal=0', 'thal=1', 'thal=2', 'thal=3'], index=['target=0(No Disease)', 'target=1(Have Disease)'])

sex_target
cp_target
fbs_target
rest_target
rest_target0, rest_target1 = [79, 59], [68, 97]

rest_target = DataFrame([rest_target0, rest_target1], columns=['restecg=0(normal)', 'restecg=1(ST-T)'], index=['target=0(No Disease)', 'target=1(Have Disease)'])

rest_target
slp_target
ex_target
ca_target
ca_target0, ca_target1 = [45, 44, 31, 18], [130, 21, 7, 7]

ca_target = DataFrame([ca_target0,ca_target1], columns=['ca=0', 'ca=1', 'ca=2', 'ca=3'], index=['target=0(No Disease)', 'target=1(Have Disease)'])

ca_target
thal_target
thal_target0, thal_target1 = [13, 36, 89], [7, 130, 28]

thal_target = DataFrame([thal_target0,thal_target1], columns=['thal=1', 'thal=2', 'thal=3'], index=['target=0(No Disease)', 'target=1(Have Disease)'])

thal_target
# code from https://m.blog.naver.com/PostView.nhn?blogId=parksehoon1971&logNo=221589203965&proxyReferer=https%3A%2F%2Fwww.google.com%2F



chi2, p, dof, expected = chi2_contingency([sex_target0, sex_target1])

print(" * sex - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([cp_target0, cp_target1])

print(" * cp - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([fbs_target0, fbs_target1])

print(" * fbs - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([rest_target0, rest_target1])

print(" * rest - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([slp_target0, slp_target1])

print(" * slp - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([ex_target0, ex_target1])

print(" * ex - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([ca_target0, ca_target1])

print(" * ca - target expected frequency")

print(expected)

print("")



chi2, p, dof, expected = chi2_contingency([thal_target0, thal_target1])

print(" * thal - target expected frequency")

print(expected)

print("")
sex_target_result = chisquare(sex_target0, f_exp=sex_target1)

cp_target_result = chisquare(cp_target0, f_exp=cp_target1)

fbs_target_result = chisquare(fbs_target0, f_exp=fbs_target1)

rest_target_result = chisquare(rest_target0, f_exp=rest_target1)

slp_target_result = chisquare(slp_target0, f_exp=slp_target1)

ex_target_result = chisquare(ex_target0, f_exp=ex_target1)

ca_target_result = chisquare(ca_target0, f_exp=ca_target1)

thal_target_result = chisquare(thal_target0, f_exp=thal_target1)
print(" * sex-target")

print(sex_target_result)

print("")

print(" * cp-target")

print(cp_target_result)

print("")

print(" * fbs-target")

print(fbs_target_result)

print("")

print(" * rest-target")

print(rest_target_result)

print("")

print(" * slp-target")

print(slp_target_result)

print("")

print(" * ex-target")

print(ex_target_result)

print("")

print(" * ca-target")

print(ca_target_result)

print("")

print(" * thal-target")

print(thal_target_result)

print("")
y = hd_data.target

X = hd_data
# choose the cols(which has p value smaller than 0.01)

filtered_col = ['age', 'sex', 'cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

y1 = y

X1 = X[filtered_col]



# split into train set and test set

train_X1, test_X1, train_y1, test_y1 = train_test_split(X1, y1, random_state=0)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, max_depth = 7)

random_forest.fit(train_X1, train_y1)

y_pred = random_forest.predict(test_X1)



acc_RandomForest = accuracy_score(test_y1, y_pred)*100

print(acc_RandomForest)
# code from https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c



estimator = random_forest.estimators_[1]



export_graphviz(estimator, out_file='tree.dot', 

                feature_names = train_X1.columns,

                class_names = ["0(No disease)","1(Have disease)"],

                rounded = True, proportion = True, 

                label='root', filled = True,

                precision = 2)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
# code fromhttps://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning



# KNN

# try ro find best k value

scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(train_X1, train_y1)

    y_pred = knn2.predict(test_X1)

    scoreList.append(accuracy_score(test_y1, y_pred))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()



accuracies = {}

acc = max(scoreList)*100

accuracies['KNN'] = acc

print("Maximum KNN Score is {:.2f}%".format(acc))



# KNN

knn = KNeighborsClassifier(n_neighbors = 13)

knn.fit(train_X1, train_y1)

y_pred = knn.predict(test_X1)

acc_knn = accuracy_score(test_y1, y_pred)*100

print(acc_knn)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(train_X1, train_y1)

y_pred = gaussian.predict(test_X1)

acc_gaussian_naive_bayes = accuracy_score(test_y1, y_pred)*100

print(acc_gaussian_naive_bayes)

# Perceptron



perceptron = Perceptron()

perceptron.fit(train_X1, train_y1)

y_pred = perceptron.predict(test_X1)

acc_perceptron = accuracy_score(test_y1, y_pred)*100

print(acc_perceptron)

# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(train_X1, train_y1)

y_pred = linear_svc.predict(test_X1)

acc_linear_svc = accuracy_score(test_y1, y_pred)*100

print(acc_linear_svc)

# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(train_X1, train_y1)

y_pred = sgd.predict(test_X1)

acc_stochastic_gradient_descent = accuracy_score(test_y1, y_pred)*100

print(acc_stochastic_gradient_descent)

# Decision Tree



decision_tree = DecisionTreeClassifier(max_depth=7)

decision_tree.fit(train_X1, train_y1)

y_pred = decision_tree.predict(test_X1)

acc_decision_tree = accuracy_score(test_y1, y_pred)*100

print(acc_decision_tree)

export_graphviz(decision_tree, out_file="tree.dot",

                feature_names=train_X1.columns, 

                class_names=["0(No disease)","1(Have disease)"], 

                rounded = True, proportion = True, 

                label='root', precision = 2,

                filled=True, impurity=True)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
# Logistic Regression



logistic_regression_model = LogisticRegression()

logistic_regression_model.fit(train_X1, train_y1)

X1=sm.add_constant(X)

model=sm.OLS(y, X1)

res=model.fit()

y_pred = logistic_regression_model.predict(test_X1)

acc_logistic_regression=accuracy_score(test_y1, y_pred)*100

print(acc_logistic_regression)

models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_knn, acc_logistic_regression, 

              acc_RandomForest, acc_gaussian_naive_bayes, acc_perceptron, 

              acc_stochastic_gradient_descent, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
plt.rcParams['figure.figsize']=(15,8)



colors = ['palegoldenrod','lightgreen','cadetblue','gold','greenyellow','aquamarine','steelblue','khaki']



ax = plt.bar(x='Model', data=models, height="Score", alpha=0.7, color=colors, edgecolor=('black'), linewidth=2)

plt.ylabel('accuracy score', fontsize=12)

plt.xlabel('model', fontsize=12)

plt.title('Accuracy score of Models', fontsize=18)

plt.show()

# code from https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model



y_pred = random_forest.predict(test_X1)

y_pred_quant = random_forest.predict_proba(test_X1)[:, 1]

y_pred_bin = random_forest.predict(test_X1)
confusion_matrix = confusion_matrix(test_y1, y_pred_bin)

confusion_matrix
total=sum(sum(confusion_matrix))



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(test_y1, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr, tpr)
perm = PermutationImportance(random_forest, random_state=1).fit(test_X1, test_y1)

eli5.show_weights(perm, feature_names = test_X1.columns.tolist())