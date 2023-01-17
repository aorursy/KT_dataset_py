# 1-Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import beta
from scipy.stats import f
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# 2-Import Data
# description --> https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv') 
data.head()
data.info()
data = data.iloc[:, 1:-1]

variables = data.iloc[:, 1:]
labels = data.iloc[:, 0]
data.describe()
# For illustration purposes
data_visualization = data.iloc[:, [0, 1, 2, 3, 4, 14, 21, 22, 23, 24]]
# Visualisation of the data using a box plot
fig=plt.figure(figsize=(16,8), dpi= 100, facecolor='w', edgecolor='k')
ax = sns.boxplot(data=data_visualization, orient="v", palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# outlier detection
Q1 = variables.quantile(0.25)
Q3 = variables.quantile(0.75)
IQR = Q3 - Q1
Lower_Whisker = Q1 - 1.5 * IQR
Upper_Whisker = Q3 + 1.5 * IQR

print('Lower Whisker:')
print(Lower_Whisker)
print('\n\n\nUpperWhisker:')
print(Upper_Whisker)

Lower_outliers = variables < Lower_Whisker
Upper_outliers = variables > Upper_Whisker

Lower_outliers_index = Lower_outliers.any(axis=1)
Upper_outliers_index = Upper_outliers.any(axis=1)
print(f'There are {sum(Lower_outliers_index)} points below the lower whisker')
print(f'There are {sum(Upper_outliers_index)} points above the upper whisker')
variables.skew()
median = variables.median()
variables = variables.where((variables >= Lower_Whisker) & (variables <= Upper_Whisker), median, axis=1)
variables.skew()
# Pair plot
sns.pairplot(data_visualization, hue='diagnosis')
correlation = variables.corr('pearson')
plt.figure(figsize=(25,25), dpi= 100, facecolor='w', edgecolor='k')
ax = sns.heatmap(correlation.round(2), cmap='RdYlGn_r', linewidths=0.5, annot=True,
                 cbar=True, square=True, fmt='0.2f')
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False, labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title('Correlation matrix')
# Observations and variables
observations_index = list(data.index)
variables_name = list(data.columns[1:])

for i in range(len(variables_name)):
    for j in range(len(variables_name)):
        if i != j & j > i:
            if correlation.iloc[i, j] > 0.9:
                print("{} and {} are {} correlated".format(variables_name[i], variables_name[j],
                                                           correlation.iloc[i, j].round(3)))
variables = variables.drop(['perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'perimeter_se'], axis=1)
# Label Encoding
labels_encoder_response = LabelEncoder()
labels = labels_encoder_response.fit_transform(labels)
# train and test split
X_train, X_test, y_train, y_test = train_test_split(variables, labels, test_size=0.2, random_state=0)

# standardization
sc_training = StandardScaler()
X_train = sc_training.fit_transform(X_train)
X_test = sc_training.transform(X_test)

# Principal component analysis
pca = PCA()
Z_train = pca.fit_transform(X_train)
Z_test = pca.fit_transform(X_test)
# Eigenvalues
Eigen_Values = pca.explained_variance_
ell = pca.explained_variance_ratio_


# Scree plot
plt.subplots(1, 2, figsize = (20, 10))

ax1 = plt.subplot(1, 2, 1)
x = np.arange(len(Eigen_Values)) + 1
ax1.plot(x, Eigen_Values / Eigen_Values.sum(), 'ro-', lw=2)
ax1.set_xticks(x, ["" + str(i) for i in x])
ax1.set_xlabel('Number of components')
ax1.set_ylabel('Explained variance')
ax1.set_title('Scree Plot')

# Pareto plot
ax2 = plt.subplot(1, 2, 2)
ind = np.arange(1, len(ell) + 1)
ax2.bar(ind, ell, align='center', alpha=0.5)
ax2.plot(np.cumsum(ell))
ax2.set_xlabel('Number of components')
ax2.set_ylabel('Cumulative explained variance')
ax1.set_title('Pareto Plot')

for x, y in zip(ind, np.cumsum(ell)):
    label = "{:.2f}".format(y)
    if float(label) >= 0.79:
        plt.annotate("cumulative explained variance: " + label + "\n" +
                     "Number of PC: " + str(x),  # this is the text
                     (x, y),  # this is the point to label
                     textcoords='figure fraction',  # how to position the text
                     xytext=(.8, 0.5),  # distance from text to points (x,y)
                     arrowprops=dict(facecolor='black', shrink=0.1),
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     ha='center')  # horizontal alignment can be left, right or center
        NUMBER_OF_PCs = x # for further use
        break
Z_train = Z_train[:, :NUMBER_OF_PCs]
Z_test = Z_test[:, :NUMBER_OF_PCs]
Model_Score = []

# Logistic Regression
classifier_lr = LogisticRegression(random_state=0)
scores = cross_val_score(classifier_lr, Z_train, y_train, cv=10, scoring='accuracy')
lr_train_score_mean = scores.mean()
lr_train_score_std = scores.std()
classifier_lr.fit(Z_train, y_train)
model_name = 'Logistic Regression'
lr_test_score = classifier_lr.score(Z_test, y_test)

score = list((model_name, lr_train_score_mean.round(4), lr_train_score_std.round(4), lr_test_score.round(4)))
Model_Score.append(score)
# Making the confusion matrix
y_predicted_lr = classifier_lr.predict(Z_test)
cm_lr = confusion_matrix(y_test, y_predicted_lr)

print(cm_lr)
# SVC
def svm(degree, kernel, gamma, x_train, x_test, train_label, test_label):
    if kernel == 'poly':
        support_vector_machine = SVC(kernel='poly', degree=degree, random_state=0)
        cv_score = cross_val_score(support_vector_machine, x_train, train_label, cv=10)
        svm_train_score_mean = cv_score.mean()
        svm_train_score_std = cv_score.std()
        support_vector_machine.fit(x_train, train_label)
        svm_test_score = support_vector_machine.score(x_test, test_label)
        name = 'SVM with ' + str(degree) + '-degree polynomial kernel'
    elif kernel == 'rbf':
        support_vector_machine = SVC(kernel='rbf', gamma=gamma, random_state=0)
        cv_score = cross_val_score(support_vector_machine, x_train, train_label, cv=10)
        svm_train_score_mean = cv_score.mean()
        svm_train_score_std = cv_score.std()
        support_vector_machine.fit(x_train, train_label)
        svm_test_score = support_vector_machine.score(x_test, test_label)
        name = 'SVM with rbf kernel and ' + str(gamma) + ' coefficient'
    else:
        support_vector_machine = SVC(kernel='sigmoid', gamma=gamma, random_state=0)
        cv_score = cross_val_score(support_vector_machine, x_train, train_label, cv=10)
        svm_train_score_mean = cv_score.mean()
        svm_train_score_std = cv_score.std()
        support_vector_machine.fit(x_train, train_label)
        svm_test_score = support_vector_machine.score(x_test, test_label)
        name = 'SVM with sigmoid kernel and ' + str(gamma) + ' coefficient'
    return support_vector_machine, list((name, svm_train_score_mean.round(4), svm_train_score_std.round(4),
                                         svm_test_score.round(4)))

svc_models = [
    [None, 'rbf', 'scale'], [None, 'rbf', 'auto'], [None, 'sigmoid', 'scale'], [None, 'sigmoid', 'auto'],
    [1, 'poly', None], [2, 'poly', None], [3, 'poly', 'None'], [4, 'poly', 'None'], [5, 'poly', 'None']
]

for i, x in enumerate(svc_models):
    _, score = svm(svc_models[i][0], svc_models[i][1], svc_models[i][2], Z_train, Z_test, y_train, y_test)
    Model_Score.append(score)
# Random Forest Classification
acc_score = []
std_score = []
max_rf_ne = 50
for ne in range(1, max_rf_ne):
    classifier_rf = RandomForestClassifier(n_estimators=ne, random_state=0)
    scores = cross_val_score(classifier_rf, Z_train, y_train, cv=10, scoring='accuracy')
    rf_train_score_mean = scores.mean()
    rf_train_score_std = scores.std()
    acc_score.append(rf_train_score_mean)
    std_score.append(rf_train_score_std)

best_rf_acc = max(acc_score)
best_rf_ne = acc_score.index(max(acc_score))

classifier_rf = RandomForestClassifier(n_estimators=best_rf_ne, random_state=0)
classifier_rf.fit(Z_train, y_train)
rf_test_score = classifier_rf.score(Z_test, y_test)

f, ax = plt.subplots()
ax.plot(range(1, max_rf_ne), acc_score, marker='o')
ax.set_title('accuracy')

model_name = 'Random Forest with ' + str(best_rf_ne) + ' estimators'

score = list((model_name, rf_train_score_mean.round(4), rf_train_score_std.round(4), rf_test_score.round(4)))
Model_Score.append(score)
# ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=NUMBER_OF_PCs))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier_ann = KerasClassifier(build_fn=build_classifier, batch_size=25, epochs=100)
accuracies = cross_val_score(estimator=classifier_ann, X=Z_train, y=y_train, cv=10)
ann_score = accuracies.mean()
ann_std = accuracies.std()

classifier_ann.fit(Z_train, y_train)
ann_test_score = classifier_ann.score(Z_test, y_test)

model_name = 'Two-hidden-layer ANN with 25 batch size and 100 epochs'
score = list((model_name, np.round(ann_score, 4), np.round(ann_std, 4), np.round(ann_test_score, 4)))
Model_Score.append(score)
# Max voting
def max_voting(estimators, x_train, x_test, train_label, test_label):
    mv_classifier = VotingClassifier(estimators=estimators, voting='hard')
    cv_score = cross_val_score(mv_classifier, x_train, train_label, cv=10)
    mv_train_score_mean = cv_score.mean()
    mv_train_score_std = cv_score.std()
    mv_classifier.fit(x_train, train_label)
    name = 'Max voting'  # Add the name of the base models (estimators)
    mv_test_score = mv_classifier.score(x_test, test_label)
    return mv_classifier, list((name, mv_train_score_mean.round(4), mv_train_score_std.round(4),
                                mv_test_score.round(4)))

model_1 = RandomForestClassifier(n_estimators=best_rf_ne)
model_2 = LogisticRegression(random_state=0)
model_3 = SVC(kernel='poly', degree=2)
model_4 = SVC(kernel='poly', degree=3)
model_5 = SVC(kernel='poly', degree=4)
model_6 = SVC(kernel='poly', degree=5)
model_7 = SVC(kernel='rbf', gamma='scale')
model_8 = SVC(kernel='rbf', gamma='auto')
model_9 = SVC(kernel='sigmoid', gamma='scale')
model_10 = SVC(kernel='sigmoid', gamma='auto')

# version 10
classifier_mv, score = max_voting([('rf', model_1), ('lr', model_2), ('SVM_2', model_3), ('SVM_3', model_4),
                                   ('SVM_4', model_5), ('SVM_5', model_6), ('SVM_rbf_scale', model_7),
                                   ('SVM_rbf_auto', model_8), ('SVM_sigmoid_scale', model_9),
                                   ('SVM_sigmoid_auto', model_10)], Z_train, Z_test, y_train, y_test)

Model_Score.append(score)
# Results
Model_Score = pd.DataFrame(Model_Score, columns=['Model', 'Train Score Average', 'Train Score SD', 'Test Score'])
Model_Score = Model_Score.sort_values(by=['Test Score'], ascending=False)
Model_Score