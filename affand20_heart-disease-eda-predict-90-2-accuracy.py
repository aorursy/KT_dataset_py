import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head(2)
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

data.columns.unique()
data.describe().round(2)
data.info()
col_class = [
    len(data.sex.unique()),
    len(data.chest_pain_type.unique()),
    len(data.resting_blood_pressure.unique()),
    len(data.serum_cholesterol.unique()),
    len(data.fasting_blood_sugar.unique()),
    len(data.rest_ecg.unique()),
    len(data.max_heart_rate.unique()),
    len(data.exercise_angina.unique()),
    len(data.st_depression.unique()),
    len(data.st_slope.unique()),
    len(data.num_major_vessels.unique()),
    len(data.thalassemia.unique()),
    len(data.target.unique()),
]

plt.figure(figsize=(6,6))
plt.barh(data.columns.unique()[1:], col_class)
plt.show()
print('Total people safe = %d' % len(data[data.target == 0]))
print('Total people diseased = %d' % len(data[data.target == 1]))

plt.figure(figsize=(5,5))
sns.countplot(data.target)
plt.ylim(0,250)
plt.legend(['Safe', 'Diseased'])
plt.show()
data['age'].hist(bins=25)
plt.figure(figsize=(18, 10))
sns.countplot(x='age', hue='target', data=data)
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
data['age'].mode()
young = data[data.age <= 40]
middle = data[(data.age > 40) & (data.age <= 55)]
old = data[data.age > 55]

print('Total young people %d' % len(young))
print('Total middle-aged people %d' % len(middle))
print('Total old people %d' % len(old))

ax = plt.figure(figsize=(8,4))
plt.barh(['young', 'middle', 'old'], [len(young), len(middle), len(old)])
plt.title('Total people in each group age')
plt.show()
age_m = [len(young[young.sex==1]), len(middle[middle.sex==1]), len(old[old.sex==1])]
age_f = [len(young[young.sex==0]), len(middle[middle.sex==0]), len(old[old.sex==0])]
xpos = np.arange(0,3)

plt.xticks(xpos, ['young', 'middle', 'old'])
plt.bar(xpos+0.2, age_m, width=0.4, label='Male')
plt.bar(xpos-0.2, age_f, width=0.4, label='Female')
plt.title('Total people in each group age by gender')
plt.ylim(0,130)
plt.legend()
plt.show()
age_m = [len(young[young.target==1]), len(middle[middle.target==1]), len(old[old.target==1])]
age_f = [len(young[young.target==0]), len(middle[middle.target==0]), len(old[old.target==0])]
xpos = np.arange(0,3)

plt.xticks(xpos, ['young', 'middle', 'old'])
plt.bar(xpos+0.2, age_m, width=0.4, label='Diseased')
plt.bar(xpos-0.2, age_f, width=0.4, label='Safe')
plt.title('Total diseased and safe people on each group age')
plt.ylim(0,130)
plt.legend()
plt.show()
plt.figure(figsize=(8,5))
plt.title('People safe and diseased by sex')
sns.countplot(y=data.sex, hue=data.target)
plt.yticks(np.arange(0,2), ['Female', 'Male'])
plt.show()
fig = plt.figure(figsize=(15,6))
plt.title('People safe and diseased by sex and group age')
plt.axis('off')

ax1 = fig.add_subplot(131)
sns.countplot(young.sex, hue=young.target, ax=ax1)
ax1.set_xticklabels(['Male', 'Female'])
ax1.set_ylim(0,100)

ax2 = fig.add_subplot(132)
sns.countplot(middle.sex, hue=middle.target, ax=ax2)
ax2.set_xticklabels(['Male', 'Female'])
ax2.set_ylim(0,100)

ax3 = fig.add_subplot(133)
sns.countplot(old.sex, hue=old.target, ax=ax3)
ax3.set_xticklabels(['Male', 'Female'])
ax3.set_ylim(0,100)

plt.show()
sns.countplot(x='chest_pain_type', hue="target", data=data)
plt.title("People safe and diseased grouped by chest pain type")
plt.show()
sns.barplot(x="target", y='resting_blood_pressure',data = data)
plt.title('People safe and diseased grouped by their blood pressure')
plt.show()
sns.barplot(x="target", y='serum_cholesterol',data = data)
plt.title('People safe and diseased grouped by their cholesterol')
plt.show()
sns.countplot(hue='fasting_blood_sugar',x ='target',data = data)
plt.title('People safe and diseased grouped by their cholesterol their Fasting Blood Sugar')
plt.show()
sns.countplot(x='rest_ecg', hue ='target', data = data)
plt.title('People safe and diseased grouped by their ECG graph')
plt.show()
sns.barplot(x="target", y='max_heart_rate',data= data)
plt.title('People safe and diseased grouped by their Maximum Heart Rate')
plt.show()
sns.countplot(x='exercise_angina', hue ='target', data = data)
plt.title('People safe and diseased grouped by their Exercise Induced Angina')
plt.show()
sns.countplot(hue='st_slope',x ='target',data = data)
plt.title('People safe and diseased grouped by their depression value')
plt.show()
sns.countplot(hue='num_major_vessels',x ='target',data = data)
plt.title('People safe and diseased grouped by their Number Blood Vessel')
plt.show()
sns.countplot(hue='thalassemia',x ='target',data = data)
plt.title('People safe and diseased grouped by their thalassemia type')
plt.show()
plt.figure(figsize=(10,8))
plt.title('Correlation between variable before outlier removal')
sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()
# sns.scatterplot('age', 'st_depression', data=data)
plt.figure(figsize=(8,4))
sns.pairplot(data[['age', 'st_depression', 'serum_cholesterol', 'resting_blood_pressure', 'max_heart_rate']])
plt.show
def assign_col(row):
    if row >= 29 and row <= 40:
        return 1
    elif row > 40 and row <= 55:
        return 2
    else:
        return 3

data['age_bin'] = data['age'].apply(assign_col)
data = data[['age', 'age_bin', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']]
data.head()
plt.figure(figsize=(7,8))
plt.title('Boxplot Initial Data')

ax1 = plt.subplot(221)
ax1.boxplot(data['st_depression'])
ax1.set_xticklabels(['st_depression'])

ax2 = plt.subplot(222)
ax2.boxplot(data['serum_cholesterol'])
ax2.set_xticklabels(['serum_cholesterol'])

ax3 = plt.subplot(223)
ax3.boxplot(data['resting_blood_pressure'])
ax3.set_xticklabels(['resting_blood_pressure'])

ax4 = plt.subplot(224)
ax4.boxplot(data['max_heart_rate'])
ax4.set_xticklabels(['max_heart_rate'])

plt.show()
def remove_outliers(df, column):
    upper = df[column].quantile(.90)
    lower = df[column].quantile(.10)
    
    out = df[(df[column] > upper) | (df[column] < lower)]
    print('Total outlier %s = %d' % (column, len(out)))
    
    df = df[(df[column] < upper) & (df[column] > lower)]
    return df
# before remove outlier
len(data)
columns = ['st_depression', 'serum_cholesterol', 'resting_blood_pressure', 'max_heart_rate']

# remove outlier
data_clean = data
for i in columns:
    data_clean = remove_outliers(data_clean, i)

data_clean = data_clean.reset_index()

# after remove outlier
print('=====================')
print('After outlier removal = %d ' % len(data_clean))
plt.figure(figsize=(7,8))

ax1 = plt.subplot(221)
ax1.boxplot(data_clean['st_depression'])
ax1.set_xticklabels(['st_depression'])

ax2 = plt.subplot(222)
ax2.boxplot(data_clean['serum_cholesterol'])
ax2.set_xticklabels(['serum_cholesterol'])

ax3 = plt.subplot(223)
ax3.boxplot(data_clean['resting_blood_pressure'])
ax3.set_xticklabels(['resting_blood_pressure'])

ax4 = plt.subplot(224)
ax4.boxplot(data_clean['max_heart_rate'])
ax4.set_xticklabels(['max_heart_rate'])

plt.show()
from scipy.stats import chi2_contingency

cat_col = ['age_bin', 'sex', 'chest_pain_type', 'st_slope', 'thalassemia']
chi2_check = []
chi_score = []

for i in cat_col:
    chi_test = chi2_contingency(pd.crosstab(data['target'], data[i]))[1]
    chi_score.append(chi_test)
    if chi_test < 0.05:
        chi2_check.append('Reject Null Hypothesis')
    else:
        chi2_check.append('Fail to Reject Null Hypothesis')
res = pd.DataFrame(data = [cat_col, chi2_check, chi_score] 
             ).T 
res.columns = ['Column', 'Hypothesis', 'Chi Score']
res
# data without outlier removal
# convert numbered label to string first
data_enc = data
data_enc.chest_pain_type = data_enc.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})

data_enc.st_slope = data_enc.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})

data_enc.thalassemia = data_enc.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})

data_enc.age_bin = data_enc.age_bin.map({1:'young', 2:'middle', 3:'old', 0:'absent'})
data_enc.head()
X = data_enc.iloc[:, 1:-1]
y = data_enc.iloc[:, -1]
X.head(1)
# Categorical columns
cat_cols = ['age_bin', 'chest_pain_type', 'st_slope', 'thalassemia']

for column in cat_cols:
    dummies = pd.get_dummies(X[column], drop_first=True)
    X[dummies.columns] = dummies
    X.drop(column, axis=1, inplace=True)
    
X.head()
from sklearn.model_selection import train_test_split, GridSearchCV

# Splitting the data into test and train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X-Train:",X_train.shape)
print("X-Test:",X_test.shape)
print("Y-Train:",y_train.shape)
print("Y-Test:",y_test.shape)
from sklearn.preprocessing import StandardScaler

num_cols = ['st_depression', 'serum_cholesterol', 'resting_blood_pressure', 'max_heart_rate']
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train.head()
# data without outlier removal
# convert numbered label to string first
data_enc_clean = data_clean
data_enc_clean.chest_pain_type = data_enc_clean.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})

data_enc_clean.st_slope = data_enc_clean.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})

data_enc_clean.thalassemia = data_enc_clean.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})

data_enc_clean.age_bin = data_enc_clean.age_bin.map({1:'young', 2:'middle', 3:'old', 0:'absent'})
data_enc_clean.head()
X_clean = data_enc_clean.iloc[:, 1:-1]
y_clean = data_enc_clean.iloc[:, -1]
X_clean.head(1)
# Categorical columns
cat_cols = ['age_bin', 'chest_pain_type', 'st_slope', 'thalassemia']

for column in cat_cols:
    dummies = pd.get_dummies(X_clean[column], drop_first=True)
    X_clean[dummies.columns] = dummies
    X_clean.drop(column, axis=1, inplace=True)
    
X_clean.head()
from sklearn.model_selection import train_test_split, GridSearchCV

# Splitting the data into test and train 
X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

print("X-Train:",X_clean_train.shape)
print("X-Test:",X_clean_test.shape)
print("Y-Train:",y_clean_train.shape)
print("Y-Test:",y_clean_test.shape)
from sklearn.preprocessing import StandardScaler

num_cols = ['st_depression', 'serum_cholesterol', 'resting_blood_pressure', 'max_heart_rate']
scaler = StandardScaler()
scaler.fit(X_clean_train[num_cols])

X_clean_train[num_cols] = scaler.transform(X_clean_train[num_cols])
X_clean_test[num_cols] = scaler.transform(X_clean_test[num_cols])

X_clean_train.head()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

accuracy_svm = accuracy_score(y_pred, y_test)
print(f"The accuracy on test set using SVM is: {np.round(accuracy_svm, 3)*100.0}%")
print(classification_report(y_test, y_pred))
cv_score = cross_val_score(svm, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)
print(cv_score, end='\n\n')
print('Mean accuracy cross validation %f ' % np.mean(cv_score))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
accuracy_nb = accuracy_score(y_pred, y_test)
print(f"The accuracy on test set using Naive Bayes is: {np.round(accuracy_nb, 3)*100.0}%")
print(classification_report(y_test, y_pred))
cv_score = cross_val_score(gnb, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)
print(cv_score, end='\n\n')
print('Mean accuracy cross validation %f ' % np.mean(cv_score))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()

# creating a list of K's for performing KNN
my_list = list(range(0,30))

# filtering out only the odd K values
neighbors = list(filter(lambda x: x % 2 != 0, my_list))

# list to hold the cv scores
cv_scores = []

# perform 10-fold cross validation with default weights
for k in neighbors:
  Knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
  scores = cross_val_score(Knn, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)
  cv_scores.append(scores.mean())

# finding the optimal k
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print("The optimal K value is with default weight parameter: ", optimal_k)
# plotting accuracy vs K
plt.plot(neighbors, cv_scores)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K Plot for normal ")
plt.grid()
plt.show()

print("Accuracy scores for each K value is : ", np.round(cv_scores, 3))
# Finding the accuracy of KNN with optimal K

from sklearn.metrics import accuracy_score

# create instance of classifier
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_k, algorithm = 'kd_tree', 
                                   n_jobs = -1)

# fit the model
knn_optimal.fit(X_train, y_train)

# predict on test vector
y_pred = knn_optimal.predict(X_test)

# evaluate accuracy score
accuracy_knn = accuracy_score(y_test, y_pred)
print(f"The accuracy on test set using KNN for optimal K = {optimal_k} is {np.round(accuracy_knn, 3)*100}%")
print(classification_report(y_test, y_pred))
cv_scores = cross_val_score(Knn, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)
print(cv_score, end='\n\n')
print('Mean accuracy cross validation %f ' % np.mean(cv_score))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
svm = SVC()
svm.fit(X_clean_train, y_clean_train)
y_clean_pred = svm.predict(X_clean_test)

accuracy_svm_clean = accuracy_score(y_clean_pred, y_clean_test)
print(f"The accuracy on test set using SVM is: {np.round(accuracy_svm_clean, 3)*100.0}%")
print(classification_report(y_clean_test, y_clean_pred))
cv_score = cross_val_score(svm, X_clean_train, y_clean_train, cv=10, scoring='accuracy', n_jobs = -1)
print(cv_score, end='\n\n')
print('Mean accuracy cross validation %f ' % np.mean(cv_score))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_clean_test, y_clean_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_clean_pred = gnb.fit(X_clean_train, y_clean_train).predict(X_clean_test)
accuracy_nb_clean = accuracy_score(y_clean_pred, y_clean_test)
print(f"The accuracy on test set using Naive Bayes is: {np.round(accuracy_nb_clean, 3)*100.0}%")
print(classification_report(y_clean_test, y_clean_pred))
cv_score = cross_val_score(gnb, X_clean_train, y_clean_train, cv=10, scoring='accuracy', n_jobs = -1)
print(cv_score, end='\n\n')
print('Mean accuracy cross validation %f ' % np.mean(cv_score))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_clean_test, y_clean_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()

# creating a list of K's for performing KNN
my_list = list(range(0,30))

# filtering out only the odd K values
neighbors = list(filter(lambda x: x % 2 != 0, my_list))

# list to hold the cv scores
cv_scores = []

# perform 10-fold cross validation with default weights
for k in neighbors:
  Knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
  scores = cross_val_score(Knn, X_clean_train, y_clean_train, cv=10, scoring='accuracy', n_jobs = -1)
  cv_scores.append(scores.mean())

# finding the optimal k
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print("The optimal K value is with default weight parameter: ", optimal_k)
# plotting accuracy vs K
plt.plot(neighbors, cv_scores)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K Plot for normal ")
plt.grid()
plt.show()

print("Accuracy scores for each K value is : ", np.round(cv_scores, 3))
# Finding the accuracy of KNN with optimal K

from sklearn.metrics import accuracy_score

# create instance of classifier
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_k, algorithm = 'kd_tree', 
                                   n_jobs = -1)

# fit the model
knn_optimal.fit(X_clean_train, y_clean_train)

# predict on test vector
y_clean_pred = knn_optimal.predict(X_clean_test)

# evaluate accuracy score
accuracy_knn_clean = accuracy_score(y_clean_test, y_clean_pred)
print(f"The accuracy on test set using KNN for optimal K = {optimal_k} is {np.round(accuracy_knn_clean, 3)*100}%")
print(classification_report(y_clean_test, y_clean_pred))
cv_scores = cross_val_score(Knn, X_clean_train, y_clean_train, cv=10, scoring='accuracy', n_jobs = -1)
print(cv_score, end='\n\n')
print('Mean accuracy cross validation %f ' % np.mean(cv_score))
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_clean_test, y_clean_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()
model_list = ['SVM (outlier)', 'Naive Bayes (outlier)', 'KNN (outlier)', 'SVM ', 'Naive Bayes', 'KNN']
acc_list = [accuracy_svm, accuracy_nb, accuracy_knn, accuracy_svm_clean, accuracy_nb_clean, accuracy_knn_clean]

df_comp = pd.DataFrame({ 'model' : model_list, 'accuracy' : acc_list })
df_comp_sort = df_comp.sort_values('accuracy', ascending=False)
df_comp_sort
plt.figure(figsize=(8,5))
plt.title('Accuracy comparison between models.')
sns.barplot(df_comp_sort['model'], df_comp_sort['accuracy'])
plt.xticks(rotation=45)
plt.show()