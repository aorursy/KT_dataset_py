import pandas as pd    # a wonderful dataframe to work with
import numpy as np     # adding a number of mathematical and science functions
import seaborn as sns  # a very easy to use statistical data visualization package
import matplotlib.pyplot as plt # a required plotting tool
import warnings
# sklearn is a big source of pre-written and mostly optimized ML algorithms.
# Here we use their Decision trees, Support Vector Machines, and the classic Perceptron. 
from sklearn import preprocessing, svm   
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
#ignore warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('../input/xAPI-Edu-Data.csv')
data.head()
data.tail()
ax = sns.countplot(x='Class', data=data, order=['L', 'M', 'H'])
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(data)), (p.get_x() + 0.24, p.get_height() + 2))
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='gender', data=data, order=['M','F'], ax=axarr[0])
sns.countplot(x='gender', hue='Class', data=data, order=['M', 'F'],hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='NationalITy', data=data, ax=axarr[0])
sns.countplot(x='NationalITy', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='PlaceofBirth', data=data, ax=axarr[0])
sns.countplot(x='PlaceofBirth', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='StageID', data=data, ax=axarr[0])
sns.countplot(x='StageID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='GradeID', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], ax=axarr[0])
sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.show()
#Students in Grade 5
data.loc[data['GradeID'] == 'G-05']
#Students in Grade 9
data.loc[data['GradeID'] == 'G-09']
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='SectionID', data=data, order=['A', 'B', 'C'], ax = axarr[0])
sns.countplot(x='SectionID', hue='Class', data=data, order=['A', 'B', 'C'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Topic', data=data, ax = axarr[0])
sns.countplot(x='Topic', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Semester', data=data, ax = axarr[0])
sns.countplot(x='Semester', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='Relation', data=data, ax = axarr[0])
sns.countplot(x='Relation', hue='Class', data=data,hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
sns.pairplot(data, hue="Class", diag_kind="kde", hue_order = ['L', 'M', 'H'], markers=["o", "s", "D"])
plt.show()
data.groupby('Topic').median()
data.groupby('GradeID').median()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='ParentAnsweringSurvey', data=data, order=['Yes', 'No'], ax = axarr[0])
sns.countplot(x='ParentAnsweringSurvey', hue='Class', data=data, order=['Yes', 'No'], hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='ParentschoolSatisfaction', data=data, order=['Good', 'Bad'], ax = axarr[0])
sns.countplot(x='ParentschoolSatisfaction', hue='Class', data=data, order=['Good', 'Bad'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
fig, axarr  = plt.subplots(2,figsize=(10,10))
sns.countplot(x='StudentAbsenceDays', data=data, order=['Under-7', 'Above-7'], ax = axarr[0])
sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, order=['Under-7', 'Above-7'],hue_order = ['L', 'M', 'H'], ax = axarr[1])
plt.show()
# Translate GradeID from categorical to numerical
gradeID_dict = {"G-01" : 1,
                "G-02" : 2,
                "G-03" : 3,
                "G-04" : 4,
                "G-05" : 5,
                "G-06" : 6,
                "G-07" : 7,
                "G-08" : 8,
                "G-09" : 9,
                "G-10" : 10,
                "G-11" : 11,
                "G-12" : 12}

data = data.replace({"GradeID" : gradeID_dict})

class_dict = {"L" : -1,
                "M" : 0,
                "H" : 1}
data = data.replace({"Class" : class_dict})

# Scale numerical fields
data["GradeID"] = preprocessing.scale(data["GradeID"])
data["raisedhands"] = preprocessing.scale(data["raisedhands"])
data["VisITedResources"] = preprocessing.scale(data["VisITedResources"])
data["AnnouncementsView"] = preprocessing.scale(data["AnnouncementsView"])
data["Discussion"] = preprocessing.scale(data["Discussion"])

# Use dummy variables for categorical fields
data = pd.get_dummies(data, columns=["gender",
                                     "NationalITy",
                                     "PlaceofBirth",
                                     "SectionID",
                                     "StageID",
                                     "Topic",
                                     "Semester",
                                     "Relation",
                                     "ParentAnsweringSurvey",
                                     "ParentschoolSatisfaction",
                                     "StudentAbsenceDays"])

# Show preprocessed data
data.head()
corr = data.corr()
corr.iloc[[5]]
perc = Perceptron(n_iter=100, eta0=0.1, random_state=15)
results = []
predMiss = []

for _ in range(1000):
    # Randomly sample our training data
    data_train = data.sample(frac=0.7)
    # train data without label
    data_train_X = data_train.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of train data 
    data_train_Y = data_train.loc[:, lambda x: "Class"]

    # The rest is test data
    data_test = data.loc[~data.index.isin(data_train.index)]
    # Test data without label
    data_test_X = data_test.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of test data 
    data_test_Y = data_test.loc[:, lambda x: "Class"]

    # Train svm
    perc.fit(data_train_X, data_train_Y)
    predMiss.append((data_test_Y != perc.predict(data_test_X)).sum())
    # Score the mean accuracy on the test data and append results in a list
    results.append(perc.score(data_test_X, data_test_Y))

# Convert results to an array and look at the minimum and the average
predErr = np.hstack(predMiss)
Final = np.hstack(results)
print('Minimum Accuracy Score:   %.8f' % Final[Final.argmin()])
print('Maximum Accuracy Score:   %.8f' % Final[Final.argmax()])
print('Average Accuracy Score:   %.8f' % np.average(Final))
print('Minimum Prediction Misses:   %d' % predErr[predErr.argmin()])
print('Maximum Prediction Misses:   %d' % predErr[predErr.argmax()])
print('Average Prediction Misses:   %.2f' % np.average(predErr))
# Create the radial basis function kernel version of a Support Vector Machine classifier
rbf_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
# Create the linear kernel version of a Support Vector Machine classifier
lin_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
# Create the polynomial kernel version of a Support Vector Machine classifier
poly_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
# Create the sigmoid kernel version of a Support Vector Machine classifier
sig_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='sigmoid',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
res_rbf = []
predMiss_rbf = []
res_lin = []
predMiss_lin = []
res_poly = []
predMiss_poly = []
res_sig = []
predMiss_sig = []

for _ in range(1000):
    # Randomly sample our training data
    data_train = data.sample(frac=0.7)
    # train data without label
    data_train_X = data_train.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of train data 
    data_train_Y = data_train.loc[:, lambda x: "Class"]

    # The rest is test data
    data_test = data.loc[~data.index.isin(data_train.index)]
    # Test data without label
    data_test_X = data_test.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of test data 
    data_test_Y = data_test.loc[:, lambda x: "Class"]

    # Train svms
    rbf_clf.fit(data_train_X, data_train_Y)
    lin_clf.fit(data_train_X, data_train_Y)
    poly_clf.fit(data_train_X, data_train_Y)
    sig_clf.fit(data_train_X, data_train_Y)
    
    #Sum the prediction misses. Since this is a smaller dataset, 
    predMiss_rbf.append((data_test_Y != rbf_clf.predict(data_test_X)).sum())
    predMiss_lin.append((data_test_Y != lin_clf.predict(data_test_X)).sum())
    predMiss_poly.append((data_test_Y != poly_clf.predict(data_test_X)).sum())
    predMiss_sig.append((data_test_Y != sig_clf.predict(data_test_X)).sum())
    # Score the mean accuracy on the test data and append results in a list
    res_rbf.append(rbf_clf.score(data_test_X, data_test_Y))
    res_lin.append(lin_clf.score(data_test_X, data_test_Y))
    res_poly.append(poly_clf.score(data_test_X, data_test_Y))
    res_sig.append(sig_clf.score(data_test_X, data_test_Y))

# Convert results and prediction lists to an array and look at the minimums and the averages
predErr_rbf = np.hstack(predMiss_rbf)
Final_rbf = np.hstack(res_rbf)
predErr_lin = np.hstack(predMiss_lin)
Final_lin = np.hstack(res_lin)
predErr_poly = np.hstack(predMiss_poly)
Final_poly = np.hstack(res_poly)
predErr_sig = np.hstack(predMiss_sig)
Final_sig = np.hstack(res_sig)


print('RBF Minimum Accuracy Score:   %.8f' % Final_rbf[Final_rbf.argmin()])
print('RBF Maximum Accuracy Score:   %.8f' % Final_rbf[Final_rbf.argmax()])
print('RBF Average Accuracy Score:   %.8f' % np.average(Final_rbf))
print('------------------------------------------------')
print('Linear Minimum Accuracy Score:   %.8f' % Final_lin[Final_lin.argmin()])
print('Linear Maximum Accuracy Score:   %.8f' % Final_lin[Final_lin.argmax()])
print('Linear Average Accuracy Score:   %.8f' % np.average(Final_lin))
print('------------------------------------------------')
print('Polynomial Minimum Accuracy Score:   %.8f' % Final_poly[Final_poly.argmin()])
print('Polynomial Maximum Accuracy Score:   %.8f' % Final_poly[Final_poly.argmax()])
print('Polynomial Average Accuracy Score:   %.8f' % np.average(Final_poly))
print('------------------------------------------------')
print('Sigmoid Minimum Accuracy Score:   %.8f' % Final_sig[Final_sig.argmin()])
print('Sigmoid Maximum Accuracy Score:   %.8f' % Final_sig[Final_sig.argmax()])
print('Sigmoid Average Accuracy Score:   %.8f' % np.average(Final_sig))
print('================================================')
#print('Minimum Prediction Misses:   %d' % predErr[predErr.argmin()])
#print('Maximum Prediction Misses:   %d' % predErr[predErr.argmax()])
print('RBF Average Prediction Misses:   %.2f' % np.average(predErr_rbf))
print('Linear Average Prediction Misses:   %.2f' % np.average(predErr_lin))
print('Polynomial Average Prediction Misses:   %.2f' % np.average(predErr_poly))
print('Sigmoid Average Prediction Misses:   %.2f' % np.average(predErr_sig))
tree3 = DecisionTreeClassifier(random_state=56, criterion='gini', max_depth=3)
tree5 = DecisionTreeClassifier(random_state=56, criterion='gini', max_depth=5)
results_3 = []
results_5 = []
predMiss_3 = []
predMiss_5 = []


for _ in range(1000):
    # Randomly sample our training data
    data_train = data.sample(frac=0.7)
    # train data without label
    data_train_X = data_train.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of train data 
    data_train_Y = data_train.loc[:, lambda x: "Class"]

    # The rest is test data
    data_test = data.loc[~data.index.isin(data_train.index)]
    # Test data without label
    data_test_X = data_test.loc[:, lambda x: [l for l in data if l != "Class"]]
    # labels of test data 
    data_test_Y = data_test.loc[:, lambda x: "Class"]

    # Train svm
    tree3.fit(data_train_X, data_train_Y)
    tree5.fit(data_train_X, data_train_Y)
    #Sum the prediction misses. Since this is a smaller dataset,
    predMiss_3.append((data_test_Y != tree3.predict(data_test_X)).sum())
    predMiss_5.append((data_test_Y != tree5.predict(data_test_X)).sum())
    # Score the mean accuracy on the test data and append results in a list
    results_3.append(tree3.score(data_test_X, data_test_Y))
    results_5.append(tree5.score(data_test_X, data_test_Y))

# Convert results to an array and look at the minimum and the average
predErr_3 = np.hstack(predMiss_3)
predErr_5 = np.hstack(predMiss_5)
Final_3 = np.hstack(results_3)
Final_5 = np.hstack(results_5)
print('3-depth Tree Minimum Accuracy Score:   %.8f' % Final_3[Final_3.argmin()])
print('3-depth Tree Maximum Accuracy Score:   %.8f' % Final_3[Final_3.argmax()])
print('3-depth Tree Average Accuracy Score:   %.8f' % np.average(Final_3))
print('------------------------------------------------')
print('5-depth Tree Minimum Accuracy Score:   %.8f' % Final_5[Final_5.argmin()])
print('5-depth Tree Maximum Accuracy Score:   %.8f' % Final_5[Final_5.argmax()])
print('5-depth Tree Average Accuracy Score:   %.8f' % np.average(Final_5))
#print('Minimum Prediction Misses:   %d' % predErr[predErr.argmin()])
#print('Maximum Prediction Misses:   %d' % predErr[predErr.argmax()])
#print('Average Prediction Misses:   %.2f' % np.average(predErr))
