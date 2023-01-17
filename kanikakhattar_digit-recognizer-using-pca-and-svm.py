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
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
numbers = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
numbers.head()
numbers.shape
numbers.info()
numbers.describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.99])
#missing value check
sum(numbers.isnull().sum(axis=0))
numbers['label'].value_counts()
np.unique(numbers['label'])
sns.countplot(numbers['label'],palette = 'icefire')
#Checking average value of all pixels
#round(numbers.drop('label', axis=1).mean(), 2).sort_values(ascending = False)
y = pd.value_counts(numbers.values.ravel()).sort_index()
width = 0.9
plt.figure(figsize=[8,8])
plt.bar(range(len(y)),y,width,color="blue")
plt.title('Pixel Value Frequency (Log Scale)')
plt.yscale('log')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')
plt.figure(figsize=[15,15])
plt.subplot(2,3,1)
sns.distplot(numbers['pixel575'],kde=False)
plt.subplot(2,3,2)
sns.distplot(numbers['pixel624'],kde=False)
plt.subplot(2,3,3)
sns.distplot(numbers['pixel572'],kde=False)
plt.subplot(2,3,4)
sns.distplot(numbers['pixel407'],kde=False)
plt.subplot(2,3,5)
sns.distplot(numbers['pixel576'],kde=False)
plt.subplot(2,3,6)
sns.distplot(numbers['pixel580'],kde=False)
plt.show()
plt.figure(figsize=[15,12])
plt.subplot(2,3,1)
sns.barplot(x='label', y='pixel575', data=numbers)
plt.subplot(2,3,2)
sns.barplot(x='label', y='pixel624', data=numbers)
plt.subplot(2,3,3)
sns.barplot(x='label', y='pixel572', data=numbers)
plt.subplot(2,3,4)
sns.barplot(x='label', y='pixel683', data=numbers)
plt.subplot(2,3,5)
sns.barplot(x='label', y='pixel576', data=numbers)
plt.subplot(2,3,6)
sns.barplot(x='label', y='pixel580', data=numbers)
plt.show()
numbers.loc[numbers['label']==1].head(10).index.values
plt.figure(figsize=[10,10])
ones_index = numbers.loc[numbers['label']==1].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[ones_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 1")
plt.figure(figsize=[10,10])
threes_index = numbers.loc[numbers['label']==3].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[threes_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 3")
plt.figure(figsize=[10,10])
fives_index = numbers.loc[numbers['label']==5].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[fives_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 5")
plt.figure(figsize=[10,10])
fours_index = numbers.loc[numbers['label']==4].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[fours_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 4")
### Heatmap
y = numbers['label']
X = numbers.drop('label',axis=1)
X.head()
y[:5]
#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2 ,test_size = 0.8, random_state=100)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#X_train_scaled = pd.DataFrame(X_train_scaled)
#round(X_train_scaled.describe(),2)
print('X_train shape:',X_train_scaled.shape)
print('y_train shape:',y_train.shape)
print('X_test shape:',X_test_scaled.shape)
print('y_test shape:',y_test.shape)
pca = PCA(random_state=42)
pca.fit(X_train_scaled)
pca.components_.shape
#pca.explained_variance_ratio_
var_cummu = np.cumsum(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=[12,8])
plt.vlines(x=200, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.91, xmax=800, xmin=0, colors="g", linestyles="--")
plt.plot(var_cummu)
plt.ylabel("Cumulative variance explained")
plt.show()
pca_final = IncrementalPCA(n_components = 200)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_train_pca.shape
pca_final.components_.shape
df_train_pca = pd.DataFrame(X_train_pca)
df_train_pca.head()
y_train_df = pd.DataFrame(y_train)
y_train_df['label'].value_counts()
new_df = pd.concat([df_train_pca,y_train_df],axis=1)
new_df['label'].value_counts().sort_index()
plt.figure(figsize=(10,10))
sns.scatterplot(x=new_df[1],y=new_df[0],hue=new_df['label'],size=10,legend='full',palette='rainbow')
sns.pairplot(data=new_df, x_vars=[0,1,2], y_vars=[0,1,2], hue = "label", size=5)
pca_final.explained_variance_ratio_
X_test_pca = pca_final.transform(X_test_scaled)
X_test_pca.shape
# linear model
model_linear = SVC(kernel='linear')
model_linear.fit(X_train_pca, y_train)

# predict
y_train_pred = model_linear.predict(X_train_pca)
y_test_pred = model_linear.predict(X_test_pca)
train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))

print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
# non-linear model
# using poly kernel, C=1, default value of gamma

# model
non_linear_model_poly = SVC(kernel='poly')
non_linear_model_poly.fit(X_train_pca, y_train)

# predict
y_train_pred = non_linear_model_poly.predict(X_train_pca)
y_test_pred = non_linear_model_poly.predict(X_test_pca)
train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))
print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
# non-linear model
# using rbf kernel, C=1, default value of gamma

# model
non_linear_model_poly = SVC(kernel='rbf')
non_linear_model_poly.fit(X_train_pca, y_train)

# predict
y_train_pred = non_linear_model_poly.predict(X_train_pca)
y_test_pred = non_linear_model_poly.predict(X_test_pca)
train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))
print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
pipe_steps = [('scaler',StandardScaler()),('pca',PCA()),('SVM',SVC(kernel='poly'))]
check_params = {
    'pca__n_components' : [195,200],
    'SVM__C':[1,10],
    'SVM__gamma':[0.01,0.001]
}

pipeline = Pipeline(pipe_steps)
folds = KFold(n_splits=3,shuffle=True,random_state=101)


#setting up GridSearchCV()
model_cv = GridSearchCV(estimator = pipeline,
                       param_grid = check_params,
                       scoring = 'accuracy',
                       cv = folds,
                       verbose = 3,
                       return_train_score=True,
                       n_jobs=-1)

#fit the model
model_cv.fit(X_train,y_train) # Considering our initial data as scaling will be handled by the pipeline.
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
# converting C to numeric type for plotting on x-axis
cv_results['param_SVM__C'] = cv_results['param_SVM__C'].astype('int')

# # plotting
plt.figure(figsize=(20,7))

# subplot 1/3
plt.subplot(121)
gamma_01 = cv_results[(cv_results['param_SVM__gamma']==0.01) & (cv_results['param_pca__n_components'] == 195)]

plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# subplot 2/3
plt.subplot(122)
gamma_001 = cv_results[(cv_results['param_SVM__gamma']==0.001) & (cv_results['param_pca__n_components'] == 195)]

plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# # plotting
plt.figure(figsize=(20,7))

# subplot 1/3
plt.subplot(121)
gamma_01 = cv_results[(cv_results['param_SVM__gamma']==0.01) & (cv_results['param_pca__n_components'] == 200)]

plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# subplot 2/3
plt.subplot(122)
gamma_001 = cv_results[(cv_results['param_SVM__gamma']==0.001) & (cv_results['param_pca__n_components'] == 200)]

plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
pca_final = IncrementalPCA(n_components = 195)

X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)
print(X_test_pca.shape)
print(X_train_pca.shape)
# model with optimal hyperparameters

# model
final_model = SVC(C=1, gamma=0.01, kernel="poly")

final_model.fit(X_train_pca, y_train)
# predict
y_train_pred = final_model.predict(X_train_pca)
y_test_pred = final_model.predict(X_test_pca)
# metrics
train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))

print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))
#import file and reading few lines
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_df.head(10)
test_df.shape
test_scaled = scaler.transform(test_df)
final_test_pca = pca_final.transform(test_scaled)
final_test_pca.shape
#model.predict
test_predict = final_model.predict(final_test_pca)
# Plotting the distribution of prediction
a = {'ImageId': np.arange(1,test_predict.shape[0]+1), 'Label': test_predict}
data_to_export = pd.DataFrame(a)
sns.countplot(data_to_export['Label'], palette = 'icefire')
# Let us visualize few of predicted test numbers
df = np.random.randint(1,test_predict.shape[0]+1,5)

plt.figure(figsize=(16,4))
for i,j in enumerate(df):
    plt.subplot(150+i+1)
    d = test_scaled[j].reshape(28,28)
    plt.title(f'Predicted Label: {test_predict[j]}')
    plt.imshow(d)
plt.show()
# Exporting the predicted values 
data_to_export.to_csv(path_or_buf='submission.csv', index=False)
submitted = pd.read_csv('submission.csv')
submitted.head()
