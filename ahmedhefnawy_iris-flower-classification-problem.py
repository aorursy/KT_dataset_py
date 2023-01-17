# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
# Load dataset
DataSet = pd.read_csv("../input/iris/Iris.csv")
DataSet.head(10)
DataSet.shape
DataSet.head(10)
DataSet.describe()
print(DataSet.groupby('Species').size())
# box and whisker plots
DataSet.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False , figsize=(16,16) , color='g')
plt.show()
DataSet['PetalWidthCm'].hist(bins=100 , figsize=(7,7) )
plt.show()
DataSet.hist(bins=25 ,figsize=(18,13))
plt.show()
# scatter plot matrix
DataSet.plot(kind='scatter', x = 'PetalLengthCm' , y= 'PetalWidthCm')
plt.show()
# scatter plot matrix
scatter_matrix(DataSet , figsize=(16,16))
plt.show()
# Split-out validation dataset
array = DataSet.values
X = array[:,0:4]  # All features
Y = array[:,4].astype('int') # The Output we need to predict
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))# 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("Algorithm Name : {} : reslt_Mean = {} % |---| result_STD = {} \n-------------------------------\n".format( name , cv_results.mean(), cv_results.std() ) )
    
# Compare Algorithms
fig = plt.figure(figsize=(10,7))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# >>>>>>> --- Make predictions on validation dataset ------ <<<<<<< #
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('The accuracy Score: ',accuracy_score(Y_validation, predictions)*100,'%', '\n')
print('The confusion matrix: ',confusion_matrix(Y_validation, predictions) , '\n')
print('_____________________\nClassification report: \n',classification_report(Y_validation, predictions))
