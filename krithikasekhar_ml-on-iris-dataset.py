# Trying out different machine learning models on the Iris Dataset.
# Load all the required python modules initially
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
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
# supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Load the dataset
dataset = pandas.read_csv("../input/Iris.csv")
# summarize the dataset
# shape
print("Shape of the dataset:")
print(dataset.shape)
# head
print("\n First 20 entries of the dataset:")
print(dataset.head(20))
# summary statistics
print("\n")
print(dataset.describe())
# class distribution
print("\n")
print(dataset.groupby('Species').size())
# drop ID column as its just an identifier and not going to help with the data analysis
dataset = dataset.drop(['Id'], axis = 1)
# visualize the dataset

# we can try to understand the dimensions of each species through boxplots by looking at the average
# petal length, petal width, sepal width and sepal length for each species

# average Petal length for each Species
sns.boxplot(x='Species', y='PetalLengthCm', data=dataset)
plt.show()
# average Sepal length for every species
sns.boxplot(x='Species', y='SepalLengthCm', data=dataset)
plt.show()
# average petal width for each species
sns.boxplot(x='Species', y='PetalWidthCm', data=dataset)
plt.show()
# average sepal width for every species
sns.boxplot(x='Species', y='SepalWidthCm', data=dataset)
plt.show()
# From the plots, we can see that on an average, Iris virgnica has the highest petal length,
# sepal length, petal width hence any iris having larger dimensions is likely to be virginica

# A hexagonal binning plot can be used to find natural clusters located within the data thereby
# helping visualize point densities within the data
dataset.plot(kind='hexbin', x=1, y=2, gridsize=10)
plt.show()
# A scatter matrix can be used to understand the relationships between the attributes
# The diagonal grouping of elements suggests a high correlation 
# From scatter matrix we can infer how features are grouped and understand their separability

labels = dataset['Species']
colors_dict = {'Iris-setosa': "red", 'Iris-versicolor': "yellow", 'Iris-virginica': "purple"}
colors = [colors_dict[c] for c in labels] 
scatter_matrix(dataset, figsize=(15, 15), marker='o', diagonal='kde', s=60, alpha=0.8, color=colors)
plt.show()
# create a validation dataset by splitting the features(X) and labels(Y) into 70% training data and
# 30% test data
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size, random_state=seed
)

# seed and evaluation metric
# Let us perform 10 fold cross validation to ensure that all the data can be used as test data
# using accuracy score as the metric to evaluate the models
seed = 10
scoring = 'accuracy'
# build models
models = []
models.append(('Logistic Regression',LogisticRegression()))
models.append(('Linear Discriminant Analysis',LinearDiscriminantAnalysis()))
models.append(('K Neighbours Classifier',KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('Support Vector Machines', SVC()))
# evaluate each model
results = []
names = ['LR', 'LDA', 'KNN', 'DT', 'GNB', 'SVM']
row_list = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv= kfold, 
                                                 scoring = scoring)
    results.append(cv_results.mean())
    data_dict = {}
    data_dict['Classifier'] = name
    data_dict['Mean Accuracy Score'] = cv_results.mean()
    row_list.append(data_dict)
df = pandas.DataFrame(row_list)
print(df)
# From the above table, it is clear that SVM gives the best accuracy for the train data
# Compare Algorithms and plot the different accuracy scores
sns.swarmplot(x=names,y=results, size=10).set_title("Accuracy of different algorithms")
plt.show()
# fit the best model on training dataset
svc = SVC()
svc.fit(X_train, Y_train)
# make predictions on validation dataset
print("Accuracy on validation data:")
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print("\n Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
print("\n Classification Report:")
print(classification_report(Y_validation, predictions))
# We obtain a final accuracy of 95.5% on the test data by fitting the train data on a 
# Support Vector Machine Model