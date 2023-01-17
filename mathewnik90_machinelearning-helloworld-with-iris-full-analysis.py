import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visualization library

import matplotlib.pyplot as plt #data visualization library

from sklearn.datasets import load_iris #sklearn library with datasets



irisRaw = load_iris() #importing iris dataset
iris = pd.DataFrame(data= np.c_[irisRaw['data'], irisRaw['target']],

                     columns= irisRaw['feature_names'] + ['target'])



# In case you wish to use the data set provided by Kaggle directly, you just need to uncomment the below code



#iris = pd.read_csv("../input/Iris.csv") # load the dataset

#iris.drop('Id',axis=1,inplace=True) # Drop id column

#iris['Species'] = iris['Species'].map({'Iris-setosa':0.0

#                                       ,'Iris-versicolor':1.0,'Iris-virginica':2.0}) # Replace values in species column

#iris.rename(columns = {'Species':'target'}, inplace = True) # Rename column as target
print("Top 5 rows\n")

print(iris.head(n=5))

print("\nStatistics for the dataset\n")

print(iris.describe())

print("\nStructure of data (rows,columns)\n")

print(iris.shape)
iris['target'].unique()
iris.columns = ['SPL','SPW','PTL','PTW','target']

iris.head(1)
print("Do null value exist")

print(pd.isnull(iris).any())

print("\n\n Count of null values")

print(pd.isnull(iris).sum())
labelGroups = iris.groupby('target')



for name, group in labelGroups:

    # print the name of the group

    print("\n\n",name)

    # print data for that group

    print(group.describe())
iris.boxplot(return_type='axes')
labelGroups['SPL'].hist(alpha=0.4)
sns.pairplot(iris, hue="target")
from pandas.plotting import parallel_coordinates

parallel_coordinates(iris, "target")

plt.show()
irisUnPvt = pd.melt(iris, "target", var_name="measurement")

sns.swarmplot(x="measurement", y="value", hue="target", data=irisUnPvt)

plt.show()
print(irisUnPvt.head(10),'\n\n')

print(irisUnPvt['measurement'].unique())
pearson = iris.corr(method='pearson')

print(pearson,"\n\n")

# assume target attr is the last, then remove corr with itself

corr_without_target = pearson.iloc[-1][:-1]

# attributes sorted from the most predictive

corr_without_target.sort_values(inplace=True)

print("Correlation of feature with the target")

print(corr_without_target)
corr_without_target[abs(corr_without_target).argsort()[::-1]]
# Set up the matplotlib figure

f, ax = plt.subplots(1,2,sharey=True,sharex=True)





# Generate a custom diverging colormap

cmap = sns.diverging_palette(10,225, as_cmap=True)



# clear the upper half of the matrix

mask = np.zeros_like(pearson, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set title for the left section

ax.flat[0].set_title("Pearson")

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(pearson, mask=mask, cmap=cmap, vmax=1,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar= False, ax=ax.flat[0],annot=True)



# Set title for the right section

ax.flat[1].set_title("Spearman")

scorrelation = iris.corr(method='spearman')

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(scorrelation, mask=mask, cmap=cmap, vmax=1,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar= False, ax=ax.flat[1],annot=True)



f.tight_layout()

plt.show()
from sklearn.decomposition import PCA # Import PCA from the library

pca = PCA(n_components=3) # Reduce the number of components to three.

pca.fit(iris[['SPL','SPW','PTL','PTW']])



print("Amount of variance explained by each component")

print(pca.explained_variance_ratio_)

print("\n\n","Amount of variance considered from each components")

print(pca.components_)



plt.figure(1, figsize=(4, 3))

plt.clf()

plt.axes([.2, .2, .7, .7])

plt.plot(pca.explained_variance_ratio_, linewidth=2)

plt.axis('tight')

plt.xlabel('n_components')

plt.ylabel('explained_variance_')

plt.show()
from sklearn.model_selection import train_test_split # import package to splid the data



X_train, X_test, y_train, y_test = train_test_split(irisRaw.data, irisRaw.target

                                                    , test_size=0.2, random_state=4) 

# We'll keep aside 20% for testing. Random state ensures repeatability.

# I have used the raw data for simplicity over here. 

# Alternatively, below approach can be used to get data from pandas data frame



#X_train, X_test, y_train, y_test = train_test_split(

#    iris[['SPL','SPW','PTL','PTW']], iris.iloc[:,4]

#    , test_size=0.6, random_state=4



print("Shape of training data \tInput:",X_train.shape,"\tExpected output:", y_train.shape)

print("Shape of test data   \tInput:",X_test.shape," \tExpected output:", y_test.shape)
from sklearn.neighbors import KNeighborsClassifier # Import KNeighbour classifier

import sklearn.metrics as mtr # Package to derive and report metrics 



knn = KNeighborsClassifier(n_neighbors=20) # Instantiate the model using the value of n as 13

knn.fit(X_train,y_train) # Train the model using the training data set



y_true, y_pred = y_test, knn.predict(X_test) # Predict values for the test data set

print(mtr.classification_report(y_true, y_pred)) # Print the classification report
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)

    

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i,format(cm[i, j],'.2f'),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Get the classesto plot the values

class_names = iris['target'].unique()



# Compute confusion matrix

cnf_matrix = mtr.confusion_matrix(y_test, y_pred)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
from sklearn.model_selection import GridSearchCV



# define the parameter values that should be searched

k_range = list(range(1, 25))



knn = KNeighborsClassifier()



# create a parameter grid: map the parameter names to the values that should be searched

param_grid = dict(n_neighbors=k_range)



# instantiate the grid

# Optimize for the best value of recall score

grid = GridSearchCV(knn, param_grid, cv=10, scoring=mtr.make_scorer(mtr.recall_score,average='micro'))



# fit the grid with data

grid.fit(X_train, y_train)



# examine the best model

print(grid.best_score_)

print(grid.best_params_)

print(grid.best_estimator_)
# create a list of the mean scores only

grid_mean_scores = grid.cv_results_['mean_test_score']



# plot the results

plt.plot(k_range, grid_mean_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated recall')

plt.show()
knn = KNeighborsClassifier(n_neighbors=10) # Create model with number of neighbours as 10

knn.fit(X_train,y_train) # Train the model

y_true, y_pred = y_test, knn.predict(X_test) # Predict the test data



# Evaluate results

print(mtr.classification_report(y_true, y_pred)) 





# Compute confusion matrix

cnf_matrix = mtr.confusion_matrix(y_test, y_pred)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.regularizers import l2

from keras.utils import np_utils



# Make test and train set

X_train, X_test, y_train, y_test = train_test_split(irisRaw.data, irisRaw.target, 

                                                    test_size=0.5,

                                                    random_state=4)



################################

# Evaluate Logistic Regression

################################

lr = LogisticRegression(C=1e4)

lr.fit(X_train, y_train)

pred_y = lr.predict(X_test)







################################

# Evaluate Keras Neural Network

################################



# Make ONE-HOT

def one_hot_encode_object_array(arr):

    '''One hot encode a numpy array of objects (e.g. strings)'''

    uniques, ids = np.unique(arr, return_inverse=True)

    return np_utils.to_categorical(ids, len(uniques))





train_y_ohe = one_hot_encode_object_array(y_train)

test_y_ohe = one_hot_encode_object_array(y_test)



# Initialize sequential model

model = Sequential()



# Stack layers

model.add(Dense(16,input_shape=(4,), 

                activation="tanh",

                kernel_regularizer=l2(0.001)))

model.add(Dropout(0.5))

model.add(Dense(3, activation="softmax"))



# Create the model

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')



# Actual modelling

model.fit(X_train, train_y_ohe,batch_size=1, epochs=100,verbose=0)



# Evaluate the models

score, accuracy = model.evaluate(X_test, test_y_ohe, batch_size=16,verbose=0)

print("Linear regression Accuracy = {:.2f}".format(lr.score(X_test, y_test)))

print("Neural Network Score = {:.2f}".format(score))

print("Neural Network Accuracy = {:.2f}".format(accuracy))