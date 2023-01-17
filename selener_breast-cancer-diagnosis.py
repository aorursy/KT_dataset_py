# here we will import the libraries used for machine learning

import sys 

import numpy as np # linear algebra

from scipy.stats import randint

import pandas as pd # data processing, CSV file I/O, data manipulation 

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. 



from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.model_selection import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.preprocessing import StandardScaler # for normalization

from sklearn.preprocessing import Imputer  # dealing with NaN

from sklearn.pipeline import Pipeline # pipeline making

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics # for the check the error and accuracy of the model

from sklearn import svm, datasets # for Support Vector Machine

from sklearn.svm import SVC





## for Deep-learing:

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import SGD 

from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
data0 = pd.read_csv("../input/data.csv")
data0.head()
data0.diagnosis.unique()
data0[data0 == '?'] = np.NaN

# Drop missing values and print shape of new DataFrame

data0.info()
# drop columns:  "Unnamed: 32" and "ID"

# To keep the same name of file, write: inplace=True

# Separating target from features (predictor variables)



y = data0.diagnosis     # target= M or B 



list = ['Unnamed: 32','id','diagnosis']

features = data0.drop(list,axis = 1,inplace = False)



list = ['Unnamed: 32','id']

data0.drop(list, axis = 1, inplace = True)
print(data0.isnull().sum())
# The frequency of cancer stages

B, M = data0['diagnosis'].value_counts()

print('Number of Malignant : ', M)

print('Number of Benign: ', B)



plt.figure(figsize=(10,6))

sns.set_context('notebook', font_scale=1.5)

sns.countplot('diagnosis',data=data0, palette="Set1")

plt.annotate('Malignant = 212', xy=(-0.2, 250), xytext=(-0.2, 250), size=18, color='red')

plt.annotate('Benign = 357', xy=(0.8, 250), xytext=(0.8, 250), size=18, color='w');
features.describe()
#data0.columns or

data0.keys()
# Standardization of features

stdX = (features - features.mean()) / (features.std())              

data_st = pd.concat([y,stdX.iloc[:,:]],axis=1)

data_st = pd.melt(data_st,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
plt.figure(figsize=(12,30))

sns.set_context('notebook', font_scale=1.5)

sns.boxplot(x="value", y="features", hue="diagnosis", data=data_st, palette='Set1')

plt.legend(loc='best');
plt.figure(figsize=(12,30))

sns.set_context('notebook', font_scale=1.5)

sns.violinplot(x="value", y="features", hue="diagnosis", data=data_st,split=True, 

               inner="quart", palette='Set1')

plt.legend(loc='best');
corr = data0.corr() # .corr is used to find corelation

f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corr, cbar = True,  square = True, annot = True, fmt= '.1f', 

            xticklabels= True, yticklabels= True

            ,cmap="coolwarm", linewidths=.5, ax=ax);
def pearson_r(x, y):

    # Compute correlation matrix: corr_mat

    corr_mat = np.corrcoef(x, y)



    # Return entry [0,1]

    return corr_mat[0,1]



# Compute Pearson correlation coefficient for 'radius_mean', 'symmetry_mean'

r1 = pearson_r(data0['radius_mean'], data0['perimeter_mean'])

r2= pearson_r(data0['radius_mean'], data0['symmetry_mean'])



name_c = []

for (i,j) in zip(range(1,31),range(1,31)):

        r = pearson_r(data0.iloc[:,1], data0.iloc[:,j])

        if abs(r) >= 0.80 and data0.columns[j]  not in name_c:

                    name_c.append(data0.columns[j]) 

print()

print('* Lenght of columns assuming r >=0.80:', len(name_c)) 

print('name_c =',name_c)
name_c = []

for (i,j) in zip(range(1,31),range(1,31)):

        r = pearson_r(data0.iloc[:,1], data0.iloc[:,j])

        if abs(r) <= 0.40 and data0.columns[j]  not in name_c:

                    name_c.append(data0.columns[j])

                            

print('* Lenght of columns assuming r <=0.40:', len(name_c)) 

print('name_c =',name_c) 
sns.lmplot(x='radius_mean', y= 'symmetry_mean', data = data0, hue ='diagnosis', 

           palette='Set1')

plt.title('Linear Regression: distinguishing between M and B', size=16)





sns.lmplot(x='radius_mean', y= 'perimeter_mean', data = data0, hue ='diagnosis', 

           palette='Set1')

plt.title('Linear Regression: Cannot distinguish between M and B', size=16);



print('Uncorrelated data are poentially more useful: discrimentory!')
plt.figure(figsize=(15,15))

sns.set_context('notebook', font_scale=1.5)

plt.subplot(2, 2, 1)

sns.boxplot(y="radius_mean", x="diagnosis", data=data0, palette="Set1") 

sns.swarmplot(x="diagnosis", y="radius_mean",data=data0, palette="Set3", dodge=True)

plt.subplot(2, 2, 2)  

sns.boxplot(y="fractal_dimension_mean", x="diagnosis", data=data0, palette="Set1")

sns.swarmplot(x="diagnosis", y="fractal_dimension_mean",data=data0, palette="Set3",

              dodge=True)

plt.subplots_adjust(wspace=0.4); 
# CDF function

def ecdf(data0):

    n=len(data0)

    x=np.sort(data0)

    y=np.arange(1, n+1)/n

    return x, y 



data2 = data0['radius_mean']

Malignant = data2[data0['diagnosis']=='M']

Benign = data2[data0['diagnosis']=='B']



x1, y1 = ecdf(Malignant)

x2, y2 = ecdf(Benign)



data3 = data0['fractal_dimension_mean']

Malignant_f = data3[data0['diagnosis']=='M']

Benign_f = data3[data0['diagnosis']=='B']



x3, y3 = ecdf(Malignant_f)

x4, y4 = ecdf(Benign_f)



plt.figure(figsize=(15,15))

#plt.close('all')

plt.subplot(2, 2,  1)

plt.subplots_adjust(wspace=0.4, hspace=2)

plt.plot(x1, y1, marker='.',linestyle='none', color='red', label='M')

plt.plot(x2, y2, marker='.',linestyle='none', color ='blue', label='B')

plt.margins(0.02)

plt.xlabel('radius_mean', size=20)

plt.ylabel('ECDF', size=20)

plt.title('Empirical Cumulative Distribution Function', size=20)

plt.legend(prop={'size':20})

#plt.show()

plt.subplot(2, 2,  2)

plt.subplots_adjust(wspace=0.4, hspace=2)

plt.plot(x3, y3, marker='.',linestyle='none', color='red', label='M')

plt.plot(x4, y4, marker='.',linestyle='none', color ='blue', label='B')

plt.margins(0.02)

plt.xlabel('fractal_dimension_mean', size=20)

plt.ylabel('ECDF', size=20)

plt.title('Empirical Cumulative Distribution Function', size=20)

plt.legend(prop={'size':20});
def permutation_sample(data1, data2):

    """Generate a permutation sample from two data sets."""



    # Concatenate the data sets: data

    data = np.concatenate((data1, data2))



    # Permute the concatenated array: permuted_data

    permuted_data = np.random.permutation(data)



    # Split the permuted array into two: perm_sample_1, perm_sample_2

    perm_sample_1 = permuted_data[:len(data1)]

    perm_sample_2 = permuted_data[len(data1):]



    return perm_sample_1, perm_sample_2







def draw_perm_reps(data_1, data_2, func, size=1):

    """Generate multiple permutation replicates."""



    # Initialize array of replicates: perm_replicates

    perm_replicates = np.empty(size)



    for i in range(size):

        # Generate permutation sample

        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)



        # Compute the test statistic

        perm_replicates[i] = func(perm_sample_1, perm_sample_2)



    return perm_replicates







def diff_of_means(data_1, data_2):

    """Difference in means of two arrays."""



    # The difference of means of data_1, data_2: diff

    diff = np.mean(data_1)-np.mean(data_2)



    return diff
diff_of_means(Malignant, Benign)
# Computing difference of mean overall acore

empirical_diff_means = diff_of_means(Malignant, Benign)



# Drawing 10,000 permutation replicates: perm_replicates

perm_replicates = draw_perm_reps(Malignant, Benign,diff_of_means, size=10000)



# Computing p-value: p

p = np.sum(perm_replicates >= empirical_diff_means)/ len(perm_replicates) 



print('p-value =', p)
# Let's map diagnosis column[object] to integer value:0, 1

# later on below I show how to use LabelEncoder(): it is better way to categorize

data=data0.copy()

data['diagnosis']=data0['diagnosis'].map({'M':1,'B':0})
# Split the data into train (0.7) and test (0.3)



## all data without dropping those with correlations

X = data.drop('diagnosis', axis=1)

y = data['diagnosis']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, 

                                                    stratify=y)



print(type(X))

print(type(y))
# Creating a k-NN classifier with 3 neighbors: knn

knn = KNeighborsClassifier(n_neighbors=3)



# Fit the classifier to the training data

knn.fit(X_train, y_train)



# Print the accuracy

print('Accuracy KNN(1): ', knn.score(X_test, y_test))
neighbors = np.arange(1, 22)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    #Compute accuracy on the training and testing sets

    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test)



plt.figure(figsize=(12,7))

sns.set_context('notebook', font_scale=1.5)

plt.title('Learning curves for k-NN: Varying Number of Neighbors', size=20)

plt.plot(neighbors, test_accuracy, marker ='o', label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, marker ='o', label = 'Training Accuracy')

plt.legend(prop={'size':15})

plt.xlabel('Number of Neighbors (k)', size=15)

plt.ylabel('Accuracy', size=15)

plt.annotate('Over-fitting', xy=(0.5, 0.94), xytext=(0.3, 0.935), size=15, color='red')

plt.annotate('Under-fitting', xy=(0.5, 0.94), xytext=(18, 0.93), size=15, color='red')

plt.xticks(np.arange(min(neighbors), max(neighbors)+1, 1.0));
## data are distributed in a wide range (below), need to be normalizded.

plt.figure(figsize=(15,3))

ax= data.drop('diagnosis', axis=1).boxplot(data.columns.name, rot=90)

plt.xticks( size=20)

ax.set_ylim([0,50]);
steps = [('scaler', StandardScaler()), 

         ('knn', KNeighborsClassifier())]



pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors' : np.arange(1, 50)}





k_nn = GridSearchCV(pipeline, param_grid=parameters)

k_nn.fit(X_train, y_train)

y_pred = k_nn.predict(X_test)



print(k_nn.best_params_)

print()

print(classification_report(y_test, y_pred))

print("Best score is: {}".format(k_nn.best_score_))



ConfMatrix = confusion_matrix(y_test,k_nn.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix");
cv_knn = cross_val_score(k_nn, X, y, cv=5, scoring='accuracy')

print('Average 5-Fold CV Score: ', cv_knn.mean(), ', Standard deviation: ', cv_knn.std())
# To Setup the pipeline

steps = [('scaler', StandardScaler()),

         ('SVM', SVC())]



pipeline = Pipeline(steps)



# Specify the hyperparameter space: C is regularization strength while gamma controls the kernel coefficient. 

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}



# Create train & test sets



# Instantiate the GridSearchCV object: cv

cv =GridSearchCV(pipeline,parameters, cv=3)



# Fit to the training set

cv.fit(X_train, y_train)



# Predict the labels of the test set: y_pred

y_pred = cv.predict(X_test)



# Compute and print metrics

print("Tuned Model Parameters: {}".format(cv.best_params_))

print("Accuracy: {}".format(cv.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

print("Best score is: {}".format(cv.best_score_))



ConfMatrix = confusion_matrix(y_test,cv.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix");
X = data.drop('diagnosis', axis=1).values[:,:2]

y = data['diagnosis'].values



h = .02  # step size in the mesh



# we create an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors

C = 1.0  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(X, y)

rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)

poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

lin_svc = svm.LinearSVC(C=C).fit(X, y)



# create a mesh to plot in

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



# title for the plots

titles = ['SVC with linear kernel',

          'LinearSVC (linear kernel)',

          'SVC with RBF kernel',

          'SVC with polynomial (degree 3) kernel']

plt.figure(figsize=(10,12))

for j, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):

# Plot the decision boundary by assigning a color to each point in the mesh 

    plt.subplot(2, 2, j + 1)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('RdBu_r'), alpha=0.6)



# Ploting  the training points

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('RdBu_r'))

    plt.xlabel('radius_mean',size=20)

    plt.ylabel('texture_mean',size=20)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.xticks(())

    plt.yticks(())

    plt.title(titles[j],size=20);
# Setup the hyperparameter grid, (not scaled data)

param_grid = {'C': np.logspace(-5, 8, 15)}



# Instantiate a logistic regression classifier: logreg

logreg = LogisticRegression()



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg,param_grid , cv=5)



# Fit it to the data

logreg_cv.fit(X_train, y_train)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 

print()

print(classification_report(y_test, y_pred))

print("Best score is {}".format(logreg_cv.best_score_))



ConfMatrix = confusion_matrix(y_test,logreg_cv.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix");
# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

#tree = DecisionTreeClassifier() # ExtraTrees is better here. 

tree= ExtraTreesClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)



# Fit it to the data

tree_cv.fit(X_train, y_train)

y_pred = tree_cv.predict(X_test)



# Print the tuned parameters and score

print("Tuned Extra Tree Parameters: {}".format(tree_cv.best_params_))

print()

print(classification_report(y_test, y_pred))

print("Best score is {}".format(tree_cv.best_score_))

# metrics.accuracy_score(y_pred,y_test) # the same as above



ConfMatrix = confusion_matrix(y_test,tree_cv.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix");
Ran = RandomForestClassifier(n_estimators=50)

Ran.fit(X_train, y_train)

y_pred = Ran.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_pred,y_test))



## 5-fold cross-validation 

cv_scores =cross_val_score(Ran, X, y, cv=5)



# Print the 5-fold cross-validation scores

print()

print(classification_report(y_test, y_pred))

print()

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)), 

      ", Standard deviation: {}".format(np.std(cv_scores)))



ConfMatrix = confusion_matrix(y_test,Ran.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix");
tree_2= ExtraTreesClassifier()

tree_2.fit(X_train, y_train)

print('Extra-Tree score:',tree_2.score(X_test, y_test))

print('Shape of original data:', X_train.shape)

print()

tree_2.feature_importances_

model_reduced = SelectFromModel(tree_2, prefit=True)

X_reduced = model_reduced.transform(X_train)

print('Shape of data with most important features:', X_reduced.shape)

print()

print(classification_report(y_test, y_pred))

print()



ConfMatrix = confusion_matrix(y_test,tree_2.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix");
cv_tree2 = cross_val_score(logreg_cv, X, y, cv=5, scoring='accuracy')

print("Average 5-Fold CV Score: {}".format(np.mean(cv_tree)), 

      "Standard deviation: {}".format(np.std(cv_tree)));
### The data is not normalized. 



## method 1 

predictors= data.drop('diagnosis', axis=1).values  # .values to conver it to array

target = to_categorical(data.diagnosis.values)

n_cols = predictors.shape[1]



#np.random.seed(1337) # for reproducibility

seed = 1337

np.random.seed(seed)



model = Sequential()



# Add layers and nodes

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))

model.add(Dense(50, activation='relu'))

model.add(Dense(2, activation='softmax'))

# Compile the model

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy']) 

    



## Fit with 0.3 splitting with early_stopping_monitor with 30 epochs

early_stopping_monitor =EarlyStopping(patience=2) 

# Fit the model



history=model.fit(predictors, target, validation_split=0.3, epochs=100, batch_size=5,

                  callbacks = [early_stopping_monitor])



# 1 epoch = one forward pass and one backward pass of all the training examples

# batch size = number of samples that going to be propagated through the network.

# The higher the batch size, the more memory space. 
#RandomForest

impor_Forest=Ran.feature_importances_

indices_1 = np.argsort(impor_Forest)[::-1]



#ExtraTree

impor_Extra_tree=tree_2.feature_importances_

indices_2= np.argsort(impor_Extra_tree)[::-1]



featimp_1 = pd.Series(impor_Forest, index=data.columns[1:]).sort_values(ascending=False)

featimp_2 = pd.Series(impor_Extra_tree, index=data.columns[1:]).sort_values(ascending=False)



Table_impor= pd.DataFrame({'ExtraTree': featimp_2,'Random-Forest': featimp_1})

Table_impor=Table_impor.sort_values('ExtraTree', ascending=False)

print(Table_impor)

print()

print('The six most important features:')

print(featimp_1[0:6])



sns.set_context('notebook', font_scale=1.5)

Table_impor.plot(kind='barh', figsize=(12,10))

plt.title('Feature importance', size=20);