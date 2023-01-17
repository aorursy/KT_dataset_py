import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_excel(r'../input/CTG.xls', sheetname=1, skiprows=1) # Get the data
dataset.head()
dataset = dataset.dropna(axis=1, how='all') # Got rid of empty columns

dataset = dataset.dropna(axis=0, how='any') # Three rows were REALLY missing information

dataset = dataset.drop(['b','e','DR','Tendency'], axis=1) # No information from such columns...
dataset.head()
import matplotlib.cm as cm # To make pretty graphs

import seaborn as sns

color = sns.color_palette()



cnt_srs = dataset['CLASS'].value_counts()

cnt_srs = cnt_srs.head(10)

plt.figure(figsize=(10,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

N = 10

ind = np.arange(N) 

plt.xticks(ind, ('A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'))

plt.ylabel('Number of Cases', fontsize=12)

plt.title('Distribution of Classes', fontsize=18)

plt.show()
import matplotlib.cm as cm # To make pretty graphs

import seaborn as sns

color = sns.color_palette()



cnt_srs = dataset['NSP'].value_counts()

cnt_srs = cnt_srs.head(10)

plt.figure(figsize=(10,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

N = 3

ind = np.arange(N) 

plt.xticks(ind, ('Normal','Suspect', 'Pathologic'))

plt.ylabel('Number of Cases', fontsize=12)

plt.title('Distribution of NSP', fontsize=18)

plt.show()
x = dataset[['CLASS','NSP']]

N = 10

zeroesddf = pd.DataFrame(data=np.zeros(10,),index=[1,2,3,4,5,6,7,8,9,10])



ind = np.arange(N)    # the x locations for the groups

width = 0.50    # the width of the bars: can also be len(x) sequence



plt.figure(figsize=(12,6))



for nsp_i in [1,2,3]:

    norm = x.loc[(x['NSP']==nsp_i)].drop('NSP', axis=1)

    norm = (pd.value_counts(norm['CLASS'].values, sort=False)).sort_index()

    norm = pd.DataFrame(data=norm, index=norm.index)

    norm = zeroesddf.add(norm,axis='index',fill_value=0) 

    norm = (norm.values).reshape(-1)

    if nsp_i == 1 :

        p1 = plt.bar(ind,norm, width)

    elif nsp_i == 2 :

        p2 = plt.bar(ind,norm,width)

    elif nsp_i == 3 :

        p3 = plt.bar(ind,norm,width)

    

plt.ylabel('Number of Cases')

plt.title('Distribution of NSP accross the classes')

plt.xticks(ind, ('A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'))

plt.legend((p1[0], p2[0], p3[0]), ('Normal', 'Suspect', 'Pathologic'))

plt.show()
sns.set(style="white")

# Generate a large random dataset

d = dataset



# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Preparing our variables

X = dataset.iloc[:,:-2].values # We got rid of the class value to focus on the FHR diagnosis

y = dataset.iloc[:,-1].values # Containing the FHR diagnosis



# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import KernelPCA



names = [

         'Linear Kernel',

         'Polynomial Kernel',

         'RBF Kernel',

         'Sigmoid Kernel',

         'Cosine Kernel'

         ]



classifiers = [

    KernelPCA(n_components = 3, kernel = 'linear'),

    KernelPCA(n_components = 3, kernel = 'poly', gamma= 0.00001),

    KernelPCA(n_components = 3, kernel = 'rbf', gamma= 0.00001),

    KernelPCA(n_components = 3, kernel = 'sigmoid', gamma= 0.00001),

    KernelPCA(n_components = 3, kernel = 'cosine')

]



models=zip(names,classifiers)

   

for name, kpca in models:

    X_PCA = kpca.fit_transform(X_train)

    

    from mpl_toolkits.mplot3d import axes3d

    from matplotlib import style

    style.use('ggplot')

    

    fig = plt.figure(figsize=(10,6))

    ax1 = fig.add_subplot(111, projection='3d')

    loc = [1,2,3]

    classes = ['Normal','Suspect','Pathologic']

    x3d = X_PCA[:,0]

    y3d = X_PCA[:,1]

    z3d = X_PCA[:,2]



    plot = ax1.scatter(x3d, y3d, z3d, c=y_train, cmap="viridis")

    ax1.set_xlabel('PC1')

    ax1.set_ylabel('PC2')

    ax1.set_zlabel('PC3')

    cb = plt.colorbar(plot)

    cb.set_ticks(loc)

    cb.set_ticklabels(classes)



    plt.title(name)

    plt.show()
#https://statcompute.wordpress.com/2017/01/15/autoencoder-for-dimensionality-reduction/



from numpy.random import seed

from sklearn.preprocessing import minmax_scale

from keras.layers import Input, Dense

from keras.models import Model



X = dataset.iloc[:,:-1].values # Put back the class value to use it later



 

# Feature scaling for the AE

sX = minmax_scale(X, axis = 0)

ncol = 36

X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(sX, y, train_size = 0.8, random_state = seed(2017))

 

Class_train_auto = X_train_auto[:, 36] # These two columns will be used for comparison

Class_test_auto = X_test_auto[:,36]    # later on



X_train_auto = np.delete(X_train_auto, 36, 1)  # But theyr are not needed

X_test_auto = np.delete(X_test_auto, 36, 1)  # as of right now



input_dim = Input(shape = (ncol, ))



# Define the dimension of the encoder

encoding_dim = 3



# Define the encoder layer

encoded = Dense(encoding_dim, activation = 'relu')(input_dim)



# Define the decoder layer

decoded = Dense(ncol, activation = 'sigmoid')(encoded)



# Combine the layers into a model to create the AE

autoencoder = Model(input = input_dim, output = decoded)



# Configure and train the AE

autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

autoencoder.fit(X_train_auto, X_train_auto, nb_epoch = 150, batch_size = 25, shuffle = True, validation_data = (X_test_auto, X_test_auto))



# Get the encoded data and reduced dimmension

encoder = Model(input = input_dim, output = encoded)

encoded_input = Input(shape = (encoding_dim, ))

encoded_out = encoder.predict(X_test_auto)
# Visualising the clusters

from mpl_toolkits.mplot3d import axes3d

from matplotlib import style

style.use('ggplot')



fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(111, projection='3d')

ax1.w_xaxis.set_pane_color((0.7, 0.7, 0.7, 1.0)) # During the previous visualisation

ax1.w_yaxis.set_pane_color((0.7, 0.7, 0.7, 1.0)) # the colors were difficult to see

ax1.w_zaxis.set_pane_color((0.7, 0.7, 0.7, 1.0)) # with the light gray background, color tuning



x3d = encoded_out[:,0]

y3d = encoded_out[:,1]

z3d = encoded_out[:,2]



plot = ax1.scatter(x3d, y3d, z3d, c=y_test_auto, cmap="viridis", marker='o')

ax1.set_xlabel('Coded_1')

ax1.set_ylabel('Coded_2')

ax1.set_zlabel('Coded_3')

cb = plt.colorbar(plot)

cb.set_ticks(loc)

cb.set_ticklabels(classes)

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(encoded_out)
# Visualising the clusters

from mpl_toolkits.mplot3d import axes3d

from matplotlib import style

style.use('ggplot')



fig = plt.figure(figsize=(10,6))

loc = [0,1,2,3,4,5,6]

classes = ['Group 1','Group 2','Group 3','Group 4','Group 5','Group 6','Group 7']

ax1 = fig.add_subplot(111, projection='3d')

ax1.w_xaxis.set_pane_color((0.7, 0.7, 0.7, 1.0)) # During the previous visualisation

ax1.w_yaxis.set_pane_color((0.7, 0.7, 0.7, 1.0)) # the colors were difficult to see

ax1.w_zaxis.set_pane_color((0.7, 0.7, 0.7, 1.0)) # with the light gray background, color tuning



x3d = encoded_out[:,0]

y3d = encoded_out[:,1]

z3d = encoded_out[:,2]



ploto = ax1.scatter(x3d, y3d, z3d, c=y_kmeans, cmap="viridis")

cb = plt.colorbar(ploto)

cb.set_ticks(loc)

cb.set_ticklabels(classes)

ax1.set_xlabel('Coded 1')

ax1.set_ylabel('Coded 2')

ax1.set_zlabel('Coded 3')

plt.title('Clustering of the groups found with the AE')

plt.show()

dataset2 = pd.DataFrame(data=X_test_auto)

dataset2['NSP'] = y_test_auto

dataset2['New Groups'] = y_kmeans

#Succesfully added our new groups to the corresponding dataframe
# Distribution of the newfound classes



import matplotlib.cm as cm # To make pretty graphs

import seaborn as sns

color = sns.color_palette()



cnt_srs = dataset2['New Groups'].value_counts()

cnt_srs = cnt_srs.head(10)

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

N = 7

ind = np.arange(N) 

plt.xticks(ind, ('Group 1','Group 2','Group 3','Group 4','Group 5','Group 6','Group 7'))

plt.ylabel('Number of Cases', fontsize=12)

plt.title('Distribution of the New Classes', fontsize=18)

plt.show()
x = dataset2[['New Groups','NSP']]

plt.figure(figsize=(12,6))

N = 7

zeroesddf = pd.DataFrame(data=np.zeros(7,),index=[0,1,2,3,4,5,6])



ind = np.arange(N)    # the x locations for the groups

width = 0.50    # the width of the bars: can also be len(x) sequence



plt.figure(figsize=(12,6))



for nsp_i in [1,2,3]:

    norm = x.loc[(x['NSP']==nsp_i)].drop('NSP', axis=1)

    norm = (pd.value_counts(norm['New Groups'].values, sort=False)).sort_index()

    norm = pd.DataFrame(data=norm, index=norm.index)

    norm = zeroesddf.add(norm,axis='index',fill_value=0) 

    norm = (norm.values).reshape(-1)

    if nsp_i == 1 :

        p1 = plt.bar(ind,norm, width)

    elif nsp_i == 2 :

        p2 = plt.bar(ind,norm,width)

    elif nsp_i == 3 :

        p3 = plt.bar(ind,norm,width)



plt.ylabel('Number of Cases')

plt.title('Distribution of NSP accross the new classes')

plt.xticks(ind, ('Group 1','Group 2','Group 3','Group 4','Group 5','Group 6','Group 7'))

plt.legend((p1[0], p2[0], p3[0]), ('Normal', 'Suspect', 'Pathologic'))

plt.show()
# Decoding the scaled data

Class_test_auto2 = Class_test_auto



for i in range(426,):

    if Class_test_auto[i]==1. :

        Class_test_auto2[i]=9

    elif Class_test_auto[i]==1/9:

        Class_test_auto2[i]=1

    elif Class_test_auto[i]==2/9:

        Class_test_auto2[i]=2

    elif Class_test_auto[i]==1/3:

        Class_test_auto2[i]=3

    elif Class_test_auto[i]==8/9:

        Class_test_auto2[i]=8

    elif Class_test_auto[i]>0.7 and Class_test_auto[i]<0.8 :

        Class_test_auto2[i]=7

    elif Class_test_auto[i]>0.4 and Class_test_auto[i]<0.5 :

        Class_test_auto2[i]=4

    elif Class_test_auto[i]>0.5 and Class_test_auto[i]<0.6 :

        Class_test_auto2[i]=5

    elif Class_test_auto[i]>0.6 and Class_test_auto[i]<0.7 :

        Class_test_auto2[i]=6
dataset2['CLASS'] = Class_test_auto2



x = dataset2[['CLASS','New Groups']]

N = 10

zeroesddf = pd.DataFrame(data=np.zeros(10,),index=[0,1,2,3,4,5,6,7,8,9])



ind = np.arange(N)    # the x locations for the groups

width = 0.50    # the width of the bars: can also be len(x) sequence



plt.figure(figsize=(12,6))



norm=[]

p=[]

for class_i in [0,1,2,3,4,5,6]:

    temp = x.loc[(x['New Groups']==class_i)].drop('New Groups', axis=1)

    temp = (pd.value_counts(temp['CLASS'].values, sort=False)).sort_index()

    temp = pd.DataFrame(data=temp, index=temp.index)

    temp = zeroesddf.add(temp,axis='index',fill_value=0) 

    temp = (temp.values).reshape(-1)

    norm.append(temp)

    p_temp = plt.bar(ind,norm[class_i], width)

    p.append(p_temp)

        

plt.ylabel('Number of Cases')

plt.title('New vs Old')

plt.xticks(ind, ('A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'))

plt.legend((p[0][0], p[1][0], p[2][0], p[3][0], p[4][0], p[5][0], p[6][0]), ('Group 1', 'Group 2',

                                                               'Group 3','Group 4','Group 5',

                                                              'Group 6','Group 7'))

plt.show()
from sklearn.svm import SVC # Importing the relevant classifier

classifier = SVC(kernel = 'linear', random_state = 0) # Choosing a linear Kernel for SVC and not K-svc

classifier.fit(X_train, y_train)# Fitting the classifier onto our dataset

score = classifier.score(X_test, y_test, sample_weight=None) # Checking how well it did



# Predicting the Test set results

y_pred = classifier.predict(X_test) # Still all pretty good



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred)



print('The score is : '+str(score))   

label = ["N","S","P"]

sns.heatmap(cm1, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")

plt.show()
# Importing all the classifiers fitting to the problem from sklearn                             

from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Setting different parameters for the grid search

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



# Turning our problem into a non linear problem

n_col=36

poly = PolynomialFeatures(2)

X_train=poly.fit_transform(X_train)

X_test=poly.fit_transform(X_test)







names = [

         'ElasticNet',

         'SVC',

         'kSVC',

         'KNN',

         'DecisionTree',

         'RandomForestClassifier',

         'GridSearchCV',

         'HuberRegressor',

         #♠'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier',

         ]



classifiers = [

    ElasticNetCV(cv=10, random_state=0),

    SVC(),

    SVC(kernel = 'rbf', random_state = 0),

    KNeighborsClassifier(n_neighbors = 1),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200),

    GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    #Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True), # Seem to have problems with the Ridge, so I'm taking it out

    Lasso(alpha=0.05),

    LassoCV(),

    Lars(n_nonzero_coefs=10),

    BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



models=zip(names,classifiers,correction)

   

for name, clf,correct in models:

    regr=clf.fit(X_train,y_train)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

    

    # Confusion Matrix

    print('--'*40)

    print(name, 'Confusion Matrix')

    conf=confusion_matrix(y_test, np.round(regr.predict(X_test) ) )     

    label = ["N","S","P"]

    sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")

    plt.show()

    

    print('--'*40)



    # Classification Report

    print(name,'Classification Report')

    classif=classification_report(y_test,np.round( regr.predict(X_test) ) )

    print(classif)





    # Accuracy

    print('--'*40)

    logreg_accuracy = round(accuracy_score(y_test, np.round( regr.predict(X_test) ) ) * 100,2)

    print(name, 'Accuracy', logreg_accuracy,'%')
from sklearn.model_selection import GridSearchCV

parameters = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



# Here we typed out all the options we want the method to test, each one in

# its own dictionary.



from sklearn.svm import SVC

grid_search = GridSearchCV(SVC(),param_grid = parameters, refit = True, verbose = 1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
parameters = {'C': [50,75, 125, 100, 500], 'gamma': [0.0001,0.00001], 'kernel': ['rbf']}



from sklearn.svm import SVC

grid_search = GridSearchCV(SVC(),param_grid = parameters, refit = True, verbose = 1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print(best_accuracy)

print(best_parameters)
# Importing the Keras libraries, models and layers

import keras

from keras.models import Sequential

from keras.layers import Dense
# In this part we are barely reprocessing the data and putting it in a form that will fit the ANN

# as we explained earlier.



# While chain-testing the different algorithms we had changed X and y to fit a polynomial form,

# so we have to import them anew.

X = dataset.iloc[:,:-2].values

y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

y_trainann = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()

y_trainann = np.delete(y_trainann, 1, 1) # Deleting a column to avoid redundancy after the OHE
# Initializing the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 36))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_trainann, batch_size = 5, nb_epoch = 20)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.7)



recovered_y = np.zeros(532,)

for i in range(0,532):

    for j in range(0,2):

        if (y_pred[i][j] == True) and j == 0:

            recovered_y[i] = 1

        elif (y_pred[i][j] == True) and j == 1:

            recovered_y[i] = 3            

for i in range(0,532):

    if recovered_y[i] == 0:

        recovered_y[i] = 2

     

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, recovered_y)

acc = accuracy_score(y_test, recovered_y)



print('The score is : '+str(acc))   

label = ["N","S","P"]

sns.heatmap(cm, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")

plt.show()
epochs = [5,10,15]

batch_sizes = [1,2,3,4,5,10,15,20]

table_results = np.zeros([(len(epochs)*len(batch_sizes)),3],)

c = 0

for aa in epochs:

    for bb in batch_sizes:

        print('The number of epochs is : '+str(aa))

        print('The batch size is : '+str(bb))

        classifier.fit(X_train, y_trainann, batch_size = bb, nb_epoch = aa)



        y_pred = classifier.predict(X_test)

        y_pred = (y_pred > 0.7)



        recovered_y = np.zeros(532,)

        for i in range(0,532):

            for j in range(0,2):

                if (y_pred[i][j] == True) and j == 0:

                    recovered_y[i] = 1

                elif (y_pred[i][j] == True) and j == 1:

                    recovered_y[i] = 3            

        for i in range(0,532):

            if recovered_y[i] == 0:

                recovered_y[i] = 2



        from sklearn.metrics import confusion_matrix, accuracy_score

        cm = confusion_matrix(y_test, recovered_y)

        acc = accuracy_score(y_test, recovered_y)

        

        table_results[c][0] = aa

        table_results[c][1] = bb

        table_results[c][2] = acc

        c += 1

        print('The accuracy is : '+str(acc))   

        label = ["N","S","P"]

        sns.heatmap(cm, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")

        plt.show()
style.use('ggplot')

    

fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(111, projection='3d')

loc = [1,2,3]

classes = ['Normal','Suspect','Pathologic']

    

x3d = table_results[:,0]

y3d = table_results[:,1]

z3d = table_results[:,2]



plot = ax1.scatter(x3d, y3d, z3d,c=z3d, cmap="jet")

ax1.set_xlabel('#epochs')

ax1.set_ylabel('Batch Size')

ax1.set_zlabel('Accuracy')





plt.title('Accuracy of the ANN in GridSearch')

plt.show()
table_results