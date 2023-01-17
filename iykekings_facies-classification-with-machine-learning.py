# From Brendon Hall, Enthought post on seg

# This notebook demonstrates how to train a machine learning algorithm to predict facies from well log data. The dataset we will use comes from a class excercise from The University of Kansas on Neural Networks and Fuzzy Systems. This exercise is based on a consortium project to use machine learning techniques to create a reservoir model of the largest gas fields in North America, the Hugoton and Panoma Fields. For more info on the origin of the data, see Bohling and Dubois (2003) and Dubois et al. (2007).



# The dataset we will use is log data from nine wells that have been labeled with a facies type based on oberservation of core. We will use this log data to train a support vector machine to classify facies types. Support vector machines (or SVMs) are a type of supervised learning model that can be trained on data to perform classification and regression tasks. The SVM algorithm uses the training data to fit an optimal hyperplane between the different classes (or facies, in our case). We will use the SVM implementation in scikit-learn.



# First we will explore the dataset. We will load the training data from 9 wells, and take a look at what we have to work with. We will plot the data from a couple wells, and create cross plots to look at the variation within the data.



# Next we will condition the data set. We will remove the entries that have incomplete data. The data will be scaled to have zero mean and unit variance. We will also split the data into training and test sets.



# We will then be ready to build the SVM classifier. We will demonstrate how to use the cross validation set to do model parameter selection.



# Finally, once we have a built and tuned the classifier, we can apply the trained model to classify facies in wells which do not already have labels. We will apply the classifier to two wells, but in principle you could apply the classifier to any number of wells that had the same log data.
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable



from pandas import set_option

set_option("display.max_rows", 10)

pd.options.mode.chained_assignment = None



training_data = pd.read_csv('../input/raw-data/facies_vectors.csv')

training_data
# To evaluate the accuracy of the classifier,

# we will remove one well from the training set so that we can compare 

# the predicted and actual facies labels.

blind = training_data[training_data['Well Name'] == 'SHANKLE']

training_data = training_data[training_data['Well Name'] != 'SHANKLE']

blind
# Let's clean up this dataset. The 'Well Name' and 'Formation' columns 

# can be turned into a categorical data type.

training_data['Well Name'] = training_data['Well Name'].astype('category')

training_data['Formation'] = training_data['Formation'].astype('category')

training_data['Well Name'].unique()
# 1=sandstone  2=c_siltstone   3=f_siltstone 

# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite

# 8=packstone 9=bafflestone

facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',

       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']



facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',

                 'WS', 'D','PS', 'BS']

#facies_color_map is a dictionary that maps facies labels

#to their respective colors

facies_color_map = {}

for ind, label in enumerate(facies_labels):

    facies_color_map[label] = facies_colors[ind]



def label_facies(row, labels):

    return labels[ row['Facies'] -1]

    

training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)

training_data.describe()
# Looking at the count values, most values have 4149 valid values except for PE, which has 3232.

# In this tutorial we will drop the feature vectors that don't have a valid PE entry.

PE_mask = training_data['PE'].notnull().values

training_data = training_data[PE_mask]
# Let's take a look at the data from individual wells in a more familiar log plot form. 

# We will create plots for the five well log variables, as well as a log for facies labels.

def make_facies_log_plot(logs, facies_colors):

    #make sure logs are sorted by depth

    logs = logs.sort_values(by='Depth')

    cmap_facies = colors.ListedColormap(

            facies_colors[0:len(facies_colors)], 'indexed')

    

    ztop=logs.Depth.min(); zbot=logs.Depth.max()

    

    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

    

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))

    ax[0].plot(logs.GR, logs.Depth, '-g')

    ax[1].plot(logs.ILD_log10, logs.Depth, '-')

    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')

    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')

    ax[4].plot(logs.PE, logs.Depth, '-', color='black')

    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',

                    cmap=cmap_facies,vmin=1,vmax=9)

    

    divider = make_axes_locatable(ax[5])

    cax = divider.append_axes("right", size="20%", pad=0.05)

    cbar=plt.colorbar(im, cax=cax)

    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 

                                'SiSh', ' MS ', ' WS ', ' D  ', 

                                ' PS ', ' BS ']))

    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    

    for i in range(len(ax)-1):

        ax[i].set_ylim(ztop,zbot)

        ax[i].invert_yaxis()

        ax[i].grid()

        ax[i].locator_params(axis='x', nbins=3)

    

    ax[0].set_xlabel("GR")

    ax[0].set_xlim(logs.GR.min(),logs.GR.max())

    ax[1].set_xlabel("ILD_log10")

    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())

    ax[2].set_xlabel("DeltaPHI")

    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())

    ax[3].set_xlabel("PHIND")

    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())

    ax[4].set_xlabel("PE")

    ax[4].set_xlim(logs.PE.min(),logs.PE.max())

    ax[5].set_xlabel('Facies')

    

    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])

    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])

    ax[5].set_xticklabels([])

    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
# We then show log plots for wells SHRIMPLIN.

make_facies_log_plot(

    training_data[training_data['Well Name'] == 'SHRIMPLIN'],

    facies_colors)
# Again for NOLAN wells

make_facies_log_plot(

    training_data[training_data['Well Name'] == 'NOLAN'],

    facies_colors)
#count the number of unique entries for each facies, sort them by

#facies number (instead of by number of entries)

facies_counts = training_data['Facies'].value_counts().sort_index()

#use facies labels to index each count

facies_counts.index = facies_labels



facies_counts.plot(kind='bar',color=facies_colors, 

                   title='Distribution of Training Data by Facies')

facies_counts
#save plot display settings to change back to when done plotting with seaborn

inline_rc = dict(mpl.rcParams)



import seaborn as sns

sns.set()

sns.pairplot(training_data.drop(['Well Name','Facies','Formation','Depth','NM_M','RELPOS'],axis=1),

             hue='FaciesLabels', palette=facies_color_map,

             hue_order=list(reversed(facies_labels)))



#switch back to default matplotlib plot style

mpl.rcParams.update(inline_rc)
# Now we extract just the feature variables we need to perform the classification. 

# The predictor variables are the five wireline values and two geologic constraining variables. 

# We also get a vector of the facies labels that correspond to each feature vector.

correct_facies_labels = training_data['Facies'].values



feature_vectors = training_data.drop(['Formation', 'Well Name', 'Depth','Facies','FaciesLabels'], axis=1)

feature_vectors.describe()
from sklearn import preprocessing



scaler = preprocessing.StandardScaler().fit(feature_vectors)

scaled_features = scaler.transform(feature_vectors)
feature_vectors
# Split to test and training data, test will be used to compare the accuracy of the model, since we the facies of the model

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

        scaled_features, correct_facies_labels, test_size=0.2, random_state=42)
#Training the classifier

from sklearn import svm



clf = svm.SVC()

clf.fit(X_train,y_train)
#Predict

predicted_labels = clf.predict(X_test)
# To simplify reading the confusion matrix, 

# a function has been written to display the matrix along with facies labels and various error metrics.

def display_cm(cm, labels, hide_zeros=False,

                             display_metrics=False):

    """Display confusion matrix with labels, along with

       metrics such as Recall, Precision and F1 score.

       Based on Zach Guo's print_cm gist at

       https://gist.github.com/zachguo/10296432

    """



    precision = np.diagonal(cm)/cm.sum(axis=0).astype('float')

    recall = np.diagonal(cm)/cm.sum(axis=1).astype('float')

    F1 = 2 * (precision * recall) / (precision + recall)

    

    precision[np.isnan(precision)] = 0

    recall[np.isnan(recall)] = 0

    F1[np.isnan(F1)] = 0

    

    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0,1))

    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0,1))

    total_F1 = np.sum(F1 * cm.sum(axis=1)) / cm.sum(axis=(0,1))

    #print total_precision

    

    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length

    empty_cell = " " * columnwidth

    # Print header

    print("    " + " Pred", end=' ')

    for label in labels: 

        print("%{0}s".format(columnwidth) % label, end=' ')

    print("%{0}s".format(columnwidth) % 'Total')

    print("    " + " True")

    # Print rows

    for i, label1 in enumerate(labels):

        print("    %{0}s".format(columnwidth) % label1, end=' ')

        for j in range(len(labels)): 

            cell = "%{0}d".format(columnwidth) % cm[i, j]

            if hide_zeros:

                cell = cell if float(cm[i, j]) != 0 else empty_cell

            print(cell, end=' ')

        print("%{0}d".format(columnwidth) % sum(cm[i,:]))

        

    if display_metrics:

        print()

        print("Precision", end=' ')

        for j in range(len(labels)):

            cell = "%{0}.2f".format(columnwidth) % precision[j]

            print(cell, end=' ')

        print("%{0}.2f".format(columnwidth) % total_precision)

        print("   Recall", end=' ')

        for j in range(len(labels)):

            cell = "%{0}.2f".format(columnwidth) % recall[j]

            print(cell, end=' ')

        print("%{0}.2f".format(columnwidth) % total_recall)

        print("       F1", end=' ')

        for j in range(len(labels)):

            cell = "%{0}.2f".format(columnwidth) % F1[j]

            print(cell, end=' ')

        print("%{0}.2f".format(columnwidth) % total_F1)

    

                  

def display_adj_cm(

        cm, labels, adjacent_facies, hide_zeros=False, 

        display_metrics=False):

    """This function displays a confusion matrix that counts 

       adjacent facies as correct.

    """

    adj_cm = np.copy(cm)

    

    for i in np.arange(0,cm.shape[0]):

        for j in adjacent_facies[i]:

            adj_cm[i][i] += adj_cm[i][j]

            adj_cm[i][j] = 0.0

        

    display_cm(adj_cm, labels, hide_zeros, 

                             display_metrics)


from sklearn.metrics import confusion_matrix



conf = confusion_matrix(y_test, predicted_labels)

display_cm(conf, facies_labels, hide_zeros=True)
def accuracy(conf):

    total_correct = 0.

    nb_classes = conf.shape[0]

    for i in np.arange(0,nb_classes):

        total_correct += conf[i][i]

    acc = total_correct/sum(sum(conf))

    return acc
# The error within these 'adjacent facies' can also be calculated. 

adjacent_facies = np.array([[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]])



def accuracy_adjacent(conf, adjacent_facies):

    nb_classes = conf.shape[0]

    total_correct = 0.

    for i in np.arange(0,nb_classes):

        total_correct += conf[i][i]

        for j in adjacent_facies[i]:

            total_correct += conf[i][j]

    return total_correct / sum(sum(conf))
print('Facies classification accuracy = %f' % accuracy(conf))

print('Adjacent facies classification accuracy = %f' % accuracy_adjacent(conf, adjacent_facies))
#model selection takes a few minutes, change this variable

#to true to run the parameter loop

do_model_selection = True



if do_model_selection:

    C_range = np.array([.01, 1, 5, 10, 20, 50, 100, 1000, 5000, 10000])

    gamma_range = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])

    

    fig, axes = plt.subplots(3, 2, 

                        sharex='col', sharey='row',figsize=(10,10))

    plot_number = 0

    for outer_ind, gamma_value in enumerate(gamma_range):

        row = int(plot_number / 2)

        column = int(plot_number % 2)

        cv_errors = np.zeros(C_range.shape)

        train_errors = np.zeros(C_range.shape)

        for index, c_value in enumerate(C_range):

            

            clf = svm.SVC(C=c_value, gamma=gamma_value)

            clf.fit(X_train,y_train)

            

            train_conf = confusion_matrix(y_train, clf.predict(X_train))

            cv_conf = confusion_matrix(y_test, clf.predict(X_test))

        

            cv_errors[index] = accuracy(cv_conf)

            train_errors[index] = accuracy(train_conf)



        ax = axes[row, column]

        ax.set_title('Gamma = %g'%gamma_value)

        ax.semilogx(C_range, cv_errors, label='CV error')

        ax.semilogx(C_range, train_errors, label='Train error')

        plot_number += 1

        ax.set_ylim([0.2,1])

        

    ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)

    fig.text(0.5, 0.03, 'C value', ha='center',

             fontsize=14)

             

    fig.text(0.04, 0.5, 'Classification Accuracy', va='center', 

             rotation='vertical', fontsize=14)


clf = svm.SVC(C=10, gamma=1)        

clf.fit(X_train, y_train)



cv_conf = confusion_matrix(y_test, clf.predict(X_test))



print('Optimized facies classification accuracy = %.2f' % accuracy(cv_conf))

print('Optimized adjacent facies classification accuracy = %.2f' % accuracy_adjacent(cv_conf, adjacent_facies))
display_cm(cv_conf, facies_labels, 

           display_metrics=True, hide_zeros=True)
display_adj_cm(cv_conf, facies_labels, adjacent_facies, 

           display_metrics=True, hide_zeros=True)
# Applying the classification model to the blind data

# We held a well back from the training, and stored it in a dataframe called blind:

blind
y_blind = blind['Facies'].values
well_features = blind.drop(['Facies', 'Formation', 'Well Name', 'Depth'], axis=1)

well_features.describe()
# Now we can transform this with the scaler we made before:

X_blind = scaler.transform(well_features)
# Now it's a simple matter of making a prediction and storing it back in the dataframe:

y_pred = clf.predict(X_blind)

blind['Prediction'] = y_pred
# Let's see how did with the confusion matrix

cv_conf = confusion_matrix(y_blind, y_pred)



print('Optimized facies classification accuracy = %.2f' % accuracy(cv_conf))

print('Optimized adjacent facies classification accuracy = %.2f' % accuracy_adjacent(cv_conf, adjacent_facies))
display_cm(cv_conf, facies_labels,

           display_metrics=True, hide_zeros=True)
# but does remarkably well on the adjacent facies predictions.

display_adj_cm(cv_conf, facies_labels, adjacent_facies,

               display_metrics=True, hide_zeros=True)
def compare_facies_plot(logs, compadre, facies_colors):

    #make sure logs are sorted by depth

    logs = logs.sort_values(by='Depth')

    cmap_facies = colors.ListedColormap(

            facies_colors[0:len(facies_colors)], 'indexed')

    

    ztop=logs.Depth.min(); zbot=logs.Depth.max()

    

    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

    cluster2 = np.repeat(np.expand_dims(logs[compadre].values,1), 100, 1)

    

    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 12))

    ax[0].plot(logs.GR, logs.Depth, '-g')

    ax[1].plot(logs.ILD_log10, logs.Depth, '-')

    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')

    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')

    ax[4].plot(logs.PE, logs.Depth, '-', color='black')

    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',

                    cmap=cmap_facies,vmin=1,vmax=9)

    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',

                    cmap=cmap_facies,vmin=1,vmax=9)

    

    divider = make_axes_locatable(ax[6])

    cax = divider.append_axes("right", size="20%", pad=0.05)

    cbar=plt.colorbar(im2, cax=cax)

    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 

                                'SiSh', ' MS ', ' WS ', ' D  ', 

                                ' PS ', ' BS ']))

    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    

    for i in range(len(ax)-2):

        ax[i].set_ylim(ztop,zbot)

        ax[i].invert_yaxis()

        ax[i].grid()

        ax[i].locator_params(axis='x', nbins=3)

    

    ax[0].set_xlabel("GR")

    ax[0].set_xlim(logs.GR.min(),logs.GR.max())

    ax[1].set_xlabel("ILD_log10")

    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())

    ax[2].set_xlabel("DeltaPHI")

    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())

    ax[3].set_xlabel("PHIND")

    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())

    ax[4].set_xlabel("PE")

    ax[4].set_xlim(logs.PE.min(),logs.PE.max())

    ax[5].set_xlabel('Facies')

    ax[6].set_xlabel(compadre)

    

    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])

    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])

    ax[5].set_xticklabels([])

    ax[6].set_xticklabels([])

    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
compare_facies_plot(blind, 'Prediction', facies_colors)
# Applying the classification model to new data

# Now that we have a trained facies classification model we can use it to

# identify facies in wells that do not have core data. 

# In this case, we will apply the classifier to two wells, 

# but we could use it on any number of wells for which we have the same set of well logs for input.



# This dataset is similar to the training data except it does not have facies labels. 

# It is loaded into a dataframe called test_data.

well_data = pd.read_csv('../input/validation-data/validation_data_nofacies.csv')

well_data['Well Name'] = well_data['Well Name'].astype('category')

well_features = well_data.drop(['Formation', 'Well Name', 'Depth'], axis=1)
# The data needs to be scaled using the same constants we used for the training data.

X_unknown = scaler.transform(well_features)
#predict facies of unclassified data

y_unknown = clf.predict(X_unknown)

well_data['Facies'] = y_unknown

well_data
well_data['Well Name'].unique()
# We can use the well log plot to view the classification results along with the well logs.

make_facies_log_plot(

    well_data[well_data['Well Name'] == 'STUART'],

    facies_colors=facies_colors)



make_facies_log_plot(

    well_data[well_data['Well Name'] == 'CRAWFORD'],

    facies_colors=facies_colors)
# Finally we can write out a csv file with the well data along with the facies classification results.

well_data.to_csv('well_data_with_facies.csv')