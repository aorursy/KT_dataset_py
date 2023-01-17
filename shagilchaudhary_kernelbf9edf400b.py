import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt



breast_cancer = pd.read_csv("../input/breast-cancer.csv")

breast_cancer = breast_cancer.drop(['Unnamed: 32', 'id'], axis = 1)

keys = breast_cancer.keys()

keys
breast_cancer.head(5)
breast_cancer.tail(5)
import seaborn as sns

sns.pairplot(breast_cancer)
def type_diagnosis():

    cancerdf = breast_cancer

    counts = cancerdf.diagnosis.value_counts(ascending = True)

    counts.index = "malignant benign".split()

    return counts

type_diagnosis()
features_mean= list(breast_cancer.columns[1:11])

bins = 12

plt.figure(figsize=(15,15))

for i, feature in enumerate(features_mean):

    rows = int(len(features_mean)/2)

    

    plt.subplot(rows, 2, i+1)

    

    sns.distplot(breast_cancer[breast_cancer['diagnosis']=='M'][feature], bins=bins, color='red', label='M');

    sns.distplot(breast_cancer[breast_cancer['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');

    

    plt.legend(loc='upper right')



plt.tight_layout()

plt.show()
diag_map = {'M':1, 'B':0}

breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map(diag_map)
def split():

    cancerdf = breast_cancer

    X = cancerdf[cancerdf.columns[1:]]

    y = cancerdf.diagnosis

    return X,y

split()
from sklearn.model_selection import train_test_split

def train_test_splitting():

    X = breast_cancer.loc[:,features_mean]

    y = breast_cancer.loc[:, 'diagnosis']



#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    return train_test_split(X, y, train_size = 426, test_size = 143, random_state = 0)

train_test_splitting()

from sklearn.neighbors import KNeighborsClassifier

def classifing_data():

    X_train, X_test, y_train, y_test = train_test_splitting()

    model = KNeighborsClassifier(n_neighbors = 5)

    model.fit(X_train, y_train)

    return model

classifing_data()
def label_prediction_in_knn():

    cancerdf = breast_cancer

    means = cancerdf.mean()[:-0].values.reshape(1, -1)

    model = classifing_data

    return model.predict(means)
def predicting_X_test_values():

    X_train, X_test, y_train, y_test = train_test_splitting()

    knn = classifing_data()   

    return knn.predict(X_test)

predicting_X_test_values()
def scoring_the_model():

    X_train, X_test, y_train, y_test = train_test_splitting()

    knn = classifing_data()    

    return knn.score(X_test, y_test)

scoring_the_model()
def accuracy_plot():

    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = train_test_splitting()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)

    mal_train_X = X_train[y_train==1]

    mal_train_y = y_train[y_train==1]

    ben_train_X = X_train[y_train==0]

    ben_train_y = y_train[y_train==0]



    mal_test_X = X_test[y_test==1]

    mal_test_y = y_test[y_test==1]

    ben_test_X = X_test[y_test==0]

    ben_test_y = y_test[y_test==0]



    knn = classifing_data()



    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 

              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

    plt.figure()



    # Plot the scores as a bar chart

    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])



    # directly label the score onto the bars

    for bar in bars:

        height = bar.get_height()

        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 

                     ha='center', color='w', fontsize=11)



    # remove all the ticks (both axes), and tick labels on the Y axis

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)



    # remove the frame of the chart

    for spine in plt.gca().spines.values():

        spine.set_visible(False)



    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);

    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
accuracy_plot()