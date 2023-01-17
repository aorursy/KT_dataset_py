# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.




from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import time





def random_forest_classifier(features, target):

    """

    To train the random forest classifier with features and target data

    :param features:

    :param target:

    :return: trained random forest classifier

    """

    clf = RandomForestClassifier()

    clf.fit(features, target)

    return clf



    

def main():

    #Assignment Step 1: 

    # Reading Input Train Data

    train_data = pd.read_csv('../input/train.csv')

    train_y = train_data[train_data.columns[0]]

    train_x = train_data[train_data.columns[1::]]

    print (train_x.head(2))

    # Reading Input Test Data

    test_data = pd.read_csv('../input/test.csv')

    test_x = test_data

    

    # Assignment step 2

    combined_x = pd.concat([train_x,test_x])

    print(combined_x.shape)

    

    pca = PCA(n_components=780)

    pca.fit(combined_x)

    

    # Explained Variance Per Component

    evpc_array = pca.explained_variance_ratio_

    total_variance_explained = 0

    for index,evp in enumerate(evpc_array):

        total_variance_explained += evp

        print(index,total_variance_explained*100)

        if total_variance_explained >= 0.995:

            n_comps_for_95 = index + 1

            break

    # Number of PCA components for 95%

    print  (n_comps_for_95)       

    pca = PCA(n_components=n_comps_for_95)

    pca.fit(combined_x)

    print("Explained Variance:", sum(pca.explained_variance_ratio_))

    

    # Transform Train Data

    train_x = pca.transform(train_x)

    

    print (test_data.head(2))

    # Start time

    start_t = time.time()

    # Create random forest classifier instance

    trained_model = random_forest_classifier(train_x, train_y)

    # End time

    end_t = time.time()



    print ("Trained model :: ", trained_model)

    print ("time collapsed:,", end_t - start_t)

    

    # Test

    predictions = trained_model.predict(pca.transform(test_x))

    # Train and Test Accuracy

    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))

    

    print (type(predictions))

    print('Predictions: %s' % predictions)

    with open('submission.csv','w') as f:

        f.write("ImageId,Label\n")

        for index,pred in enumerate(predictions):

            f.write("%d,%d\n" % (index+1,pred))

        

    

        

    #pred_data = pd.DataFrame(data = predictions)

    #pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]) 

    #pred_data.to_csv('../submission.csv')





if __name__ == "__main__":

    main()