# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.learning_curve import validation_curve

from sklearn import preprocessing

import matplotlib.pyplot as plt



labeled_images = pd.read_csv("../input/train.csv")

train_image, test_image, train_label, test_label = train_test_split(labeled_images.iloc[0:500, 1:]

                                                        , labeled_images.iloc[0:500, 0], test_size = 0.2)



######--------------------------cross-validation for model param-----------------------------------######

'''

param_range = np.arange(1, 10)

train_loss, cv_loss = validation_curve(KNeighborsClassifier(), train_image, 

            train_label, param_name = 'n_neighbors', param_range = param_range, cv = 5, scoring = 'accuracy')



train_loss_mean = np.mean(train_loss, axis = 1)

cv_loss_mean = np.mean(cv_loss, axis = 1)



plt.plot(param_range, train_loss_mean, 'o-', color = 'r', label = 'Training')

plt.plot(param_range, cv_loss_mean, 'o-', color = 'g', label = 'Cross-validation') 



plt.xlabel('n-neighbors')

plt.ylabel('Loss')

plt.legend(loc = 'best')

plt.show()

'''

######-------------------------cross-validation for model param-----------------------------------######





######-------------------------training----------------------------######



knn = KNeighborsClassifier(n_neighbors = 4)

train_image[train_image > 0] = 1

test_image[test_image > 0] = 1

#train_image = preprocessing.scale(train_image)

#test_image = preprocessing.scale(test_image)

knn.fit(train_image, train_label)

#print(knn.predict(test_image))

#print(test_label)

print(knn.score(test_image, test_label))



######------------------------training----------------------------######



######------------------------output------------------------------######

test_data = pd.read_csv('../input/test.csv')

test_data[test_data > 0] = 1

results = knn.predict(test_data[0:100])



df = pd.DataFrame(results)

df.index += 1

df.columns = ['Label']

df.to_csv('results1.csv')