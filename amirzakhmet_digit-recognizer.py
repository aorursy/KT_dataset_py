# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.neural_network import MLPClassifier







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
training_set = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_set = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



y_training = training_set.iloc[:,0].tolist()
X_training = training_set.iloc[:,1:785]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 10), random_state=1)

clf.fit(X_training, y_training)
the_prediction = clf.predict(test_set)

the_result = []



for index, record in test_set.iterrows():

    the_id = record[0]

    temp = []

    temp.append(index)

    temp.append(the_prediction[index])

    the_result.append(temp)



f = open("sample_submission_3.csv", "a")

f.write('ImageId,Label\n')

for record in the_result:

    the_id = record[0]

    predicted = record[1]

    f.write(str(int(the_id)) + ',' + str(int(predicted)) + '\n' )

f.close()