# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
colors = 10*['g','r','c','b','k']
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#reading in testing and training data
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv('../input/test.csv')
training_data = training_data.dropna()
#testing_data = testing_data.dropna()
#testing_data.dropna(axis = 0, how='any')
#Print out data labels if needed
#print(list(testing_data.columns.values), list(training_data.columns.values))
# Any results you write to the current directory are saved as output.

#Split training and testing data into labels and features
def split_to_labels(data):
    data_ids = data['PassengerId']
    labels = data['Survived']
    #removing data that Sklearn cannot handle
    features = data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
    return data_ids, labels, features

train_ID, labels_train, features_train = split_to_labels(training_data)
#Function is not used for test data because there are no labels for the test data
test_ID = testing_data['PassengerId']
features_test = testing_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
#Converting Categorical Data to Numeric Data
#Encoding Female to 0 Male to 1
features_train['Sex'] = pd.get_dummies(features_train['Sex'])

features_train['Embarked'] = pd.get_dummies(features_train['Embarked'])
features_test['Sex'] = pd.get_dummies(features_test['Sex'])

features_test['Embarked'] = pd.get_dummies(features_test['Embarked'])

for column_name in list(features_test.columns.values):
    average = np.mean(features_test[column_name])
    #features_test[column_name]=features_test[column_name].fillna(average)

'''clf = svm.SVC()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)

submission_csv = pd.DataFrame()
submission_csv['PassengerId'] = test_ID
submission_csv['Survived'] = predictions

submission_csv.to_csv()'''
#Mean Shift Part 41
class Mean_Shift:
    def __init__(selfd, radius=4):
        self.bandwidth = bandwidth
        
    def fit(self,data):
        centroids = {}
        
        for i in range(len(data)):
            centroids[i] = data[i]
            
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.lialg.norm(featureset-centroid) < self.radius:
                        in_badwidth.append(featureset)
                        
                new_centroid = np.average(in_bandwidth, axis = 0)
                new_centroids.append(tuple(new_centroid))
                
            uniques = sorted(list(set(new_centroids)))
            
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import  MeanShift
from sklearn import preprocessing
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

def df_cleaning(df):
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    #print(df.head())

    def handle_non_numerical_data(df):
        columns = df.columns.values

        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]

            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1

                df[column] = list(map(convert_to_int, df[column]))

        return df

    df = handle_non_numerical_data(df)
    return df
training_data = df_cleaning(training_data)
testing_data = df_cleaning(testing_data)


### In Progre
clf = MeanShift()
clf.fit(features_train)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
labels = clf.labels_


labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
plt.scatter(cluster_centers[:, 5],cluster_centers[:, 1],s=100,c='red')
plt.scatter(training_data.values[:, 5],training_data.values[:,9],s=20,c='blue')
plt.show()
