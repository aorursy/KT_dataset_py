# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sys

import pickle

sys.path.append("../tools/")

import pandas

import numpy as np

#from tester import dump_classifier_and_data

import matplotlib.pyplot as plt

import scipy

%matplotlib inline
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):

    return_list = []

    if isinstance(sort_keys, str):

        import pickle

        keys = pickle.load(open(sort_keys, "rb"))

    elif sort_keys:

        keys = sorted(dictionary.keys())

    else:

        keys = dictionary.keys()



    for key in keys:

        tmp_list = []

        for feature in features:

            try:

                dictionary[key][feature]

            except KeyError:

                print ("error: key ", feature, " not present")

                return

            value = dictionary[key][feature]

            if value=="NaN" and remove_NaN:

                value = 0

            tmp_list.append( float(value) )



        # Logic for deciding whether or not to add the data point.

        append = True

        # exclude 'poi' class as criteria.

        if features[0] == 'poi':

            test_list = tmp_list[1:]

        else:

            test_list = tmp_list

        ### if all features are zero and you want to remove

        ### data points that are all zero, do that here

        if remove_all_zeroes:

            append = False

            for item in test_list:

                if item != 0 and item != "NaN":

                    append = True

                    break

        ### if any features for a given data point are zero

        ### and you want to remove data points with any zeroes,

        ### handle that here

        if remove_any_zeroes:

            if 0 in test_list or "NaN" in test_list:

                append = False

        ### Append the data point if flagged for addition.

        if append:

            return_list.append( np.array(tmp_list) )



    return np.array(return_list)
def targetFeatureSplit( data ):

    target = []

    features = []

    for item in data:

        target.append( item[0] )

        features.append( item[1:] )



    return target, features
email_features_list=['from_messages',

    'from_poi_to_this_person',

    'from_this_person_to_poi',

    'shared_receipt_with_poi',

    'to_messages',

    ]



financial_features_list=['bonus',

    'deferral_payments',

    'deferred_income',

    'director_fees',

    'exercised_stock_options',

    'expenses',

    'loan_advances',

    'long_term_incentive',

    'other',

    'restricted_stock',

    'restricted_stock_deferred',

    'salary',

    'total_payments',

    'total_stock_value',

]



features_list = ['poi']+email_features_list + financial_features_list 

### Load the dictionary containing the dataset

import pickle

original = "/kaggle/input/enron-person-of-interest-dataset/final_project_dataset.pkl"

destination = "final_project_dataset_unix.pkl"



content = ''

outsize = 0

with open(original, 'rb') as infile:

    content = infile.read()

with open(destination, 'wb') as output:

    for line in content.splitlines():

        outsize += len(line) + 1

        output.write(line + str.encode('\n'))

        

data_dict = pickle.load(open("final_project_dataset_unix.pkl", "rb"))

data_dict
#Dataset exploration

print ('Exploratory Data Analysis')

data_dict.keys()

print ('Total number of data points= {0}'.format(len(data_dict.keys())))



count_poi=0

for name in data_dict.keys():

    if data_dict[name]['poi']==True:

        count_poi+=1



print ('Number of Persons of Interest: {0}'.format(count_poi))

print ('Number of Non-Person of Interest: {0}'.format(len(data_dict.keys())-count_poi))
all_features=data_dict['BAXTER JOHN C'].keys()

print ('Total Features everyone on the list has:', len(all_features))



missing={}

for feature in all_features:

    missing[feature]=0



for person in data_dict:

    records=0

    for feature in all_features:

        if data_dict[person][feature]=='NaN':

            missing[feature]+=1

        else:

            records+=1



print ('Number of Missing Values for each Feature:')

for feature in all_features:

    print (feature, missing[feature])
def PlotOutlier(data_dict, ax, ay):

    data = featureFormat(data_dict, [ax,ay,'poi'])

    for point in data:

        x = point[0]

        y = point[1]

        poi=point[2]

        if poi:

            color='blue'

        else:

            color='green'



        plt.scatter( x, y, color=color )

    plt.xlabel(ax)

    plt.ylabel(ay)

    plt.show()

PlotOutlier(data_dict, 'from_poi_to_this_person','from_this_person_to_poi')

PlotOutlier(data_dict, 'total_payments', 'total_stock_value')

PlotOutlier(data_dict, 'from_messages','to_messages')

PlotOutlier(data_dict, 'salary','bonus')





def remove_outliers(data_dict, outliers):

    for outlier in outliers:

        data_dict.pop(outlier, 0)

outliers =['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHARD EUGENE E']#latter two has almost all nan values

remove_outliers(data_dict, outliers)

data_dict
my_dataset = data_dict

def computeFraction( poi_messages, all_messages ):

    """ given a number messages to/from POI (numerator) 

        and number of all messages to/from a person (denominator),

        return the fraction of messages to/from that person

        that are from/to a POI

   """

    fraction = 0.

    if all_messages =='NaN':

        return fraction

    if poi_messages=='NaN':

        return fraction

        

    fraction=float(poi_messages)/float(all_messages)

    return fraction

submit_dict={}

for name in my_dataset:



    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]

    to_messages = data_point["to_messages"]

    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )

    data_point["fraction_from_poi"] = fraction_from_poi





    from_this_person_to_poi = data_point["from_this_person_to_poi"]

    from_messages = data_point["from_messages"]

    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,

                       "from_this_person_to_poi":fraction_to_poi}

    data_point["fraction_to_poi"] = fraction_to_poi

my_feature_list=features_list+['fraction_from_poi','fraction_to_poi']



for x in range(len(my_feature_list)): 

    print (my_feature_list[x])
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import cross_validate

from sklearn.svm import SVC





def getkbest(data_dict, features_list, k):

    data=featureFormat(my_dataset, features_list)

    labels, features = targetFeatureSplit(data)

    selection=SelectKBest(k=k).fit(features,labels)

    scores=selection.scores_

    unsorted_pairs = zip(features_list[1:], scores)

    sorted_pairs=list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))

    selection_best = dict(sorted_pairs[:k])

    return selection_best

num=12 

best_features = getkbest(my_dataset, my_feature_list, num)

print ('Selected features and their scores: ', best_features)

l=best_features

def getList(dict): 

    list = [] 

    for key in dict.keys(): 

        list.append(key) 

          

    return list

l=getList(l)      

my_feature_list = ['poi'] + l

#type(l)

print ("{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:]))
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

from sklearn import preprocessing

scaler=preprocessing.MinMaxScaler()

features=scaler.fit_transform(features)
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier



clf_d=Pipeline([

    ('standardscaler',StandardScaler()),

    ('pca',PCA()),

    ('clf_d',DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=2, min_samples_split=7, splitter='best',random_state=42))])



from sklearn.linear_model import LogisticRegression



clf_p=Pipeline([

    ('standardscaler', StandardScaler()),

    ('classifier', LogisticRegression(penalty='l2', tol=0.001, C=0.0000001, random_state=42))])



from sklearn.cluster import KMeans

clf_k=Pipeline([

    ('standardscaler',StandardScaler()),

    ('pca',PCA()),

    ('clf_k',KMeans(n_clusters=2, random_state=42, tol=0.001))])



from sklearn.svm import SVC

clf_s=Pipeline([

    ('standardscaler',StandardScaler()),

    ('pca',PCA()),

    ('clf_s',SVC(kernel='rbf',C = 1000,random_state = 42))])





from sklearn.naive_bayes import GaussianNB

clf_g=Pipeline(steps=[

    ('standardscaler',StandardScaler()),

    ('pca',PCA()),

    ('clf_g',GaussianNB())])



from sklearn.ensemble import RandomForestClassifier

clf_rf =Pipeline( [

    ('standardscaler',StandardScaler()),

    ('pca',PCA()),

    ('clf_rf',RandomForestClassifier())])
from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report,accuracy_score

def evaluate(clf, features, labels, num=1):

    print (clf)

    accuracy=[]

    precision=[]

    recall=[]

    for trial in range(num):

        features_train, features_test, labels_train, labels_test=train_test_split(features, labels, test_size=0.3, random_state=42)

        clf=clf.fit(features_train, labels_train)

        pred=clf.predict(features_test)

        accuracy.append(accuracy_score(labels_test,pred))

        precision.append(precision_score(labels_test, pred))

        recall.append(recall_score(labels_test, pred))

    print ('precision: {}'.format(np.mean(precision)))

    print ('recall: {}'.format(np.mean(recall)))

    print ('accuracy: {}'.format(np.mean(accuracy)))

    return np.mean(precision), np.mean(recall), confusion_matrix(labels_test, pred),classification_report(labels_test, pred)
print ('KMeans: ',evaluate(clf_k, features, labels))

print ('Gaussian: ',evaluate(clf_g, features, labels))

print ('Linear Regression: ', evaluate(clf_p, features, labels))
print ('Random Forest: ',evaluate(clf_rf, features, labels))
print ('SVC: ', evaluate(clf_s, features, labels))
print ('Decision Tree: ', evaluate(clf_d, features, labels))