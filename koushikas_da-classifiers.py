# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
import tensorflow as tf

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
Accuracy={}

CSV_COLUMN_NAMES = ['Date','Time','NO2(GT)','T','RH','AH','Weekday','hour']
train_path="../input/train.csv"
label_name='NO2(GT)'
train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0,  # ignore the first row of the CSV file.
                        skipinitialspace=True,
                        #skiprows=1
                       )
    
train.pop('Time')
train.pop('Date')
train.pop('T')
train['hour'] = train['hour'].astype(str)
train.Weekday= train.Weekday.astype(str)
# train['T']= train['T'].astype(float)
train_features, train_label = train, train.pop(label_name)

print(train_features.head())
CSV_COLUMN_NAMES = ['Date','Time','NO2(GT)','T','RH','AH','Weekday','hour']
test_path="../input/test.csv"
label_name='NO2(GT)'
test = pd.read_csv(filepath_or_buffer=test_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0,  # ignore the first row of the CSV file.
                        skipinitialspace=True,
                        #skiprows=1
                       )
    
test.pop('Time')
test.pop('Date')
test.pop('T')
test['hour'] = test['hour'].astype(str)
test.Weekday= test.Weekday.astype(str)
# test['T']= test['T'].astype(float)
test_features, test_label = test, test.pop(label_name)


clf = DecisionTreeClassifier()
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of Decison tree")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'DecisionTree':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
clf = RandomForestClassifier()
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of Random Forest")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'RandomForest':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
clf = svm.SVC(gamma='auto')
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of SVM Classifier")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'SVM':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
clf = GaussianNB()
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of Naive Bayes")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'Naive_Bayes':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
clf = AdaBoostClassifier()
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of AdaBoost")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'AdaBoost':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
clf = GradientBoostingClassifier()
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of GradientBoosting")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'GradientBoosting':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
clf = KMeans(n_clusters=2)
clf = clf.fit(train_features, train_label)
predict_label=clf.predict(test_features)
print("Results of Kmeans Clustering")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'Kmeans_Clustering':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
#creating normal features and crossed features for the nn model

Weekday = tf.feature_column.categorical_column_with_vocabulary_list(
    'Weekday', ['0', '1', '2', '3', '4','5','6'])
hour = tf.feature_column.categorical_column_with_vocabulary_list(
    'hour', [ '0', '1', '2', '3', '4','5','6','7','8','10','12','13','14','15','16','17','18','19','20','21','22','23'])
# T = tf.feature_column.numeric_column(key='T',dtype=tf.float64)
RH = tf.feature_column.numeric_column(key='RH',dtype=tf.float64)
AH = tf.feature_column.numeric_column(key='AH',dtype=tf.float64)

base_columns = [
    tf.feature_column.indicator_column(Weekday),
    tf.feature_column.indicator_column(hour),
#     T,
    RH,
    AH
]

Weekday_x_hour = tf.feature_column.crossed_column(
    ['Weekday', 'hour'], hash_bucket_size=1000)

crossed_columns = [
     tf.feature_column.indicator_column(Weekday_x_hour)
]

#running Dnn classifer model with the features designed above and with and input layer with 4 nodes
classifier = tf.estimator.DNNClassifier(feature_columns=base_columns+crossed_columns,hidden_units=[5],n_classes=2)
#training the nnmodel

def train_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        
        dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
        
        return dataset.make_one_shot_iterator().get_next()


classifier.train(
        input_fn=lambda:train_input_fn(train_features, train_label, 50 ),
        steps=1000)
#eval_input_fn() is similar to train_input_fn()  it helps in shuffling data and providing input as batches 
def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


test_df=pd.read_csv("../input/test.csv",parse_dates=True)

predict_x= { 
#     'T':[],
    'Weekday':[],
    'hour':[ ],
    'RH':[ ],
    'AH':[ ],
}


# predict_x['T'] = test_df['T'].astype(float)
predict_x['RH'] = test_df['RH'].astype(float)
predict_x['AH'] = test_df['AH'].astype(float)
predict_x['hour']= test_df.hour.astype(str)
predict_x['Weekday']= test_df.Weekday.astype(str)

#predicting using classifier. 
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x,
                                  labels=None,
                                  batch_size=50))

i=0
predict_label=[]
for pred_dict in zip(predictions):
    if pred_dict[0]['classes'] == [b'1']:      
        predict_label.append(1)
        
    else:
        predict_label.append(0)
    i=i+1


print("Results of Neural Network")
print("Confusion Matrix: ")
print(confusion_matrix( test_label, predict_label)) 
      
print ("Accuracy : ")
print(accuracy_score(test_label,predict_label)*100) 
Accuracy.update({'DNN':accuracy_score(test_label,predict_label)*100})

print("Report : ")
print(classification_report(test_label, predict_label)) 
y_pos = np.arange(len(Accuracy.keys()))
plt.barh(y_pos,list(Accuracy.values()), align='center', alpha=0.5)
plt.yticks(y_pos, list(Accuracy.keys()))

plt.xlabel('Accuracy')
plt.title('Classifiers Comparsion')
plt.xlim(0, 100) 
plt.show()

"""
1) All ensemble learning methods (such as AdaBoost, Gradient Boost, Random forest) have performed better 
with an accuracy around 80%. With Gradient boost being the best with accuracy of 79.00326%. 

2) K Mean clustering with only 2  clusters has an accuracy of  42.40196%. The reason for low accuracy 
could be explained by “Curse of Increase Dimensionality”. Since KNN performs poorly as no of input 
fields increases, five input features  has decreased the accuracy and accuracy  continues to decreases
if we increase the no of clusters.


"""