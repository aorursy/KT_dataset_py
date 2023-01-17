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



import tensorflow as tf



## Load Data



data = pd.read_csv("../input/loan.csv", low_memory=False)



#data.info()

#data.shape



## Clean data.



clean_data = data.dropna(thresh=len(data),axis=1)

#clean_data.shape

list(clean_data)
#clean_data.loan_status.str.contains("Fully Paid").astype(int)



#clean_data.loan_status[clean_data.loan_status.str.contains("Fully Paid") == True] = 1

#clean_data.loan_status[clean_data.loan_status.str.contains("Fully Paid") == False] = 0
## Remove data that does not meet the credit policy.

clean_data = clean_data[clean_data.loan_status.str.contains("Does not meet the credit policy") == False]



#clean_data.loan_status[clean_data.loan_status.str.contains("Fully Paid")].astype(int)

clean_data.loan_status[clean_data.loan_status.str.contains("Fully Paid") == True] = 1

clean_data.loan_status[clean_data.loan_status.str.contains("Fully Paid") == False] = 0
clean_data.loan_status.unique()



clean_data.shape



clean_data_orig = clean_data.copy()



list(clean_data)
## Split Data

ratio = 0.7

msk = np.random.rand(len(clean_data)) < ratio

train_data = clean_data[msk]

test_data = clean_data[~msk]



## Use loan status as label for loan defaulters

y_label['loan_status'] = clean_data['loan_status'][msk]

y_test_label['loan_status'] = clean_data['loan_status'][~msk]



train_data = train_data.select_dtypes(exclude=[np.object])

test_data = test_data.select_dtypes(exclude=[np.object])





len(train_data)

len(test_data)



#train_data['loan_amnt'].hist()
##Vizualization



import matplotlib.pyplot as plt



#train_data.plot()



#plt.figure(); train_data.plot(); plt.legend(loc='best')
#y_label[y_label.str.contains("Does not") == True].size
list(train_data)



#train_data.drop('id', axis=1, inplace=True)

#train_data.drop('member_id', axis=1, inplace=True)

train_data.drop('funded_amnt_inv', axis=1, inplace=True)

#train_data.drop('url', axis=1, inplace=True)

#train_data.drop('loan_status', axis=1, inplace=True)

#train_data.drop('application_type', axis=1, inplace=True)





#test_data.drop('id', axis=1, inplace=True)

#test_data.drop('member_id', axis=1, inplace=True)

test_data.drop('funded_amnt_inv', axis=1, inplace=True)

#test_data.drop('url', axis=1, inplace=True)

#test_data.drop('loan_status', axis=1, inplace=True)

#test_data.drop('application_type', axis=1, inplace=True)

train_data.shape
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
unique, counts = np.unique(msk, return_counts=True)

counts
y_label.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(train_data, y_label)

Y_pred = logreg.predict(test_data)

acc_log = round(logreg.score(train_data, y_label) * 100, 2)

acc_log

train_data.info()
import numpy as np

def get_series_ids(x):

    '''Function returns a pandas series consisting of ids, 

       corresponding to objects in input pandas series x

       Example: 

       get_series_ids(pd.Series(['a','a','b','b','c'])) 

       returns Series([0,0,1,1,2], dtype=int)'''



    values = np.unique(x)

    values2nums = dict(zip(values,range(len(values))))

    return x.replace(values2nums)
x = tf.placeholder(tf.float32, shape=[len(train_data), None])

y = tf.placeholder(tf.float32, shape=[None, 2])



W = tf.Variable(tf.zeros([len(train_data),2]))

b = tf.Variable(tf.zeros([2]))

learning_rate = 0.01

training_epochs = 25

batch_size = 100

display_step = 1



pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax



# Minimize error using cross entropy

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



# Initializing the variables

init = tf.global_variables_initializer()



# Launch the graph

with tf.Session() as sess:

    sess.run(init)



    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        total_batch = len(train_data)

        # Loop over all batches

        for i in range(total_batch):

            batch_xs = train_data

            batch_ys = y_label

            # Run optimization op (backprop) and cost op (to get loss value)

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,

                                                          y: batch_ys})

            # Compute average loss

            avg_cost += c / total_batch

        # Display logs per epoch step

        if (epoch+1) % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))



    print("Optimization Finished!")



    # Test model

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy:", accuracy.eval({x: test_data, y: y_test_labels}))