import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
# Loading datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/sample_submission.csv")
train.head()
# print "before",test.shape

test["comment_text"].fillna("fillanything").values

# print "after", test.shape



train_comment, test_comment = train["comment_text"], test["comment_text"]

allcomment = pd.concat([train_comment,test_comment])

# allcomment = allcomment["comment_text"].fillna("fillanything").values # filling if any value is left blank
col = np.array(train.columns)

y_lable = train[col[2:]].astype(np.int32)           # ground truth

print (y_lable.shape[1])
vectorizer = TfidfVectorizer(stop_words='english', lowercase = True, strip_accents='unicode', ngram_range=(1,3), encoding = 'utf-8', decode_error = 'strict', max_features = 1000)

vectorizer.fit_transform(allcomment)

# Once done save it into pickle format

# joblib.dump(vectorizing_all_comment, "comment_1000.pkl")

# vectorizing_all_comment = joblib.load("comment_1000.pkl")
trainVectFeatures = vectorizer.transform(train["comment_text"])

testVectFeatures = vectorizer.transform(test["comment_text"])
from scipy.sparse import csr_matrix

trainVectFeatures = trainVectFeatures.todense()
#trainVectFeatures= joblib.load("trainVectFeatures1000.pkl").astype(np.int64)

# joblib.load( "testVectFeatures1000.pkl").astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(trainVectFeatures, y_lable, test_size=0.2, random_state=42)
input_size = trainVectFeatures.shape[1]

output_size = y_train.shape[1] # 6

hidden_unit1 = 1100



X_input = tf.placeholder(tf.float32, [None, input_size] ) # input 1000 

y_output = tf.placeholder(tf.float32, [None, output_size] )  # y=6 



w1 = tf.Variable(tf.random_normal([input_size, hidden_unit1]), name = "Weights_1") #Initaliazing weight_1

b1 = tf.Variable(tf.random_normal([hidden_unit1]), name = "Bias_1")



w2 = tf.Variable(tf.random_normal([hidden_unit1, output_size]),  name = "Weights_2") #Initaliazing weight_2

b2 = tf.Variable(tf.random_normal([output_size]), name = "Bias_2")
hidden_layer = tf.nn.sigmoid(tf.matmul(X_input, w1)+b1) 

output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, w2) + b2)



losses = tf.losses.mean_squared_error(y_train, output_layer)

optimizer = tf.train.GradientDescentOptimizer(0.09).minimize(losses)



init = tf.global_variables_initializer()

sess = tf.InteractiveSession() 

sess.run(tf.global_variables_initializer())



step_size = 100

for step in range(step_size):  

    a,b,c,d = sess.run([hidden_layer,output_layer,losses,optimizer], feed_dict={X_input:X_train,y_output:y_train})



    if step%20==0:

        print ("losses after per 20 iteration: ",c)



correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(y_output,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



print ("Accuracy on the model: ",accuracy.eval(feed_dict={X_input:X_train, y_output:y_train}))