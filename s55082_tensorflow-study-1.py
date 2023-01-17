import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from sklearn import datasets, metrics, cross_validation

import tensorflow as tf

import math

data = pd.read_csv('../input/train.csv')
#对年龄进行处理，对于年龄为空的情况，填充为均值

def PrepareAge(data):

    age = data['Age'].copy()

    mean_age = age.mean()

    print('Mean age: {}'.format(mean_age))

    age[age.isnull()] = mean_age

    return age

#替换字符型数据，转为哑变量

def PrepareCat(data, feature):

    f = data[feature]

    classes = sorted(x for x in set(f) if type(x) != float or not math.isnan(x))

    print(classes)

    if len(classes) <= 2:

        out = pd.DataFrame({feature:(f == classes[0]).map({True:1, False:0})})

        return out

    ones = np.ones(len(f))

    out = pd.DataFrame({('{}_{}'.format(feature, c)):(f == c).map({True:1, False:0}) for c in classes})

    return out



#合并两份数据

def Merge(d1, d2):

    output = pd.DataFrame()

    for f in d1:

        output[f] = d1[f].copy()

    for f in d2:

        output[f] = d2[f].copy()

    return output
#构造交叉变量

def CrossCat(data, features):

    name = '*'.join(features)

    p = None

    for f in features:

        s = pd.Series(str(x) for x in data[f])

        p = s if p is None else (p + '*' + s)

        p[data[f].isnull()] = math.nan

    out = pd.DataFrame()

    out[name] = p

    return PrepareCat(out, name)



def PrepareFeatures(data):

    output = pd.DataFrame()

    output['Age'] = PrepareAge(data) / 80

    output['Fare'] = data['Fare'] / 512

    output['Parch'] = data['Parch'] / 6

    output['SibSp'] = data['SibSp'] / 8

    output = Merge(output, PrepareCat(data, 'Sex'))

    output = Merge(output, PrepareCat(data, 'Pclass'))

    output = Merge(output, PrepareCat(data, 'Embarked'))

    output['Sex_Age'] = output['Age']*output['Sex']

    output['Sex_Fare'] = output['Fare']*output['Sex']

    output = Merge(output, CrossCat(data, ['Sex', 'Embarked']))

    output = Merge(output, CrossCat(data, ['Sex', 'Pclass']))

    return output



#labe转换为array形式

def PrepareTarget(data):

    return np.array(data.Survived, dtype='int8').reshape(-1, 1)



#训练数据预处理

training_data = PrepareFeatures(data)

target_training_data = PrepareTarget(data)

training_data = np.array(training_data, dtype='float32')
ITERATIONS = 40000

LEARNING_RATE = 1e-4

nodes = 20



def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)

 

def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)

# Let's train the model

feature_count = training_data.shape[1]

x = tf.placeholder('float', shape=[None, feature_count], name='x')

y_ = tf.placeholder('float', shape=[None, 1], name='y_')

print(x.get_shape())

w1 = weight_variable([feature_count, nodes])

b1 = bias_variable([nodes])

l1 = tf.nn.relu(tf.matmul(x, w1) + b1)



w2 = weight_variable([nodes, 1])

b2 = bias_variable([nodes])

l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)



w3 = weight_variable([nodes, 1])

b3 = bias_variable([1])

y = tf.nn.sigmoid(tf.matmul(l2, w3) + b3)





cross_entropy = -tf.reduce_mean(y_*tf.log(tf.maximum(0.00001, y)) + (1.0 - y_)*tf.log(tf.maximum(0.00001, 1.0-y)))

reg = 0.01 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2)))

#设置预测准确率，修改这个地方

predict = (y > 0.5)

correct_prediction = tf.equal(predict, (y_ > 0.5))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy + reg)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)



for i in range(ITERATIONS):

    feed={x:training_data, y_:target_training_data}

    sess.run(train_step, feed_dict=feed)

    if i % 1000 == 0 or i == ITERATIONS-1:

        print('{} {} {:.2f}%'.format(i, sess.run(cross_entropy, feed_dict=feed), sess.run(accuracy, feed_dict=feed)*100.0))
test_data = pd.read_csv('../input/test.csv')

test_features = PrepareFeatures(test_data)

#predicted = sess.run(predict, feed_dict={x:test_features})

predicted = sess.run(y, feed_dict={x:test_features})
predicted
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