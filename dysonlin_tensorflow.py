# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import pylab as P

import tensorflow as tf



data = pd.read_csv('../input/train.csv')



print(data.describe(include = 'all'))



print(data.dtypes)



print(data.info())


def slice_by_survived(f):

    survived_feature = data['Survived']

    alive = data[(survived_feature == 1) & (f.isnull() == False)][feature]

    dead = data[(survived_feature == 0) & (f.isnull() == False)][feature]



    return np.array(alive, dtype='float'), np.array(dead, dtype='float')

    

numerical_features = [

    ['Age'],

    ['Fare'],

    ['Parch'],

    ['SibSp'],

]

    

for features in numerical_features:

    f = None

    for feature in features:

        if f is None:

            f = data[feature]

        else:

            f *= data[feature]

    alive, dead = slice_by_survived(f)

    plt.figure()

    plt.title('X'.join(features))

    plt.bar([0, 1], [alive.mean(), dead.mean()], color=['blue', 'red'])

    plt.figure()
for feature in numerical_features:

    data[feature].hist()

    P.show()
def CrossFeature(features):

    f = None

    d = data[features + ['Survived']].dropna()

    out = pd.DataFrame()

    for feature in features:

        s = pd.Series(str(x) for x in data[feature])

        if f is None:

            f = s

        else:

            f += s

    name = 'X'.join(features)

    out['Survived'] = d['Survived']

    out[name] = f

    return out[name], out['Survived']



categorical_features = [

    ['Pclass'],

    ['Sex'],

    ['Embarked'],

    ['Pclass', 'Sex'],

]



for features in categorical_features:

    print(features)

    f, surv = CrossFeature(features)

    classes = list(x for x in set(f) if type(x) != float or not math.isnan(x))

    distribution = [(f == c).sum() for c in classes]

    outcome = [(len(f[(f == c) & (surv == 1)]),

                len(f[(f == c) & (surv == 0)])) for c in classes]

    positions = []

    label_positions = []

    offset = 0

    heights = []

    color = []

    for out in outcome:

        positions.append(0.2 + offset)

        positions.append(1.0 + offset)

        color += ['b', 'r']

        heights.extend(out)

        label_positions.append(1 + offset)

        offset += 2

    

    fig, ax = plt.subplots()

    ax.set_title('x'.join(features))

    rects = ax.bar(positions, heights, color=color)

    ax.set_xticks(label_positions)

    ax.set_xticklabels(classes)
# Prepare Age feature

def PrepareAge(data):

    age = data['Age'].copy()

    mean_age = age.mean()

    print('Mean age: {}'.format(mean_age))

    age[age.isnull()] = mean_age

    return age



# Prepare categorical feature

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



def Merge(d1, d2):

    output = pd.DataFrame()

    for f in d1:

        output[f] = d1[f].copy()

    for f in d2:

        output[f] = d2[f].copy()

    return output

      



print(PrepareCat(data, 'Embarked').describe())
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



def PrepareTarget(data):

    return np.array(data.Survived, dtype='int8').reshape(-1, 1)



training_data = PrepareFeatures(data)



target_training_data = PrepareTarget(data)



print(training_data[training_data['Age'].isnull()])



print(training_data.describe())

print(training_data.info())



training_data = np.array(training_data, dtype='float32')

print (training_data)
ITERATIONS = 40000

LEARNING_RATE = 1e-4



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



nodes = 20



w1 = weight_variable([feature_count, nodes])

b1 = bias_variable([nodes])

l1 = tf.nn.relu(tf.matmul(x, w1) + b1)



w2 = weight_variable([nodes, 1])

b2 = bias_variable([1])

y = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)



cross_entropy = -tf.reduce_mean(y_*tf.log(tf.maximum(0.00001, y)) + (1.0 - y_)*tf.log(tf.maximum(0.00001, 1.0-y)))

reg = 0.01 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2)))



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



print(test_data.describe(include='all'))



test_features = PrepareFeatures(test_data)



print(test_features)



predicted = sess.run(predict, feed_dict={x:test_features})



# Write data



sol = pd.DataFrame()

sol['PassengerId'] = test_data['PassengerId']

sol['Survived'] = pd.Series(predicted.reshape(-1)).map({True:1, False:0})

print(sol)

sol.to_csv('solution3.csv', index=False)