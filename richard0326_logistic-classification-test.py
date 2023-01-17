import pandas as pd
import tensorflow as tf

def load_file(path):
    data = pd.read_csv(path)
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Sex"] = data["Sex"].apply(lambda sex: 1 if sex == "male" else 0)
    
    data["Embarked"] = data["Embarked"].fillna("S")
    
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
    
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
    return data

train, test = load_file("../input/train.csv"), load_file("../input/test.csv")
train.head()
train_x_data = train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
print(train_x_data.shape, train_x_data.head())
test_x_data = test.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
print(test_x_data.shape, test_x_data.head())
train_y_data = pd.DataFrame(train.loc[:, "Survived"])
print(train_y_data.shape, train_y_data.head())
print(train_x_data.isnull().sum())
print(train_y_data.isnull().sum())
print(test_x_data.isnull().sum())
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([7, 1], mean=0, stddev=0.001), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
sess = tf.Session()
# Initialize TensorFlow variables
sess.run(tf.global_variables_initializer())

feed = {X:train_x_data, Y:train_y_data}
for step in range(10001):
    sess.run(train, feed_dict=feed)
    if step % 100 == 0:
        print(step, 'cost : ', sess.run(cost, feed_dict=feed))
test_y = sess.run(hypothesis, feed_dict={X:test_x_data})

### 0.5 기준으로 이하일 경우 사망, 이상일 경우 생존
prediction = list()
for i in range(len(test_y)):
    ans =1
    if test_y[i] <= 0.5:
        ans = 0
    prediction.extend([ans])

print(prediction)
index = list()
for i in range(len(test_y)):
    index.extend([i+892])
    
output = pd.DataFrame({
         "PassengerId": index,
         "Survived": prediction
    })
output.to_csv("logistic_regression.csv", index=False)