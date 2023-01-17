import pandas as pd
import json
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

color = sns.color_palette()

%matplotlib inline
df=pd.read_json(open('../input/train.json','r'))
print(df.shape)

df.head()

int_level = df['interest_level'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Interest level', fontsize=12)
plt.show()
cnt_srs = df['bathrooms'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('bathrooms', fontsize=12)
plt.show()
cnt_srs = df['bedrooms'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('bedrooms', fontsize=12)
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(x='bedrooms', hue='interest_level', data=df)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('bedrooms', fontsize=12)
plt.show()
ulimit = np.percentile(df.price.values, 99)
df['price'].loc[df['price']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(df.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()
def feat_engg(x_df):

  x_df['num_photos']=x_df['photos'].apply(len)
  x_df["num_features"] = x_df["features"].apply(len)
  x_df["num_description_words"] = x_df["description"].apply(lambda x: len(x.split(" ")))
  x_df["created"] = pd.to_datetime(x_df["created"])
  x_df["created_year"] = x_df["created"].dt.year
  x_df["created_month"] = x_df["created"].dt.month
  x_df["created_day"] = x_df["created"].dt.day
  
  features = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
  
  return x_df[features]


input_data=feat_engg(df)
input_labels=df['interest_level']
cnt_srs = df['num_photos'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.xlabel('Number of Photos', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
df['num_photos'].loc[df['num_photos']>12] = 12
plt.figure(figsize=(12,6))
sns.violinplot(x="num_photos", y="interest_level", data=df, order =['low','medium','high'])
plt.xlabel('Number of Photos', fontsize=12)
plt.ylabel('Interest Level', fontsize=12)
plt.show()
# number of features variable and its distribution

df["num_features"] = df['features'].apply(len)
cnt_srs = df['num_features'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of features', fontsize=12)
plt.show()
input_data.head()
input_labels.head()
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transformed_labels = encoder.fit_transform(input_labels)
print(transformed_labels) #notice that first column stands for 'high', second column for 'low' and third column for 'medium'
X_train, X_val, y_train, y_val = train_test_split(input_data, transformed_labels, test_size=0.33)

y_train.shape
X_train.shape
import tensorflow as tf
sess=tf.InteractiveSession()

n_hidden_1=60
n_hidden_2=60
n_hidden_3=60
n_hidden_4=60

x=tf.placeholder(tf.float32,shape=[None,11])
y_= tf.placeholder(tf.float32, shape=[None, 3])


w=tf.Variable(tf.zeros([11,3]))
b=tf.Variable(tf.zeros([3]))

def multilayer_perceptron(x,weights,biases):
    
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.sigmoid(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.sigmoid(layer_2)
    
    layer_3=tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3=tf.nn.sigmoid(layer_3)
    
    layer_4=tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
    layer_4=tf.nn.relu(layer_4)
    
    
    out_layer=tf.add(tf.matmul(layer_4,weights['out']),biases['out'])
    return out_layer

weights={
            'h1':tf.Variable(tf.truncated_normal([11,n_hidden_1])),
            'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
            'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
            'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
            'out':tf.Variable(tf.truncated_normal([n_hidden_4,3]))
}

biases={
            'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),
            'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
            'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
            'out':tf.Variable(tf.truncated_normal([3]))
}

y=multilayer_perceptron(x,weights,biases)
probs=tf.nn.softmax(y)  #to calculate probabilities of classes
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #ADAM Optimizer is used

sess.run(tf.global_variables_initializer())
mse_history=[]
cost_history=[]
accuracy_history=[]
training_epochs=1000

for epoch in range(training_epochs):
    sess.run(train_step,feed_dict={x:X_train,y_:y_train})
    cost=sess.run(cross_entropy,feed_dict={x:X_train,y_:y_train})
    cost_history=np.append(cost_history,cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    pred_y=sess.run(y,feed_dict={x:X_val})
    mse=tf.reduce_mean(tf.square(pred_y - y_val))
    mse_=sess.run(mse)
    mse_history.append(mse_)
    accuracy=sess.run(accuracy,feed_dict={x:X_train,y_:y_train})
    accuracy_history.append(accuracy)
    
    print('epoch:',epoch,'-','cost:',cost,'-MSE:',mse_,'-train accuracy:',accuracy)
correct_predictioncorrect_ =tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Validation accuracy:',sess.run(accuracy,feed_dict={x:X_val,y_:y_val}))
sns.distplot(accuracy_history)
test_df=pd.read_json(open('../input/test.json','r'))
test_df.head()
test_data=feat_engg(test_df)
test_data.head()
result=sess.run(probs,feed_dict={x:test_data}) #we have to submit probabilites of classes
result 
#in the above result array the respective columns represents ['high',low', 'medium'] but the desired is ['high','medium','low'] so we will swap the second and third columns

result[:,[1, 2]] =result[:,[2, 1]]
result

result.shape
labels2idx = {'high': 0, 'medium': 1, 'low': 2}

sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = result[:, labels2idx[label]]
sub.to_csv("mysubmission.csv", index=False)
sub