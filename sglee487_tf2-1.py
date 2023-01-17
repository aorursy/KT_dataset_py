import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print (os.getcwd())
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt;

import seaborn as sns

import time

import os

from IPython.display import clear_output

%matplotlib inline
import tensorflow as tf



print('tensorflow version: {}'.format(tf.__version__))

print('GPU 사용 가능 여부: {}'.format(tf.test.is_gpu_available()))

print(tf.config.list_physical_devices('GPU'))
train_data = pd.read_csv('../input/titanic/train.csv')
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train_data.isnull().sum().sort_values(ascending=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train_data,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train_data,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train_data,palette='rainbow')
train_data['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train_data)
sns.countplot(x='Parch',data=train_data)
train_data['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train_data,palette='winter')
train_data[["Pclass","Age"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Age", ascending=False)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 30



        else:

            return 25



    else:

        return Age
train_data['Age'] = train_data[['Age', 'Pclass']].apply(impute_age,axis=1)
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Embarked',data=train_data)
train_data['Embarked'] = train_data['Embarked'].fillna('S')
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train_data.isnull().sum().sort_values(ascending=False)
train_data.drop(["Name"], axis=1)

train_data['Sex'] = train_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_data = train_data.drop(["Name","Ticket","Cabin"],axis=1)
train_data.info()
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} ).astype(int)
train_data.head()
def make_dataset(pandas_data):

    

    # 입력, 타겟(라벨) 데이터 생성

    

    sequences = pandas_data.drop(["PassengerId", "Survived"], axis=1)

    labels = pandas_data["Survived"]

    

    return np.array(sequences), np.array(labels)
train_sequences, train_labels = make_dataset(train_data)
print(train_sequences)

print(len(train_sequences))
print(train_sequences, train_sequences.shape, len(train_sequences))

print(train_labels.shape)
batch_size = 6

max_epochs = 130

learning_rate = 0.00001

num_classes = 2
BUFFER_SIZE = len(train_sequences)

train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))

train_dataset = train_dataset.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.batch(128)
print(train_dataset)
class DNNModel(tf.keras.Model):

    def __init__(self, num_classes):

        super(DNNModel, self).__init__()

        ## 코드 시작 ##

        self.dense1 = tf.keras.layers.Dense(512, activation='relu')

        self.bn1 = tf.keras.layers.BatchNormalization()

        # self.relu1 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.relu1 = tf.keras.layers.ReLU()

        self.dense2 = tf.keras.layers.Dense(200, activation='relu')

        self.bn2 = tf.keras.layers.BatchNormalization()

        self.relu2 = tf.keras.layers.ReLU()

        self.dense3 = tf.keras.layers.Dense(num_classes)

        ## 코드 종료 ##



    def call(self, inputs, training=False):

        """Run the model."""

        ## 코드 시작 ##

        dense1_out = self.dense1(inputs)

        bn1_out = self.bn1(dense1_out, training=training)

        relu1_out = self.relu1(bn1_out)

        dense2_out = self.dense2(relu1_out)

        bn2_out = self.bn2(dense2_out, training=training)

        relu2_out = self.relu2(bn2_out)

        dense3_out = self.dense3(relu2_out)

        ## 코드 종료 ##

        return dense3_out
model = DNNModel(num_classes)
for person, labels in train_dataset.take(1):

    print(person[0:3])

    print("predictions: ", model(person[0:3]))

    

model.summary()
loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
mean_accuracy = tf.keras.metrics.Accuracy("accuracy")

mean_loss = tf.keras.metrics.Mean("loss")
mean_accuracy_ex = tf.keras.metrics.Accuracy("example_accuracy")

print(mean_accuracy_ex([0, 1, 2, 4], [0, 1, 3, 4])) # 0.75 = 3/4

print(mean_accuracy_ex([5, 1, 7, 3], [5, 0, 2, 7])) # 0.5 = 4/8 위의 계산에서 부터 누적
mean_ex = tf.keras.metrics.Mean("example_mean")

# mean(values) 형태로 사용합니다.

print(mean_ex([0, 1, 2, 3])) # 1.5 = (0 + 1 + 2 + 3) / 4

print(mean_ex([4, 5])) # 2.5 = (0 + 1 + 2 + 3 + 4 + 5) / 6
num_batches_per_epoch = len(list(train_dataset))

for epoch in range(max_epochs):

    for step, (persons, labels) in enumerate(train_dataset):

        start_time = time.time()



        with tf.GradientTape() as tape:

            ## 코드 시작 ##

            predictions = model(persons)    # 위의 설명 1. 을 참고하여 None을 채우세요.

            labels_onehot = tf.one_hot(labels,num_classes)

            loss_value = loss_object(y_true=labels_onehot,y_pred=predictions)     # 위의 설명 2. 를 참고하여 None을 채우세요.



        gradients = tape.gradient(loss_value, model.trainable_variables)                   # 위의 설명 3. 을 참고하여 None을 채우세요.

        optimizer.apply_gradients(zip(gradients,model.trainable_variables))    # 위의 설명 4. 를 참고하여 None을 채우세요.

        ## 코드 종료 ##

        

        mean_loss(loss_value)

        mean_accuracy(labels, tf.argmax(predictions, axis=1))

        

        if (step+1) % 5 == 0:

#             clear_output(wait=True)

            epochs = epoch + step / float(num_batches_per_epoch)

            duration = time.time() - start_time

            examples_per_sec = batch_size / float(duration)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, ({:.2f} examples/sec; {:.3f} sec/batch)'.format(

                epoch+1, max_epochs, step+1, num_batches_per_epoch, mean_loss.result(), 

                mean_accuracy.result() * 100, examples_per_sec, duration))



    # clear the history

    mean_loss.reset_states()

    mean_accuracy.reset_states()



print("training done!")
test_batch_size = 25

batch_index = np.random.choice(

    len(train_data), size=test_batch_size, replace=False)



batch_xs = train_sequences[batch_index]

batch_ys = train_labels[batch_index]

y_pred_ = model(batch_xs, training=False)



print(batch_ys[:])

print(y_pred_[:,:])
def save_model(model, epoch, train_dir):

    model_name = 'my_model_' + str(epoch)

    model.save_weights(os.path.join(train_dir, model_name))
model.save_weights(os.path.join("./model_checkpoint/", "my_model_loss_0.3824"))
test_data = pd.read_csv('../input/titanic/test.csv')
test_data['Age'] = test_data[['Age', 'Pclass']].apply(impute_age,axis=1)
test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data.drop(["Name"], axis=1)

test_data['Sex'] = test_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_data = test_data.drop(["Name","Ticket","Cabin"],axis=1)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} ).astype(int)
test_data = test_data.drop(["PassengerId"], axis=1)
test_input = np.array(test_data)
test_result = model(test_input)
test_result_merge = np.zeros((1309-892+1,2))
for i,j in zip(range(0,(1309-892+1)),range(892,1309+1)):

    test_result_merge[i,0] = j

    test_result_merge[i,1] = np.argmax(test_result[i])
test_result_merge = test_result_merge.astype(int)
test_result_pandas = pd.DataFrame(test_result_merge, columns=["PassengerId","Survived"])
print(test_result_pandas)
test_result_pandas.to_csv('predictions_loss_03769.csv' , index=False)