import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv", index_col="ID")
df
# Kategorik veriler için kurallar

categoric_attribute_information = {

    "SEX": [1,2],

    "EDUCATION": [1, 2, 3, 4],

    "MARRIAGE": [1, 2, 3]

}



for column in categoric_attribute_information.keys():

    unique_values = list(set(df[column].tolist()))

    if unique_values == categoric_attribute_information[column]:

        print("DOĞRU", column, categoric_attribute_information[column], "==", unique_values, "\n")

    else:

        print("HATA!", column, categoric_attribute_information[column], "!=", unique_values, "\n")
print("En küçük yaş:", min(df["AGE"].tolist()), "\n")

print("En büyük yaş:", max(df["AGE"].tolist()), "\n")
columns_starts_with_pay = [column for column in df.columns if column[:4] == "PAY_" and len(column) == 5]



for column in columns_starts_with_pay:

    unique_values = list(set(df[column].tolist()))

    for value in unique_values:

        if not ((value == -1) or (value > 0)):

            print("HATA!", column, "=", value)
def combine_categories(x):

    if x in [4, 5, 6]:

        return 0

    else:

        return x



df["EDUCATION"] = df["EDUCATION"].apply(

    lambda x: combine_categories(x)

)
list(set(df["EDUCATION"].tolist()))
df.info()
# Eğitim seviyesi kategorileri

education_categories = {

    0: "others",

    1: "graduate_student",

    2: "university",

    3: "high_school",

}

# Medeni durum kategorileri

# Bu kategoriler yukarıda linkini verdiğim kaynağa göre veri sağlayıcısı tarafından sonradan düzeltilen kategorilerdir

marriage_categories = {

    0: "others",

    1: "married",

    2: "single",

    3: "divorce"

}



for category in education_categories.keys():

    df["EDUCATION"][df["EDUCATION"] == category] = education_categories[category]

    

for category in marriage_categories:

    df["MARRIAGE"][df["MARRIAGE"] == category] = marriage_categories[category]

    

df
# One Hot Encoder

categoric_variables = df.select_dtypes(

    include=[np.object]

).columns.tolist()

df = pd.get_dummies(df, prefix=categoric_variables)



# Hedef değişkeni sayısaldan kategorik tipe dönüştürüldü

df["default.payment.next.month"][df["default.payment.next.month"] == 1] = "Yes"

df["default.payment.next.month"][df["default.payment.next.month"] == 0] = "No"



df
# 0 - 1 Normalizasyon

numeric_variables = df.select_dtypes(

    exclude=[np.object]

).columns.tolist()

for variable in numeric_variables:

    data = list(set(df[variable].tolist()))

    min_value = min(data)

    max_value = max(data)

    df[variable] = df[variable].apply(

        lambda x: (x-min_value)/(max_value-min_value)

    )

    

df
count_yes = df["default.payment.next.month"].tolist().count("Yes")

count_no = df["default.payment.next.month"].tolist().count("No")

print("Yes sayısı:", count_yes)

print(" No sayısı:", count_no)

print(" Yes oranı:", count_yes / (count_yes+count_no))
df = df.rename(columns={"default.payment.next.month": "target"})



X = df.drop(columns=["target"]).values

y = df.filter(["target"])

y = pd.get_dummies(y, prefix=["target"]).values



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42

)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
split = int(len(y_test)/2)

inputX = X_train

inputY = y_train

inputX_valid = X_test[:split]

inputY_valid = y_test[:split]

inputX_test = X_test[split:]

inputY_test = y_test[split:]
input_nodes = inputX.shape[1]



multiplier = 3



hidden_nodes1 = input_nodes

hidden_nodes2 = round(hidden_nodes1 * multiplier)

hidden_nodes3 = round(hidden_nodes2 * multiplier)



pkeep = tf.placeholder(tf.float32)



# Input

x = tf.placeholder(tf.float32, [None, input_nodes])



# Layer 1

W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = 0.15))

b1 = tf.Variable(tf.zeros([hidden_nodes1]))

y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)



# Layer 2

W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = 0.15))

b2 = tf.Variable(tf.zeros([hidden_nodes2]))

y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)



# Layer 3

W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev = 0.15)) 

b3 = tf.Variable(tf.zeros([hidden_nodes3]))

y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

y3 = tf.nn.dropout(y3, pkeep)



# Layer 4

W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev = 0.15)) 

b4 = tf.Variable(tf.zeros([2]))

y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)



# Output

y = y4

y_ = tf.placeholder(tf.float32, [None, 2])



training_epochs = 100

training_dropout = 0.9

display_step = 10

n_samples = y_train.shape[0]

batch_size = 2048

learning_rate = 0.01



# Cross entropy

cost = -tf.reduce_sum(y_ * tf.log(y))



optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



accuracy_summary = []

cost_summary = []

valid_accuracy_summary = [] 

valid_cost_summary = [] 

stop_early = 0
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    

    for epoch in range(training_epochs):

        for batch in range(int(n_samples/batch_size)):

            batch_x = inputX[batch*batch_size : (1+batch)*batch_size]

            batch_y = inputY[batch*batch_size : (1+batch)*batch_size]

            

            sess.run([optimizer], feed_dict={

                x: batch_x,

                y_: batch_y,

                pkeep: training_dropout

            })

            

        if (epoch) % display_step == 0:

            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={

                x: inputX,

                y_: inputY,

                pkeep: training_dropout

            })

            

            valid_accuracy, valid_newCost = sess.run([accuracy, cost], feed_dict={

                x: inputX_valid,

                y_: inputY_valid,

                pkeep: 1

            })

            

            print(

                "Epoch:", epoch,

                "Acc =", "{:.5f}".format(train_accuracy),

                "Cost =", "{:.5f}".format(newCost),

                "Valid_Acc =", "{:.5f}".format(valid_accuracy),

                "Valid_Cost = ", "{:.5f}".format(valid_newCost)

            )

            

            accuracy_summary.append(train_accuracy)

            cost_summary.append(newCost)

            valid_accuracy_summary.append(valid_accuracy)

            valid_cost_summary.append(valid_newCost)

            

            if valid_accuracy < max(valid_accuracy_summary) and epoch > 100:

                stop_early += 1

                if stop_early == 15:

                    break

            else:

                stop_early = 0

                

    print("Optimization Finished!")
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))



axes[0].set_ylabel("Cost", fontsize=14)

axes[0].plot(cost_summary, color='blue')

axes[0].plot(valid_cost_summary, color='green')



axes[1].set_ylabel("Accuracy", fontsize=14)

axes[1].set_xlabel("Epoch", fontsize=14)

axes[1].plot(accuracy_summary, color='blue')

axes[1].plot(valid_accuracy_summary, color='green')

plt.show()