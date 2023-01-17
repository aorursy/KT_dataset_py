# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
csv_path = '../input/all0813-edited.csv'
# Any results you write to the current directory are saved as output.
stateList=['-','AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
def raceNum(key):
    return ord(key[-1:])-ord('a')

def stateNum(key):
    return 0.02*stateList.index(key.upper());

pmt=['year','borninus','educ','gender','hhsize_at_a','hhsize_at_b','hhsize_at_c','hisp','married','state','ideo','party','church','relig','employ']
n_pmt=len(pmt)+1;
key=[];
cnt=[];
required_key=['jaq3','jaq5'];
feature=[];
label=[];
pmt_nm_loct=[1,8,10,11,13,14,15,17,19,310,330,335,342,414]#14
pmt_bias_loct=[-2007,1,0,0,1,1,1,1,1,0,0,0,0,-9]
#pmt_nmr=[1/6,1/2,1/9,1/2,1/100,1/100,1/100,1/2,1/10,1/9,1/99,1/9,1/9,1/13]
with open(csv_path) as f:
    reader=csv.reader(f);
    for i,row in enumerate(reader):
        if(i==0):
            key=row;
            cnt=[0]*len(row);
            #print(key)
            print([key[v] for v in pmt_nm_loct])
        else:
            if(row[24]==8):
                continue;
            x=[];
            y=[];
            for i,v in enumerate(pmt_nm_loct):
                x.append((int(row[v])+pmt_bias_loct[i]));
            x.append(stateNum(row[131]))
            tmp=(int(row[24])*1+int(row[25])*2+int(row[26])*3+int(row[27])*4+int(row[28])*5)
            x.append(tmp);
            if(row[214]!='' and row[216]!=''):
                y=[int(row[214]),int(row[216])]
                feature.append(x);
                label.append(y);
                #print(len(x),len(y))

features=np.array(feature);
labels=np.array(label)
print(features.shape,labels.shape)
Xtrain=np.array(feature[:2800])
Ytrain=np.array(label[:2800])
Xtest=np.array(feature[2800:])
Ytest=np.array(label[2800:])
numFeatures=Xtrain.shape[1]
numLabels=Ytrain.shape[1]
print(numFeatures, numLabels)
print(Xtrain[10:15],Ytrain[10:15])
numEpochs=500
learningRate=tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=Xtrain.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)
# X = X-matrix / feature-matrix / data-matrix... It's a tensor to hold our email
# data. 'None' here means that we can hold any number of emails
X = tf.placeholder(tf.float32, [None, numFeatures])
# yGold = Y-matrix / label-matrix / labels... This will be our correct answers
# matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here 
# means that we can hold any number of emails
yGold = tf.placeholder(tf.float32, [None, numLabels])
# Values are randomly sampled from a Gaussian with a standard deviation of:
#     sqrt(6 / (numInputNodes + numOutputNodes + 1))

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6/numFeatures+
                                                         numLabels+1)),
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6/numFeatures+numLabels+1)),
                                    name="bias"))
######################
### PREDICTION OPS ###
######################

# INITIALIZE our weights and biases
init_OP = tf.initialize_all_variables()

# PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
#####################
### EVALUATION OP ###
#####################

# COST FUNCTION i.e. MEAN SQUARED ERROR
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")
#######################
### OPTIMIZATION OP ###
#######################

# OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
###########################
### GRAPH LIVE UPDATING ###
###########################
import matplotlib.pyplot as plt
epoch_values=[]
accuracy_values=[]
cost_values=[]
# Turn on interactive plotting
plt.ion()
# Create the main, super plot
fig = plt.figure()
# Create two subplots on their own axes and give titles
ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()
#####################
### RUN THE GRAPH ###
#####################
import time
# Create a tensorflow session
sess = tf.Session()

# Initialize all tensorflow variables
sess.run(init_OP)

## Ops for vizualization
# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)
# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
# Merge all summaries
all_summary_OPS = tf.summary.merge_all()
# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

# Initialize reporting variables
cost = 0
diff = 1
print("Epochs: ",numEpochs)
# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: Xtrain, yGold: Ytrain})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP], 
                feed_dict={X: Xtrain, yGold: Ytrain}
            )
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Write summary stats to writer
            writer.add_summary(summary_results, i)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))

            # Plot progress to our two subplots
            accuracyLine, = ax1.plot(epoch_values, accuracy_values)
            costLine, = ax2.plot(epoch_values, cost_values)
            fig.canvas.draw()
            time.sleep(1)


# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                     feed_dict={X: Xtest, 
                                                                yGold: Ytest})))
print(weights.name)
weights.get_shape()

print(sess.run(weights))