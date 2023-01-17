import pandas as pd 

import numpy as np

from mxnet import nd

mnist_train = pd.read_csv('../input/train.csv')

mnist_test  = pd.read_csv('../input/test.csv')
# Initialize the counter

xcount = nd.ones((784, 10))

ycount = nd.ones((10))
for i in range(len(mnist_train.iloc[:])):

    y = int(mnist_train.iloc[i][0])

    ycount[y] += 1

    # for j in range(len(mnist_train.iloc[i][1:])):

    #    xcount[j,y] += nd.array(mnist_train.iloc[i][j+1])

    xcount[:,y] += nd.floor(nd.array(mnist_train.iloc[i][1:].values)/128)

    
py = ycount / ycount.sum()

px = (xcount / ycount.reshape(1,10))
import matplotlib.pyplot as plt

fig, figarr = plt.subplots(1, 10, figsize=(10, 10))

for i in range(10):

    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')

    figarr[i].axes.get_xaxis().set_visible(False)

    figarr[i].axes.get_yaxis().set_visible(False)



plt.show()

print('Class probabilities', py)
mnist_test.iloc[0][:].shape
# Get the first test item

data_first_test= nd.array(mnist_test.iloc[0][:].values).reshape((784,1))



# Compute the per pixel conditional probabilities

x_prob = px * data_first_test + (1 - px) * (1 - data_first_test)

# Compute the serial multiplication with py

x_prob_y = x_prob.prod(0) * py

print('Unormalized probabilities', x_prob_y)

# Normalization

x_prob_y_normal = x_prob_y / x_prob_y.sum()

print('Normalized probabilities', x_prob_y_normal)
logpx = nd.log(px)

logpxneg = nd.log(1-px)

logpy = nd.log(py)
def bayespost(data):

    # We need to incorporate the prior probability p(y) since p(y|x) is proportional to p(x|y) p(y)

    logpost = logpy.copy()

    logpost += (logpx * data + logpxneg * (1-data)).sum(0)

    # Normalize to prevent overflow or underflow by subtracting the largest value

    logpost -= nd.max(logpost)

    # Compute the softmax using logpx

    post = nd.exp(logpost).asnumpy()

    post /= np.sum(post)

    return post



# Get all prediction results

prediction_results = [] # each one is prior_probalility * likelihood

for i in range(len(mnist_test.iloc[:])):

    x = nd.array(mnist_test.iloc[i][:]).reshape((784,1))

    post = bayespost(x)

    the_list = post.tolist()

    if not float(1) in the_list:

        the_list = [round(i) for i in the_list]

    prediction_results.append(the_list.index(float(1.0)))

    
# Generate submission csv_file and fill it with predictions 

output_file = 'submission.csv'

with open(output_file, 'w') as f:

    f.write('ImageId,Label\n')

    for i in range(len(prediction_results)):

        f.write(''.join([str(i+1), ',', str(prediction_results[i]), '\n']))