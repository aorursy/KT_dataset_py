import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('../input/train.csv').as_matrix()

test_data = pd.read_csv('../input/test.csv').as_matrix()

x_train = train_data[0:,1:]

y_train = train_data[0:,0]

x_test = test_data
m = x_train.shape[0]

n = x_train.shape[1]

labels = np.unique(y_train)

labels_count = labels.shape[0]
# plot single image

def plot_image(array, size=28):

    plt.imshow(array.reshape(size, size))

    plt.show()
# Creating and plotting average digits

average_digits = np.empty((0, n+1))



plt.figure(figsize=(8,7))

plt.gray()



for label in labels:

    digits = x_train[y_train.flatten() == label]

    average_digit = digits.mean(0)   

    average_digits = np.vstack((average_digits, np.append(average_digit, label)))

    image = average_digit.reshape(28, 28)

    plt.subplot(3,4,label+1)

    plt.imshow(image)

    plt.title('Average '+str(label))

plt.show()



average_digits_x = average_digits[:,:-1]

average_digits_y = average_digits[:,-1]
# predicting the label by proximity to the avearge



# for the training set

predictions = np.empty(0)

for digit in x_train:

    d = ((average_digits_x - digit)**2).sum(axis=1)

    index = np.argmin(d)

    prediction = average_digits_y[index]

    predictions = np.append(predictions, prediction)

predictions = predictions.astype('int')

#accuracy on the training set

accuracy = np.sum(predictions == y_train) / m

print("Accuracy on the training set is: {:f}".format(accuracy))
# predicting the label by proximity to the avearge



# for the test set

predictions = np.empty(0)

for digit in x_test:

    d = ((average_digits_x - digit)**2).sum(axis=1)

    index = np.argmin(d)

    prediction = average_digits_y[index]

    predictions = np.append(predictions, prediction)

predictions = predictions.astype('int')



# writing predictions to .csv

df = pd.DataFrame(columns=['ImageId','Label'])

df['ImageId'] = np.arange(1,28001)

df['Label'] = predictions

df.to_csv('test_predictions.csv', index=False)