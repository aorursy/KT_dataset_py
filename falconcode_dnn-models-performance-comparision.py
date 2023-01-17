import numpy as np

import matplotlib.pyplot as plt
data = [[1.0, 1.0, .92, .63, .70], # acc

        [0.97, .99, .93, .75, .46]]  # val_acc

X = np.arange(5)

ticks = ['Model1', 'Model2', 'Model3', 'model4', 'Model5']

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(X - 0.125, data[0], color = 'b', width = 0.25, tick_label =ticks)

ax.bar(X + 0.125, data[1], color = 'y', width = 0.25, )

ax.legend(labels=['training accuracy', 'validation accuracy'])

ax.set_xticks(X)

ax.set_title('Best performance of the studied models')

plt.show()
data = [[0.01, 0.0, .58, .64, .59], # loss

        [0.1, 0.0, .62, .58, .74]]  # val_loss

X = np.arange(5)

ticks = ['Model1', 'Model2', 'Model3', 'model4', 'Model5']

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(X - 0.125, data[0], color = 'b', width = 0.25, tick_label =ticks)

ax.bar(X + 0.125, data[1], color = 'y', width = 0.25, )

ax.set_xticks(X)

ax.legend(labels=['training loss', 'validation loss'])

ax.set_title('Best performance of the studied models')

plt.show()
data = [[0.96, 0.98, 0.93, 0.75, 0.45], # Prediction_Accuracy

        [0.89, 0.99, 0.15, 0.34, 0.73], # Sensitivity

        [0.97, 0.98, 0.98, 0.78, 0.44], # Specificity

        [0.66, 0.82, 0.36, 0.09, 0.07]] # Precision



X = np.arange(5)

ticks = ['Model1', 'Model2', 'Model3', 'model4', 'Model5']

fig = plt.figure()#figsize = (10.0, 6.0))

ax = fig.add_axes([0,0,1,1])

ax.bar(X - 0.3, data[0], color = 'b', width = 0.2)

ax.bar(X - 0.1, data[1], color = 'y', width = 0.2, tick_label =ticks)

ax.bar(X + 0.1, data[2], color = 'g', width = 0.2)

ax.bar(X + 0.3, data[3], color = 'r', width = 0.2)



ax.set_xticks(X)

ax.set_yticks(np.arange(0, 1.4, 0.2))

ax.legend(labels=['Prediction_Accuracy', 'Sensitivity', 'Specificity', 'Precision'])

ax.set_title('Prediction metrics of the studied models')

plt.show()