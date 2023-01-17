import numpy as np

import math



def logloss(true_label, predicted, eps=1e-15):

    p = np.clip(predicted, eps, 1 - eps)

    if true_label == 1:

        return -math.log(p)

    else:

        return -math.log(1 - p)
print("The label to be predicted is {} and the model's prediction is {}. Hence Logloss is {}".format(1,0.5,logloss(1,0.5)))



print("The label to be predicted is {} and the model's prediction is {}. Hence Logloss is {}".format(1,0.1,logloss(1,0.9)))



print("The label to be predicted is {} and the model's prediction is {}. Hence Logloss is {}".format(1,0.9,logloss(1,0.1)))
import matplotlib.pyplot as plt 

# predicted probabilities

p = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

lgloss = [logloss(1,i) for i in p]

plt.plot(p,lgloss)

plt.xlabel("probalities")

plt.ylabel("LogLoss")

plt.title("LogLoss for Different Probabilities")
