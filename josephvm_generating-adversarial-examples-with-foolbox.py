batch = 10 # how many adversarial examples to find
!pip install foolbox
# Imports

import time, foolbox, torch

import matplotlib.pyplot as plt

import numpy as np

import torchvision.models as models



%matplotlib inline



# Foolbox defaults to a GPU
# sets a bunch of variables - could have an if-else to mess with different networks

network = models.inception_v3(pretrained=True) # grab the model from torchvision

dataset = 'imagenet'

channels = 3 # RGB

size = 224 # image size

classes = 1000 

network.eval();
# Foolbox time 

# Convert the model to a Foolbox model

fnetwork = foolbox.models.PyTorchModel(network, bounds=(0, 1), num_classes=classes, channel_axis=1) 



# get source image and label

# images are just an array

images, labels = foolbox.utils.samples(dataset=dataset, batchsize=batch, data_format='channels_first', bounds=(0, 1))

images = images.reshape(batch, channels, size, size)



print(images.shape)

print("Labels:      ", labels)

predictions = fnetwork.forward(images).argmax(axis=-1)

print("Predictions: ", predictions) # original prediction

print("Accuracy: ", np.mean(predictions == labels)) # Accuracy of original images

already_correct = np.sum(predictions != labels) # keep track of how many were already correct
attack = foolbox.attacks.DeepFoolL2Attack(fnetwork, distance=foolbox.distances.Linfinity)



t1 = time.time()

adversarials = attack(images, labels,  unpack=False) # get the adversarial examples

t2 = time.time()



avg_time = (t2 - t1) / batch

print("Avg Time: ", avg_time, "seconds")
# this cell & next 2 are mostly from a Foolbox tutorial

adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])

print("Labels: ", labels)

print("Adv. Labels: ", adversarial_classes)

print("Classification Acc: ", np.mean(adversarial_classes == labels)) # should be 0.0
_sum = 0

for i in range(batch):

    _sum += adversarials[i].distance.value

avg_Linf = _sum / (batch-already_correct)

print("Avg L-inf distance: ", avg_Linf)
# The 'Adversarial' objects also provide a 'distance' attribute. Note that the distances

# can be 0 (misclassified without perturbation) and inf (attack failed).

distances = np.asarray([a.distance.value for a in adversarials])

print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))

print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))

print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))
# Plot the original, adversarial, & the difference between them

for i in range(batch):



    image = images[i]

    adversarial = adversarials[i].perturbed



    # CHW to HWC (switching the channels)

    image = image.transpose(1, 2, 0)

    adversarial = adversarial.transpose(1, 2, 0)

    if image.shape[2] == 1: # for MNIST (only one color channel)

        image = image.reshape(28, 28)

        adversarial = adversarial.reshape(28, 28)

    

    plt.figure()



    # Original

    plt.subplot(1, 3, 1)

    plt.title('Label: '+ str(labels[i]))

    plt.imshow(image)

    plt.axis('off')



    # Adversarial

    plt.subplot(1, 3, 2)

    plt.title('Adversarial: ' + str(adversarials[i].adversarial_class))

    plt.imshow(adversarial)

    plt.axis('off')



    # Difference

    plt.subplot(1, 3, 3)

    plt.title('Difference')

    difference = adversarial - image

    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)

    plt.axis('off')



    plt.show()