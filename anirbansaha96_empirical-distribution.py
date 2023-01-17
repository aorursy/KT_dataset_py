import numpy as np

dataset1=np.random.normal(0,1,20)

dataset1.sort()
dataset1
def empirical_distribution(x, dataset):

    counter=0;

    for i in range(len(dataset)):

        if dataset[i]<=x:

            counter=counter+1;

        else:

            continue;

    return counter/len(dataset);
empirical_distribution(-0.5, dataset1)