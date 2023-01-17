import numpy

from pandas import read_csv

from sklearn.utils import resample

from sklearn.metrics import accuracy_score

from matplotlib import pyplot

%matplotlib inline



# load dataset

x = numpy.array([180,162,158,172,168,150,171,183,165,176]) 





# configure bootstrap

n_iterations = 1000

n_size = int(len(x))



# run bootstrap

medians = list()

for i in range(n_iterations):

    # prepare train and test sets

    s = resample(x, n_samples=n_size);

    m = numpy.median(s);

    #print(m)

    medians.append(m)



# plot scores

pyplot.hist(medians)

pyplot.show()



# confidence intervals

alpha = 0.95

p = ((1.0-alpha)/2.0) * 100

lower =  numpy.percentile(medians, p)



p = (alpha+((1.0-alpha)/2.0)) * 100

upper =  numpy.percentile(medians, p)

print('%.1f confidence interval %.1f and %.1f' % (alpha*100, lower, upper))


