import numpy as np # linear algebra

from lwlwrap import *
# Random test data.

num_samples = 100

num_labels = 20



truth = np.random.rand(num_samples, num_labels) > 0.5

# Ensure at least some samples with no truth labels.

truth[0:1, :] = False



scores = np.random.rand(num_samples, num_labels)
per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)

print("lwlrap from per-class values=", np.sum(per_class_lwlrap * weight_per_class))

print("lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
# Test of accumulator version.

accumulator = lwlrap_accumulator()

batch_size = 12

for base_sample in range(0, scores.shape[0], batch_size):

  accumulator.accumulate_samples(

      truth[base_sample : base_sample + batch_size, :], 

      scores[base_sample : base_sample + batch_size, :])

print("cumulative_lwlrap=", accumulator.overall_lwlrap())

print("total_num_samples=", accumulator.total_num_samples)


