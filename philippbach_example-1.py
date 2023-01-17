import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import math

np.random.seed(0)

mu = 5
sigma = math.sqrt(1)
alpha = 0.05
sample_size = 1000
sample = np.random.normal(mu, sigma, sample_size)
sample_mean = sample.mean()

print("Sample Mean:", sample_mean)
quantile = stats.norm.ppf(q = 1-alpha/2)
print("Quantile:", quantile)              # Check the quantile                        

confidence_interval = (sample_mean - (sigma/math.sqrt(sample_size))*quantile,
                                      sample_mean + (sigma/math.sqrt(sample_size))*quantile)
print("Confidence interval:", confidence_interval)
