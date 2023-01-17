import numpy as np



norm_with_exponents = lambda data: np.exp(data) / np.sum(np.exp(data))

norm_by_sum = lambda data: data / np.sum(data)



data = np.array([10, 20, 35])

print("data\t\t", data)

print("norm w/ exp\t", norm_with_exponents(data))

print("norm by sum\t",norm_by_sum(data))



print("\nReadable notation")

for exp_notation in norm_with_exponents(data):

    print(f"{exp_notation}\t= {exp_notation:.13f}")



data = np.array([1, 2, 3.5])

print("\n\n\ndata\t\t", data)

print("norm w/ exp\t", norm_with_exponents(data))

print("norm by sum\t",norm_by_sum(data))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
# gen uniform rvs data of 100 samples

data = np.random.uniform(100,200,100)
sns.distplot(data)



plt.title("data")

plt.xlabel("data")

plt.ylabel("counts")

plt.show()
data_norm = (data - data.min()) / (data.max() - data.min())
data_std = (data - data.mean()) / (data.std())
data_div_by_sum = data / data.sum()
data_norm_w_exponents = np.exp(data) / np.sum(np.exp(data))
fig, axarr = plt.subplots(1, 4)

fig.set_size_inches(15,4)



sns.distplot(data_norm, kde=False, ax=axarr[0])

axarr[0].title.set_text("data_norm")

axarr[0].set_xlabel("data_norm")

axarr[0].set_ylabel("counts")



sns.distplot(data_std, ax=axarr[1])

axarr[1].title.set_text("data_std")

axarr[1].set_xlabel("data_std")

axarr[1].set_ylabel("counts")



sns.distplot(data_div_by_sum, ax=axarr[2])

axarr[2].title.set_text("data_div_by_sum")

axarr[2].set_xlabel("data_div_by_sum")

axarr[2].set_ylabel("counts")



sns.distplot(data_norm_w_exponents, ax=axarr[3])

axarr[3].title.set_text("data_norm_w_exponents")

axarr[3].set_xlabel("Note: because of exp scaling\nn_high_conf << n_low_conf")

axarr[3].set_ylabel("counts")



plt.show()





plt.figure(figsize=(15,7))

plt.hist(data_norm, fc=(0, 1, 0, 0.5))

plt.hist(data_std, fc=(0, 0, 1, 0.5))

plt.hist(data_div_by_sum, fc=(1, 0, 0, 1))

plt.hist(data_norm_w_exponents, fc=(0, 0, 0, 0.5))



plt.ylabel("counts")

plt.xlabel("feature")

plt.legend(["data_norm", "data_std", "data_div_by_sum", "Normalisation by exponents"])

plt.grid()



plt.show()
errors1 = np.random.normal(0,5, 100) 

errors2 = np.random.normal(0,5, 100) 

feat1 = np.arange(100, 200, 1) + errors1

feat2 = np.arange(500, 400, -1) + errors2



plt.scatter(feat1, feat2)

plt.xlabel("feat1")

plt.ylabel("feat2")

plt.title("2 feats are used as axes")

plt.show()
# normalisation

feat1_norm = (feat1 - feat1.min()) / (feat1.max() - feat1.min())

feat2_norm = (feat2 - feat2.min()) / (feat2.max() - feat2.min())



# standardistion

feat1_std = (feat1 - feat1.mean()) / (feat1.std())

feat2_std = (feat2 - feat2.mean()) / (feat2.std())



# div by sum

feat1_div_by_sum = (feat1) / (feat1.sum())

feat2_div_by_sum = (feat2) / (feat2.sum())
fig, axarr = plt.subplots(1, 3)

fig.set_size_inches(18,4)



sns.scatterplot(feat1_norm, feat2_norm, ax=axarr[0])

axarr[0].title.set_text("data_norm")

axarr[0].set_xlabel("feat1_norm")

axarr[0].set_ylabel("feat2_norm")



sns.scatterplot(feat1_std, feat2_std, ax=axarr[1])

axarr[1].title.set_text("data_std")

axarr[1].set_xlabel("feat1_std")

axarr[1].set_ylabel("feat2_std")



sns.scatterplot(feat1_div_by_sum, feat2_div_by_sum, ax=axarr[2])

axarr[2].title.set_text("data_div_by_sum")

axarr[2].set_xlabel("feat1_div_by_sum")

axarr[2].set_ylabel("feat1_div_by_sum")



plt.show()



plt.figure(figsize=(10,6))

plt.scatter(feat1_norm, feat2_norm, fc=(0, 1, 0, 0.5))

plt.scatter(feat1_std, feat2_std, fc=(0, 0, 1, 0.5))

plt.scatter(feat1_div_by_sum, feat2_div_by_sum, fc=(1, 0, 0, 0.5))



plt.ylabel('feat_2')

plt.xlabel('feat_1')

plt.legend(["feat1_norm, feat2_norm", "feat1_std, feat2_std", "feat1_div_by_sum, feat2_div_by_sum"])

plt.title("Note Scales")

plt.grid()

plt.show()