import numpy as np
np.bincount([1,2,3,3,3])



# number of

# 0s : 0

# 1s : 1

# 2s : 1

# 3s : 3
def _replaceZeroes(data, reduction_factor=0.999999):

  """

  avoid log(0) undefined error becasue `p_xis` can be 0

  and  we have to calculate log(p_xis)

  

  Replace all zeroes by a number lower than the lowest

  """

  min_nonzero = np.min(data[np.nonzero(data)])

  data[data == 0] = min_nonzero - (reduction_factor*min_nonzero)

  return data 





def _get_pxis(X):

    """

    - classes labels must be whole numbers. 0, 1, 2 ....

        + because np.bincount works like that

    - len(output) = num_classes

    """

    num_samples = len(X)

    count_xis   = np.bincount(X)

    p_xis       = count_xis / num_samples

    

    return _replaceZeroes(p_xis) # avoid log error





def entropy(X):

    """

    - X is a 1-D vec

    """

    p_xis = _get_pxis(X)

    return -1 * np.sum(p_xis * np.log2(p_xis))





def gini_impurity(X):

    """

    - X is a 1-D vec

    """

    p_xis = _get_pxis(X)

    return 1 - np.sum(p_xis * p_xis)



def neg_lg_ps(X):

    """

    - X is a 1-D vec

    """

    p_xis = _get_pxis(X)

    return -1 * np.sum(np.log2(p_xis))
#2 classes

a = [1,1,1,1,1,1,1,1,1,1] # zero randomness

b = [0,1,1,1,0,1,1,1,1,0] # some randomness

c = [0,1,0,1,0,1,0,1,0,1] # extreme randomness
H_a = entropy(a)

H_b = entropy(b)

H_c = entropy(c)



G_a = gini_impurity(a)

G_b = gini_impurity(b)

G_c = gini_impurity(c)



L_a = neg_lg_ps(a)

L_b = neg_lg_ps(b)

L_c = neg_lg_ps(c)



print(f"H_a: {H_a:.2f}\tH_b: {H_b:.2f}\tH_c: {H_c:.2f}")

print(f"G_a: {G_a:.2f}\tG_b: {G_b:.2f}\tG_c: {G_c:.2f}")

print(f"L_a: {L_a:.2f}\tL_b: {L_b:.2f}\tL_b: {L_b:.2f}")
import matplotlib.pyplot as plt

import seaborn as sns



fig, axarr = plt.subplots(1,3)

fig.set_size_inches(15,3)



x_labels = ["zero randomness", "some randomness", "extreme randomness"]



axarr[0].barh(x_labels, [H_a, H_b, H_c])

axarr[1].barh(x_labels, [G_a, G_b, G_c])

axarr[2].barh(x_labels, [L_a, L_b, L_c])



axarr[0].set_title("Entropy")

axarr[1].set_title("Gini Impurity")

axarr[2].set_title("Neg Log Sum of Ps\nNote: works oppositely")





fig.tight_layout(pad=3.0)

plt.show()
#3 classes

a = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3] # zero randomness

b = [0,1,1,1,2,1,1,1,3,2,1,1,1,0,1,1] # some randomness

c = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3] # extreme randomness
H_a = entropy(a)

H_b = entropy(b)

H_c = entropy(c)



G_a = gini_impurity(a)

G_b = gini_impurity(b)

G_c = gini_impurity(c)



print(f"H_a: {H_a:.2f}\tH_b: {H_b:.2f}\tH_c: {H_c:.2f}")

print(f"G_a: {G_a:.2f}\tG_b: {G_b:.2f}\tG_c: {G_c:.2f}")