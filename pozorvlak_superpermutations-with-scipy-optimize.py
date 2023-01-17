import numpy as np # linear algebra

from scipy import optimize

from itertools import permutations
n = 3

id_n = np.diag(np.ones(n))



def tensor_from_array(t):

    m = int(t.size / n)

    return t.reshape((m, n))



def prob_perm_occurs(t, p):

    m = len(t)

    return np.product([t[i:(m-n+i+1), p[i]] for i in range(n)])



def all_perms_occur_once(t):

    l = 0

    for p in permutations(range(n)):

        l += (prob_perm_occurs(t, p).sum() - 1) ** 2

    return l



def no_repeats(t):

    return (t[:-1] * t[1:]).sum()



def f(t):

    t = tensor_from_array(t)

    return all_perms_occur_once(t) + no_repeats(t)



def probs_sum_to_1(t):

    return 1 - t.sum(1)



def starts_with_abc(t):

    return (t[:n] - id_n).reshape((n*n,))



def eqcons(t):

    t = tensor_from_array(t)

    return np.concatenate([probs_sum_to_1(t), starts_with_abc(t)])
x0 = np.concatenate([id_n, np.zeros([6, 3])])
eqcons(x0)
(x0[:-1] * x0[1:])
all_perms_occur_once(x0)
bounds = [(0, 1) for i in range(x0.size)]
optimize.fmin_slsqp(f, x0, bounds=bounds, f_eqcons=eqcons).reshape([9,3])