import torch

from torch import optim

from torch import Tensor

import numpy as np

from itertools import permutations
start = torch.tensor([[0.99, 0.01, 0.01], [0.01, .99, 0.01], [0.01, 0.01, .99]], requires_grad=False, dtype=torch.float32)
rest = torch.rand([6, 3], requires_grad=True)

rest
def probs_sum_to_1(t):

    return ((t.sum(1) - 1) ** 2).sum()



def probs_within_0_and_1(t):

    return (t[t < 0] ** 2).sum() + (t[t > 1] ** 2).sum()



def prob_perm_occurs(t, p):

    n = len(p)

    m = len(t)

    return np.product([t[i:(m-n+i+1), p[i]] for i in range(n)])



def all_perms_occur_once(t):

    n = t.size(1)

    l = 0

    for p in permutations(range(n)):

        l += (prob_perm_occurs(t, p).sum() - 1) ** 2

    return l



def no_repeats(t):

    return (t[:-1] * t[1:]).sum()



def loss_fn(t):

    return (probs_sum_to_1(t)

            + probs_within_0_and_1(t)

            + all_perms_occur_once(t)

            + no_repeats(t))



def tensor_to_string(t):

    indices = t.max(1).indices

    return "".join(chr(i + ord('a')) for i in indices)
optimizer = optim.SGD([start, rest], lr=0.001, momentum=0.9)
for i in range(10000):

    optimizer.zero_grad()

    t = torch.cat((start, rest))

    loss = loss_fn(t)

    loss.backward()

    # print(rest.grad.data.sum())

    optimizer.step()

[t,

tensor_to_string(t),

loss.item()]