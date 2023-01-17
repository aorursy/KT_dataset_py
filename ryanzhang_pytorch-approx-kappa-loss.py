import torch

import numpy as np

from sklearn.metrics import cohen_kappa_score
a = np.array([1,2,3,0,4])

b = np.array([[5, 1, 1, 1, 1], [1, 1, 5, 1, 1], [1, 1, 1, 5, 1], [1, 5, 1, 1, 1], [1, 1, 1, 1, 5]])

c = np.array([[5, 1, 1, 1, 1], [1, 1, 5, 1, 1], [1, 1, 1, 5, 1], [5, 1, 1, 1, 1], [1, 1, 1, 1, 5]])

d = np.array([[1, 5, 1, 1, 1], [1, 1, 5, 1, 1], [1, 1, 1, 5, 1], [5, 1, 1, 1, 1], [1, 1, 1, 1, 5]])


def qwk(logits, labels, num_classes=5, epsilon=1e-10):

    probas = torch.nn.functional.softmax(logits.float(), 0).float()

    labels = torch.nn.functional.one_hot(labels, num_classes).float()

    repeat_op = torch.arange(0, num_classes).view(num_classes, 1).repeat(1, num_classes).float()

    repeat_op_sq = torch.pow((repeat_op - repeat_op.transpose(0, 1)), 2)

    weights = repeat_op_sq / (num_classes - 1) ** 2



    pred_ = probas ** 2

    pred_norm = pred_ / (epsilon + pred_.sum(1).view(-1, 1))



    hist_rater_a = pred_norm.sum(0)

    hist_rater_b = labels.sum(0)

    conf_mat = torch.matmul(pred_norm.transpose(0, 1), labels)



    nom = (weights * conf_mat).sum()

    denom = (weights * torch.matmul(hist_rater_a.view(num_classes, 1), hist_rater_b.view(1, num_classes)) / labels.shape[0]).sum()

    return nom / (denom + epsilon)
print(cohen_kappa_score(a, b.argmax(1), weights='quadratic'))

print(qwk(torch.from_numpy(b), torch.from_numpy(a)).numpy())
print(cohen_kappa_score(a, c.argmax(1), weights='quadratic'))

print(qwk(torch.from_numpy(c), torch.from_numpy(a)).numpy())
print(cohen_kappa_score(a, d.argmax(1), weights='quadratic'))

print(qwk(torch.from_numpy(d), torch.from_numpy(a)).numpy())