import torch
def F_score(logit, label, threshold=0.5, beta=2):
    prob = torch.sigmoid(logit)
    prob = prob > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
def label2onehot(label, num_classes=7178):
    onehot = label.new(label.size(0), num_classes)
    onehot.zero_()
    onehot.scatter_(1, label, 1)
    return onehot
logit = torch.FloatTensor([[0.1,0.2,-0.5,0.8],[-0.3,0.9,-0.1,0.1]])
label = torch.LongTensor([[1,2],[0,3]])
onehot = label2onehot(label,4)
print(onehot)
print(logit)
fscore = F_score(logit, onehot)
print(fscore)
