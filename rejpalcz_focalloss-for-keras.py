from keras import backend as K
import tensorflow as tf

def KerasFocalLoss(target, input):
    
    gamma = 2.
    input = tf.cast(input, tf.float32)
    
    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    
    return K.mean(K.sum(loss, axis=1))
import numpy as np
from fastai.conv_learner import *
from fastai.dataset import *


# credits: https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb#
# credits originally: https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()
# define some results
Y_true = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
Y_pred = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1]], dtype=np.float32)
fc = FocalLoss()

print(fc.forward(torch.from_numpy(Y_pred), torch.from_numpy(Y_true.astype(np.float32))))
print(K.eval(KerasFocalLoss(Y_true, Y_pred)))