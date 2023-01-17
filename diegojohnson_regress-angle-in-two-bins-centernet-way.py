import numpy as np

# get 𝜃𝑙 (Counter clockwise is positive)

def get_alpha(rot):

  # output: (Batch, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 

  #                     bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]

  # return rot[:, 0]

    idx = rot[:, 1] > rot[:, 5]

    

    # alpha1 is relative to -90º

    alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)

    

    # alpha2 is relative to +90º

    alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)

    

    return alpha1 * idx + alpha2 * (1 - idx)
# you need to do some finetune to get real 𝜃ray

# np.arctan2(x, z)