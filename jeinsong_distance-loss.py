import numpy as np

import torch
def dist_between(start_logits, end_logits, device='cpu', max_seq_len=128):

    """get dist btw. pred & ground_truth"""



    linear_func = torch.tensor(np.linspace(0, 1, max_seq_len, endpoint=False), requires_grad=False)

    linear_func = linear_func.to(device)



    start_pos = (start_logits*linear_func).sum(axis=1)

    end_pos = (end_logits*linear_func).sum(axis=1)



    diff = end_pos-start_pos



    return diff.sum(axis=0)/diff.size(0)





def dist_loss(start_logits, end_logits, start_positions, end_positions, device='cpu', max_seq_len=128, scale=1):

    """calculate distance loss between prediction's length & GT's length

    

    Input

    - start_logits ; shape (batch, max_seq_len{128})

        - logits for start index

    - end_logits

        - logits for end index

    - start_positions ; shape (batch, 1)

        - start index for GT

    - end_positions

        - end index for GT

    """

    start_logits = torch.nn.Softmax(1)(start_logits) # shape ; (batch, max_seq_len)

    end_logits = torch.nn.Softmax(1)(end_logits)

    

    start_one_hot = torch.nn.functional.one_hot(start_positions, num_classes=max_seq_len).to(device)

    end_one_hot = torch.nn.functional.one_hot(end_positions, num_classes=max_seq_len).to(device)

    

    pred_dist = dist_between(start_logits, end_logits, device, max_seq_len)

    gt_dist = dist_between(start_one_hot, end_one_hot, device, max_seq_len) # always positive

    diff = (gt_dist-pred_dist)



    rev_diff_squared = 1-torch.sqrt(diff*diff) # as diff is smaller, make it get closer to the one

    loss = -torch.log(rev_diff_squared) # by using negative log function, if argument is near zero -> inifinite, near one -> zero



    return loss*scale

start_logits = torch.zeros(10)

start_logits[0] = 1

start_logits[1] = 8

start_logits[2] = 1



end_logits = torch.zeros(10)

end_logits[2] = 2

end_logits[3] = 6

end_logits[4] = 2



start_pos = torch.tensor(1)

end_pos = torch.tensor(8)



dist_loss(

    torch.unsqueeze(start_logits, 0),

    torch.unsqueeze(end_logits, 0),

    torch.unsqueeze(start_pos, 0),

    torch.unsqueeze(end_pos, 0),

    max_seq_len=10,

)
start_logits = torch.zeros(10)

start_logits[0] = 1

start_logits[1] = 8

start_logits[2] = 1



end_logits = torch.zeros(10)

end_logits[7] = 1

end_logits[8] = 8

end_logits[9] = 1



start_pos = torch.tensor(1)

end_pos = torch.tensor(1)



dist_loss(

    torch.unsqueeze(start_logits, 0),

    torch.unsqueeze(end_logits, 0),

    torch.unsqueeze(start_pos, 0),

    torch.unsqueeze(end_pos, 0),

    max_seq_len=10,

)
start_logits = torch.zeros(10)

start_logits[0] = 1

start_logits[1] = 8

start_logits[2] = 1



end_logits = torch.zeros(10)

end_logits[7] = 1

end_logits[8] = 8

end_logits[9] = 1



start_pos = torch.tensor(1)

end_pos = torch.tensor(8)



dist_loss(

    torch.unsqueeze(start_logits, 0),

    torch.unsqueeze(end_logits, 0),

    torch.unsqueeze(start_pos, 0),

    torch.unsqueeze(end_pos, 0),

    max_seq_len=10,

)