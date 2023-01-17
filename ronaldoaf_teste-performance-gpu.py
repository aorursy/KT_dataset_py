import torch

import datetime

a=torch.Tensor([2, 2, 2,2,2,2,2,2,2,2,2 ])

b=torch.Tensor([2, 2, 2,2,2,2,2,2,2,2,2 ])

a
b


torch.device='cpu'

inicio=datetime.datetime.now()

for i in range(5000000): a*b

(datetime.datetime.now()-inicio).seconds
torch.device='cuda'

inicio=datetime.datetime.now()

for i in range(5000000): a*b

(datetime.datetime.now()-inicio).seconds