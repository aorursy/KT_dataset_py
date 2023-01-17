from __future__ import print_function
import torch
# construct a 3x3 matric uninitialized

x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
x = torch.tensor([4.5444, 5.5323])
print(x)
x.size()
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))
