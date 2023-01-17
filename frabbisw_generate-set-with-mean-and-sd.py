import torch
x=torch.rand(10,1, requires_grad=True)
y1=torch.tensor([.2])
y2=torch.tensor([1.0])
lr=.1
for i in range(1000):
    z1=torch.mean(x)
    z2=torch.std(x)
    error1 = torch.mean((z1 - y1) ** 2)
    error1.backward()
    error2 = torch.mean((z2 - y2) ** 2)
    error2.backward()
    
    if i%100 == 0:
        print(error1,error2)
    
    with torch.no_grad():
        x=x-x.grad*lr
    x.requires_grad=True
print(torch.mean(x), torch.std(x))
