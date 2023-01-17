import torch
a = torch.rand(3, 5, 9, 17)

b = torch.rand(3, 5, 17, 8)

print(a.size())

print(b.size())

print(b.mean((0,1)).size())

print(b.mean((0,2)).size())

print(b.mean((0,3)).size())

print(b.mean((1,2)).size())

print(b.mean((1,3)).size())

print(b.mean((2,3)).size())
print(a.size())

a = a.view(15,9,17)

print(b.size())

b = b.view(15,17,8)

c = torch.bmm(a, b)

print(c.size())

c = c.view(3, 5, 9, 8)

print(c.size())
print(b-b.mean((0,1)))
d = torch.rand(5)

print(d)

d = d.diag()

print(d)



d = torch.rand(2,5)

print(d)

d = d.diag()

print(d)
a = torch.rand(2, 3)

print(a, a.size())

b = torch.eye(a.size(1))

print(b)

c = a.unsqueeze(2).expand(*a.size(), a.size(1))

print("unsqueeze0", a.unsqueeze(0), a.unsqueeze(0).size())

print("unsqueeze1",a.unsqueeze(1), a.unsqueeze(1).size())

print("unsqueeze2",a.unsqueeze(2), a.unsqueeze(2).size())

print(c)

d = c * b

print(d)
a=torch.rand(1, 2, 3, 4)

b = 1/a

print(a)

print(b)

print(a * b)

print(a*(a>0.1).float())
