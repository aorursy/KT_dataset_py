import torch
a = [1.0, 2.0, 1.0]
a[0]
a[2] = 3.0

a
a = torch.ones(3)  # Creates a one-dimensional tensor of size 3 filled with ones

a
a[1]
float(a[1])
a[2] = 2.0

a
points = torch.zeros(6)  # Using .zeros is just a way to get an appropriately sized array.

points[0] = 4.0          # We overwrite those zeros with the values we actually want.

points[1] = 1.0

points[2] = 5.0

points[3] = 3.0

points[4] = 2.0

points[5] = 1.0

points
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])

points
float(points[0]), float(points[1])
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

points
points.shape
points = torch.zeros(3, 2)

points
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

points
points[0, 1]
points[0]
some_list = list(range(6))

some_list[:]               # All elements in the list

some_list[1:4]             # From element 1 inclusive to element 4 exclusive

some_list[1:]              # From element 1 inclusive to the end of the list

some_list[:4]              # From the start of the list to element 4 exclusive

some_list[:-1]             # From the start of the list to one before the last element

some_list[1:4:2]           # From element 1 inclusive to element 4 exclusive, in steps of 2

some_list
points[1:]                # All rows after the first; implicitly all columns

points[1:, :]             # All rows after the first; all columns

points[1:, 0]             # All rows after the first; first column

points[None]              # Adds a dimension of size 1 , just like unsqueeze
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]

weights = torch.tensor([0.2126, 0.7152, 0.0722])

img_t, weights
batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]

batch_t
img_gray_naive = img_t.mean(-3)

batch_gray_naive = batch_t.mean(-3)

img_gray_naive.shape, batch_gray_naive.shape
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)

img_weights = (img_t * unsqueezed_weights)

batch_weights = (batch_t * unsqueezed_weights)

img_gray_weighted = img_weights.sum(-3)

batch_gray_weighted = batch_weights.sum(-3)

batch_weights.shape, batch_t.shape, unsqueezed_weights.shape
img_gray_weighted_fancy = torch.einsum('...chw,c->...hw', img_t, weights)

batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw', batch_t, weights)

batch_gray_weighted_fancy.shape
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])

weights_named
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')

batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')

print("img named:", img_named.shape, img_named.names)

print("batch named:", batch_named.shape, batch_named.names)
weights_aligned = weights_named.align_as(img_named)

weights_aligned.shape, weights_aligned.names
gray_named = (img_named * weights_aligned).sum('channels')

gray_named.shape, gray_named.names
# gray_named = (img_named[..., :3] * weights_named).sum('channels')
gray_plain = gray_named.rename(None)

gray_plain.shape, gray_plain.names
double_points = torch.ones(10, 2, dtype=torch.double)

short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
short_points.dtype
double_points = torch.zeros(10, 2).double()

short_points = torch.ones(10, 2).short()
double_points = torch.zeros(10, 2).to(torch.double)

short_points = torch.ones(10, 2).to(dtype=torch.short)
points_64 = torch.rand(5, dtype=torch.double)  # rand initializes the tensor elements to random numbers between 0 and 1 .

points_short = points_64.to(torch.short)

points_64 * points_short                       # works from PyTorch 1.3 onwards
a = torch.ones(3, 2)

a_t = torch.transpose(a, 0, 1)



a.shape, a_t.shape
a = torch.ones(3, 2)

a_t = a.transpose(0, 1)



a.shape, a_t.shape
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

points.storage()
points_storage = points.storage()

points_storage[0]
points.storage()[1]
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

points_storage = points.storage()

points_storage[0] = 2.0

points
a = torch.ones(3, 2)

a
a.zero_()

a
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

second_point = points[1]

second_point.storage_offset()
second_point.size()
second_point.shape
points.stride()
second_point = points[1]

second_point.size()
second_point.storage_offset()
second_point.stride()
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

second_point = points[1]

second_point[0] = 10.0

points
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

second_point = points[1].clone()

second_point[0] = 10.0

points
second_point
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

points
points_t = points.t()

points_t
id(points.storage()) == id(points_t.storage())
points.stride()
points_t.stride()
some_t = torch.ones(3, 4, 5)

transpose_t = some_t.transpose(0, 2)

some_t.shape
some_t
transpose_t.shape
transpose_t
some_t.stride()
transpose_t.stride()
points.is_contiguous()
points_t.is_contiguous()
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

points_t = points.t()

points_t
points_t.storage()
points_t.stride()
points_t_cont = points_t.contiguous()

points_t_cont
points_t_cont.stride()
points_t_cont.storage()
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
points_gpu = points.to(device='cuda')
points_gpu
points_gpu = points.to(device='cuda:0')
points = 2 * points                        # Multiplication performed on the CPU

points_gpu = 2 * points.to(device='cuda')  # Multiplication performed on the ! GPU
points_gpu = points_gpu + 4
points_cpu = points_gpu.to(device='cpu')
points_gpu = points.cuda()      # Defaults to GPU index 0

points_gpu = points.cuda(0)

points_cpu = points_gpu.cpu()
points = torch.ones(3, 4)

points_np = points.numpy()

points_np
points = torch.from_numpy(points_np)

points
torch.save(points, './ourpoints.t')
with open('./ourpoints.t','wb') as f:

    torch.save(points, f)
points = torch.load('./ourpoints.t')
with open('./ourpoints.t','rb') as f:

    points = torch.load(f)
# !conda install h5py
import h5py
f = h5py.File('./ourpoints.hdf5', 'w')

dset = f.create_dataset('coords', data=points.numpy())

f.close()
f = h5py.File('./ourpoints.hdf5', 'r')

dset = f['coords']

last_points = dset[-2:]

last_points
last_points = torch.from_numpy(dset[-2:])

f.close()
# creating tensor a

a = torch.tensor(list(range(9)))

a
# size of tensor a

a.size()
a.shape
# offset of tensor a

a.storage_offset()
# stride of tensor a

a.stride()
# creating new tensor b

b = a.view(3, 3)

b
# tensors a and b share the same storage

a.storage().data_ptr() == b.storage().data_ptr()
id(a.storage()) == id(b.storage())
# create tensor c

c = b[1:,1:]

c
# size of tensor c

c.size()
c.shape
# offset of tensor c

c.storage_offset()
# stride of tensor c

c.stride()
# torch.cos(a)

# # cos_vml_cpu not implemented for 'Long'
# torch.sqrt(a)

# # sqrt_vml_cpu not implemented for 'Long'
# we need to change dtype

# dtype=torch.float64

# a = torch.tensor(a, dtype=torch.float64)

a = a.to(dtype=torch.float64)

a
torch.cos(a)
torch.sqrt(a)