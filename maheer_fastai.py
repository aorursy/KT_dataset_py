!pip install fastai --upgrade -q
from fastai.vision.all import *
# download a sample of MNIST that contains images of just 3 and 7
path = untar_data(URLs.MNIST_SAMPLE)
# displays the count of items, before listing the items. 
path.ls()
# the dataset has seperate training and valid sets.
# lets look into training set
(path/'train').ls()
# Let's take a look into one of its folders
# Here, we use sorted to ensure we all get the same order of files
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

# let's see number of images each folder contains
len(threes), len(sevens)
im3_path = threes[1]
im3 = Image.open(im3_path)

im3
# NumPy array
# Here, 4:10 indicates we requested the rows from index 4 (inclusice) to 10 (exclusive)
array(im3)[4:10,4:10]
# PyTorch tensor
tensor(im3)[4:10,4:10]
im3_t =tensor(im3)

df = pd.DataFrame(im3_t[4:25,4:25])

df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]

# let's check the length of each category again.
len(seven_tensors), len(three_tensors)

show_image(three_tensors[1])
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255


stacked_threes.shape
len(stacked_sevens.shape), stacked_sevens.ndim
mean3 = stacked_threes.mean(0)

print("This is the ideal number 3!!")
show_image(mean3)
mean7 = stacked_sevens.mean(0)

print("This is the ideal number 7!!")
show_image(mean7)
# Let's take a sample 3.
a_3 = stacked_threes[1]

# display the image.
show_image(a_3)
# With ideal 3.
dist_3_l1 = (a_3 - mean3).abs().mean()
dist_3_l2 = ((a_3 - mean3)**2).mean().sqrt()

print("Dist with ideal 3: L1",dist_3_l1,', L2',dist_3_l2)

# With ideal 7.
dist_7_l1 = (a_3 - mean7).abs().mean()
dist_7_l2 = ((a_3 - mean7)**2).mean().sqrt()

print("Dist with ideal 37: L1",dist_7_l1,', L2',dist_7_l2)
F.l1_loss(a_3.float(),mean3), F.mse_loss(a_3,mean3).sqrt()
valid_3_tensors = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tensors = valid_3_tensors.float()/255

valid_7_tensors = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tensors = valid_7_tensors.float()/255

valid_3_tensors.shape, valid_7_tensors.shape
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))

mnist_distance(a_3, mean3)
valid_3_dist = mnist_distance(valid_3_tensors, mean3)

valid_3_dist, valid_3_dist.shape
def is_3(x):
    return mnist_distance(x,mean3) < mnist_distance(x,mean7)

is_3(a_3), is_3(a_3).float()
# Now, let's check for the whole validation set.
is_3(valid_3_tensors)
acc_3 = is_3(valid_3_tensors).float().mean()
acc_7 = 1- is_3(valid_7_tensors).float().mean()

acc_3, acc_7, (acc_3 + acc_7)/2
def pr_eight(x,w):
    (x*w).sum()
def f(x):
    return x**2


def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)

# Plot a graph of this function.
plot_function(f,'x','x**2')
# picking a random value for x and plotting it on the graph.
plot_function(f,'x','x**2')
plt.scatter(-1.5,f(-1.5),color = 'red')
xt = tensor(3.).requires_grad_()

# we calcualte our function with that value.
yt = f(xt)
yt
yt.backward()

# Now, let's veiw the gradients.
xt.grad
xt = tensor([3.,4.,10.]).requires_grad_()

# we'll add sum to our function so that it can take a vector and return a scalar.
def f(x):
    return (x**2).sum()

yt= f(xt)
print(yt)

# calculating gradients.
yt.backward()
xt.grad
# train and valid sets.
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1,28*28)
valid_x = torch.cat([valid_3_tensors, valid_7_tensors]).view(-1,28*28)


# Labels. 1 for 3's and 0 for 7's
train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
valid_y = tensor([1] * len(valid_3_tensors) + [0] * len(valid_7_tensors)).unsqueeze(1)
dset = list(zip(train_x, train_y))
valid_dset = list(zip(valid_x, valid_y))

# lets check first item
x,y = dset[0]
x.shape, y
def init_params(size, std = 1.0):
    return (torch.randn(size)*std).requires_grad_()

weights = init_params((28*28,1))

bias = init_params(1)
def linear1(xb):
    return xb@weights + bias

preds = linear1(train_x)
preds
corrects = (preds > 0.0).float() == train_y
corrects
# now accuracy for the first epoch( still we didn't back propogate)
corrects.float().mean().item()
dl = DataLoader(dset, batch_size = 256)
valid_dl = DataLoader(valid_dset, batch_size= 256)
def mnist_loss(preds , trgts):
    preds = preds.sigmoid()
    return torch.where(trgts ==1, 1-preds, preds).mean()

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
def batch_acc(xb, yb):
    preds = xb.sigmoid()
    crct = (preds > 0.5) == yb
    return crct.float().mean()


def validate_epoch(model):
    accs = [batch_acc(model(xb),yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(),4)
lr = 1
params = weights, bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1))
    
linear_model= nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(),lr)

def train_epoch(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model))
        

        
train_model(linear_model, 20)
dls = DataLoaders(dl, valid_dl)


learn = Learner(dls, linear_model, opt_func= SGD,
               loss_func= mnist_loss, metrics= batch_acc)

learn.fit(10, lr =lr)