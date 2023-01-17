!pip install fastai --upgrade -q

from fastai.vision.all import *



from tqdm import tqdm
path = untar_data(URLs.MNIST)
trainpath = path/'training'

testpath = path/'testing'
def dataset(path, bs=256):

    stacks_x, stacks_y = [], []

    for i in tqdm((path).ls().sorted()):

        stack = torch.stack(

            [tensor(Image.open(o)) for o in (i).ls().sorted()]

                            ).float()/255

        stacks_x.append(stack)

        stacks_y.append(len(i.ls()))

        

    x = torch.cat(stacks_x).view(-1, 28*28)

    label = [stacks_y[i] * [i] for i in tqdm(range(len(stacks_y)))]

    y = tensor([j for sub in label for j in tqdm(sub)]).unsqueeze(1)

    dset = list(zip(x, y))

    return DataLoader(dset, batch_size=bs, shuffle=True)
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()



def linear1(xb):return xb@weights + bias
# def softmax(data):return torch.exp(data)/torch.exp(data).sum(dim=1)
dl = dataset(trainpath) 

valid_dl = dataset(testpath)
weights = init_params((28*28,10))

bias = init_params(10)



d1 = list(dl)[0][0][:4]

l1 = list(dl)[0][1][:4]



train1 = linear1(d1)

d1.shape, l1.shape 
def mnist_loss(predictions, targets):

    predictions =  F.softmax(predictions,1) # softmax(predictions)predictions.sigmoid()

    return torch.where(targets.squeeze()==predictions.argmax(1), 1-torch.max(predictions,1)[0], torch.max(predictions,1)[0]).mean()



def calc_grad(xb, yb, model):

    preds = model(xb)

    loss = mnist_loss(preds, yb)

    loss.backward()



def train_epoch(model, lr, params):

    for xb,yb in dl:

        calc_grad(xb, yb, model)

        for p in params:

            p.data -= p.grad*lr

            p.grad.zero_()



def batch_accuracy(predictions, targets):

    preds = []

    prediction = F.softmax(predictions,1) # predictions[i].sigmoid() # softmax(predictions)

    return tensor(targets.squeeze()==predictions.argmax(1)).float().mean()



def validate_epoch(model):

    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]

    return round(torch.stack(accs).mean().item(), 4)
mnist_loss(train1,l1)
lr = 1.

params = weights,bias

train_epoch(linear1, lr, params)

validate_epoch(linear1)
for i in range(20):

    train_epoch(linear1, lr, params)

    print(validate_epoch(linear1), end=' ')
for i in range(40):

    train_epoch(linear1, lr, params)

    print(validate_epoch(linear1), end=' ')