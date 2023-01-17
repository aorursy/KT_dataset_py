# Install the fastai v2 library

!pip install fastai2 --quiet



# Import required functions

from fastai2.vision.all import *

matplotlib.rc('image', cmap='Greys')
# Download data

path = untar_data(URLs.MNIST_SAMPLE)

Path.BASE_PATH = path # For eaiser printing

path.ls()
# Look inside the train folder

(path/'train').ls()
# List of filenames for 3s and 7s

threes = (path/'train'/'3').ls().sorted()

sevens = (path/'train'/'7').ls().sorted()

threes, sevens
# View an image using PIL

im3_path = threes[1]

im3 = Image.open(im3_path)

im3
# View an image as a numpy array

array(im3)[4:10,4:10]
# View an image as a PyTorch tensor

tensor(im3)[4:10,4:10]
# Overlaying pixels with the color

im3_t = tensor(im3)

df = pd.DataFrame(im3_t[4:15,4:22])

df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
# Load 3s and 7s and tensors

seven_tensors = [tensor(Image.open(o)) for o in sevens]

three_tensors = [tensor(Image.open(o)) for o in threes]

print('No. of 3s:', len(three_tensors))

print('No. of 7s:', len(seven_tensors))



# Sample 3

show_image(three_tensors[1]);
# Stack list of tensors into a single tensor

stacked_sevens = torch.stack(seven_tensors).float()/255

stacked_threes = torch.stack(three_tensors).float()/255

print(stacked_threes.shape, stacked_sevens.shape)
# View the average 3

mean3 = stacked_threes.mean(0)

show_image(mean3);
# View the average 7

mean7 = stacked_sevens.mean(0)

show_image(mean7);
# Pick a sample '3'

a_3 = stacked_threes[1]

show_image(a_3);
# Calculate L1 & L2 distance from the mean 3

dist_3_abs = (a_3 - mean3).abs().mean()

dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()

dist_3_abs,dist_3_sqr
# Calculate L1 & L2 distance from the mean 7

dist_7_abs = (a_3 - mean7).abs().mean()

dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()

dist_7_abs,dist_7_sqr
# Using PyTorch built in functions to calculate L1 & L2 distance

F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
# Create tensors from the validation set

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()]).float()/255

valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()]).float()/255

valid_3_tens.shape,valid_7_tens.shape
# Helper function to calculate the L1 distance from mean

def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))

mnist_distance(a_3, mean3)
# Use it for the entire validation set (thanks to broadcasting)

valid_3_dist = mnist_distance(valid_3_tens, mean3)

valid_3_dist, valid_3_dist.shape
# Helper function to predict whether an image is a 3

def is_3(x): 

    return mnist_distance(x,mean3) < mnist_distance(x,mean7)
# Predict for a single image

is_3(a_3), is_3(a_3).float()
# Predict for the entire validation set

is_3(valid_3_tens)
# Look at an image that the model failed to classify

show_image(valid_3_tens[1]);
# Calculate overall accuracy

accuracy_3s =      is_3(valid_3_tens).float() .mean()

accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()



accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
# Input data (quardatic function with some random noise)

time = torch.arange(0,20).float()

speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1



print(time)

plt.scatter(time,speed);
# Model - simple polynomial function

def f(t, params):

    a,b,c = params

    return a*(t**2) + (b*t) + c



# Loss function - MSE

def mse(preds, targets): 

    return ((preds-targets)**2).mean()



# Parameters initialized randomly

params = torch.randn(3).requires_grad_()

orig_params = params.clone()

params
# Calculate predictions

preds = f(time, params)



# Helper function to plot predictions

def show_preds(preds, ax=None):

    if ax is None: ax=plt.subplots()[1]

    ax.scatter(time, speed)

    ax.scatter(time, to_np(preds), color='red')

    ax.set_ylim(-300,100)

    

show_preds(preds)
loss = mse(preds, speed)

loss
loss.backward()

print('params:', params)

print('params.grad:', params.grad)
# Update the weights

lr = 1e-5

params.data -= lr * params.grad.data

params.grad = None



# Check the new params & loss

print('updated params:', params)

preds = f(time,params)

print('updated loss:', mse(preds, speed))

show_preds(preds)
# Helper function to apply step

def apply_step(params, prn=True):

    preds = f(time, params)

    loss = mse(preds, speed)

    loss.backward()

    params.data -= lr * params.grad.data

    params.grad = None

    if prn: print(loss.item())

    return preds
for i in range(10): apply_step(params)
params = orig_params.detach().requires_grad_()

_,axs = plt.subplots(1,5,figsize=(15,3))

for ax in axs: 

    show_preds(apply_step(params, False), ax)

plt.tight_layout()
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project='fastai2-lesson3', environment=None)