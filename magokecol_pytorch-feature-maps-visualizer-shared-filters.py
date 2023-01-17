# set this to True before commiting

COMMIT = True



if COMMIT:

    upscaling_steps=13

    opt_steps=20

    verbose=1

    INPUT_IMAGES = 100

    TOP = 5

    COMMON = 40

    

else:

    upscaling_steps=8

    opt_steps=10

    verbose=2

    INPUT_IMAGES = 10

    TOP = 1

    COMMON = 4

    

INTERACTIVE = not COMMIT
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torchvision import models, transforms

import cv2

import matplotlib.pyplot as plt

import zipfile

import PIL

import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

if INTERACTIVE:

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
blenheim_spaniel_dir = '/kaggle/input/stanford-dogs-dataset/images/Images/n02086646-Blenheim_spaniel'
imgsize = 224



def training_data(label,data_dir):

    X = []

    for img in os.listdir(data_dir):

        path = os.path.join(data_dir,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img,(imgsize,imgsize))        

        X.append(np.array(img))

    return X

X = training_data('blenheim_spaniel',blenheim_spaniel_dir)

print(len(X))
fig = plt.figure(figsize=(16, 16))

for i in range(16):

    ax = plt.subplot(4,4, i+1)

    ax.imshow(X[i])

plt.show()
# VGG19 BN

model = models.vgg19_bn()

model.load_state_dict(torch.load('/kaggle/input/pytorch-pretrained-models/vgg19_bn-c79401a0.pth'))

model.eval()

model = model.double()
# list cnn layers

layers = [layer for layer in model.children()]

print('Layers: {}'.format(len(layers)))

print('Layers[0]: {}'.format(len(layers[0])))

if INTERACTIVE:

    for l in layers:

        print(l)
# global variable to work with GPU if possible

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available(): print('Thanks Kaggle! Guau!!!')

print(device)
class SaveFeatures():

    def __init__(self, module, device=None):

        # we are going to hook some model's layer (module here)

        self.hook = module.register_forward_hook(self.hook_fn)

        self.device = device



    def hook_fn(self, module, input, output):

        # when the module.forward() is executed, here we intercept its

        # input and output. We are interested in the module's output.

        self.features = output.clone()

        if self.device is not None:

            self.features = self.features.to(device)

        self.features.requires_grad_(True)



    def close(self):

        # we must call this method to free memory resources

        self.hook.remove()
class FeatureMapVisualizer():

    def __init__(self, cnn, device, channels=3, layers_base=None, norm=None, denorm=None, save=None):

        self.model = cnn



        if layers_base is None:

            self.layers = self.model

        else:

            self.layers = layers_base

        

        self.channels = channels

        self.device = device



        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)

        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

            

        self.norm = norm

        self.denorm = denorm



        if norm is None:

            self.norm = transforms.Normalize(mean=mean.tolist(), std=std.tolist())



        if denorm is None:

            self.denorm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

            

        self.save = save



    def set_layers_base(self, layers):

        # sometime we want to access to layers in deeper levels

        # so we could call something like:

        # featureMap.set_layers_base([module for module in model.children()][5][1])

        self.layers = layers

        

    def optimize_img(self, activations, filter, img, learning_rate, opt_steps, verbose):

        

        size = img.shape[1]

        img = torch.from_numpy(img.astype(np.float32).transpose(2,0,1))

        

        img = self.norm(img).double()

        img_input = img.clone().detach().reshape(1, self.channels, size, size).to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([img_input], lr=learning_rate, weight_decay=1e-6)



        for n in range(opt_steps):

            optimizer.zero_grad()

            self.output = self.model(img_input)

            # TODO: the idea is to find an input image that

            #       'illuminate' ONLY ONE feature map (filter here)

            # TODO: 1 test a loss function that punish current

            #       activation filter with the rest of the

            #       filters mean values in the layer

            # TODO: 2 test a loss function that punish current activation

            #       filter with all the rest of the filters mean value

            #       of more layers (all?)

            loss = -1 * activations.features[0, filter].mean()

            loss.backward()

            if verbose > 1:

                print('.', end='')

            #print(loss.clone().detach().cpu().item())

            optimizer.step()

        if verbose > 1:

            print()

        img = self.denorm(img_input.clone().detach()[0].type(torch.float32))

        img = img.cpu().numpy().transpose(1,2,0)

        return img

        



    def visualize(self, layer, filter, size=56, upscaling_steps=12, upscaling_factor=1.2, lr=0.1, opt_steps=20, blur=None, verbose=2):

        training = self.model.training

        self.model.eval()

        self.model = self.model.double().to(self.device)

        # generate random image

        img = np.uint8(np.random.uniform(100, 160, (size, size, self.channels)))/255

        # register hook

        activations = SaveFeatures(self.layers[layer], self.device)

        if verbose > 0:

            print('Processing filter {}...'.format(filter))



        for i in range(upscaling_steps):

            if verbose > 1:

                print('{:3d} x {:3d}'.format(size,size), end='')



            img = self.optimize_img(activations, filter, img, learning_rate=lr, opt_steps=opt_steps, verbose=verbose)



            if i < upscaling_steps-1:

                size = int(size*upscaling_factor)

                # scale image up

                img = cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC)

                # blur image to reduce high frequency patterns

                if blur is not None: img = cv2.blur(img,(blur,blur))

            img = np.clip(img, 0, 1)



        if verbose > 0:

            print('preparing image...')

        activations.close()

        self.model.train(training)

        if self.save != None:

            self.save("layer_{:02d}_filter_{:03d}.jpg".format(layer, filter), img)

        return img

    

    # We return the mean of every activation value, but this could

    # be other metric based on convolutional output values.

    def get_activations(self, monitor, input, mean=True):



        training = self.model.training

        self.model.eval()

        self.model = self.model.double().to(self.device)



        activations = {}

        mean_acts = {}



        print('hooking layers {}'.format(monitor))

        for layer in monitor:

            activations[layer] = SaveFeatures(self.layers[layer], device=self.device)



        self.output = self.model(input.to(self.device))



        for layer in activations.keys():

            filters = activations[layer].features.size()[1]

            mean_acts[layer] = [activations[layer].features[0,i].mean().item() for i in range(filters)]



        print('unhooking layers.')        

        for layer in activations.keys():

            activations[layer].close()

            

        self.model.train(training)

        

        if mean:

            return mean_acts

        

        return activations
# We will save all generated images in this directory

images_dir = './images/'

# create images directory

if not os.path.exists(images_dir):

    os.makedirs(images_dir)

!ls
# We save images only when commiting

if COMMIT:

    def save(name, img):

        global image_dir

        plt.imsave(images_dir + name, img)

else:

    # do not waste time saving images

    save = None
with open('/kaggle/input/imagenet-1000-labels/imagenet1000_clsidx_to_labels.txt','r') as f:

    class_labels = eval(f.read())
# we save in the variable 'monitor' every ReLU layers that appears

# after every convolutional layer (they present non negative data)

# but skip shallow layers (> 5)

#monitor = [2,5,9,12,16,19,22,25,29,32,35,38,42,45,48,51]

#monitor = [i for i, layer in enumerate(layers[0]) if isinstance(layer, torch.nn.Conv2d)]

monitor = [i for i, layer in enumerate(layers[0]) if isinstance(layer, torch.nn.ReLU) and i > 5]



# define mean and std used for most famous images datasets

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)

std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)



# define global transformations based on previous mean and std

normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

denormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())



# The input images will be prepared with this transformation

# Minimum image size recommended for input is 224

img2tensor = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
def top(k, output, labels):

    values, indices = torch.topk(output, k, 1)

    values, indices = values[0].tolist(), indices[0].tolist()

    print('Top {} predicted classes:'.format(k))

    for i, idx in enumerate(indices):

        print('- {}: {}'.format(labels[idx], values[i]))
def show_filters(mean_act, filter, layer=None):

    filters = len(mean_act)

    plt.figure(figsize=(16,6))

    extraticks=[filter]

    act = plt.bar(list(range(filters)), mean_act)

    ax = act[0].axes

    ax.set_xticks([0,int(filters/2),filters] + extraticks)

    ax.set_xlim(0,filters)

    ax.plot(filter, mean_act[filter], 'ro')

    plt.axvline(x=filter, color='grey', linestyle='--')

    ax.set_xlabel("feature map")

    ax.set_ylabel("mean activation")

    if layer is not None:

        ax.set_title('Features maps of layer {}'.format(layer))

    plt.show()
input = img2tensor(PIL.Image.fromarray(np.uint8(X[0]*255))).unsqueeze(0).double()

fmv = FeatureMapVisualizer(model, device, layers_base=layers[0], save=None)

activations = fmv.get_activations(monitor, input, mean=False)
# create the global counter

counter = {}

total_filters = 0

for layer in activations.keys():

    filters = activations[layer].features.size(1)

    total_filters += filters

    #print('Layer {}: {} filters'.format(layer, filters))

    for f in range(filters):

        counter[(layer, f)] = [0, 0]

        

print('Filters to track:', total_filters)

for i in range(INPUT_IMAGES):

    top_filters = {}

    input = img2tensor(PIL.Image.fromarray(np.uint8(X[i]*255))).unsqueeze(0).double()

    mean_acts = fmv.get_activations(monitor, input, mean=True)

    #top(3, fmv.output, class_labels)

    for j, layer in enumerate(mean_acts.keys()):

        top_filters = sorted(range(len(mean_acts[layer])), key=lambda idx: (mean_acts[layer][idx], layer), reverse=True)[:TOP]

        for filter in top_filters:

            counter[(layer, filter)][0] += 1

            counter[(layer, filter)][1] += mean_acts[layer][filter]

    
idxs = sorted(counter, key=lambda item: (counter[item][0], item[0], counter[item][1]), reverse=True)[0:COMMON*2]

print(idxs)

for i in idxs:

    print(i, counter[i])
# TODO: plot found filters
# Lets generate images that activate some top common filters found

fmv = FeatureMapVisualizer(model, device, layers_base=layers[0], save=save)



fms = []



for i in range(COMMON):

    layer, filter = idxs[i]

    

    fms.append(fmv.visualize(

                            layer=layer, filter=filter, size=56,

                            upscaling_steps=upscaling_steps, opt_steps=opt_steps, blur=3, verbose=verbose

                        )

              )
for i, img in enumerate(fms):

    fig = plt.figure(figsize=(8,8))

    plt.title('L:{:02d} F:{:03d}'.format(*idxs[i]))

    plt.imshow(img)

    plt.show()
def zip_images(name):

    ziph = zipfile.ZipFile('{}.zip'.format(name), 'w', zipfile.ZIP_DEFLATED)

    # ziph is zipfile handle

    for root, dirs, files in os.walk('./images/'):

        for file in files:

            ziph.write(os.path.join(root, file))

    ziph.close()
zip_images('dog_common_filters')
# remove all image files

!rm ./images/*.*

!rmdir ./images/ 