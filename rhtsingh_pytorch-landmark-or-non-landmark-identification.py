!wget https://raw.githubusercontent.com/CSAILVision/places365/master/wideresnet.py
import pandas as pd

from tqdm import tqdm

import cv2

import torch

from torch.autograd import Variable as V

import torchvision.models as models

from torchvision import transforms as trn

from torch.nn import functional as F

import os

import numpy as np

import cv2

from PIL import Image

import gc

gc.enable()

import matplotlib.pyplot as plt
def load_labels():

    # prepare all the labels

    # scene category relevant

    file_name_category = 'categories_places365.txt'

    if not os.access(file_name_category, os.W_OK):

        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'

        os.system('wget ' + synset_url)

    classes = list()

    with open(file_name_category) as class_file:

        for line in class_file:

            classes.append(line.strip().split(' ')[0][3:])

    classes = tuple(classes)



    # indoor and outdoor relevant

    file_name_IO = 'IO_places365.txt'

    if not os.access(file_name_IO, os.W_OK):

        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'

        os.system('wget ' + synset_url)

    with open(file_name_IO) as f:

        lines = f.readlines()

        labels_IO = []

        for line in lines:

            items = line.rstrip().split()

            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor

    labels_IO = np.array(labels_IO)

    return classes, labels_IO
def returnTF():

# load the image transformer

    tf = trn.Compose([

        trn.ToPILImage(),

        trn.Resize((224,224)),

        trn.ToTensor(),

        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    return tf
def hook_feature(module, input, output):

    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    

def recursion_change_bn(module):

    if isinstance(module, torch.nn.BatchNorm2d):

        module.track_running_stats = 1

    else:

        for i, (name, module1) in enumerate(module._modules.items()):

            module1 = recursion_change_bn(module1)

    return module



def load_model():

    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'

    if not os.access(model_file, os.W_OK):

        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)

        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')



    import wideresnet

    model = wideresnet.resnet18(num_classes=365)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

    model.load_state_dict(state_dict)

    

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...

    for i, (name, module) in enumerate(model._modules.items()):

        module = recursion_change_bn(model)

    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)



    model.eval()

    # hook the feature extractor

    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet

    for name in features_names:

        model._modules.get(name).register_forward_hook(hook_feature)

    return model
classes, labels_IO = load_labels()

features_blobs = []

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model = load_model()

model = model.to(device)



tf = returnTF()



params = list(model.parameters())

weight_softmax = params[-2].data.cpu().numpy()

weight_softmax[weight_softmax<0] = 0
MIN_SAMPLES_PER_CLASS = 50

train = pd.read_csv('../input/landmark-recognition-2020/train.csv')

test = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

print(train.shape)

counts = train.landmark_id.value_counts()

selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index

num_classes = selected_classes.shape[0]

print('classes with at least N samples:', num_classes)

train = train.loc[train.landmark_id.isin(selected_classes)]

print(train.shape)
io_test = []

for i, img_id in tqdm(enumerate(test.id), total=len(test)):

    image_path = f"../input/landmark-recognition-2020/test/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    input_img = V(tf(img).unsqueeze(0))

    logit = model.forward(input_img.cuda())

    h_x = F.softmax(logit, 1).data.squeeze()

    probs, idx = h_x.sort(0, True)

    probs = probs.cpu().numpy()

    idx = idx.cpu().numpy()



    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor

    if io_image < 0.5:

        io_test.append(0) 

    else:

        io_test.append(1) 

        

    del input_img

    del img

    del image_path

    del logit

    del probs

    del idx

    del io_image

    if i%1000 ==0:

        gc.collect()
test['io'] = io_test

test.to_csv('test_io.csv',index=False)
def return_img(img_id):

    image_path = f"../input/landmark-recognition-2020/test/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"

    img = np.array(Image.open(image_path).resize((224, 224), Image.LANCZOS))

    return img
# Get 'landmark' images

n = 16

landmark_images =  test[test['io'] == 1]['id'][:n]



fig = plt.figure(figsize = (16, 16))

for i, img_id in enumerate(landmark_images):

    image = return_img(img_id)

    fig.add_subplot(4, 4, i+1)

    plt.title(img_id)

    plt.imshow(image)
# Get non 'landmark' images

n = 16

landmark_images =  test[test['io'] == 0]['id'][:n]

# landmark_indexes = landmark_images[:n].index.values



# Plot image examples

fig = plt.figure(figsize = (16, 16))

for i, img_id in enumerate(landmark_images):

    image = return_img(img_id)

    fig.add_subplot(4, 4, i+1)

    plt.title(img_id)

    plt.imshow(image)