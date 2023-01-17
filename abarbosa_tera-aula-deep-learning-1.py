# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
PATH_TO_IMAGES = "../input/imgs/imgs/"
size=224
from fastai.imports import *
from fastai.transforms import *
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from torchvision import models, transforms
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
torch.cuda.is_available(), torch.backends.cudnn.enabled
import numpy as np
import os
fnames = np.array([f'{f}' for f in sorted(os.listdir(f'{PATH_TO_IMAGES}'))])
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
fnames
# Let's get our class labels.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}
l = range(1,6)
{key: labels[key] for key in labels.keys() & l}
# Initialize the pre-trained model
model = models.resnet50(pretrained=True)
#default is train mode we need to change it
model.eval();
img = plt.imread(f'{PATH_TO_IMAGES}{fnames[1]}')
plt.imshow(img);
# Image pre-processing transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    normalize
])
# Apply transforms
img_tensor = preprocess(img)
# Add a batch dimension
img_tensor.unsqueeze_(0)
# Forward pass without activation
fc_out = model(Variable(img_tensor))
from torch.nn import functional as f

h_x = f.softmax(fc_out, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
idx = idx.tolist()
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], labels[idx[i]]))