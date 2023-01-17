import os

# for dirname, _, filenames in os.walk('/kaggle/input/stanford-dogs-dataset/images/Images/n02088094-Afghan_hound'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



import torch



import requests

import urllib.request



from PIL import Image

from io import BytesIO
torch.cuda.is_available()
from torchvision import models
# dir(models)
alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

# resnet
from torchvision import transforms





preprocess = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )])
# img = Image.open("/kaggle/input/stanford-dogs-dataset/images/Images/n02109961-Eskimo_dog/n02109961_658.jpg")  # ERROR ('Siberian husky', 50.530677795410156)

img = Image.open("/kaggle/input/stanford-dogs-dataset/images/Images/n02088094-Afghan_hound/n02088094_294.jpg")  # ('Afghan hound, Afghan', 98.31031799316406)



# url = "https://cdn.mos.cms.futurecdn.net/QjuZKXnkLQgsYsL98uhL9X-1200-80.jpg"  # ('Pomeranian', 99.84607696533203)

# url = "https://images.wagwalkingweb.com/media/articles/dog/jack-russell-terrier-allergies/jack-russell-terrier-allergies.jpg"  # ERROR ('beagle', 91.52595520019531)

# url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.pets4homes.co.uk%2Fimages%2Fbreeds%2F44%2Flarge%2Ff71f3b4752554e2bad635759c5cdc45c.jpg&f=1&nofb=1"  # ('toy terrier', 70.98729705810547)



# response = requests.get(url)

# img = Image.open(BytesIO(response.content))



img
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)

# out
target_url = 'https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch2/imagenet_classes.txt'



data = urllib.request.urlopen(target_url)

labels = [line.strip().decode('utf-8') for line in data.readlines()]



# labels
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

labels[index[0]], percentage[index[0]].item()
_, indices = torch.sort(out, descending=True)

[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]