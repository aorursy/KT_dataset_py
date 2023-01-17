# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
%matplotlib inline
def get_transforms():
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)])
    
    transforms_dict = {
        'train': train_transforms,
        'valid': valid_transforms,
        'test': test_transforms
    }
    
    return transforms_dict

transforms_dict = get_transforms()
def load_datasets(data_path):
    transforms_dict = get_transforms()
    train_dir = data_path + 'train'
    valid_dir = data_path + 'valid'
    test_dir = data_path + 'test'
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms_dict['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transforms_dict['valid'])
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms_dict['test'])
    
    datasets_dict = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }
    
    return datasets_dict

datasets_dict = load_datasets('../input/flower-recognition/')
def get_data_loaders(data_path):
    datasets = load_datasets(data_path)
    train_loader = DataLoader(datasets['train'], batch_size=64, shuffle=True)
    valid_loader = DataLoader(datasets['valid'], batch_size=32)
    test_loader = DataLoader(datasets['test'], batch_size=32)
    
    data_loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    
    return data_loaders

data_loaders = get_data_loaders('../input/flower-recognition/')
# class number to name
cat_to_name = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
def process_image(image):
    '''
    Turn image to numpy array for use in PyTorch model
    '''
    with Image.open(image) as img:
        input_img = transforms_dict['test'](img).numpy()
    return input_img

# show images with category name
def imshow(image_full_path_name):
    fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = process_image(image_full_path_name)
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks loke noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.set_title(cat_to_name[image_full_path_name.split('/')[-2]])
    ax.imshow(image)

imshow('../input/flower-recognition/train/1/image_06734.jpg')
def create_model(hidden_units=[4096, 1024], arch='vgg16', checkpoint=None):
    model_arch = checkpoint['arch'] if checkpoint else arch
    if 'vgg11' == model_arch:
        model = models.vgg11(pretrained=True)
    elif 'vgg13' == model_arch:
        model = models.vgg13(pretrained=True)
    elif 'vgg19' == model_arch:
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    input_size = checkpoint['input_size'] if checkpoint else model.classifier[0].in_features
    hidden_layers = checkpoint['hidden_layers'] if checkpoint else hidden_units
    
    od = OrderedDict()
    od['fc1'] = nn.Linear(input_size, hidden_layers[0])
    od['relu1'] = nn.ReLU()
    od['dropout1'] = nn.Dropout()
    count = 2
    for i, hidden_layer in enumerate(hidden_layers[1:]):
        od['fc{}'.format(count)] = nn.Linear(hidden_layers[i], hidden_layers[i+1])
        od['relu{}'.format(count)] = nn.ReLU()
        od['dropout{}'.format(count)] = nn.Dropout()
        count += 1
    od['fc{}'.format(count)] = nn.Linear(hidden_layers[-1], 102)
    od['output'] = nn.LogSoftmax(dim=1)
    
    classifier = nn.Sequential(od)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier
    
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    
    return model
def validation(model, valid_loader, criterion):
    model.eval()
    model.to('cuda')
    valid_loss = 0
    valid_accuracy = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model.forward(inputs)
            valid_loss += criterion(outputs, labels).item()
            
            ps = torch.exp(outputs).data
            equality = (labels.data == ps.max(1)[1])
            valid_accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)
def train(epochs, learning_rate, train_loader, valid_loader):
    print_every = 40
    steps = 0
    
    model = create_model()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to('cuda')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validation(model, valid_loader, criterion)
                train_loss /= print_every
                
                print('Epoch: {0}/{1}... '.format(epoch+1, epochs),
                     'Train Loss: {:.4f}'.format(train_loss),
                     'Validation Loss: {0:.4f}, Validation accuracy: {1:.4f}'.format(valid_loss, valid_accuracy))
                
                train_loss = 0
                model.train()
    
    return model
def test(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the test images: {:.2f}%'.format(100 * correct / total))
def save_model(model, arch, train_dataset, hidden_layers):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'arch': arch,
        'input_size': model.classifier[0].in_features,
        'hidden_layers': hidden_layers,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    torch.save(checkpoint, arch + '_checkpoint.pth')
trained_model = train(3, 0.001, data_loaders['train'], data_loaders['valid'])
test(trained_model, data_loaders['test'])
save_model(trained_model, 'vgg16', datasets_dict['train'], [4096, 1024])
