%matplotlib inline



import pandas as pd

import numpy as np



from collections import OrderedDict



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import torch

from torch import optim, nn

from torchvision import datasets, transforms, models

from PIL import Image



import random, os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
print(os.listdir("../input"))

data_dir = '../input'

label_file = '../input/monkey_labels.txt'

train_dir = data_dir + '/training/training'

valid_dir = data_dir + '/validation/validation'

print(os.listdir(train_dir))
labelCols = ["label", "latin_name", "common_name", "train_images", "valid_images"]

monkeyLabels = pd.read_csv(label_file, skiprows = 1, sep=',', names=labelCols)

monkeyLabels
for i in ['label', 'latin_name', 'common_name']:

    monkeyLabels[i] = monkeyLabels[i].map(lambda x: x.lstrip(' \n\t').rstrip(' \n\t'))

monkeyLabels
"""Constants"""

CATEGORIES = len(monkeyLabels['label'])

ARCHITECTURE = 'resnet18'

LAYERS = 3

AVAILABLE_MODELS = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet', 'resnet18']

EPOCHS = 40

BATCH_SIZE = 64

LEARN_RATE = 0.001

MEANS = [0.485, 0.456, 0.406]

STANDARD_DEVIATIONS = [0.229, 0.224, 0.225]

CROP_DIMENSION = 224

RESIZE_DIMENSION = 256

NAME_DISPLAY_TYPE = 'common_name'





train_loss, validation_loss, train_accuracy, validation_accuracy = [], [], [], []
def create_classifier(hidden_units, current_in_features):

    """



    :param hidden_units:

    :param current_in_features:

    :return:

    """

    print("Creating Classifier")

    hidden_layers = []

    hidden_steps = int((hidden_units - CATEGORIES)/LAYERS)

    start = current_in_features

    end = hidden_units

    for i in range(1, LAYERS + 1):

        hidden_layers.append(('fc{}'.format(i), nn.Linear(start, end)))

        hidden_layers.append(('dropout{}'.format(i), nn.Dropout(p=0.1)))

        hidden_layers.append(('relu{}'.format(i), nn.ReLU()))

        start = end

        end = start - hidden_steps



    hidden_layers.append(('outputs', nn.Linear(start, CATEGORIES)))

    hidden_layers.append(('logsoftmax', nn.LogSoftmax(dim=1)))



    classifier = nn.Sequential(OrderedDict(hidden_layers))

    return classifier





def build_neural_net(architecture, hidden_units):

    """



    :param architecture:

    :param hidden_units:

    :return:

    """

    print('Building Neural Network')

    if architecture in AVAILABLE_MODELS:

        # Download the pretrained model based on architecture

        neural_network_model = getattr(models, architecture)(pretrained=True)

        

        # Freeze model weights

        for param in neural_network_model.parameters():

            param.requires_grad = False

            

        # Acquire the current in features of the pretrained classifier for new feed forward network

        if architecture == 'alexnet':

            current_in_features = neural_network_model.classifier._modules['1'].in_features

            neural_network_model.classifier = create_classifier(hidden_units, current_in_features)

        elif architecture == 'resnet18':

            current_in_features = neural_network_model.fc.in_features

            neural_network_model.fc = create_classifier(hidden_units, current_in_features)

        else:

            current_in_features = neural_network_model.classifier._modules['0'].in_features

            neural_network_model.classifier = create_classifier(hidden_units, current_in_features)

    else:

        # Raise exception if architecture doesn't match available models

        raise Exception("Architecture of requested pretrained model is not available")



    return neural_network_model



def random_photo(directory):

    random_dir = random.choice([x for x in os.listdir(directory)])

    random_filename = random.choice([x for x in os.listdir(directory + '/' + random_dir)])

    full_path = "%s/%s/%s" % (directory,random_dir,random_filename)

    return full_path



def plot_images(tensor, count=1, normalize=True):

    

    images, labels = next(tensor)

    a = np.floor(count**0.5).astype(int)

    b = np.ceil(1.*count/a).astype(int)

    fig = plt.figure(figsize=(3.*b,3.*a))

    for i in range(1,count+1):

        ax = fig.add_subplot(a,b,i)

        ax.plot([1,2,3],[1,2,3])

        forDisplay = images[i].numpy().transpose((1, 2, 0))



        if normalize:

            mean = np.array(MEANS)

            standard_deviation = np.array(STANDARD_DEVIATIONS)

            forDisplay = standard_deviation * forDisplay + mean

            forDisplay = np.clip(forDisplay, 0, 1)

        ax.imshow(forDisplay)

        ax.set_title(monkeyLabels.loc[labels[i].item(), NAME_DISPLAY_TYPE])

        ax.set_axis_off()

    

    plt.show()

    

def plot_performance(train_loss, validation_loss, train_accuracy, validation_accuracy):

    plt.subplot(1, 2, 1)

    plt.plot(train_loss, label='Training loss')

    plt.plot(validation_loss, label='Validation loss')

    plt.title('Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')



    # Put a nicer background color on the legend.

    legend.get_frame().set_facecolor('C7')



    plt.subplot(1, 2, 2)

    plt.plot(train_accuracy, label='Training accuracy')

    plt.plot(validation_accuracy, label='Validation accuracy')

    plt.title('Accuracy')

    plt.ylabel('Percent Accurate')

    plt.xlabel('Epochs')

    legend = plt.legend(loc='upper left', shadow=True, fontsize='small')



    # Put a nicer background color on the legend.

    legend.get_frame().set_facecolor('C7')

    

    plt.tight_layout()

    

def train_model(model, criterion, optimizer, num_epochs=EPOCHS):

    model = model.to(device)

    

    for epoch in range(num_epochs):

        for phase in ['train', 'validation']:

            if phase == 'train':

                model.train()

                print("Training: Epoch %d" % epoch)

            else:

                model.eval()

                print("Validation: Epoch %d" % epoch)

                

            running_loss = 0.0

            running_corrects = 0



            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                outputs = model(inputs)

                loss = criterion(outputs, labels)



                if phase == 'train':

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()



                _, preds = torch.max(outputs, 1)

                running_loss += loss.detach() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            

            if phase == 'validation':

                validation_loss.append(epoch_loss.item())

                validation_accuracy.append(epoch_acc.item() * 100)

            else:

                train_loss.append(epoch_loss.item())

                train_accuracy.append(epoch_acc.item() * 100)

            

    

    plot_performance(train_loss, validation_loss, train_accuracy, validation_accuracy)
def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,

        returns an Numpy array'''

    

    process_transforms = transforms.Compose([transforms.Resize(RESIZE_DIMENSION),

                                             transforms.CenterCrop(CROP_DIMENSION),

                                             transforms.ToTensor(),

                                             transforms.Normalize(MEANS, STANDARD_DEVIATIONS),

                                            ])

    

    pil_image = Image.open(image)

    tensor_image = process_transforms(pil_image)

    numpy_image = tensor_image.numpy()

    

    return numpy_image



def imshow(image, ax=None, title=None):

    if ax is None:

        fig, ax = plt.subplots()

    

    # PyTorch tensors assume the color channel is the first dimension

    # but matplotlib assumes is the third dimension

    image = image.transpose((1, 2, 0))

    

    # Undo preprocessing

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed

    image = np.clip(image, 0, 1)

    

    if title is not None:

        ax.set_title(title)

    ax.axis('off')    

    ax.imshow(image)

    

    return ax





def convert_labels(labels):

    names_array = np.array([])

    for i in np.nditer(labels.cpu().numpy()):        

        #print(monkeyLabels.loc[i.item(), NAME_DISPLAY_TYPE])

        names_array = np.append(names_array, monkeyLabels.loc[i.item(), NAME_DISPLAY_TYPE])

    return names_array



def process_image_tensor(image_path):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,

        returns an Numpy array'''

    

    process_transforms = transforms.Compose([transforms.Resize(RESIZE_DIMENSION),

                                             transforms.CenterCrop(CROP_DIMENSION),

                                             transforms.ToTensor(),

                                             transforms.Normalize(MEANS, STANDARD_DEVIATIONS),

                                            ])

    

    pil_image = Image.open(image_path)

    tensor_image = process_transforms(pil_image)

    

    return tensor_image



def predict(image_path, model, topk=5):

    ''' Predict the class (or classes) of an image using a trained deep learning model.

    '''

    model.to(device)

    model.eval()

    image_tensor = process_image_tensor(image_path).unsqueeze_(0)

    

    with torch.no_grad():

        image_tensor = image_tensor.to(device)

        output = model.forward(image_tensor)

    

    probabilities = torch.exp(output)

    probs, labels = probabilities.topk(topk)

    return probs.cpu().numpy()[0], convert_labels(labels);



def view_class_probability(image_path, neural_network_model):

    probabilities, classes = predict(image_path, neural_network_model)

    

    

    image_elements = image_path.split('/')    

    category_num = image_elements[-2]

    

    fig, (ax1, ax2) = plt.subplots(figsize = (6,10), ncols=2)

    ax1 = plt.subplot(2,1,1)

    imshow(process_image(image_path), ax1)

    

    

    

    # Set up title

    title = monkeyLabels.loc[int(category_num[-1]), NAME_DISPLAY_TYPE]

    

    # Plot image

    img = process_image(image_path)

    imshow(img, ax1, title);

    ax2 = fig.add_subplot(2,1,2)

    y_pos = np.arange(len(classes))

    error = np.random.rand(len(classes))



    ax2.barh(y_pos, probabilities, align='center',

        color='blue')

    ax2.set_yticks(y_pos)

    ax2.set_yticklabels(classes)

    ax2.invert_yaxis()  # labels read top-to-bottom

    ax2.set_xlabel('Probability')

    ax2.set_title('Class Probability')

    plt.tight_layout()

       
# Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.CenterCrop(CROP_DIMENSION),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.RandomVerticalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize(MEANS, STANDARD_DEVIATIONS)

                                      ])



valid_transforms = transforms.Compose([transforms.Resize(RESIZE_DIMENSION), 

                                      transforms.CenterCrop(CROP_DIMENSION), 

                                      transforms.ToTensor(), 

                                      transforms.Normalize(MEANS, STANDARD_DEVIATIONS)

                                     ])



# Load the datasets with ImageFolder

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)



# Using the image datasets and the trainforms, define the dataloaders

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)



dataloaders = {

    "train": train_loader,

    "validation": valid_loader

}
plot_images(iter(train_loader), 9)
plot_images(iter(valid_loader), 9)
neural_network_model = build_neural_net(ARCHITECTURE,LAYERS)

neural_network_model
criterion = nn.NLLLoss()



if ARCHITECTURE == 'resnet18':

    optimizer = optim.Adam(neural_network_model.fc.parameters(), lr=LEARN_RATE)

else:

    optimizer = optim.Adam(neural_network_model.classifier.parameters(), lr=LEARN_RATE)

train_model(neural_network_model, criterion, optimizer)
view_class_probability(random_photo(valid_dir), neural_network_model)
view_class_probability(random_photo(valid_dir), neural_network_model)