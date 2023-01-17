from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://www.who.int/malaria/map-pbo-nets-hompepage.jpg")
import seaborn as sns

# PyTorch

from torchvision import transforms, datasets, models

import torch

from torch import optim, cuda

from torch.utils.data import DataLoader, sampler

import torch.nn as nn



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)



# Data science tools

import numpy as np

import pandas as pd

import os



# Image manipulations

from PIL import Image

# Useful for examining network

# Timing utility

from timeit import default_timer as timer





# Visualizations

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['font.size'] = 14
import os

print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/"))
traindir_parasitized = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized/"

traindir_uninfected = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected/"
def imshow(image):

    """Display Infected image"""

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.axis('off')

    plt.show()





# Example image

x = Image.open(traindir_parasitized + '/C136P97ThinF_IMG_20151005_140538_cell_96.png')

np.array(x).shape

imshow(x)
# Uninfected image

x = Image.open(traindir_uninfected + '/C128P89ThinF_IMG_20151004_131632_cell_18.png')

np.array(x).shape

imshow(x)
def imshow_tensor(image, ax=None, title=None):

    """Imshow for Tensor."""



    if ax is None:

        fig, ax = plt.subplots()



    # Set the color channel as the third dimension

    image = image.numpy().transpose((1, 2, 0))



    # Reverse the preprocessing steps

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean



    # Clip the image pixel values

    image = np.clip(image, 0, 1)



    ax.imshow(image)

    plt.axis('off')



    return ax, image
save_file_name = 'resnet50-transfer-4.pt'

checkpoint_path = 'resnet50-transfer-4.pth'



# Change to fit hardware

batch_size = 128



# Whether to train on a gpu

train_on_gpu = cuda.is_available()

print(f'Train on gpu: {train_on_gpu}')



# Number of gpus

if train_on_gpu:

    gpu_count = cuda.device_count()

    print(f'{gpu_count} gpus detected.')
image_transformers = {'train':

    transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomVerticalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406], 

                                                            [0.229, 0.224, 0.225])]),



'test':transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], 

                                                           [0.229, 0.224, 0.225])]),



'val':transforms.Compose([transforms.Resize(256),

                                            transforms.CenterCrop(224),

                                            transforms.ToTensor(),

                                            transforms.Normalize([0.485, 0.456, 0.406], 

                                                                 [0.229, 0.224, 0.225])])}
data = {'train':datasets.ImageFolder(root ='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',transform=image_transformers['train']),

        'val':datasets.ImageFolder(root ='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',transform=image_transformers['val']),

        'test':datasets.ImageFolder(root ='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',transform=image_transformers['test'])}



# Dataloader iterators

dataloaders = {

    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),

    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),

    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)

}
trainiter = iter(dataloaders['train'])

features, labels = next(trainiter)

features.shape, labels.shape
model = models.resnet50(pretrained=True)

model
for param in model.parameters():

    param.requires_grad = False
n_inputs = model.fc.in_features



# Add on classifier

model.fc = nn.Sequential(

    nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),

    nn.Linear(256, 2), nn.LogSoftmax(dim=1))



model.fc
total_params = sum(p.numel() for p in model.parameters())

print(f'{total_params:,} total parameters.')

total_trainable_params = sum(

    p.numel() for p in model.parameters() if p.requires_grad)

print(f'{total_trainable_params:,} training parameters.')
if train_on_gpu:

    model = model.to('cuda')
model.class_to_idx = data['train'].class_to_idx

model.idx_to_class = {

    idx: class_

    for class_, idx in model.class_to_idx.items()}



list(model.idx_to_class.items())
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters())
for p in optimizer.param_groups[0]['params']:

    if p.requires_grad:

        print(p.shape)
def train(model,criterion,optimizer,train_loader,

          valid_loader,save_file_name,max_epochs_stop=3,

          n_epochs=20,print_every=1):

    

    """Train a PyTorch Model



    Params

    --------

        model (PyTorch model): cnn to train

        criterion (PyTorch loss): objective to minimize

        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters

        train_loader (PyTorch dataloader): training dataloader to iterate through

        valid_loader (PyTorch dataloader): validation dataloader used for early stopping

        save_file_name (str ending in '.pt'): file path to save the model state dict

        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping

        n_epochs (int): maximum number of training epochs

        print_every (int): frequency of epochs to print training stats



    Returns

    --------

        model (PyTorch model): trained cnn with best weights

        history (DataFrame): history of train and validation loss and accuracy

    """



    # Early stopping intialization

    epochs_no_improve = 0

    valid_loss_min = np.Inf



    valid_max_acc = 0

    history = []



    # Number of epochs already trained (if using loaded in model weights)

    try:

        print(f'Model has been trained for: {model.epochs} epochs.\n')

    except:

        model.epochs = 0

        print(f'Starting Training from Scratch.\n')



    overall_start = timer()



    # Main loop

    for epoch in range(n_epochs):



        # keep track of training and validation loss each epoch

        train_loss = 0.0

        valid_loss = 0.0



        train_acc = 0

        valid_acc = 0



        # Set to training

        model.train()

        start = timer()



        # Training loop

        for ii, (data, target) in enumerate(train_loader):

            # Tensors to gpu

            if train_on_gpu:

                data, target = data.cuda(), target.cuda()



            # Clear gradients

            optimizer.zero_grad()

            # Predicted outputs are log probabilities

            output = model(data)



            # Loss and backpropagation of gradients

            loss = criterion(output, target)

            loss.backward()



            # Update the parameters

            optimizer.step()



            # Track train loss by multiplying average loss by number of examples in batch

            train_loss += loss.item() * data.size(0)



            # Calculate accuracy by finding max log probability

            _, pred = torch.max(output, dim=1)

            correct_tensor = pred.eq(target.data.view_as(pred))

            # Need to convert correct tensor from int to float to average

            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            # Multiply average accuracy times the number of examples in batch

            train_acc += accuracy.item() * data.size(0)



            # Track training progress

            print(

                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',

                end='\r')



        # After training loops ends, start validation

        else:

            model.epochs += 1



            # Don't need to keep track of gradients

            with torch.no_grad():

                # Set to evaluation mode

                model.eval()



                # Validation loop

                for data, target in valid_loader:

                    # Tensors to gpu

                    if train_on_gpu:

                        data, target = data.cuda(), target.cuda()



                    # Forward pass

                    output = model(data)



                    # Validation loss

                    loss = criterion(output, target)

                    # Multiply average loss times the number of examples in batch

                    valid_loss += loss.item() * data.size(0)



                    # Calculate validation accuracy

                    _, pred = torch.max(output, dim=1)

                    correct_tensor = pred.eq(target.data.view_as(pred))

                    accuracy = torch.mean(

                        correct_tensor.type(torch.FloatTensor))

                    # Multiply average accuracy times the number of examples

                    valid_acc += accuracy.item() * data.size(0)



                # Calculate average losses

                train_loss = train_loss / len(train_loader.dataset)

                valid_loss = valid_loss / len(valid_loader.dataset)



                # Calculate average accuracy

                train_acc = train_acc / len(train_loader.dataset)

                valid_acc = valid_acc / len(valid_loader.dataset)



                history.append([train_loss, valid_loss, train_acc, valid_acc])



                # Print training and validation results

                if (epoch + 1) % print_every == 0:

                    print(

                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'

                    )

                    print(

                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'

                    )



                # Save the model if validation loss decreases

                if valid_loss < valid_loss_min:

                    # Save model

                    torch.save(model.state_dict(), save_file_name)

                    # Track improvement

                    epochs_no_improve = 0

                    valid_loss_min = valid_loss

                    valid_best_acc = valid_acc

                    best_epoch = epoch



                # Otherwise increment count of epochs with no improvement

                else:

                    epochs_no_improve += 1

                    # Trigger early stopping

                    if epochs_no_improve >= max_epochs_stop:

                        print(

                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'

                        )

                        total_time = timer() - overall_start

                        print(

                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'

                        )



                        # Load the best state dict

                        model.load_state_dict(torch.load(save_file_name))

                        # Attach the optimizer

                        model.optimizer = optimizer



                        # Format history

                        history = pd.DataFrame(

                            history,

                            columns=[

                                'train_loss', 'valid_loss', 'train_acc',

                                'valid_acc'

                            ])

                        return model, history



    # Attach the optimizer

    model.optimizer = optimizer

    # Record overall time and print out stats

    total_time = timer() - overall_start

    print(

        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'

    )

    print(

        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'

    )

    # Format history

    history = pd.DataFrame(

        history,

        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    return model, history
model, history = train(model,criterion,optimizer,dataloaders['train'],

                       dataloaders['val'],save_file_name=save_file_name,

                       max_epochs_stop=3,n_epochs=2,print_every=1)
plt.figure(figsize=(8, 6))

for c in ['train_loss', 'valid_loss']:

    plt.plot(

        history[c], label=c)

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Average Negative Log Likelihood')

plt.title('Training and Validation Losses')
plt.figure(figsize=(8, 6))

for c in ['train_acc', 'valid_acc']:

    plt.plot(

        100 * history[c], label=c)

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Average Accuracy')

plt.title('Training and Validation Accuracy')
def save_checkpoint(model, path):

    """Save a PyTorch model checkpoint



    Params

    --------

        model (PyTorch model): model to save

        path (str): location to save model. Must start with `model_name-` and end in '.pth'



    Returns

    --------

        None, save the `model` to `path`



    """



    model_name = path.split('-')[0]

    assert (model_name in ['vgg16', 'resnet50'

                           ]), "Path must have the correct model name"



    # Basic details

    checkpoint = {'class_to_idx': model.class_to_idx,

                  'idx_to_class': model.idx_to_class,

                  'epochs': model.epochs}



    # Extract the final classifier and the state dictionary

    if model_name == 'resnet50':

        checkpoint['fc'] = model.fc

        checkpoint['state_dict'] = model.state_dict()



    # Add the optimizer

    checkpoint['optimizer'] = model.optimizer

    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()



    # Save the data to the path

    torch.save(checkpoint, path)
save_checkpoint(model, path=checkpoint_path)
def load_checkpoint(path):

    """Load a PyTorch model checkpoint



    Params

    --------

        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'



    Returns

    --------

        None, save the `model` to `path`



    """



    # Get the model name

    model_name = 'resnet50'



    # Load in checkpoint

    checkpoint = torch.load(path)



    if model_name == 'resnet50':

        model = models.resnet50(pretrained=True)

        # Make sure to set parameters as not trainable

        for param in model.parameters():

            param.requires_grad = False

        model.fc = checkpoint['fc']

        

    # Load in the state dict

    model.load_state_dict(checkpoint['state_dict'])



    total_params = sum(p.numel() for p in model.parameters())

    print(f'{total_params:,} total parameters.')

    total_trainable_params = sum(

        p.numel() for p in model.parameters() if p.requires_grad)

    print(f'{total_trainable_params:,} total gradient parameters.')



    # Move to gpu

    if train_on_gpu:

        model = model.to('cuda')



    # Model basics

    model.class_to_idx = checkpoint['class_to_idx']

    model.idx_to_class = checkpoint['idx_to_class']

    model.epochs = checkpoint['epochs']



    # Optimizer

    optimizer = checkpoint['optimizer']

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



    return model, optimizer
!pip install torchsummary

from torchsummary import summary
model, optimizer = load_checkpoint(path=checkpoint_path)

summary(model, input_size=(3, 224, 224), batch_size=batch_size)
model, history = train(model,criterion,optimizer,dataloaders['train'],dataloaders['val'],

                       save_file_name=save_file_name,max_epochs_stop=3,n_epochs=20,print_every=1)
def process_image(image_path):

    """Process an image path into a PyTorch tensor"""



    image = Image.open(image_path)

    # Resize

    img = image.resize((224, 224))



    # Center crop

    width = 256

    height = 256

    new_width = 224

    new_height = 224



    left = (width - new_width) / 2

    top = (height - new_height) / 2

    right = (width + new_width) / 2

    bottom = (height + new_height) / 2

    img = img.crop((left, top, right, bottom))



    # Convert to numpy, transpose color dimension and normalize

    img = np.array(img)/ 256



    # Standardization

    means = np.array([0.485, 0.456, 0.406]).reshape((3,))

    stds = np.array([0.229, 0.224, 0.225]).reshape((3,))



    img = img - means

    img = img / stds



    img_tensor = torch.Tensor(img)



    return img_tensor
x = process_image(traindir_uninfected + '/C100P61ThinF_IMG_20150918_144348_cell_71.png')

x.shape
def predict(image_path, model, topk=2):

    """Make a prediction for an image using a trained model



    Params

    --------

        image_path (str): filename of the image

        model (PyTorch model): trained model for inference

        topk (int): number of top predictions to return



    Returns



    """

    real_class = image_path.split('/')[-2]



    # Convert to pytorch tensor

    img_tensor = process_image(image_path)



    # Resize

    if train_on_gpu:

        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()



    # Set to evaluation

    with torch.no_grad():

        model.eval()

        # Model outputs log probabilities

        out = model(img_tensor)

        ps = torch.exp(out)



        # Find the topk predictions

        topk, topclass = ps.topk(topk, dim=1)



        # Extract the actual classes and probabilities

        top_classes = [model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]

        top_p = topk.cpu().numpy()[0]



        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class
img, top_p, top_classes, real_class = predict(traindir_uninfected + '/C100P61ThinF_IMG_20150918_144348_cell_71.png', model)

img.shape
def display_prediction(image_path, model, topk):

    """Display image and preditions from model"""



    # Get predictions

    img, ps, classes, y_obs = predict(image_path, model, topk)

    # Convert results to dataframe for plotting

    result = pd.DataFrame({'p': ps}, index=classes)



    # Show the image

    plt.figure(figsize=(16, 5))

    ax = plt.subplot(1, 2, 1)

    # Uninfected image

    #x = Image.open(image_path)

    #np.array(x).shape

    #imshow(x)

    ax, img = imshow_tensor(img, ax=ax)



    # Set title to be the actual class

    ax.set_title(y_obs, size=20)



    ax = plt.subplot(1, 2, 2)

    # Plot a bar plot of predictions

    result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)

    plt.xlabel('Predicted Probability')

    plt.tight_layout()
display_prediction(traindir_uninfected + '/C100P61ThinF_IMG_20150918_144348_cell_71.png', model,2)