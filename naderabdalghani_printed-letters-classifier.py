import torch

import torchvision.models as models

import torch.optim as optim

from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split

import numpy as np

from os import path

from matplotlib import pyplot as plt

from PIL import Image
use_cuda = torch.cuda.is_available()
def init_data():

    batch_size = 20

    data_transforms = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    dataset = datasets.ImageFolder('../input/camerataken-images-of-printed-english-alphabet/dataset', transform=data_transforms)



    dataset_len = len(dataset)

    train_len = int(0.6 * dataset_len)

    valid_len = int(0.2 * dataset_len)

    test_len = dataset_len - train_len - valid_len



    train_data, valid_data, test_data = random_split(dataset, [train_len, valid_len, test_len])



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    test_loader = DataLoader(test_data, batch_size=batch_size)



    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    class_names = [item[0] for item in dataset.classes]

    return loaders, class_names
def init_model():

    model = models.wide_resnet50_2(pretrained=True)

    for param in model.parameters():

        param.requires_grad = False

    n_inputs = model.fc.in_features

    alphabet_length = 52

    last_layer = torch.nn.Linear(n_inputs, alphabet_length)

    model.fc = last_layer

    model.fc.requires_grad = True

    if use_cuda:

        model = model.cuda()

    return model
def train(n_epochs, loaders, model, use_cuda):

    valid_loss_min = np.Inf

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9)



    for epoch in range(1, n_epochs + 1):

        train_loss = 0.0

        valid_loss = 0.0



        ###################

        # train the model #

        ###################

        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss

            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)

            optimizer.step()

            # update training loss

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))



        ######################

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['valid']):

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss

            loss = criterion(output, target)

            # update average validation loss

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))



        # print training/validation statistics

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch,

            train_loss,

            valid_loss

        ))



        # save the model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                valid_loss))

            torch.save(model.state_dict(), "model.pt")

            valid_loss_min = valid_loss

    # return trained model

    return model
def test(loaders, model, use_cuda):

    test_loss = 0.

    correct = 0.

    total = 0.

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):

        if use_cuda:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # update average test loss

        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class

        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label

        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

        total += data.size(0)



    print('Test Loss: {:.6f}\n'.format(test_loss))



    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (

        100. * correct / total, correct, total))
def load_trained_model(model):

    if path.exists("model.pt"):

        model.load_state_dict(torch.load('model.pt'))

        print("Model loaded successfully")

        return

    print("Model needs to be trained first!")

    return
def predict(model, class_names, img):

    model.eval()

    pil_img = Image.fromarray(img.astype(float) * 255).convert('RGB')

    pil_img = Image.open(img_path).convert('RGB')

    pil_img.show()

    img_transforms = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    transformed_img = img_transforms(pil_img)

    transformed_img = torch.unsqueeze(transformed_img, 0)

    output = model(transformed_img)

    _, prediction = torch.max(output, 1)

    print(prediction)

    return class_names[prediction.item()]
def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
loaders, class_names = init_data()
dataiter = iter(loaders['train'])

images, indices = dataiter.next() # _ for no labels

# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(20, 4))

plot_size=20

for idx in np.arange(plot_size):

    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])

    ax.set_title(class_names[indices[idx]], color="green")

    imshow(images[idx])
model = init_model()
model = train(5, loaders, model, use_cuda)
load_trained_model(model)
test(loaders, model, use_cuda)
dataiter = iter(loaders['test'])

images, indices = dataiter.next()

model = model.cpu()

model.eval()

output = model(images)

_, preds = torch.max(output, 1)



fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    ax.set_title("{} ({})".format(str(class_names[preds[idx].item()]), str(class_names[indices[idx].item()])),

                 color=("green" if preds[idx]==indices[idx] else "red"))

    imshow(images[idx])