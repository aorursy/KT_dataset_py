#!ls ../input/padhaivisdata

!ln -s ../input/padhaivisdata/ data

!ls data/
#reading the labels of data we uploaded

with open("data/imagenet_labels.txt") as f:

    classes = eval(f.read())

#type(classes)

print(list(classes.values())[0:5])
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import torch 

import torch.nn as nn

import torchvision

import torchvision.datasets as datasets



import torchvision.models as models

import torchvision.transforms as transforms
# parameters



batch_size = 1 #batch size

cuda = True
#defining the transformations for the data



transform = transforms.Compose([

    transforms.Resize(224),

    transforms.ToTensor(),

    #normalize the images with imagenet data mean and std

    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])
#define the data we uploaded as evaluation data and apply the transformations

evalset = torchvision.datasets.ImageFolder(root = "./data/imagenet", transform = transform)



#create a data loader for evaluation

evalloader = torch.utils.data.DataLoader(evalset, batch_size = batch_size, shuffle = True)
#looking at data using iter

dataiter = iter(evalloader)

images, labels = dataiter.next()



#shape of images bunch

print(images.shape)



#label of the image

print(labels[0].item())
#for visualization we will use vgg16 pretrained on imagenet data



model = models.vgg16(pretrained=True)

if cuda: model.cuda()



model.eval()
def imshow(img, title):

  """Custom function to display the image using matplotlib"""

  

  #define std correction to be made

  std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

  

  #define mean correction to be made

  mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

  

  #convert the tensor img to numpy img and de normalize 

  if cuda: img = img.cpu()

  npimg = np.multiply(img.numpy(), std_correction) + mean_correction

  

  #plot the numpy image

  plt.figure(figsize = (batch_size * 4, 4))

  plt.axis("off")

  plt.imshow(np.transpose(npimg, (1, 2, 0)))

  plt.title(title)

  plt.show()





def show_batch_images(dataloader):

  """custom function to fetch images from dataloader"""



  images,_ = next(iter(dataloader))

  if cuda: images = images.cuda()

  

  #run the model on the images

  outputs = model(images)

  if cuda: outputs = outputs.cpu()

  

  #get the maximum class 

  _, pred = torch.max(outputs.data, 1)

  

  #make grid

  img = torchvision.utils.make_grid(images)

  

  #call the function

  imshow(img, title=[classes[x.item()] for x in pred])

  

  return images, pred
images, pred = show_batch_images(evalloader)
#running inference on the images without occlusion



#vgg16 pretrained model

if cuda: images = images.cuda()

outputs = model(images)



#passing the outputs through softmax to interpret them as probability

outputs = nn.functional.softmax(outputs, dim = 1)



#getting the maximum predicted label

prob_no_occ, pred = torch.max(outputs.data, 1)



#get the first item

prob_no_occ = prob_no_occ[0].item()



print(prob_no_occ)
def occlusion(model, image, label, occ_size = 50, occ_stride = 50, occ_pixel = 0.5):

    """custom function to conduct occlusion experiments"""

  

    #get the width and height of the image

    width, height = image.shape[-2], image.shape[-1]

  

    #setting the output image width and height

    output_height = int(np.ceil((height-occ_size)/occ_stride))

    output_width = int(np.ceil((width-occ_size)/occ_stride))

  

    #create a white image of sizes we defined

    heatmap = torch.zeros((output_height, output_width))

    

    #iterate all the pixels in each column

    for h in range(0, height):

        for w in range(0, width):

            

            h_start = h*occ_stride

            w_start = w*occ_stride

            h_end = min(height, h_start + occ_size)

            w_end = min(width, w_start + occ_size)

            

            if (w_end) >= width or (h_end) >= height:

                continue

            

            input_image = image.clone().detach()

            

            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location

            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            if cuda: input_image = input_image.cuda()

            

            #run inference on modified image

            output = model(input_image)

            output = nn.functional.softmax(output, dim=1)

            prob = output.tolist()[0][label]

            

            #setting the heatmap location to probability value

            heatmap[h, w] = prob 



    return heatmap
heatmap = occlusion(model, images, pred[0].item(), 32, 14)
#displaying the image using seaborn heatmap and also setting the maximum value of gradient to probability

imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=prob_no_occ)

figure = imgplot.get_figure()    

figure.savefig('svm_conf.png', dpi=400)
#for filter visualization, we will use alexnet pretrained with imagenet data



alexnet = models.alexnet(pretrained=True)

#if cuda: alexnet.cuda()



print(alexnet)
def plot_filters_single_channel_big(t):

    

    #setting the rows and columns

    nrows = t.shape[0]*t.shape[2]

    ncols = t.shape[1]*t.shape[3]

    

    

    npimg = np.array(t.numpy(), np.float32)

    npimg = npimg.transpose((0, 2, 1, 3))

    npimg = npimg.ravel().reshape(nrows, ncols)

    

    npimg = npimg.T

    

    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    

    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)

    



def plot_filters_single_channel(t):

    

    #kernels depth * number of kernels

    nplots = t.shape[0]*t.shape[1]

    ncols = 12

    

    nrows = 1 + nplots//ncols

    #convert tensor to numpy image

    npimg = np.array(t.numpy(), np.float32)

    

    count = 0

    fig = plt.figure(figsize=(ncols, nrows))

    

    #looping through all the kernels in each channel

    for i in range(t.shape[0]):

        for j in range(t.shape[1]):

            count += 1

            ax1 = fig.add_subplot(nrows, ncols, count)

            npimg = np.array(t[i, j].numpy(), np.float32)

            npimg = (npimg - np.mean(npimg)) / np.std(npimg)

            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))

            ax1.imshow(npimg)

            ax1.set_title(str(i) + ',' + str(j))

            ax1.axis('off')

            ax1.set_xticklabels([])

            ax1.set_yticklabels([])

   

    plt.tight_layout()

    plt.show()

    



def plot_filters_multi_channel(t):

    

    #get the number of kernals

    num_kernels = t.shape[0]    

    

    #define number of columns for subplots

    num_cols = 12

    #rows = num of kernels

    num_rows = num_kernels

    

    #set the figure size

    fig = plt.figure(figsize=(num_cols,num_rows))

    

    #looping through all the kernels

    for i in range(t.shape[0]):

        ax1 = fig.add_subplot(num_rows,num_cols,i+1)

        

        #for each kernel, we convert the tensor to numpy 

        npimg = np.array(t[i].numpy(), np.float32)

        #standardize the numpy image

        npimg = (npimg - np.mean(npimg)) / np.std(npimg)

        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))

        npimg = npimg.transpose((1, 2, 0))

        ax1.imshow(npimg)

        ax1.axis('off')

        ax1.set_title(str(i))

        ax1.set_xticklabels([])

        ax1.set_yticklabels([])

        

    plt.savefig('myimage.png', dpi=100)    

    plt.tight_layout()

    plt.show()

    

def plot_weights(model, layer_num, single_channel = True, collated = False):

  

  #extracting the model features at the particular layer number

  layer = model.features[layer_num]

  

  #checking whether the layer is convolution layer or not 

  if isinstance(layer, nn.Conv2d):

    #getting the weight tensor data

    weight_tensor = model.features[layer_num].weight.data

    

    if single_channel:

      if collated:

        plot_filters_single_channel_big(weight_tensor)

      else:

        plot_filters_single_channel(weight_tensor)

        

    else:

      if weight_tensor.shape[1] == 3:

        plot_filters_multi_channel(weight_tensor)

      else:

        print("Can only plot weights with three channels with single channel = False")

        

  else:

    print("Can only visualize layers which are convolutional")
#visualize weights for alexnet - first conv layer



plot_weights(alexnet, 0, single_channel = False)
#plotting single channel images



plot_weights(alexnet, 0, single_channel = True)
#plot for 3rd layer -> 2nd conv layer



plot_weights(alexnet, 3, single_channel = True)
plot_weights(alexnet, 0, single_channel = True, collated = True)
plot_weights(alexnet, 3, single_channel = True, collated = True)
plot_weights(alexnet, 6, single_channel = True, collated = True)