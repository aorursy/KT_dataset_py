import torch

import torch.nn as nn

import torchvision.models as models

import torchvision.transforms as transforms

import os

import PIL.Image as Image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

model = models.resnet50(pretrained=True)

torch.save(model, 'resnet50.pth')
model.eval()
transform = transforms.Compose([

 transforms.Resize(256),

 transforms.CenterCrop(224),

 transforms.ToTensor(),

 transforms.Normalize(

 mean=[0.485, 0.456, 0.406],

 std=[0.229, 0.224, 0.225] 

 )])
def classify(image):

    img_t = transform(image)

    batch_t = torch.unsqueeze(img_t, 0)

    out = model(batch_t)

    with open('../input/imagenetclasses/imagenet1000.txt') as f:

      imagenet_classes = [line.strip() for line in f.readlines()]

    top3_values,top3_indices = out.topk(3)

    top3_indices_as_array = top3_indices.numpy().flatten()

    top3_predicted_classes = []

    for indice in top3_indices_as_array:

        top3_predicted_classes.append(imagenet_classes[indice])

 

    return top3_predicted_classes



def show_image(image):

    #img=mpimg.imread('your_image.png')

    imgplot = plt.imshow(image)

    plt.show()

    
im1 = Image.open("../input/mediamarkt-images/mediamarkt_images/im1.jpeg")

im2 = Image.open("../input/mediamarkt-images/mediamarkt_images/im2.jpeg")

im3 = Image.open("../input/mediamarkt-images/mediamarkt_images/im3.jpeg")

im4 = Image.open("../input/mediamarkt-images/mediamarkt_images/im4.jpeg")

im5 = Image.open("../input/mediamarkt-images/mediamarkt_images/im5.jpeg")

im6 = Image.open("../input/mediamarkt-images/mediamarkt_images/im6.jpeg")

im7 = Image.open("../input/mediamarkt-images/mediamarkt_images/im7.jpeg")

im8 = Image.open("../input/mediamarkt-images/mediamarkt_images/im8.jpeg")

im9 = Image.open("../input/mediamarkt-images/mediamarkt_images/im9.jpeg")

im10 = Image.open("../input/mediamarkt-images/mediamarkt_images/im10.jpeg")

im11 = Image.open("../input/mediamarkt-images/mediamarkt_images/im11.jpeg")

im12 = Image.open("../input/mediamarkt-images/mediamarkt_images/im12.jpeg")

im1_top3_predicted_classes = classify(im1)

im2_top3_predicted_classes = classify(im2)

im3_top3_predicted_classes = classify(im3)

im4_top3_predicted_classes = classify(im4)

im5_top3_predicted_classes = classify(im5)

im6_top3_predicted_classes = classify(im6)

im7_top3_predicted_classes = classify(im7)

im8_top3_predicted_classes = classify(im8)

im9_top3_predicted_classes = classify(im9)

im10_top3_predicted_classes = classify(im10)

im11_top3_predicted_classes = classify(im11)

im12_top3_predicted_classes = classify(im12)
print("im1 top 3 predictions: ")    

print(im1_top3_predicted_classes)

show_image(im1)



print("im2 top 3 predictions: ")    

print(im2_top3_predicted_classes)   

show_image(im2)



print("im3 top 3 predictions: ")    

print(im3_top3_predicted_classes) 

show_image(im3)



print("im4 top 3 predictions: ")    

print(im4_top3_predicted_classes)   

show_image(im4)
print("im5 top 3 predictions: ")    

print(im5_top3_predicted_classes) 

show_image(im5)



print("im6 top 3 predictions: ")    

print(im6_top3_predicted_classes)   

show_image(im6)



print("im7 top 3 predictions: ")    

print(im7_top3_predicted_classes)  

show_image(im7)



print("im8 top 3 predictions: ")    

print(im8_top3_predicted_classes)  

show_image(im8)

print("im9 top 3 predictions: ")    

print(im9_top3_predicted_classes) 

show_image(im9)



print("im10 top 3 predictions: ")    

print(im10_top3_predicted_classes)  

show_image(im10)



print("im11 top 3 predictions: ")    

print(im11_top3_predicted_classes)   

show_image(im11)



print("im12 top 3 predictions: ")    

print(im12_top3_predicted_classes)  

show_image(im12)