import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray"

random_seed = 1
torch.manual_seed(random_seed)

transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

train_set = datasets.ImageFolder(data_dir + "/train", transform = transformations)
val_set = datasets.ImageFolder(data_dir + "/val", transform = transformations)
test_set = datasets.ImageFolder(data_dir + "/test", transform = transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

classes = train_set.classes
device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
model = models.resnet50(pretrained=True, progress=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):
    running_loss = 0.0
    counter = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        counter += 1
        if counter % 10 == 0 or counter == len(train_loader):
            print(counter, "/", len(train_loader))
        
    print(running_loss)
print('Finished Training')
class_correct = [0, 0]
class_total = [0, 0]
with torch.no_grad():
    for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
def process_image(image_path):
    size = 256
    
    # Load Image
    img = Image.open(image_path).convert('RGB')
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    img = img.resize((size, int(size*(height/width))) if width < height else (int(size*(width/height)), size))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/size
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    image = image.to(device)
    
    return image
# Using model to predict label
def predict(image, model):
    # Pass the image through model
    output = model.forward(image)
    
    # Reverse the log function in output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()
# Show Image
def show_image(image):
    # Convert image to numpy
    image = image.cpu().numpy()
    
    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445
    
    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))
import os

accurate = 0
false_neg = 0
true_neg = 0
false_pos = 0
true_pos = 0
total = 0

classes = ["NORMAL", "PNEUMONIA"]

normal_total = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL'):
    for filename in filenames:
        # Process Example Image
        image = process_image(dirname + "/" + filename)
        # Give image to model to predict output
        top_prob, top_class = predict(image, model)
        
        if top_class == 0:
            accurate += 1 
        else:
            false_pos += 1
        normal_total += 1
        
        # Print the results
        # print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", classes[top_class]  )
total += normal_total

pneumonia_total = 0
for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA'):
    for filename in filenames:
        # Process Example Image
        image = process_image(dirname + "/" + filename)
        # Give image to model to predict output
        top_prob, top_class = predict(image, model)
            
        if top_class == 1:
            accurate += 1
        else:
            false_neg += 1
        pneumonia_total += 1
    
        # Print the results
        # print("The model is ", top_prob*100, "% certain that the image " + filename + " has a predicted class of ", classes[top_class]  )
total += pneumonia_total

true_neg = normal_total
true_pos = pneumonia_total

print("\n\nFalse Positives / True Positives : " + str(false_pos) + " / " + str(normal_total) + " = " + str(false_pos * 1.0 / normal_total))
print("False Negatives / True Negatives: " + str(false_neg) + " / " + str(pneumonia_total) + " = " + str(false_neg * 1.0 / pneumonia_total))

precision = true_pos * 1.0 / (true_pos + false_pos)
print("Precision (PPV): " + str(true_pos) + " / " + str(true_pos + false_pos) + " = " +  str(precision))

sensitivity = true_pos * 1.0 / (true_pos + false_neg)
print("Sensitivity (Recall): " + str(true_pos) +  " / " + str(true_pos + false_neg) + " = " + str(sensitivity))

recall = true_neg * 1.0 / (true_neg + false_pos)
print("Specificity: " + str(true_neg) + " / " + str(true_neg + false_pos) + " = " + str(recall))

npv = true_neg * 1.0 / (true_neg + false_neg)
print("NPV: " + str(true_neg) + " / " + str(true_neg + false_neg) + " = " + str(npv))

accuracy = accurate * 1.0 / total
print("Accuracy: " + str(accurate) + " / " + str(total) + " = " + str(accuracy))

f1 = (2 * precision * recall) / (precision + recall)
print("F-1 Score: " + str(f1))