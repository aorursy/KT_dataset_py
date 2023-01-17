import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import shutil
import warnings
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchvision
import cv2
import math
torch.cuda.is_available()
warnings.filterwarnings('ignore')
def seed_system(seed=4):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
seed = seed_system()
def flush_folders(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory) 
    return True
data_inpath = '/kaggle/input/indian-dance-form-classification/train/'
classes_data_path = sorted(os.listdir(data_inpath))
classes = [i for i in classes_data_path]

working_dir = '/kaggle/working/store/'
make_folders = ['train/','valid/']

def create_folder_structure(classes_data_path,home=working_dir,make_folders=make_folders):
    if not os.path.exists(home):
        os.makedirs(home)
    for path in tqdm(make_folders):
        for c in classes_data_path:
            data_path = home+path+c
            if not os.path.exists(data_path):
                os.makedirs(data_path)
    return True

def copy_data(files,destination):
    for f in files:
        shutil.copy(f,destination)
    return

def stratify_sample(class_name,data_path,split_ratio=0.8):
    all_files = os.listdir(data_path)
    num_files = len(all_files)
    print('Class {} having {} samples'.format(class_name,num_files))
    train_valid_indices = list(range(0,num_files)) 
    random.shuffle(train_valid_indices)
    nnp = round(len(train_valid_indices)*split_ratio)
    train_indices = train_valid_indices[:nnp]
    valid_indices = [i for i in train_valid_indices if i not in train_indices]
    train_files = [data_path+'/'+all_files[i] for i in train_indices]
    valid_files = [data_path+'/'+all_files[i] for i in valid_indices]
    
    return num_files,train_files,valid_files

flush = flush_folders(directory=working_dir)
create_folder = create_folder_structure(classes_data_path=classes_data_path)
distribution = []
for class_name in classes_data_path:
    num_files,train_files,valid_files = stratify_sample(class_name=class_name,data_path=data_inpath+class_name)
    train_destination,valid_destination = working_dir+'train/'+class_name,working_dir+'valid/'+class_name
    copy_train = copy_data(files=train_files,destination=train_destination)
    copy_valid = copy_data(files=valid_files,destination=valid_destination)
    distribution.append([class_name,num_files,len(train_files),len(valid_files)])
    
df_train_valid = pd.DataFrame(distribution)
df_train_valid.columns=['class_name','total_population','train_size','test_size']
def visualize_images(path, n_images,class_name,is_random=True, figsize=(16, 16)):
    plt.figure(figsize=figsize)
    w = int(n_images ** .5)
    h = math.ceil(n_images / w)
    
    all_names = os.listdir(path)
    image_names = all_names[:n_images]   
    if is_random:
        image_names = random.sample(all_names, n_images)
            
    for ind, image_name in enumerate(image_names):
        img = cv2.imread(os.path.join(path, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_shape = img.shape
        plt.subplot(h, w, ind + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(class_name+' '+str(img_shape))
    
    plt.show()

for path in classes_data_path:
    class_path = data_inpath+path
    visualize_images(path=class_path,n_images=9,class_name=path)
#Plot Data Distribution
# plt.close()
#plt.figure(figsize=(20,20))
df_train_valid.index=df_train_valid.class_name
df_train_valid.plot.bar(x='class_name', y=['train_size','test_size'], color = ['orange','grey'],rot=90)
plt.title('Train-Validation Data Distribution(Split Ratio 80:20)')
plt.xlabel('Class Name')
plt.ylabel('Population Size')
plt.show()
# import cv2
# img = cv2.imread('../input/indian-dance-form-classification/train/bharatanatyam/aug_0_1136.png')
# img.shape
image_size = 256
train_transforms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                       #transforms.RandomCrop(256)
                                       #transforms.RandomSizedCrop(image_size),
                                       
                                       transforms.RandomRotation(20),
                                       #transforms.RandomGrayscale(p=0.1),
                                       transforms.RandomHorizontalFlip(),
                                       #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                       #transforms.RandomAffine(degrees=(10,20),translate=(0.25, 0.25),scale=(1.2, 2.0)),
                                       #transforms.RandomPerspective(distortion_scale=0.1, p=0.1, interpolation=3, fill=0),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                            
                                      ])

test_transforms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                       transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                       ])

# # Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(working_dir+'/train', transform=train_transforms)
test_data = datasets.ImageFolder(working_dir+'/valid', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True)
#Summarize Train and Validation Datasets
display(trainloader.dataset)
display(testloader.dataset)
#Visualize train data before fitting to model
images, labels = iter(trainloader).next()
plt.close()
plt.figure(figsize=(20,20))
grid = torchvision.utils.make_grid(images[:16])
plt.title('Training Data')
plt.axis('off')
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.show()
#Visualize validation data before fitting to model
images, labels = iter(testloader).next()
plt.close()
plt.figure(figsize=(20,20))
grid = torchvision.utils.make_grid(images[:16])
plt.title('Validation Data')
plt.axis('off')
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.show()
#Load Pre-trained Model for Transfer Learning
model = models.vgg11(pretrained=True)
model.classifier
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
n_classes = len(classes_data_path)
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 512)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(512,64)),
                        ('relu', nn.ReLU()),
    
        ('fc3', nn.Linear(64,n_classes)),
                         ('output', nn.LogSoftmax(dim=1))
                          ]))


lr = 0.0005
step_size = 5
gamma = 0.75

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
#lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.25, patience=4,verbose=True)# step_size=step_size, gamma=gamma)


device = 'cuda'

model.to(device)
epochs = 30
steps = 0

train_losses, test_losses = [], []
epoch_lrs = []
iteration_lr = []
for e in range(epochs):
    #print(e)
    running_loss = 0
    epoch_lr = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        #print(lr_scheduler)
        
        iteration_lr.append(lr_scheduler.get_lr()[0])
        running_loss += loss.item()
        epoch_lr += lr_scheduler.get_lr()[0]
        
    else:
        test_loss = 0
        accuracy = 0
        #print('Validation Starts!')
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train(test_loss)
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        epoch_lrs.append(epoch_lr/len(trainloader))
        lr_scheduler.step()

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "LR: {:.5f}.. ".format(epoch_lrs[-1]),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
#Save Model
checkpoint = {'model': model,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
#Load Model for Inference
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint('checkpoint.pth')
#Visualize Prediction for Validation/Test Data
samples, _ = iter(testloader).next()
samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
fig.tight_layout()
output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
# ad = {0:'cat', 1:'dog'}
ad = {}
for i in range(0,n_classes):
    ad[i] = classes_data_path[i]
for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
def view_classify(img, ps,classes):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(16,12), ncols=2)
    ax1.imshow(img.cpu().resize_(1, 256, 256).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(len(classes)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_yticklabels(classes,size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
plt.close()
fig.tight_layout()

samples, _ = iter(testloader).next()
samples = samples.to(device)
img_idx = 0
sample = samples[img_idx,:].view(1,3,256,256)
output = model(sample)
ps = torch.exp(output).cpu()
view_classify(sample, ps,classes=classes)
plt.show()
# track test loss 

test_loss = 0.0
class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
y_pred = []
y_test = []

model.eval() # eval mode

# iterate over test data
for data, target in testloader:
    # move tensors to GPU if CUDA is available
    
    data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    y_pred.extend(list(pred.cpu().data.numpy()))
    y_test.extend(list(target.cpu().data.numpy()))
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(64):
        try:
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
        except:
            pass

# calculate avg test loss
test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(8):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
from sklearn.metrics import confusion_matrix
import seaborn as sn

y_pred = [classes[i] for i in y_pred]
y_test = [classes[i] for i in y_test]
y_pred = np.array(y_pred)
cm_array = confusion_matrix(y_test, y_pred,labels=classes)
df_cm = pd.DataFrame(cm_array,columns=classes)
df_cm.index=classes
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
def compute_accuracies(c_mat):
    accuracies = c_mat.astype('float') / c_mat.sum(axis=1)
    accuracies = accuracies.diagonal()
    accuracies = {k:v for k, v in zip(labels, accuracies)}
    return accuracies
inf_data_path = '/kaggle/input/indian-dance-form-classification/test/'
random_img = random.sample(os.listdir(inf_data_path),1)
random_img
test_image_path = '/kaggle/input/'

from PIL import Image
img = Image.open(inf_data_path+random_img[0])
transformed_img = test_transforms(img)
transformed_img = transformed_img.unsqueeze(0)
display(img)
output = model(transformed_img.to(device))
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
print(prediction_score,pred_label_idx)
pred_label_idx.squeeze_()
print(pred_label_idx)
predicted_label = classes[pred_label_idx]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
!pip install captum
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
integrated_gradients = IntegratedGradients(model.cpu())
attributions_ig = integrated_gradients.attribute(transformed_img, target=pred_label_idx, n_steps=20)
default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)
noise_tunnel = NoiseTunnel(integrated_gradients)
attributions_ig_nt = noise_tunnel.attribute(transformed_img.cpu(), n_samples=1, nt_type='smoothgrad_sq', target=pred_label_idx)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)
gradient_shap = GradientShap(model)

# Defining baseline distribution of images
rand_img_dist = torch.cat([transformed_img.cpu() * 0, transformed_img.cpu() * 1])

attributions_gs = gradient_shap.attribute(transformed_img.cpu(),
                                          n_samples=10,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=pred_label_idx)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)