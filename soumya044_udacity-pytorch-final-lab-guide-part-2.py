!wget "YOUR_COPIED_LINK_TO_MODEL_STATE_CHECKPOINT_FROM_PART_1"
# You will get model.pt as the download result
import torch
import torch.nn as nn
from torchvision import models

#Load your model to this variable
model = models.vgg16(pretrained = True) ## Change this if you don't use VGG16

#Add Last Linear Layer, n_inputs -> 102 flower classses
n_inputs = model.classifier[6].in_features ## ResNet and Inception Code may differ slightly, refer Part 1
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, 102)
model.classifier[6] = last_layer


#Add Loss Function (Categorical Cross Entropy)
criterion = nn.CrossEntropyLoss()

#Specify Optimizer: SGD with LR 0.01
import torch.optim as optim
optimizer = optim.SGD(model.classifier.parameters(), lr = 0.01) ## ResNet, Inception may differ Slightly
# USE model.parameters() in ResNet or Inception

#Load Model State Dictionary from Downloaded model.pt file
Model_State_Path = '/home/workspace/model.pt'
# Convert GPU Model State Dictionary to CPU based
model.load_state_dict(torch.load(Model_State_Path, map_location=lambda storage, loc: storage),strict=False)
model.eval() #Don't Forget to add this Line

#### DEFAULT THINGS FOR UDACITY WORKSPACE #######
# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224 #For Inception v3 it is 299
# Values you used for normalizing the images. Default here are for 
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
!wget "LINK_TO_CLASSIFIER_PICKLE_PATH"
# You will get the whole model path in .pt or .pth format
#Load your model to this variable
model = torch.load("PATH_TO_MODEL_SAVE_FILE", map_location=lambda storage, loc: storage)
#Add Loss Function (Categorical Cross Entropy)
criterion = nn.CrossEntropyLoss()

#Specify Optimizer: SGD with LR 0.01
import torch.optim as optim
optimizer = optim.SGD(model.classifier.parameters(), lr = 0.01) ## ResNet, Inception may differ Slightly
# USE model.parameters() in ResNet or Inception

model.eval() #Don't Forget to add this Line

#### DEFAULT THINGS FOR UDACITY WORKSPACE #######
# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224 #For Inception v3 it is 299
# Values you used for normalizing the images. Default here are for 
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]