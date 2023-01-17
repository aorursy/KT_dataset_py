import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from skimage import io
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import math
import os
from tqdm.notebook import tqdm
from torch.autograd import Variable
class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_features, out_channels, dw_kernel, dw_stride):
        super(DepthWiseSeparableConv, self).__init__()
        self.stride = dw_stride
        self.dw_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=dw_kernel, stride=dw_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=in_features)
        self.pw_conv = nn.Conv2d(in_channels=in_features, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x):
        dims = x.dim()
        if dims < 4:
            raise ValueError("Expected input to be atleast 4 dimensions, found: {}".format(dims))
        sizes = x.size()
        # View (N, C, H, W) as a stack of matrices / images containing a single channel (stack of channels across batch)
        x = x.view(-1, 1, sizes[2], sizes[3])
        # Simply put, we could think of x being a batch of single channel images where batch_size (or) batch = N * C
        x = self.dw_conv(x)
        x = x.view(sizes[0], sizes[1], sizes[2] // self.stride, sizes[3] // self.stride)
        x = F.relu(self.bn1(x))
        # 1 x 1 Conv Module
        x = F.relu(self.bn2(self.pw_conv(x)))
        return x
# Expects images of the shape (224, 224, 3)
class MobileNet(nn.Module):
    def __init__(self, num_classes=37, bounding_box_points=4):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.dw_conv1 = DepthWiseSeparableConv(in_features=32, out_channels=64, dw_kernel=3, dw_stride=1)
        self.dw_conv2 = DepthWiseSeparableConv(in_features=64, out_channels=128, dw_kernel=3, dw_stride=2)
        self.dw_conv3 = DepthWiseSeparableConv(in_features=128, out_channels=128, dw_kernel=3, dw_stride=1)
        self.dw_conv4 = DepthWiseSeparableConv(in_features=128, out_channels=256, dw_kernel=3, dw_stride=2)
        self.dw_conv5 = DepthWiseSeparableConv(in_features=256, out_channels=256, dw_kernel=3, dw_stride=1)
        self.dw_conv6 = DepthWiseSeparableConv(in_features=256, out_channels=512, dw_kernel=3, dw_stride=2)
        self.dw_conv7 = DepthWiseSeparableConv(in_features=512, out_channels=512, dw_kernel=3, dw_stride=1)
        self.dw_conv8 = DepthWiseSeparableConv(in_features=512, out_channels=512, dw_kernel=3, dw_stride=1)
        self.dw_conv9 = DepthWiseSeparableConv(in_features=512, out_channels=512, dw_kernel=3, dw_stride=1)
        self.dw_conv10 = DepthWiseSeparableConv(in_features=512, out_channels=512, dw_kernel=3, dw_stride=1)
        self.dw_conv11 = DepthWiseSeparableConv(in_features=512, out_channels=512, dw_kernel=3, dw_stride=1)
        self.dw_conv12 = DepthWiseSeparableConv(in_features=512, out_channels=1024, dw_kernel=3, dw_stride=2)
        self.dw_conv13 = DepthWiseSeparableConv(in_features=1024, out_channels=1024, dw_kernel=3, dw_stride=1)
        
        self.avg_pool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.fc_classification = nn.Linear(in_features=1 * 1 * 1024, out_features=num_classes)
        self.fc_bounding_box = nn.Linear(in_features=1 * 1 * 1024, out_features=bounding_box_points)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.dw_conv5(x)
        x = self.dw_conv6(x)
        x = self.dw_conv7(x)
        x = self.dw_conv8(x)
        x = self.dw_conv9(x)
        x = self.dw_conv10(x)
        x = self.dw_conv11(x)
        x = self.dw_conv12(x)
        x = torch.flatten(self.avg_pool(self.dw_conv13(x)), start_dim=1)
        y_classification = self.fc_classification(x)
        y_bounding_box = self.fc_bounding_box(x)
        return [y_classification, y_bounding_box]
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False).to(device)
mobilenet = MobileNet()
mobilenet.to(device=device)
mobilenet(x)
# need to install torchviz dependency for visualization
!pip install torchviz graphviz
from torchviz import make_dot
from graphviz import Source
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False).to(device)
y_classification = torch.zeros(1, dtype=torch.long, requires_grad=False).to(device)
y_bounding_box = torch.zeros(1, 4, dtype=torch.float, requires_grad=False).to(device)
mobilenet = MobileNet()
mobilenet.to(device=device)
out = mobilenet(x)
classification_loss = nn.CrossEntropyLoss()(out[0], y_classification)
bounding_box_regression_loss = nn.MSELoss()(out[1], y_bounding_box)
loss = classification_loss + bounding_box_regression_loss
model_arch = make_dot(loss)
Source(model_arch).render("../working/mobilenet_architecture_mulitple_loss")
from lxml import etree
def generate_dictionaries(root_dir="../input/the-oxfordiiit-pet-dataset/annotations/annotations/", train_val_split=0.8):
    train_ids = []
    train_val_dict = {}
    labels = {}
    classes_set = set()
    curr_len = len(classes_set)
    idx2class = {}
    with open(os.path.join(root_dir, "trainval.txt"), "r") as tv_file:
            for line in tv_file:
                try:
                    if not line.startswith('#'):
                        line_split = line.split(" ")
                        img_filename = line_split[0] + ".jpg"
                        breed_label = int(line_split[1])
                        species_label = int(line_split[2]) - 1
                        if 1 <= breed_label <= 25:
                            species_name = "cat"
                        else:
                            species_name = "dog"
                        classes_set.add(species_name)
                        if len(classes_set) > curr_len:
                            curr_len = len(classes_set)
                            idx2class[species_label] = species_name
                        with open("../input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls/{}.xml".format(line_split[0])) as xml_file:
                            xml = xml_file.read()
                        root = etree.fromstring(xml)
                        # just to check if the image exists
                        with open("../input/the-oxfordiiit-pet-dataset/images/images/{}".format(img_filename)) as f:
                            l = []
                        # iterate through xml tree structure
                        for element in root.iter():
                            l.append(element.text)
                        l = l[-5:-1]
                        l = [int(x)/1. for x in l]
                        train_ids.append(img_filename)
                        labels[img_filename] = [species_label, l]
                except Exception as e:
#                     print(e)
                    # print the files that don't have xml files
                    print(line_split[0])
                    continue
    random.shuffle(x=train_ids)
    train_val_dict['train'] = train_ids[:int(math.ceil(train_val_split*len(train_ids)))]
    train_val_dict['val'] = train_ids[int(math.ceil(train_val_split*len(train_ids))):]
    return train_val_dict, labels, idx2class
# sanity check
train_val_dict, labels, idx2class = generate_dictionaries()
len(train_val_dict['train']) + len(train_val_dict['val']), len(labels)
class PetsDataset(Dataset):
    def __init__(self, list_ids, labels, idx2class, root_dir, transforms=None):
        self.list_ids = list_ids
        self.labels = labels
        self.root_dir = root_dir
        self.idx2class = idx2class
        self.transforms = transforms
    
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        file_name = self.list_ids[index]
        class_label = torch.tensor(self.labels[file_name][0], dtype=torch.long)
        bbox_ground_truth = torch.tensor(self.labels[file_name][1], dtype=torch.float32)
        # To get image path, join root dir, class folder name, and file_name
        img_path = os.path.join(self.root_dir, file_name)
        image = io.imread(img_path)
        old_x = image.shape[0]
        old_y = image.shape[1]
        if self.transforms:
            image = self.transforms(image)
        scaled_x = 224. / old_x
        scaled_y = 224. / old_y
        for idx, coords in enumerate(bbox_ground_truth):
            if idx % 2 == 0:
                bbox_ground_truth[idx] = bbox_ground_truth[idx] *  scaled_x
            else:
                bbox_ground_truth[idx] = bbox_ground_truth[idx] *  scaled_y
        return [image, class_label, bbox_ground_truth]
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    # Don't forget to toggle to eval mode!
    model.eval()
    
    with torch.no_grad():
        losses = []
        for data, target_classify, target_bbox in tqdm(loader):
            data = data.to(device=device)
            target_classify = target_classify.to(device=device)
            target_bbox = target_bbox.to(device=device)

            scores = model(data)
            loss2 = bounding_box_regression_loss(scores[1], target_bbox)
            losses.append(loss2)
            _, predictions = scores[0].max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print("Correct: {}, Total: {}, Accuracy: {}".format(num_correct, num_samples, int(num_correct) / int(num_samples)))
        print("Regression Loss: {}".format(sum(losses) / len(losses)))
    # Don't forget to toggle back to model.train() since you're done with evaluation
    model.train()
if __name__ == '__main__':
    
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    # Training for ~100 epochs might give you better performance
    EPOCHS = 20
    NUM_CLASSES = 2
    BOUNDING_BOX_POINTS = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_val_dict, labels, idx2class = generate_dictionaries()
    train_data = PetsDataset(list_ids=train_val_dict['train'], 
                             labels=labels,
                             idx2class=idx2class, 
                             root_dir="../input/the-oxfordiiit-pet-dataset/images/images/", 
                             transforms=transform_img)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    val_data = PetsDataset(list_ids=train_val_dict['val'],
                             labels=labels,
                             idx2class=idx2class, 
                             root_dir="../input/the-oxfordiiit-pet-dataset/images/images/", 
                             transforms=transform_img)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    
    mobilenet = MobileNet(num_classes=NUM_CLASSES)
    mobilenet.to(device)
    classification_loss = nn.CrossEntropyLoss()
    bounding_box_regression_loss = nn.MSELoss()
    optimizer = optim.Adam(mobilenet.parameters(), lr=LEARNING_RATE)
    data, target_classify, target_bbox = next(iter(train_loader))
    for epoch in tqdm(range(EPOCHS)):
        losses = []
        with tqdm(total=len(train_val_dict['train']) // BATCH_SIZE) as pbar:
            for batch_idx, (data, target_classify, target_bbox) in enumerate(train_loader):
                data = data.to(device=device)
                target_classify = target_classify.to(device=device)
                target_bbox = target_bbox.to(device=device)

                scores = mobilenet(data)
                loss1 = classification_loss(scores[0], target_classify)
                loss2 = bounding_box_regression_loss(scores[1], target_bbox)
                loss = loss1 + loss2
                losses.append(loss)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
#                 print(loss.item())
        print("Cost at epoch {} is {}".format(epoch, sum(losses) / len(losses)))
#         print("Calculating Train Accuracy...")
#         check_accuracy(train_loader, mobilenet)
#         print("Calculating Validation Accuracy...")
#         check_accuracy(val_loader, mobilenet)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def see_prediction(path="../input/the-oxfordiiit-pet-dataset/images/images/Abyssinian_114.jpg"):
    image = plt.imread(path)
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    transformed_image = transform_img(image).unsqueeze(0).to(device=device)
    out = mobilenet(transformed_image)
    print(out[0])
    _, prediction = out[0].max(1)
    print(prediction)
    print(idx2class[prediction.item()])
#     breed_name = idx2class[prediction.item()]
#     print(breed_name)

    out_item = [i.item() for i in out[1][0]]
    plot_scale_x = image.shape[0] / 224.
    plot_scale_y = image.shape[1] / 224.
    for idx, coords in enumerate(out_item):
        if idx % 2 == 0:
            out_item[idx] = out_item[idx] *  plot_scale_x
        else:
            out_item[idx] = out_item[idx] *  plot_scale_y

    fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((out_item[0],out_item[1]),abs(out_item[3] - out_item[1]), abs(out_item[2] - out_item[0]),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()
see_prediction("../input/the-oxfordiiit-pet-dataset/images/images/Abyssinian_100.jpg")
see_prediction("../input/the-oxfordiiit-pet-dataset/images/images/Abyssinian_201.jpg")
see_prediction("../input/the-oxfordiiit-pet-dataset/images/images/german_shorthaired_103.jpg")