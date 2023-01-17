import torch

from torch import nn

from torchvision import datasets, models, transforms

from pathlib import Path

import time

import csv

import sys
print(torch.__version__)

print(sys.version)
PHASE_VALID = 'valid'

PHASE_TEST = 'test'

PHASES = [PHASE_VALID, PHASE_TEST]



ROOT_DIR = Path('../input')

FLOWER_DATA_DIR = ROOT_DIR / 'oxford-102-flower-pytorch' / 'flower_data' / 'flower_data'

MODEL_DIR = ROOT_DIR / 'over-98-pytorch-model-for-oxford-102-flower'

batch_size = 1



# lets see what we have here

print([f.name for f in ROOT_DIR.iterdir()])

print([f.name for f in FLOWER_DATA_DIR.iterdir()])

# in uploaded model dir

print([f.name for f in MODEL_DIR.iterdir()])
# we need to "hack" test dir, and add it to fake "0" class in order to be able to use it with PyTorch datasets

! rm -rf "./test_dir"

! mkdir "./test_dir"

! cp -r "{FLOWER_DATA_DIR}/test" "./test_dir/0"
! ls "./test_dir/"

! ls "./test_dir/0"
data_dir = {

    PHASE_VALID: FLOWER_DATA_DIR / PHASE_VALID,

    PHASE_TEST: "./test_dir/"

}
# This subclass of standard PyTorch datasets.ImageFolder will let us to have access

# to dataset filename (you can easilly modify code to return full path instead)

class ImageFolderWithPaths(datasets.ImageFolder):

    """Custom dataset that includes image file paths. Extends

    torchvision.datasets.ImageFolder

    """



    # override the __getitem__ method. this is the method dataloader calls

    def __getitem__(self, index):

        # this is what ImageFolder normally returns 

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path

        path = self.imgs[index][0]

        filename = self.imgs[index][0].split('/')[-1]

        # make a new tuple that includes original and the path

        tuple_with_path = original_tuple + (filename,)

        return tuple_with_path
data_transforms = transforms.Compose([

                        transforms.Resize(224),

                        transforms.CenterCrop(224),

                        transforms.ToTensor(),

                        transforms.Normalize((0.485, 0.456, 0.406), 

                                             (0.229, 0.224, 0.225))])



image_datasets = {mode: ImageFolderWithPaths(root=data_dir[mode], transform=data_transforms) 

                  for mode in PHASES}

data_loaders = {mode: torch.utils.data.DataLoader(image_datasets[mode], batch_size=batch_size, shuffle=False)

                for mode in PHASES}

dataset_sizes = {x: len(image_datasets[x]) for x in PHASES}



# mapping internal PyTorch dataset ID to target class ID (ID from directory name)

TARGET_CLASS = image_datasets[PHASE_VALID].classes
print(dataset_sizes)
def gen_new_model():

    resnet_model = models.resnet50(pretrained=True)

    for param in resnet_model.parameters():

        param.requires_grad = False



    num_ftrs = resnet_model.fc.in_features

    num_classes = 102

    num_hidden = 1024

    resnet_model.fc = nn.Sequential(nn.Linear(num_ftrs, num_hidden),

                                    nn.ReLU(),

                                    nn.Dropout(0.1),

                                    nn.Linear(num_hidden, num_classes),

                                    nn.LogSoftmax(dim=1))

    

    return resnet_model



# load model from checkpoint

checkpoint = torch.load(MODEL_DIR / 'SDG_pretrained_unfreeze_3_7_WD0.002_do0.15_bs6_lr0.001.pth')

# Load your model to this variable

model = gen_new_model()

model.load_state_dict(checkpoint['model_state_dict'])

model.cuda()

print()
def check_model(model):

    model.eval()

    test_predictions = []



    for phase in PHASES:

        since = time.time()

        running_corrects = 0

        # Iterate over data.

        for inputs, labels, (filename,) in data_loaders[phase]:

            inputs = inputs.cuda() 

            labels = labels.cuda()

            with torch.set_grad_enabled(False):

                outputs = model(inputs)

                _, preds = torch.max(outputs, dim=1)

            if phase == PHASE_VALID:

                running_corrects += torch.sum(preds == labels.data)

            else:

                test_predictions.append((filename, TARGET_CLASS[preds.item()]))



        if phase == PHASE_VALID:

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"Valid acc: {epoch_acc}")



        time_elapsed = time.time() - since

        print('Elapsed ({}): {:.0f}m {:.0f}s'

              .format(phase, time_elapsed // 60, time_elapsed % 60))

        

    return test_predictions
predictions = check_model(model)
with open('predictions.csv', 'w', newline='') as f:

    writer = csv.writer(f)

    writer.writerow(['file_name', 'id'])

    writer.writerows(predictions)
# lets remove ./test_dir from working directory, as we don't need it anymore

! rm -rf "./test_dir"