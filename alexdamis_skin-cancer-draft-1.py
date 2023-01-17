%matplotlib inline



# python libraries

import os, cv2, itertools

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from glob import glob

from PIL import Image



# pytorch libraries

import torch

from torch import optim,nn

from torch.autograd import Variable

from torch.utils.data import DataLoader,Dataset

from torchvision import models,transforms



# sklearn libraries

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# to make sure the results are reproducible

np.random.seed(10)

torch.manual_seed(10)

torch.cuda.manual_seed(10)



print(os.listdir("../input"))
data_dir = '../input'



# puts all image paths in a single list

all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))



# creates a dictionary with filename keys and full path values

# ex: {img5 : input/ham_img_part_03/img5.jpeg}

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}



# creates a dictionary with short hands for skin cancer types as keys and 

# the full names of the diseases as the values

lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'dermatofibroma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}
def compute_img_mean_std(image_paths):

    """

        computing the mean and std of three channel on the whole dataset,

        first we should normalize the image from 0-255 to 0-1

    """



    img_h, img_w = 224, 224

    imgs = []

    means, stdevs = [], []



    # for every image in image_paths (all jpegs in the data set)

    for i in tqdm(range(len(image_paths))):

        

        # read in an individual image as img

        img = cv2.imread(image_paths[i])

        

        # resizes the image to the proper measurements

        img = cv2.resize(img, (img_h, img_w))

        

        # add the current image to the imgs list

        imgs.append(img)



    # converts our 4D python list of images, into a 4-dimensional np array

    # for the ability to later do powerful numpy opperations

    imgs = np.array(imgs)



    # normalizes each pixel in the 4D np array to be between 0 and 1,

    # this will make obtaining the mean and std dev easier to compute

    imgs = imgs.astype(np.float32) / 255.0



    # this for loop itterates through each color of our images (blue, green, red)

    for i in range(3):

        

        # grabs an inidvidual color from each image in the image stack 

        # and converts it to a super long 1D array

        pixels = imgs[:, :, :, i].ravel()

        

        # adds each colors mean and std dev to their respective lists

        means.append(np.mean(pixels))

        stdevs.append(np.std(pixels))

        

    # reverses the three values in each list to become from BGR --> RGB

    means.reverse()

    stdevs.reverse()



    # prints and returns the mean and std dev of each color for each pixel 

    # in the images

    print("normMean = {}".format(means))

    print("normStd = {}".format(stdevs))

    return means,stdevs
# run the previously created function

norm_mean,norm_std = compute_img_mean_std(all_image_path)
# reads the metadata csv

df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))



# adds a new 'path' column to our df by itterating through the image_id column

# and getting the value associated with it in the image path dictionary

df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)



# adds a new 'cell_type' column to our df by itterating through the diagnosis

# column (dx) and getting the value associated with in in the lesion type dictionary

df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)



# adds a new 'cell_type_idx' column to our df by itterating through the diagnosis

# column to create and lable each diagnosis type with a unique integer code (7 total)

df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes





df_original.head()
# this will tell us how many images are associated with each lesion_id, every 

# column other than the 'lesion_id' column will be populated with the duplicate count

df_undup = df_original.groupby('lesion_id').count()



# now we filter out lesion_id's that have only one image associated with it

df_undup = df_undup[df_undup['image_id'] == 1]



# after chopping up the data frame, this line adds gives each row a consecutive indicie

df_undup.reset_index(inplace=True)

df_undup.head()
# here we identify lesion_id's that have duplicate images and those that have only one image.

def get_duplicates(x):

    

    # creates a list of the unduplicated lesion ids

    unique_list = list(df_undup['lesion_id'])

    

    # checks if an arbitrary lesion id is in the list of lesions without duplicates and 

    # returns the a string indicating if the lesion has a duplicate or not

    if x in unique_list:

        return 'unduplicated'

    else:

        return 'duplicated'



    

# create a new colum that is a copy of the lesion_id column

df_original['duplicates'] = df_original['lesion_id']



# apply the function to this new column

# the function will itterate through each row of this column and update its 'duplicate status'

df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

df_original.head()
# prints number of lesions with and without duplicates

df_original['duplicates'].value_counts()
# now we filter out images that don't have duplicates

df_undup = df_original[df_original['duplicates'] == 'unduplicated']

df_undup.shape
# now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set



# y holds the lesion code of each index in the unduplicate df

y = df_undup['cell_type_idx']



# df_val is the test set; test_size dictates the percent of our data we want to dedicate to testing;

# stratify = y means we want to split each disease catagory eaqually beteween test and train 

# (ex: if 20% of the data has lesion code 4, then 20% of test should also have lesion code 4)

_, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

df_val.shape
df_val['cell_type_idx'].value_counts()
# This set will be df_original excluding all rows that are in the val set

# This function identifies if an image is part of the train or val set.

def get_val_rows(x):

    # create a list of all the lesion_id's in the val set

    val_list = list(df_val['image_id'])

    if str(x) in val_list:

        return 'val'

    else:

        return 'train'



# identify train and val rows

# create a new colum that is a copy of the image_id column

df_original['train_or_val'] = df_original['image_id']

# apply the function to this new column

df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)

# filter out train rows

df_train = df_original[df_original['train_or_val'] == 'train']

print(len(df_train))

print(len(df_val))
df_train['cell_type_idx'].value_counts()
df_val['cell_type_idx'].value_counts()
# Copy fewer class to balance the number of 7 classes

data_aug_rate = [15,10,5,50,0,40,5]

for i in range(7):

    if data_aug_rate[i]:

        # takes a given row with a specified disease code and duplicates the row a certian number of times,

        # then appends the end of the data frame

        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)

df_train['cell_type_idx'].value_counts()
# re-aligns the indiceis of the train and test data frames

df_train = df_train.reset_index()

df_val = df_val.reset_index()



df_val.head()
# feature_extract is a boolean that defines if we are finetuning or feature extracting. 

# If feature_extract = False, the model is finetuned and all model parameters are updated. 

# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.

def set_parameter_requires_grad(model, feature_extracting):

    if feature_extracting:

        for param in model.parameters():

            param.requires_grad = False
def initialize_model(num_classes, feature_extract, use_pretrained=True):



    # creates a pretrained densenet-121 model

    model_ft = models.densenet121(pretrained=use_pretrained)

    

    # implements the 'requires grad' function above

    set_parameter_requires_grad(model_ft, feature_extract)

    

    # asks the net for the number of input features we are taking into account

    # for our classifier layer (the final layer of the neural net)

    num_ftrs = model_ft.classifier.in_features

    

    # reconstructs our classifier layer with the same number of input nodes (features) as 

    # before, but a newly specified number of output choices/nodes/classifications

    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    

    # input size is the length of pixels for height and width 

    input_size = 224



    return model_ft, input_size
# build the densenet model



# num classes is the number of output possibilities, its 7 because we have 7 different types of

# skin diseases

num_classes = 7



# set feature extract to false so we can reweight every layer of our densenet neural net

feature_extract = False



# Initialize the model

model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)



# choose a gpu to use as the device for our model

device = torch.device('cuda:0')



# Put the model on the device

model = model_ft.to(device)
# norm_mean = (0.49139968, 0.48215827, 0.44653124)

# norm_std = (0.24703233, 0.24348505, 0.26158768)

# define the transformation of the train images.

train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),

                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),

                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),

                                      transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

# define the transformation of the val images.

val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),

                                    transforms.Normalize(norm_mean, norm_std)])
# Define a pytorch dataloader for this dataset

class HAM10000(Dataset):

    def __init__(self, df, transform=None):

        self.df = df

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):

        # Load data and get label

        X = Image.open(self.df['path'][index])

        y = torch.tensor(int(self.df['cell_type_idx'][index]))



        if self.transform:

            X = self.transform(X)



        # returns X, the transformed image ready for training

        # and y, the correct image classification, as a single int tensor

        return X, y
# Define the training set using the table train_df and using our defined transitions (train_transform)

training_set = HAM10000(df_train, transform=train_transform)



# look into dataloader function/batch size later

# batch size == # of images to process befor updating weights

# num_workers == # of simutaneous proccessing paths

train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)





# Same for the validation set:

validation_set = HAM10000(df_val, transform=train_transform)

val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)
# we use Adam optimizer

# ??what is lr??

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# use cross entropy loss as our loss function 

# ??what is CrossEntropyLoss??

criterion = nn.CrossEntropyLoss().to(device)
# this function is used during training process, to calculate the loss and accuracy

class AverageMeter(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    # val is effectively the final value processed

    # sum adds the new values to sum since last reset

    # count adds number of values processed since last reset

    # avg is a live average of the sum give the count we've proccessed so far

    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
total_loss_train, total_acc_train = [],[]

def train(train_loader, model, criterion, optimizer, epoch):

    

    # put model into train mode, (model will update weights / parameters)

    model.train()

    

    # initailizes an average meter to keep note of losses and

    # accurate guesses (wins)

    train_loss = AverageMeter()

    train_acc = AverageMeter()

    

    # curr iterator holds the cour of total images to proccess

    curr_iter = (epoch - 1) * len(train_loader)

    

    # loop through all images in our train loader  

    for i, data in enumerate(train_loader):

        # data is the get item retval for each image in our dataset

        # returns a 32 image batch

        images, labels = data

        

        # will be the batch size which is 32

        N = images.size(0)

        # print('image shape:',images.size(0), 'label shape',labels.size(0))

        

        # preps the inputs and outputs for this batch

        images = Variable(images).to(device)

        labels = Variable(labels).to(device)



        # reset the gradients for this batch

        optimizer.zero_grad()

        

        # runs this batch through the model and store the outputs

        outputs = model(images)

       



        # compare outputs and correct answers to calculate gradients

        loss = criterion(outputs, labels)

        

        # pairs each gradient value with the parameter

        loss.backward()

        

        # updates each parameters using backwards' returned values

        optimizer.step()

        

        # grabs the highest val of the output row (dim 1) of each image

        # prediction becomes a list like of the inputs results

        # [1] means we are grabbing the prediction, not how strongly we predict it

        prediction = outputs.max(1, keepdim=True)[1]

        

        # formats lables as prediction and compares the values

        # after we compare our wins we sum them up, then divide it by the 

        # number of images in our batch

        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

        

        # grabs the number of losses this batch

        train_loss.update(loss.item())

        

        # add 1 to the curr_iter

        curr_iter += 1

        

        # every hundred batches we:

        if (i + 1) % 100 == 0:

            

            # print which epoch we are on, which batch itteration of the total number of batch

            # itterations per epoch, the average % of losses per batch and wins per batch 

            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (

                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))

            

            # save this itterations loss and win % (results) to the list 

            total_loss_train.append(train_loss.avg)

            total_acc_train.append(train_acc.avg)

            

    # returns the average loss rate and win rate for the full epoch, 

    # since we never reset in the for loop

    return train_loss.avg, train_acc.avg
def validate(val_loader, model, criterion, optimizer, epoch):

    

    # set model to evalute, this means we will not update the weights

    model.eval()

    

    # create new average meters for loss and win

    val_loss = AverageMeter()

    val_acc = AverageMeter()

    

    # lets the backprop mechanism no we wont use it,

    # saves time, energy, and space

    with torch.no_grad():

        

        # i is the batch # and data is the set of items to be processed

        for i, data in enumerate(val_loader):

            

            # uncoupe data with the __getitem__ module 

            images, labels = data

            

            # returns the number of images in our tensor (data set)

            N = images.size(0)

            

            # loads images and lables to the device for model processing

            images = Variable(images).to(device)

            labels = Variable(labels).to(device)



            # stores results of model outputs

            outputs = model(images)

            

            # prediction set refines output into the indicies of whichever

            # classification the model was most confident in (the disease code)

            prediction = outputs.max(1, keepdim=True)[1]



            # update the win % by viewing the labels tensor as prediction

            # then comparing the values and summing up the number of times our prediction 

            # was correct, next we divide by n to get the % of times the model was right.

            # that percentage gets added to the running val_acc sum, and divided by the 

            # running val_acc count

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

            

            # updates val_loss with the percentage of predictions the model gets incorrect

            val_loss.update(criterion(outputs, labels).item())



    # every epoch, after recording how well the model predicted the each batch

    # we print an update of the status to the screen

    # this includes the current epoch, and this epochs win and lose ratios

    print('------------------------------------------------------------')

    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))

    print('------------------------------------------------------------')

    

    # returns the finalized win and loss averages

    return val_loss.avg, val_acc.avg
# number of times we want to loop through the data

# a full epoch includes:

#       training on the full training set    (approx 35000 images)

#       testing on the full validation set    (approx 1000 images)

epoch_num = 10



# we create an int starting at 0 to document when we find a test set

# with an accuracy level higher than ever documented previously.

# once we find a layer that breaks our record we overwrite this value

# with the new found max value

best_val_acc = 0



# creates our lists which will hold the average loss and win values of 

# the test set after every epoch

total_loss_val, total_acc_val = [],[]



# loop through every epoch

for epoch in range(1, epoch_num+1):

    

    # train this epochs 35,000 training images 

    loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)

    

    # test this epochs 1000 validation images

    loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)

    

    # add the testing loss and win values to the list of epoch testing history

    total_loss_val.append(loss_val)

    total_acc_val.append(acc_val)

    

    # if this epoch we peformed better than we ever hav

    # print out some information on how we did

    # and save this win rate as the new best win rate

    if acc_val > best_val_acc:

        best_val_acc = acc_val

        print('*****************************************************')

        print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))

        print('*****************************************************')
# creates a figure and sets the number graphs that will populate it

fig = plt.figure(num = 2)



# the first two args denote the organization of the plot

# (both figure delcarations set our plot to have 2 rows and 1 column)

# the third arg denotes the which cell this subplot will populate

fig1 = fig.add_subplot(2,1,1)

fig2 = fig.add_subplot(2,1,2)



# on figure 1, plot each epoch's training average loss and accuracy

fig1.plot(total_loss_train, label = 'training loss')

fig1.plot(total_acc_train, label = 'training accuracy')



# on figure 2, plot each epoch's testing average loss and accuracy

fig2.plot(total_loss_val, label = 'validation loss')

fig2.plot(total_acc_val, label = 'validation accuracy')



# plot the legend 

plt.legend()



# show the plot

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    

    # need to figure out

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    

    # sets the plot name to be the title

    plt.title(title)

    

    # initializes the colorbar

    plt.colorbar()

    

    # creates a np array with the points we need to add tick marks at

    tick_marks = np.arange(len(classes))

    

    # associates the positions we need tick marks with the class values

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    # if normalize we divide the value by the

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    # plots the value and color of each matrix cell

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    # set the layout to tight

    plt.tight_layout()

    

    # set the name of the labels to x and y

    plt.ylabel('True label')

    plt.xlabel('Predicted label')


# sets model to not update values

model.eval()

y_label = []

y_predict = []



# puts device in read mode (doesnt bother computing weight update values)

with torch.no_grad():

    

    # itterate through each loader batch

    for i, data in enumerate(val_loader):

        # unpacks each batch into a tensor of images and labels

        # (inputs and expected outputs)

        images, labels = data

        

        # gets the size of the batch

        N = images.size(0)

        

        # uploads images to the device and saves

        # the models estimates

        images = Variable(images).to(device)

        outputs = model(images)

        

        # condenses the model output tensor to show the disease idx

        # as opposed to the confidences of each class

        prediction = outputs.max(1, keepdim=True)[1]

        

        # idk bluh

        y_label.extend(labels.cpu().numpy())

        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))



# compute the confusion matrix

confusion_mtx = confusion_matrix(y_label, y_predict, normalize=True)

# plot the confusion matrix

plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']

plot_confusion_matrix(confusion_mtx, plot_labels)
# Generate a classification report

report = classification_report(y_label, y_predict, target_names=plot_labels)

print(report)
# itterates through the matrix and divides the amount of correct

# guesses for each given class by the amount of total guesses

# then subracts the percentage correct from 1 to obtain the error percent

label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)



# plots the 7 error rates (one for each class)

plt.bar(np.arange(7),label_frac_error)



# labels the x and y axis

plt.xlabel('True Label')

plt.ylabel('Fraction classified incorrectly')