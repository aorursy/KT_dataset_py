import torch
import os
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
PATH = "../input/104-flowers-garden-of-eden/jpeg-224x224"
TRAIN_DIR  = PATH + '/train'
VAL_DIR  = PATH + '/val'
TEST_DIR  = PATH + '/test'
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
data_train = tv.datasets.ImageFolder(TRAIN_DIR, transform)
data_val = tv.datasets.ImageFolder(VAL_DIR, transform)
data_test = [ TEST_DIR + "/" + i for i in os.listdir(TEST_DIR)]
data_train_loader = DataLoader(data_train, 100)
data_val_loader = DataLoader(data_val,100)
# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

N = 5

def sample_pictures(n,dataset=data_train.imgs):
    size = len(dataset)
    samples = np.random.choice(size, n)
    
    return np.asarray(dataset)[samples]
    

def display_flowers(n, title=True, dataset = data_train.imgs):
    
    fig, ax = plt.subplots(1, n, figsize=(15, 6))
   
    images = sample_pictures(n, dataset)

    for i in range(n):
     
        path = images[i][0] if len(images[i]) ==2  else images[i]
        image = Image.open(path)
        ax[i].imshow(image)
        
        if title:
            label = images[i][1]
            ax[i].set_title(CLASSES[int(label)])

    plt.axis('off')
    
    plt.show()
    

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

    
    
display_flowers (N)
display_flowers (N, dataset=data_val.imgs)
display_flowers (N, False, data_test)
model = tv.models.resnet18(True)


for param in model.parameters():
    param.required_grad=False
    
model.fc = torch.nn.Linear(512, len(CLASSES))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)
n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(data_val)
N_train=len(data_train)

#n_epochs

Loss=0
cpt = 1

for epoch in range(n_epochs):
    for x, y in data_train_loader:
        #print (x.size())
        model.train() 
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
   
        z = model(x)
        # calculate loss 
        loss = criterion(z, y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        loss_list.append(loss.data)
        print ("%d / %d"%(cpt, len(data_train)/100))
        cpt +=1
    correct=0
    for x_test, y_test in data_val_loader:
        # set model to eval 
        model.eval()
        #make a prediction 
        yhat = model(x_test)
        #find max 
        yhat = torch.max(yhat, 1)[1]
       
        #Calculate misclassified  samples in mini-batch 
        #hint +=(yhat==y_test).sum().item()
        correct += (yhat==y_test).sum().item()

    accuracy=correct/N_test
plt.plot(loss_list)