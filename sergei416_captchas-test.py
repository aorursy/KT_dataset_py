match = 0

nonmatch = 0
import os

import random
tot_size = len(os.listdir('/kaggle/input/captcha-images/'))

print(tot_size)

num = random.randint(0, tot_size)

print(num)
import random

randomlist = []

for i in range(0, 500):

    n = random.randint(0,tot_size)

    randomlist.append(n)

print(randomlist)
for m in randomlist:

    import os

    import torch

    import pandas as pd

    import numpy as np

    from torch.utils.data import Dataset, random_split, DataLoader

    from PIL import Image

    import torchvision.models as models

    from tqdm.notebook import tqdm

    import torchvision.transforms as T

    from sklearn.metrics import f1_score

    import torch.nn.functional as F

    import torch.nn as nn

    from torchvision.utils import make_grid

    from torchvision.datasets import ImageFolder

    import random

    import PIL



    import matplotlib.pyplot as plt

    %matplotlib inline







    def seed_everything(seed=2020):

        random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)



    seed_everything(42)

    print('ENVIRONMENT READY')





    dataset = ImageFolder(root='/kaggle/input/captchas-segmented/data')



    dataset_size = len(dataset)

    dataset_size



    # Data augmentation

    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



    train_tfms = T.Compose([

        T.RandomCrop(128, padding=8, padding_mode='reflect'),

         #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

        T.Resize((128, 128)),

        T.RandomHorizontalFlip(), 

        T.RandomRotation(10),

        T.ToTensor(), 

         T.Normalize(*imagenet_stats,inplace=True), 

        #T.RandomErasing(inplace=True)

    ])



    valid_tfms = T.Compose([

         T.Resize((128, 128)), 

        T.ToTensor(), 

         T.Normalize(*imagenet_stats)

    ])



    def get_default_device():

        """Pick GPU if available, else CPU"""

        if torch.cuda.is_available():

            return torch.device('cuda')

        else:

            return torch.device('cpu')



    def to_device(data, device):

        """Move tensor(s) to chosen device"""

        if isinstance(data, (list,tuple)):

            return [to_device(x, device) for x in data]

        return data.to(device, non_blocking=True)



    class DeviceDataLoader():

        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):

            self.dl = dl

            self.device = device



        def __iter__(self):

            """Yield a batch of data after moving it to device"""

            for b in self.dl: 

                yield to_device(b, self.device)



        def __len__(self):

            """Number of batches"""

            return len(self.dl)



    device = get_default_device()

    device





    test_size = 200

    nontest_size = len(dataset) - test_size



    nontest_df, test_df = random_split(dataset, [nontest_size, test_size])

    len(nontest_df), len(test_df)



    val_size = 200

    train_size = len(nontest_df) - val_size



    train_df, val_df = random_split(nontest_df, [train_size, val_size])

    len(train_df), len(val_df)



    test_df.dataset.transform = valid_tfms

    val_df.dataset.transform = valid_tfms



    train_df.dataset.transform = train_tfms





    batch_size = 64



    train_dl = DataLoader(train_df, batch_size, shuffle=True, 

                          num_workers=3, pin_memory=True)

    val_dl = DataLoader(val_df, batch_size*2, 

                        num_workers=2, pin_memory=True)

    test_dl = DataLoader(test_df, batch_size*2, 

                        num_workers=2, pin_memory=True)





    train_dl = DeviceDataLoader(train_dl, device)

    val_dl = DeviceDataLoader(val_dl, device)

    test_dl = DeviceDataLoader(test_dl, device)







    diction = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',

           11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',

           21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',

           31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',

           41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 

           51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',

           61: 'z'}







    def accuracy(outputs, labels):

        _, preds = torch.max(outputs, dim=1)

        return torch.tensor(torch.sum(preds == labels).item() / len(preds))





    class ImageClassificationBase(nn.Module):

        def training_step(self, batch):

            images, labels = batch 

            out = self(images)                  # Generate predictions

            loss = F.cross_entropy(out, labels) # Calculate loss

            return loss



        def validation_step(self, batch):

            images, labels = batch 

            out = self(images)                    # Generate predictions

            loss = F.cross_entropy(out, labels)   # Calculate loss

            acc = accuracy(out, labels)           # Calculate accuracy

            return {'val_loss': loss.detach(), 'val_acc': acc}



        def validation_epoch_end(self, outputs):

            batch_losses = [x['val_loss'] for x in outputs]

            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

            batch_accs = [x['val_acc'] for x in outputs]

            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}



        def epoch_end(self, epoch, result):

            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))



    class CnnModel2(ImageClassificationBase):

        def __init__(self):

            super().__init__()

            # Use a pretrained model

            self.network = models.wide_resnet101_2(pretrained=True)

            # Replace last layer

            num_ftrs = self.network.fc.in_features

            self.network.fc = nn.Linear(num_ftrs,62)



        def forward(self, xb):

            return torch.sigmoid(self.network(xb))



    model = to_device(CnnModel2(), device)

    @torch.no_grad()

    def evaluate(model, val_loader):

        model.eval()

        outputs = [model.validation_step(batch) for batch in val_loader]

        return model.validation_epoch_end(outputs)



    model = to_device(CnnModel2(), device)

    model.load_state_dict(torch.load('/kaggle/input/captcha-solver-ml/captcha.pth'))



    print(evaluate(model, val_dl)['val_acc'])



    def predict_image(img, model):

        xb = to_device(img.unsqueeze(0), device)

        yb = model(xb)

        _, preds  = torch.max(yb, dim=1)

        return preds[0].item()



    import cv2

    import os

    import matplotlib.pyplot as plt

    import numpy as np

    import random









    import os

    import torch

    import pandas as pd

    import numpy as np

    from torch.utils.data import Dataset, random_split, DataLoader

    from PIL import Image

    import torchvision.models as models

    from tqdm.notebook import tqdm

    import torchvision.transforms as T

    from sklearn.metrics import f1_score

    import torch.nn.functional as F

    import torch.nn as nn

    from torchvision.utils import make_grid

    from torchvision.datasets import ImageFolder

    import random

    import PIL



    import matplotlib.pyplot as plt

    %matplotlib inline









    # Data augmentation

    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



    train_tfms = T.Compose([

        T.RandomCrop(128, padding=8, padding_mode='reflect'),

         #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

        T.Resize((128, 128)),

        T.RandomHorizontalFlip(), 

        T.RandomRotation(10),

        T.ToTensor(), 

         T.Normalize(*imagenet_stats,inplace=True), 

        #T.RandomErasing(inplace=True)

    ])



    valid_tfms = T.Compose([

         T.Resize((128, 128)), 

        T.ToTensor(), 

         T.Normalize(*imagenet_stats)

    ])







    diction = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',

           11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',

           21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',

           31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',

           41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 

           51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',

           61: 'z'}





    alli = list(os.listdir('/kaggle/input/captcha-images/'))

    file = alli[m]

    #print(file)

    try:

        solution = file.split('.')[0]

        hi = cv2.imread('/kaggle/input/captcha-images/' + file)

        # convert to RGB



        #plt.imshow(hi, cmap="gray")

        #plt.show()



        # convert to RGB

        image = cv2.cvtColor(hi, cv2.COLOR_BGR2RGB)

        # convert to grayscale

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



        # create a binary thresholded image

        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

        # show it

        #plt.imshow(binary, cmap="gray")

        #plt.show()



        # find the contours from the thresholded image

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw all contours

        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)



        # show the image with the drawn contours

        #plt.imshow(image)

        #plt.show()



        x = {}

        for m in range(len(contours)):

            mini = 1000

            for k in range(len(contours[m])):

                first = contours[m][k][0][0]

                if first < mini:

                    mini = first

            mini1 = mini

            #print(mini1)



            mini = 1000

            for k in range(len(contours[m])):

                first = contours[m][k][0][1]

                if first < mini:

                    mini = first

            mini2 = mini

            #print(mini2)



            maxi = 0

            for k in range(len(contours[m])):

                first = contours[m][k][0][1]

                if first > maxi:

                    maxi = first

            maxi2 = maxi

            #print(maxi2)



            maxi = 0

            for k in range(len(contours[m])):

                first = contours[m][k][0][0]

                if first > maxi:

                    maxi = first

            maxi1 = maxi

            #print(maxi1)

            x[m] = maxi2 - mini2



        biggie = sorted(x, key=x.get)



        s = {}

        def plotting(num):

            m = biggie[num]

            mini = 1000

            for k in range(len(contours[m])):

                first = contours[m][k][0][0]

                if first < mini:

                    mini = first

            mini1 = mini

            #print(mini1)



            mini = 1000

            for k in range(len(contours[m])):

                first = contours[m][k][0][1]

                if first < mini:

                    mini = first

            mini2 = mini

           # print(mini2)



            maxi = 0

            for k in range(len(contours[m])):

                first = contours[m][k][0][1]

                if first > maxi:

                    maxi = first

            maxi2 = maxi

            #print(maxi2)



            maxi = 0

            for k in range(len(contours[m])):

                first = contours[m][k][0][0]

                if first > maxi:

                    maxi = first

            maxi1 = maxi

            #print(maxi1)

            ret = [mini2, maxi2, mini1, maxi1]

            s[num] = mini1

            return ret



        fig, axs = plt.subplots(2, 5)

        wow = plotting(-1)

        # axs[0, 0].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[0,0].axis('off')

        wow = plotting(-2)

        # axs[0, 1].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[0,1].axis('off')

        wow = plotting(-3)

        #axs[0,2].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[0,2].axis('off')

        wow = plotting(-4)

        #axs[0,3].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[0,3].axis('off')

        wow = plotting(-5)

        #axs[0,4].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[0,4].axis('off')

        wow = plotting(-6)

        # axs[1,0].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[1,0].axis('off')

        wow = plotting(-7)

        #axs[1,1].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[1,1].axis('off')

        wow = plotting(-8)

        #axs[1,2].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[1,2].axis('off')

        wow = plotting(-9)

        #axs[1,3].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[1,3].axis('off')

        wow = plotting(-10)

        #axs[1,4].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        axs[1,4].axis('off')

        wow = plotting(-11)





        siggie = sorted(s, key=s.get)



        def checkforerror(wow):

            white = cv2.countNonZero(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2])

            total = (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[0] * (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[1]

            div = white/total

            if div > 0.63: 

                if (wow[1] - wow[0]) - (wow[3] - wow[2]) > -5:

                    return True

            elif div < 0.29:

                return True



            else: return False

        def checkforerrorout(wow):

            white = cv2.countNonZero(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2])

            total = (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[0] * (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[1]

            div = white/total

            return div

        def checkfordoubles(wow):

            if (wow[1] - wow[0]) - (wow[3] - wow[2]) < -4: 

                if checkforerrorout(wow) < 0.56: return True

            elif (wow[1] - wow[0]) - (wow[3] - wow[2]) < 1:

                if checkforerrorout(wow) < 0.40: return True

            else: return False









        pos = 0

        a = 0

        im = 0

        desired = 10

        fig, axs = plt.subplots(2, 5)

        first = False

        while a < desired:

            try:

                idx = pos

                #print('*' * 40)

                #print('pos', pos, 'true value is ', solution[pos])



                foldername = str('/kaggle/working/%s' % pos)

                #print(idx, foldername)

                if os.path.isdir(foldername) == False:

                    os.mkdir(foldername)

                #print('a = ', a, 'desired = ', desired)



                n1 = pos // 5

                n2 = pos % 5

                #print('using position', n1, n2)

                wow = plotting(siggie[im])

                #print('trying to plot ', im, 'checking for errors...')

                err = checkforerror(wow)

                errs = checkfordoubles(wow)

                #print('% = ', checkforerrorout(wow))

                print(err, errs)

                if err:

                    #print('Error 1 at', a)

                    #plt.imshow(binary[wow[0]-2:wow[1] + 2, wow[2]-2:wow[2] + half +7], cmap = 'gray')

                    #inp = str(input('Skip this image? '))

                    im += 1

                    #print('im bumped to', im)

                    wow = plotting(siggie[im])

                    #print('trying to plot ', im, 'checking for errors...')

                    err = checkforerror(wow)

                    errs = checkfordoubles(wow)

                    #print('% = ', checkforerrorout(wow))

                    if errs:





                        #print('Error 2 at bumped image')

                        if first == False:

                            fig = plt.figure()

                            half = abs(wow[0] - wow[1]) // 2

                            plt.imshow(binary[wow[0]-2:wow[1] + 2, wow[2]-2:wow[2] + half +7], cmap = 'gray')



                            plt.axis('off')

                            fig.savefig('/kaggle/working/%s/%s.png' %(pos, idx), dpi=fig.dpi, bbox_inches='tight', pad_inches=0, transparent=True)



                            im -= 1

                            first = True

                        else:

                            fig = plt.figure()

                            half = abs(wow[0] - wow[1]) // 2

                            plt.imshow(binary[wow[0]-2:wow[1] + 2, wow[2] + half + 7:wow[3]+2], cmap = 'gray')



                            plt.axis('off')

                            fig.savefig('/kaggle/working/%s/%s.png' %(pos, idx), dpi=fig.dpi, bbox_inches='tight', pad_inches=0, transparent=True)

                            first = False







                    else:

                        #print('No errors in bumped image')

                        fig = plt.figure()

                        plt.imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')



                        plt.axis('off')

                        fig.savefig('/kaggle/working/%s/%s.png' %(pos, idx), dpi=fig.dpi, bbox_inches='tight', pad_inches=0, transparent=True)

                elif errs:

                    #print('Error 2 at', a)

                    if first == False:

                        fig = plt.figure()

                        half = abs(wow[0] - wow[1]) // 2

                        plt.imshow(binary[wow[0]-2:wow[1] + 2, wow[2]-2:wow[2] + half +8], cmap = 'gray')



                        plt.axis('off')

                        fig.savefig('/kaggle/working/%s/%s.png' %(pos, idx), dpi=fig.dpi, bbox_inches='tight', pad_inches=0, transparent=True)

                        im -= 1

                        first = True

                    else:

                        fig = plt.figure()



                        plt.imshow(binary[wow[0]-2:wow[1] + 2, wow[2] + half + 8:wow[3]+2], cmap = 'gray')



                        plt.axis('off')

                        fig.savefig('/kaggle/working/%s/%s.png' %(pos, idx), dpi=fig.dpi, bbox_inches='tight', pad_inches=0, transparent=True)

                        xx = predict_image(fig, model)

                        print(xx)



                        first = False



                else:

                   # print('No errors found!')

                    fig = plt.figure()

                    plt.imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

                    plt.axis('off')

                    fig.savefig('/kaggle/working/%s/%s.png' %(pos, idx), dpi=fig.dpi, bbox_inches='tight', pad_inches=0, transparent=True)

                    #xx = predict_image(fig, model)

                    #print(xx)

                    #print('DONE PLOTTING - im', im)





                a += 1

                pos += 1

               # print('bumped pos to ', pos)

                im += 1

            except:

               # print('fail case')

                a += 1

                continue

        plt.show()





        test_dataset = ImageFolder(root='/kaggle/working/', transform = valid_tfms)



        test_dataset_size = len(test_dataset)

        test_dataset_size





        predicted = ''



        for k in range(10):

            img_seed = k

            img = test_dataset[img_seed][0]

            label = solution[img_seed]

            plt.imshow(img[0], cmap='gray')

            #print(img.shape)

            #print('Label:', label, ', Predicted:', diction[predict_image(img, model)])

            predicted += (diction[predict_image(img, model)])



        plt.imshow(hi, cmap="gray")

        plt.show()



        #print(predicted, solution)

        

    except: 

        predicted = 'isuck'



    if predicted == solution:

        match += 1

    else: nonmatch += 1
match, nonmatch
print('The final accuracy is', match/500)