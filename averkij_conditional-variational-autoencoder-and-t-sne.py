%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

import matplotlib

import numpy as np

import pandas as pd

import torch

import torchvision

from torch import nn

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torch.utils import data

import torch.nn.functional as F

from torchvision import transforms

from torchvision.datasets import MNIST

from torchvision.utils import save_image

from sklearn.model_selection import train_test_split

from tqdm import tqdm, tqdm_notebook

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print('Training on',DEVICE)
SEED = 42

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
batch_size = 128

learning_rate = 0.005

input_size = 28*28

hidden_size = 12

labels_length = 10
transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])
dataset = MNIST('./data', transform=transform, download=True)

train_data, test_data = data.random_split(dataset, (50000,10000))



train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#helper functions

def one_hot(x, max_x):

    return torch.eye(max_x + 1)[x]



def plot_gallery(images, h, w, n_row=3, n_col=6):

    plt.figure(figsize=(2 * n_col, 2 * n_row))

    for i in range(n_row * n_col):

        plt.subplot(n_row, n_col, i + 1)

        plt.axis("off")

        plt.imshow(images[i].reshape(h, w), cmap = matplotlib.cm.binary)

    plt.show()

    

def plot_loss(history):

    loss, val_loss = zip(*history)

    plt.figure(figsize=(15, 9))

    plt.plot(loss, label="train_loss")

    plt.plot(val_loss, label="val_loss")

    plt.legend(loc='best')

    plt.xlabel("epochs")

    plt.ylabel("loss")

    plt.show()
class autoencoder(nn.Module):

    def __init__(self):

        super(autoencoder, self).__init__()



        self.encoder = nn.Sequential(

            nn.Linear(input_size + labels_length, 64),

            #nn.ReLU(True),

            #nn.Linear(128, 64),

            nn.ReLU(True),

            nn.Linear(64, hidden_size))

             

        self.decoder = nn.Sequential(

            nn.Linear(hidden_size + labels_length, 64),

            nn.ReLU(True),

            #nn.Linear(64, 128),

            #nn.ReLU(True),

            nn.Linear(64, input_size),

            nn.Tanh())



    def encode(self, x, labels):

        x = x.view(-1, 1*28*28)

        x = torch.cat((x,labels),dim=1)

        return self.encoder(x)

    

    def decode(self, x, labels):

        x = torch.cat((x,labels),dim=1)

        return self.decoder(x)



    def forward(self, x, labels):

        x = self.encode(x,labels)

        x = self.decode(x,labels)

        return x
model = autoencoder().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def fit_epoch(model, train_loader, criterion, optimizer, is_cvae):

    running_loss = 0.0

    running_corrects = 0

    processed_data = 0

  

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)

        labels = one_hot(labels,9).to(DEVICE)

        optimizer.zero_grad()

        if is_cvae:

            outputs, mu, logvar = model(inputs,labels)

            loss = vae_loss_fn(inputs.view(-1,28*28), outputs, mu, logvar)

            loss.backward()

        else:

            outputs = model(inputs,labels)

            loss = criterion(outputs, inputs.view(-1,28*28))

            loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        processed_data += inputs.size(0)

              

    train_loss = running_loss / processed_data

    return train_loss

  

def eval_epoch(model, val_loader, criterion, is_cvae):

    model.eval()

    running_loss = 0.0

    processed_size = 0

    inp,out = [],[]

    for inputs, labels in val_loader:

        inputs = inputs.to(DEVICE)

        labels = one_hot(labels,9).to(DEVICE)



        with torch.set_grad_enabled(False):

            if is_cvae:

                outputs, mu, logvar = model(inputs,labels)

                loss = vae_loss_fn(inputs.view(-1,28*28), outputs, mu, logvar)

                loss.backward()

            else:

                outputs = model(inputs,labels)

                loss = criterion(outputs, inputs.view(-1,28*28))

                inp,out = inputs, outputs



        running_loss += loss.item() * inputs.size(0)

        processed_size += inputs.size(0)

        

    with torch.set_grad_enabled(False):

        plot_gallery([inp[0].cpu(),out[0].cpu()],28,28,1,2)



    val_loss = running_loss / processed_size

    return val_loss

  

def train(train_loader, val_loader, model, epochs, batch_size, is_cvae=False):

    history = []

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:        

        opt = torch.optim.Adam(model.parameters())

        criterion = nn.MSELoss()

        for epoch in range(epochs):

            train_loss = fit_epoch(model, train_loader, criterion, opt, is_cvae)

            print("loss", train_loss)            

            val_loss = eval_epoch(model, val_loader, criterion, is_cvae)

            history.append((train_loss, val_loss))            

            pbar_outer.update(1)

            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss, v_loss=val_loss))            

    return history
history = train(train_dataset, val_dataset, model=model, epochs=200, batch_size=128, is_cvae=False)
plot_loss(history)
with torch.set_grad_enabled(False):

    number_for_generation=[0,1,2,3,4,5,6,7,8,9]

    inputs = torch.FloatTensor(np.random.randn(10*28*28).reshape(-1,28,28))

    inputs = inputs.to(DEVICE)

    label = one_hot(number_for_generation,9).to(DEVICE)

    outputs = model(inputs,label)

    plot_gallery(outputs.cpu(),28,28,1,10)
def get_latent_data(net, count=1000, is_cvae=False):

    latent_vectors = []

    latent_labels = []

    img_inputs = []

    rounds = count/100

    i=0

    with torch.set_grad_enabled(False):

        dataset_loader = DataLoader(dataset, batch_size=100, shuffle=True)

        for inputs,labels in dataset_loader:

            inputs = inputs.to(DEVICE)

            labels_one_hot = one_hot(labels,9).to(DEVICE)

            if is_cvae:

                outputs, mu, logvar = net(inputs,labels_one_hot)

            else:

                outputs = net(inputs,labels_one_hot)

            outputs = outputs.cpu()

            if i==0:

              latent_vectors = outputs

              latent_labels = labels

              img_inputs = inputs

            else:

              latent_vectors = torch.cat((latent_vectors,outputs),0)

              latent_labels = torch.cat((latent_labels,labels),0)

              img_inputs = torch.cat((img_inputs,inputs),0)

            if i>rounds:

              break

            i+=1

    return img_inputs, latent_vectors, latent_labels



def plot_tsne(net, mode, count, is_cvae=False):

    img_inputs,latent_vectors,latent_labels = get_latent_data(net=net, count=count, is_cvae=is_cvae)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title('t-SNE')

    coords = TSNE(n_components=2,random_state=42).fit_transform(latent_vectors)

    if mode == 'imgs':

        for image, (x, y) in zip(img_inputs.cpu(), coords):

            im = OffsetImage(image.reshape(28, 28), zoom=1, cmap='gray')

            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)

            ax.add_artist(ab)

        ax.update_datalim(coords)

        ax.autoscale()

    elif mode == 'dots':

        classes = latent_labels

        plt.scatter(coords[:, 0], coords[:, 1], c=classes)

        plt.colorbar()

        for i in range(10):

            class_center = np.mean(coords[classes == i], axis=0)

            text = TextArea('{}'.format(i))

            ab = AnnotationBbox(text, class_center, xycoords='data', frameon=True)

            ax.add_artist(ab)

    plt.show()
plot_tsne(net=model, mode='dots', count=1000)
plot_tsne(net=model, mode='imgs', count=300)
class CVAE(nn.Module):

    def __init__(self, input_size, hidden_size=20):

        super(CVAE, self).__init__()

        input_size_with_label = input_size + labels_length

        hidden_size += labels_length

        

        self.fc1 = nn.Linear(input_size_with_label,512)

        self.fc21 = nn.Linear(512, hidden_size)

        self.fc22 = nn.Linear(512, hidden_size)

        

        self.relu = nn.ReLU()

        

        self.fc3 = nn.Linear(hidden_size, 512)

        self.fc4 = nn.Linear(512, input_size)

    

    def encode(self, x, labels):

        x = x.view(-1, 1*28*28)

        x = torch.cat((x, labels), 1)

        x = self.relu(self.fc1(x))

        return self.fc21(x), self.fc22(x)

        

    def decode(self, z, labels):

        torch.cat((z, labels), 1)

        z = self.relu(self.fc3(z))

        return torch.sigmoid(self.fc4(z))

        

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 *logvar)

        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

        

    def forward(self,x, labels):

        #targets = one_hot(targets,labels_length-1).float().to(DEVICE)

        mu, logvar = self.encode(x, labels)

        z = self.reparameterize(mu, logvar)

        x = self.decode(z, labels)

        return x, mu, logvar



def train_cvae(net, dataloader, test_dataloader, flatten=True, epochs=20):

    validation_losses = []

    optim = torch.optim.Adam(net.parameters())



    log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:  

        for i in range(epochs):

            for batch, labels in dataloader:

                batch = batch.to(DEVICE)

                labels = one_hot(labels,9).to(DEVICE)



                if flatten:

                    batch = batch.view(batch.size(0), 28*28)



                optim.zero_grad()

                x,mu,logvar = net(batch, labels)

                loss = vae_loss_fn(batch, x[:, :784], mu, logvar)

                loss.backward()

                optim.step()

            evaluate(validation_losses, net, test_dataloader, flatten=True)

            pbar_outer.update(1)

            tqdm.write(log_template.format(ep=i+1, v_loss=validation_losses[i]))

    plt.show()

    return validation_losses
cvae = CVAE(28*28).to(DEVICE)
def vae_loss_fn(x, recon_x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD



def evaluate(losses, autoencoder, dataloader, flatten=True):

    model = lambda x, y: autoencoder(x, y)[0]    

    loss_sum = []

    inp, out = [],[]

    loss_fn = nn.MSELoss()

    for inputs, labels in dataloader:

        inputs = inputs.to(DEVICE)

        labels = one_hot(labels,9).to(DEVICE)



        if flatten:

            inputs = inputs.view(inputs.size(0), 28*28)



        outputs = model(inputs, labels)

        loss = loss_fn(inputs, outputs)            

        loss_sum.append(loss)

        inp = inputs

        out = outputs



    with torch.set_grad_enabled(False):

        plot_gallery([inp[0].detach().cpu(),out[0].detach().cpu()],28,28,1,2)    



    losses.append((sum(loss_sum)/len(loss_sum)).item())
def train_cvae(net, dataloader, test_dataloader, flatten=True, epochs=50):

    validation_losses = []

    optim = torch.optim.Adam(net.parameters())



    log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:  

        for i in range(epochs):

            for batch, labels in dataloader:

                batch = batch.to(DEVICE)

                labels = one_hot(labels,9).to(DEVICE)



                if flatten:

                    batch = batch.view(batch.size(0), 28*28)



                optim.zero_grad()

                x,mu,logvar = net(batch, labels)

                loss = vae_loss_fn(batch, x[:, :784], mu, logvar)

                loss.backward()

                optim.step()

            evaluate(validation_losses, net, test_dataloader, flatten=True)

            pbar_outer.update(1)

            tqdm.write(log_template.format(ep=i+1, v_loss=validation_losses[i]))

    plt.show()

    return validation_losses
history = train_cvae(cvae, train_dataset, val_dataset)
val_loss = history

plt.figure(figsize=(15, 9))

plt.plot(val_loss, label="val_loss")

plt.legend(loc='best')

plt.xlabel("epochs")

plt.ylabel("loss")

plt.show()
plot_tsne(net=cvae, mode='dots', count=2000, is_cvae=True)
plot_tsne(net=cvae, mode='imgs', count=300, is_cvae=True)