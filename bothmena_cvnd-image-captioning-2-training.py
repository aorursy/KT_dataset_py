!pip install pycocotools
import sys

sys.path.append('../input/image-captioning-project-files')
import torch

from torch import nn, optim

from torchvision import transforms

from pycocotools.coco import COCO

from data_loader import get_loader

from model import EncoderCNN, DecoderRNN

import math



import torch.utils.data as data

import numpy as np

import os

import requests

import time
## TODO #1: Select appropriate values for the Python variables below.

batch_size = 10                  # batch size

vocab_threshold = 5              # minimum word count threshold

vocab_from_file = False          # if True, load existing vocab file

embed_size = 256                 # dimensionality of image and word embeddings

hidden_size = 512                # number of features in hidden state of the RNN decoder

start_epoch = 3

num_epochs = 2                   # number of training epochs

save_every = 1                   # determines frequency of saving model weights

print_every = 100                # determines window for printing average loss

log_file = 'train_log.txt'       # name of file with saved training loss and perplexity
encoder_path = os.path.join('../input/cvnd-image-captioning-2-training/models', 'encoder-{}.pkl'.format(start_epoch - 1))

decoder_path = os.path.join('../input/cvnd-image-captioning-2-training/models', 'decoder-{}.pkl'.format(start_epoch - 1))



assert os.path.isfile(encoder_path)

assert os.path.isfile(decoder_path)
# (Optional) TODO #2: Amend the image transform below.

transform_train = transforms.Compose([ 

    transforms.Resize(256),                          # smaller edge of image resized to 256

    transforms.RandomCrop(224),                      # get 224x224 crop from random location

    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5

    transforms.ToTensor(),                           # convert the PIL Image to a tensor

    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model

                         (0.229, 0.224, 0.225))])
# Build data loader.

data_loader = get_loader(transform=transform_train,

                         mode='train',

                         batch_size=batch_size,

                         vocab_threshold=vocab_threshold,

                         vocab_from_file=vocab_from_file)
# The size of the vocabulary.

vocab_size = len(data_loader.dataset.vocab)
# Initialize the encoder and decoder. 

encoder = EncoderCNN(embed_size)

decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.load_state_dict(torch.load(encoder_path))

encoder.eval()



decoder.load_state_dict(torch.load(decoder_path))

decoder.eval()
# Move models to GPU if CUDA is available. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder.to(device)

decoder.to(device)
# Define the loss function. 

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
# TODO #3: Specify the learnable parameters of the model.

params = list(decoder.parameters()) + list(encoder.embed.parameters())
# TODO #4: Define the optimizer.

optimizer = optim.Adam(params=params, lr=0.001)
# Set the total number of training steps per epoch.

total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
# Open the training log file.

f = open(log_file, 'w')
# old_time = time.time()

# response = requests.request("GET", 

#                             "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", 

#                             headers={"Metadata-Flavor":"Google"})
os.mkdir('models')
# total_step = 3
encoder.train()

decoder.train()



for epoch in range(start_epoch, num_epochs+start_epoch):

    

    for i_step in range(1, total_step+1):

        

#         if time.time() - old_time > 60:

#             old_time = time.time()

#             requests.request("POST", 

#                              "https://nebula.udacity.com/api/v1/remote/keep-alive", 

#                              headers={'Authorization': "STAR " + response.text})

        

        # Randomly sample a caption length, and sample indices with that length.

        indices = data_loader.dataset.get_train_indices()

        # Create and assign a batch sampler to retrieve a batch with the sampled indices.

        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)

        data_loader.batch_sampler.sampler = new_sampler

        

        # Obtain the batch.

        images, captions = next(iter(data_loader))



        # Move batch of images and captions to GPU if CUDA is available.

        images = images.to(device)

        captions = captions.to(device)

        

        # Zero the gradients.

        decoder.zero_grad()

        encoder.zero_grad()

        

        # Pass the inputs through the CNN-RNN model.

        features = encoder(images)

        outputs = decoder(features, captions)

        

        # Calculate the batch loss.

        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        

        # Backward pass.

        loss.backward()

        

        # Update the parameters in the optimizer.

        optimizer.step()

            

        # Get training statistics.

        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs+start_epoch-1, i_step, total_step, loss.item(), np.exp(loss.item()))

        

        # Print training statistics (on same line).

        print('\r' + stats, end="")

        sys.stdout.flush()

        

        # Print training statistics to file.

        f.write(stats + '\n')

        f.flush()

        

        # Print training statistics (on different line).

        if i_step % print_every == 0:

            print('\r' + stats)

            

    # Save the weights.

    if epoch % save_every == 0:

        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))

        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))
!ls models/
# Close the training log file.

f.close()
# (Optional) TODO: Validate your model.