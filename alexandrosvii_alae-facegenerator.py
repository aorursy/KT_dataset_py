!git clone https://github.com/podgorskiy/ALAE.git

%cd ALAE

%set_env PYTHONPATH=/project/pylib/src:/env/python

!pip install -r requirements.txt

#Dowload Mdels

!python training_artifacts/download_all.py

#Upload Pictures:
%load_ext autoreload

%autoreload 2

!git clone https://github.com/podgorskiy/ALAE.git

!conda install torch  



import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import torch.utils.data

os.chdir('/kaggle/working/ALAE')

!conda install requirements

from net import *

from model import Model

from launcher import run

from checkpointer import Checkpointer

from dlutils.pytorch import count_parameters

from defaults import get_cfg_defaults

import lreq

import logging

from PIL import Image

import bimpy

import cv2



import matplotlib.pyplot as plt

%matplotlib inline



lreq.use_implicit_lreq.set(True)





indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]



labels = ["gender",

          "smile",

          "attractive",

          "wavy-hair",

          "young",

          "big lips",

          "big nose",

          "chubby",

          "glasses",

          ]
def loadNext(index=0):

    img = np.asarray(Image.open(path + '/' + paths[index]))

    current_file.value = paths[index]



    if len(paths) == 0:

        paths.extend(paths_backup)



    if img.shape[2] == 4:

        img = img[:, :, :3]

    im = img.transpose((2, 0, 1))

    x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.

    if x.shape[0] == 4:

        x = x[:3]



    needed_resolution = model.decoder.layer_to_resolution[-1]

    while x.shape[2] > needed_resolution:

        x = F.avg_pool2d(x, 2, 2)

    if x.shape[2] != needed_resolution:

        x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))



    img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()



    latents_original = encode(x[None, ...].cuda())

    latents = latents_original[0, 0].clone()

    latents -= model.dlatent_avg.buff.data[0]

    

    for v, w in zip(attribute_values, W):

        v.value = (latents * w).sum()



    for v, w in zip(attribute_values, W):

        latents = latents - v.value * w



    return latents, latents_original, img_src





def loadRandom():

        latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)

        lat = torch.tensor(latents).float().cuda()

        dlat = mapping_fl(lat)

        layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]

        ones = torch.ones(layer_idx.shape, dtype=torch.float32)

        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)

        dlat = torch.lerp(model.dlatent_avg.buff.data, dlat, coefs)

        x = decode(dlat)[0]

        img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()

        latents_original = dlat

        latents = latents_original[0, 0].clone()

        latents -= model.dlatent_avg.buff.data[0]

        

        for v, w in zip(attribute_values, W):

            v.value = (latents * w).sum()



        for v, w in zip(attribute_values, W):

            latents = latents - v.value * w



        return latents, latents_original, img_src

    

def update_image(w, latents_original):

    with torch.no_grad():

        w = w + model.dlatent_avg.buff.data[0]

        w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)



        layer_idx = torch.arange(model.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]

        cur_layers = (7 + 1) * 2

        mixing_cutoff = cur_layers

        styles = torch.where(layer_idx < mixing_cutoff, w, latents_original)



        x_rec = decode(styles)

        resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)

        resultsample = resultsample.cpu()[0, :, :, :]

        return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)
torch.cuda.set_device(0)

torch.set_default_tensor_type('torch.cuda.FloatTensor')



cfg = get_cfg_defaults()

cfg.merge_from_file("./configs/ffhq.yaml")



logger = logging.getLogger("logger")

logger.setLevel(logging.DEBUG)






model = Model(

    startf=cfg.MODEL.START_CHANNEL_COUNT,

    layer_count=cfg.MODEL.LAYER_COUNT,

    maxf=cfg.MODEL.MAX_CHANNEL_COUNT,

    latent_size=cfg.MODEL.LATENT_SPACE_SIZE,

    truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,

    truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,

    mapping_layers=cfg.MODEL.MAPPING_LAYERS,

    channels=cfg.MODEL.CHANNELS,

    generator=cfg.MODEL.GENERATOR,

    encoder=cfg.MODEL.ENCODER)



model.cuda()

model.eval()

model.requires_grad_(False)



decoder = model.decoder

encoder = model.encoder

mapping_tl = model.mapping_tl

mapping_fl = model.mapping_fl

dlatent_avg = model.dlatent_avg



logger.info("Trainable parameters generator:")

count_parameters(decoder)



logger.info("Trainable parameters discriminator:")

count_parameters(encoder)



arguments = dict()

arguments["iteration"] = 0



model_dict = {

    'discriminator_s': encoder,

    'generator_s': decoder,

    'mapping_tl_s': mapping_tl,

    'mapping_fl_s': mapping_fl,

    'dlatent_avg': dlatent_avg

}



checkpointer = Checkpointer(cfg,

                            model_dict,

                            {},

                            logger=logger,

                            save=False)



extra_checkpoint_data = checkpointer.load()



model.eval()



layer_count = cfg.MODEL.LAYER_COUNT





def encode(x):

    Z, _ = model.encode(x, layer_count - 1, 1)

    Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

    # print(Z.shape)

    return Z





def decode(x):

    layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]

    ones = torch.ones(layer_idx.shape, dtype=torch.float32)

    coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)

    # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)

    return model.decoder(x, layer_count - 1, 1, noise=True)
path = 'dataset_samples/faces/realign1024x1024'



paths = list(os.listdir(path))

paths.sort()

paths_backup = paths[:]





randomize = bimpy.Bool(True)

current_file = bimpy.String("")



ctx = bimpy.Context()



attribute_values = [bimpy.Float(0) for i in indices]



# W: 9x512

W = [torch.tensor(np.load("principal_directions/direction_%d.npy" % i), dtype=torch.float32) for i in indices]



rnd = np.random.RandomState(5)
im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

seed = 0



#image_index = 6 # image index 

slider_vals = np.linspace(-20, 20, 10) # simulate the slider form interactive demo



for image_index in range(10):

    for target_attr in range(len(labels)):

        latents, latents_original, img_src = loadNext(image_index)  



        fig, ax = plt.subplots(1, len(slider_vals)+1, figsize=(25, 6))

        fig.suptitle(f"Variation across: {labels[target_attr]}", y=0.7)

        ax[0].imshow(img_src)

        ax[0].set_title("Original image")

        ax[0].axis('off')



        for i, val in enumerate(slider_vals):

            attribute_values[target_attr].value = val

            new_latents = latents + sum([v.value * w for v, w in zip(attribute_values, W)])

            new_im = update_image(new_latents, latents_original)



            ax[i+1].imshow(new_im)

            ax[i+1].set_title(round(val, 1))

            ax[i+1].axis('off')