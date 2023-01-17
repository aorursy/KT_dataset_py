!git clone https://gitlab.com/pw_neural_nets/numpy_gan.git
import os

os.chdir('numpy_gan')
!ls . #../../input/cropped-faces-celeba
# SEED = 420

# import random

# import numpy as np

# import torch

# import torchvision.utils as vutils

# import matplotlib.pyplot as plt

# import torchvision.transforms as transforms

# from torchvision import datasets



# from gan_utils import data_loading

# import hparams

# from DCGAN import models

# from DCGAN import pipeline

# from DCGAN import losses



# np.random.seed(SEED)

# random.seed(SEED)

# torch.manual_seed(SEED)



# if __name__ == "__main__":

#     device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

    

#     hparams.loader_type = 'load_CELEB'

#     hparams.data_path = '../../input/cropped-faces-celeba'#'../../input/celeba-dataset'#/img_align_celeba' #

#     hparams.epochs = 40

#     #hparams.grad_penalty = 5.0

#     #hparams.disc_bn = [False, True, True, True, False]

#     #hparams.weight_clipping = 0.01

#     hparams.disc_steps = 5

#     hparams.gen_lr = 0.00005

#     hparams.disc_lr = 0.00005

    

    

#     data_loader = data_loading.get_data_loader(

#         loader_type=hparams.loader_type,

#         data_path=hparams.data_path,

#         bs=hparams.bs,

#         image_size=hparams.image_size

#     )



#     generator = models.Generator(channels_shapes=hparams.gen_channels_shapes).to(device)

#     discriminator = models.Discriminator(channels_shapes=hparams.disc_channels_shapes).to(device)

    

#     generator.apply(models.weights_init)

#     discriminator.apply(models.weights_init)



#     gen_optimizer = torch.optim.RMSprop(generator.parameters(), lr=hparams.gen_lr)

#     disc_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=hparams.disc_lr)



# #     gen_optimizer = torch.optim.Adam(generator.parameters(), lr=hparams.gen_lr, betas=hparams.betas)

# #     disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=hparams.disc_lr, betas=hparams.betas)

    

#     gen_loss_fn = losses.WassersteinGeneratorLoss()#torch.nn.BCELoss()#

#     disc_loss_fn = losses.WassersteinDiscriminatorLoss()#torch.nn.BCELoss()#

    

# #     disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=0.99)

# #     gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer, gamma=0.99)

    

#     wgan = pipeline.WGANGP(

#         generator=generator,

#         discriminator=discriminator,

#         gen_loss_fn=gen_loss_fn,

#         disc_loss_fn=disc_loss_fn,

#         gen_optimizer=gen_optimizer,

#         disc_optimizer=disc_optimizer,

#         data_loader=data_loader,

# #         disc_lr_scheduler=disc_lr_scheduler,

# #         gen_lr_scheduler=gen_lr_scheduler,

#         device=device,

#     )

    

#     print(wgan.__dict__.items())



#     wgan.train_model()



#     plt.figure(figsize=(10, 5))

#     plt.title("Generator and Discriminator Loss During Training")

#     plt.plot(wgan.gen_losses, label="G")

#     plt.plot(wgan.disc_losses, label="D")

#     plt.xlabel("iterations")

#     plt.ylabel("Loss")

#     plt.legend()

#     plt.show()



#     # Grab a batch of real images from the dataloader

#     real_batch = next(iter(data_loader))



#     # Plot the real images

#     plt.figure(figsize=(15, 15))

#     plt.subplot(1, 2, 1)

#     plt.axis("off")

#     plt.title("Real Images")

#     plt.imshow(

#         np.transpose(vutils.make_grid(

#             real_batch[0].to(device)[:128],

#             padding=5,

#             normalize=True

#         ).cpu(), (1, 2, 0)))



#     # Plot the fake images from the last epoch

#     plt.subplot(1, 2, 2)

#     plt.axis("off")

#     plt.title("Fake Images")

#     plt.imshow(np.transpose(wgan.img_list[-5], (1, 2, 0)))

#     plt.show()



#     plt.figure(figsize=(10, 5))

#     plt.title("Generator and Discriminator Loss During Training")

#     plt.plot(wgan.gen_losses, label="G")

#     plt.plot(wgan.disc_losses, label="D")

#     plt.xlabel("iterations")

#     plt.ylabel("Loss")

#     plt.legend()

#     plt.show()



#     # Grab a batch of real images from the dataloader

#     real_batch = next(iter(data_loader))



#     # Plot the real images

#     plt.figure(figsize=(15, 15))

#     plt.subplot(1, 2, 1)

#     plt.axis("off")

#     plt.title("Real Images")

#     plt.imshow(

#         np.transpose(vutils.make_grid(

#             real_batch[0].to(device)[:128],

#             padding=5,

#             normalize=True

#         ).cpu(), (1, 2, 0)))



#     # Plot the fake images from the last epoch

#     plt.subplot(1, 2, 2)

#     plt.axis("off")

#     plt.title("Fake Images")

#     plt.imshow(np.transpose(wgan.img_list[-1], (1, 2, 0)))

#     plt.show()

# torch.save(wgan.img_list, './img_list.pytorch')
# fake = generator(wgan.fixed_noise[0:1]).detach().cpu()