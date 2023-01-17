!git clone https://gist.github.com/belkhir-nacim/5230ccfcab05f30c35abb03444f6a216 dataset_util
!pip install git+https://github.com/belkhir-nacim/generative_model_toolbox
!pip install pytorch-lightning
!pip install transformers
from collections import OrderedDict
import os,h5py
import numpy as np
import pandas as pd
from argparse import Namespace
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn,optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import pytorch_lightning as pl
from dataset_util.kaggle_textile_texuture_dataset import TextureDataset
from functools import partial
import math
from generative_models_toolbox.algos.vqvae import VQVAEModel
from transformers import GPT2LMHeadModel, GPT2Config
from generative_models_toolbox.utils.device import Cudafy
cudafy = Cudafy(0)
class TrainerVQVae(pl.LightningModule):

    def __init__(self, hparams):
        super(TrainerVQVae, self).__init__()
        #init model
        self.model = self.init_model(hparams=hparams)
        self.hparams = hparams
        self.last_imgs = None
        
    def init_model(self,hparams):
        num_hiddens = 128  if hparams.num_hiddens is None else hparams.num_hiddens
        num_residual_hiddens = 32 if hparams.num_residual_hiddens is None else hparams.num_residual_hiddens
        num_residual_layers = 2 if hparams.num_residual_layers is None else hparams.num_residual_layers
        embedding_dim = 64 if hparams.embedding_dim is None else hparams.embedding_dim
        num_embeddings = 128 if hparams.num_embeddings is None else hparams.num_embeddings
        commitment_cost = 0.25 if hparams.commitment_cost is None else hparams.commitment_cost
        decay = 0.99 if hparams.decay is None else hparams.decay
        return VQVAEModel(num_hiddens=num_hiddens,num_residual_hiddens=num_residual_hiddens,num_residual_layers=num_residual_layers, embedding_dim=embedding_dim,num_embeddings=num_embeddings,commitment_cost=commitment_cost,decay=decay)
        
    def prepare_data(self):
        
        self.data_mean = 0.3541
        self.data_variance = 0.1352
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize([self.data_mean], [self.data_variance]) ])
        assert((self.hparams.root is not None) and (os.path.exists(self.hparams.root)))
        self.hparams.patch_size = 64 if self.hparams.patch_size is None else  self.hparams.patch_size
        
        self.hparams.sub_sample_train = 1000 if self.hparams.sub_sample_train is None else self.hparams.sub_sample_train
        self.hparams.sub_sample_valid = 500 if self.hparams.sub_sample_valid is None else self.hparams.sub_sample_valid
        self.hparams.sub_sample_test = 2000 if self.hparams.sub_sample_test is None else self.hparams.sub_sample_test
        self.keep_angles = False if self.hparams.keep_angles is None else self.hparams.keep_angles
        self.batch_size = 256 if self.hparams.batch_size is None else self.hparams.batch_size
        self.dataset_train = TextureDataset(self.hparams.root,train=True,patch_size=self.hparams.patch_size, keep_angles=self.keep_angles, keep_defects=False, transformation=transformation, sub_sample=self.hparams.sub_sample_train)
        self.dataset_valid = TextureDataset(self.hparams.root,train=False, patch_size=self.hparams.patch_size, keep_angles=self.keep_angles, keep_defects=False, transformation=transformation,sub_sample=self.hparams.sub_sample_valid)
        self.dataset_test = TextureDataset(self.hparams.root,train=False, patch_size=self.hparams.patch_size, keep_angles=self.keep_angles, keep_defects=True, transformation=transformation, sub_sample=self.hparams.sub_sample_test)
        
    def configure_optimizers(self):
        self.hparams.lr = 5e-3  if self.hparams.lr is  None else self.hparams.lr
        if self.hparams.optim is not None:
            if self.hparams.optim =='adam':
                optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,amsgrad=False)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=lr)
        return optim
    
    def _step_run(self,z):
        z = self(z)
        z = self.model(z)
        return self(z)
    
    def forward(self, x):
        return x

    def encode(self, x):
        return self.model.encode(x)

    def reconstruct_from_code(self, code):
        return self.model.reconstruct_from_code(code)
    
    
    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    def training_step(self, batch, batch_nb):
        X, _, _ = batch
        commit_loss, XR, perplexity = self.model(X)
        recons_loss = F.mse_loss(X, XR)/self.data_variance
        loss = recons_loss + commit_loss
        return OrderedDict({"loss": loss, "log": {"loss": loss,"commit_loss": commit_loss,"recons_loss": recons_loss,}})
        
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tqdm_dict = {'train_loss': avg_loss}
        return {'train_loss': avg_loss, 'log': tqdm_dict, 'progress_bar': tqdm_dict}
    
            
    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    def validation_step(self, batch, batch_nb):
        data, _, _ = batch
        vq_loss, data_recon, perplexity = self._step_run(data)
        loss = F.mse_loss(data_recon, data) / self.data_variance
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss}
        return {'val_loss ': avg_loss, 'log': tqdm_dict, 'progress_bar': tqdm_dict}

        
    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    def test_step(self, batch, batch_nb):
        data, _, _ = batch
        if self.last_imgs is None: self.last_imgs = data
        vq_loss, data_recon, perplexity = self._step_run(data)
        loss = F.mse_loss(data_recon, data) / self.data_variance
        return {'test_loss': loss}
    
    def test_epoch_end(self, output):
        if self.last_imgs is not None:
            # log sampled images
            _, sample_imgs, _ = self.model(self.last_imgs)
            grid = make_grid(sample_imgs,16)*self.data_variance + self.data_mean
            self.logger.experiment.add_image(f'GeneratedImages', grid, self.current_epoch)
            grid = make_grid(self.last_imgs,16)*self.data_variance + self.data_mean
            self.logger.experiment.add_image(f'InputImages', grid, self.current_epoch)
        avg_loss = torch.stack([x['test_loss'] for x in output]).mean()
        tqdm_dict = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tqdm_dict, 'progress_bar': tqdm_dict}
def run_train_vqvae(params= None):
    if params is None:
        params = {'root':'/kaggle/input/textiledefectdetection','patch_size':64, 'keep_angles':True, 'sub_sample_train':2000, 'sub_sample_valid': 512, 'sub_sample_test':1024, 'batch_size':256, 'lr': 1e-3,  'num_hiddens':64, 'num_residual_hiddens': 32, 'num_residual_layers' : 2, 'embedding_dim' :64, 'num_embeddings' : 128, 'commitment_cost' : 0.25,'decay' : 0.99, 'optim':'adam'}
    hparams = Namespace(**params)
    vqvae_model = TrainerVQVae(hparams)
    vqvae_trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, show_progress_bar=False )
    vqvae_trainer.fit(vqvae_model)
    return vqvae_model, vqvae_trainer
run_train_vqvae(params = {'root':'/kaggle/input/textiledefectdetection','patch_size':64, 'keep_angles':True, 'sub_sample_train':2000, 'sub_sample_valid': 512, 'sub_sample_test':1024, 'batch_size':256, 'lr': 1e-3,  'num_hiddens':128 ,
                          'num_residual_hiddens': 32, 'num_residual_layers' : 2, 'embedding_dim' :64, 'num_embeddings' : 256, 'commitment_cost' : 0.25,'decay' : 0.99, 'optim':'adam','check_val_every_n_epoch':10,'max_epochs':300,
                          'gpus':1 if torch.cuda.is_available() else 0})
def display_result_images(path,checkpoint):
    vqvae_model = TrainerVQVae.load_from_checkpoint(checkpoint)
    vqvae_trainer = pl.Trainer(gpus=1, max_epochs=300, check_val_every_n_epoch=3, show_progress_bar=False )
    vqvae_trainer.test(vqvae_model)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    input_images = event_acc.Images('InputImages')[0]
    generated_images = event_acc.Images('GeneratedImages')[0]
    import tensorflow as tf
    i = tf.image.decode_image(input_images.encoded_image_string)
    d = tf.image.decode_image(generated_images.encoded_image_string)
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2, figsize=(20,20))
    axes[0].imshow(i.numpy())
    axes[1].imshow(d.numpy())
import torch, gc
gc.collect()
torch.cuda.empty_cache()
# display_result_images('/kaggle/working/lightning_logs/version_1/',checkpoint='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=284.ckpt')
class Generator(pl.LightningModule):
    def __init__(self, hparams, load_dataset=True):
        super().__init__()
        if load_dataset:
            self.dataset = self.load_dataset(hparams)
            hparams.vocab_size = self.dataset.vocab_size
            hparams.height, hparams.width = self.dataset.shape[1:]
            hparams.max_length = self.dataset.length
            hparams.start_token = self.dataset.start_token
        self.model = self.build_model(hparams)
        self.hparams = hparams
        
    def load_dataset(self, hparams):
        print("Loading the dataset of codes into memory...")
        device = "cpu"
        vqvae = TrainerVQVae.load_from_checkpoint(hparams.vqvae_model_path)
        vqvae = vqvae.to(device)
        vqvae.prepare_data()
        codes = []
        nb = 0
        for X, _, _ in vqvae.train_dataloader():
            X = X.to(device)
            zinds = vqvae.encode(X)
            if nb==0:
                print('zinds', zinds.shape)
            codes.append(zinds.data.cpu())
            nb += len(zinds)
            if hasattr(hparams,'nb_examples') and hparams.nb_examples is not None and  nb >= hparams.nb_examples:
                break
        codes = torch.cat(codes)
        print('Code shape', codes.shape, type(codes),'min',codes.min(),'max', codes.max())
        print(np.unique(codes))
        if hparams.nb_examples and len(codes) >= hparams.nb_examples:
            codes = codes[:hparams.nb_examples]
        vocab_size = vqvae.model.num_embeddings + 1
        start_token = vqvae.model.num_embeddings
        codes_ = codes.view(len(codes), -1)
        codes_ = torch.cat([(torch.ones(len(codes_), 1).long() * start_token), codes_.long(),], dim=1)
        print('codes', codes_.shape)
        length = codes_.shape[1]
        dataset = TensorDataset(codes_)
        dataset.vocab_size = vocab_size
        dataset.shape = codes.shape

        dataset.length = length
        dataset.start_token = start_token
        print("Done loading dataset")
        return dataset

    def forward(self, x):
        return self.model(x)

    def build_model(self, hparams):
        config = GPT2Config(vocab_size=hparams.vocab_size, n_positions=hparams.max_length, n_ctx=hparams.max_length, n_embd=256, n_layer=4, n_head=4, resid_pdrop=0.2,embd_pdrop=0.2,attn_pdrop=0.2)
        return GPT2LMHeadModel(config)

    def generate(self, nb_examples=1, **kwargs):
        
        input_ids = torch.zeros(nb_examples, 1).long().to(self.device)
        input_ids[:] = self.hparams.vocab_size - 1
        result = self.model.generate(input_ids, max_length=self.hparams.max_length, **kwargs)
        print(result.shape,'output of model')
        result = result[:, 1:]
        result = result.contiguous()
        print(result.shape, 'result afeter reshaping')
        result = result.view(nb_examples, self.hparams.height, self.hparams.width)
        return result

    def training_step(self, batch, batch_idx):
        (X,) = batch
        loss, *rest = self.model(X, labels=X)
        output = OrderedDict({"loss": loss, "log": {"loss": loss,},})
        return output

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
def train_transformer_generator(hparams=None):
    if hparams is None: 
        hparams = {'folder': 'out/generator','epochs': 200,'vqvae_model_path':None, 'lr': 5e-3, 'weight_decay': 0, 'scheduler_gamma': 1, 'batch_size': 64, 'num_workers': 1, 'gpus': 1,'nb_examples':2000}
    hparams = Namespace(**hparams)
    model = Generator(hparams)
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.folder, name="logs")
    trainer = pl.Trainer(default_root=hparams.folder, max_epochs=hparams.epochs,show_progress_bar=False,gpus=hparams.gpus,logger=logger, resume_from_checkpoint=hparams.resume_from_checkpoint)
    trainer.fit(model)
    return model, trainer
hparams = {'resume_from_checkpoint':None,
           'folder': 'out/generator', 'epochs': 200,  'lr': 1e-3, 'weight_decay': 0, 'scheduler_gamma':  1, 'batch_size': 64,
           'num_workers': 1, 'nb_examples':3000, 'vqvae_model_path':'/kaggle/working/lightning_logs/version_0/checkpoints/epoch=299.ckpt', 'gpus': 1 if torch.cuda.is_available() else 0}
generator, trainergenerator = train_transformer_generator(hparams=hparams)
import matplotlib.pyplot as plt
def random_generate_from_GPT2(path_generator='/kaggle/working/out/generator/logs/version_3/checkpoints/epoch=18.ckpt',
                              path_vqvae='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=9.ckpt', nb_examples = 2):
    vqvae_model = TrainerVQVae.load_from_checkpoint(path_vqvae)
    vqvae_model = cudafy(vqvae_model)
    vqvae_model.eval()
    generator = Generator.load_from_checkpoint(path_generator)
    generator = cudafy(generator)
    generator.eval()

    input_ids = cudafy(torch.randint(0,generator.hparams.vocab_size-1, (nb_examples**2, 1)).long())
    result = generator.model.generate(input_ids, max_length=generator.hparams.max_length, do_sample=True)
    result = result[:, 1:]
    result = result.contiguous()
    result = result.view(nb_examples**2, generator.hparams.height, generator.hparams.width)
    result[result>=generator.hparams.vocab_size-1] = generator.hparams.vocab_size-2
    with torch.no_grad():
        generated = vqvae_model.reconstruct_from_code(result)

        fig, axes = plt.subplots(1,2, figsize=(25,25))
        img = make_grid(generated, nb_examples).permute(1,2,0)*0.1352 + 0.3541
        axes[0].imshow(cudafy.get(img).detach().cpu().numpy())
        axes[0].set_title('Generated from the sampled codes')
        vqvae_model.model.eval()
        _, XR, _ = vqvae_model.model(generated)
        img = make_grid(XR, nb_examples).permute(1,2,0)*0.1352 + 0.3541
        axes[1].imshow(cudafy.get(img).detach().cpu().numpy())
        axes[1].set_title('Reconstructed from the generated images')
    
random_generate_from_GPT2(path_generator='/kaggle/working/out/generator/logs/version_0/checkpoints/epoch=199.ckpt', path_vqvae='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=299.ckpt', nb_examples = 16)
def plot_compare_original():
    path_vqvae='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=299.ckpt'
    vqvae = TrainerVQVae.load_from_checkpoint(path_vqvae)
    vqvae.eval()
    vqvae.prepare_data()
    codes = []
    nb = 0
    with torch.no_grad():
        for X, _, _ in vqvae.val_dataloader():
            fig, axes = plt.subplots(1,2, figsize=(25,25))
            img = make_grid(X, 16).permute(1,2,0)*0.1352 + 0.3541
            axes[0].imshow(cudafy.get(img).detach().cpu().numpy())
            axes[0].set_title('Original Images (without defects)')

            vqvae.model.eval()
            _, XR, _ = vqvae.model(X)
            img = make_grid(XR, 16).permute(1,2,0)*0.1352 + 0.3541
            axes[1].imshow(cudafy.get(img).detach().cpu().numpy())
            axes[1].set_title('Reconstructed Images (without defects)')        
            break
    with torch.no_grad():
        for X, _, _ in vqvae.test_dataloader():
            fig, axes = plt.subplots(1,2, figsize=(25,25))
            img = make_grid(X, 16).permute(1,2,0)*0.1352 + 0.3541
            axes[0].imshow(cudafy.get(img).detach().cpu().numpy())
            axes[0].set_title('Original Images (with defects)')

            vqvae.model.eval()
            _, XR, _ = vqvae.model(X)
            img = make_grid(XR, 16).permute(1,2,0)*0.1352 + 0.3541
            axes[1].imshow(cudafy.get(img).detach().cpu().numpy())
            axes[1].set_title('Reconstructed Images (with defects)')        
            break

    erasor = transforms.RandomErasing(p=1., value='random')

    with torch.no_grad():
        for X, _, _ in vqvae.test_dataloader():

            for i in range(X.shape[0]):
                X[i] = erasor(X[i])

            fig, axes = plt.subplots(1,2, figsize=(25,25))
            img = make_grid(X, 16).permute(1,2,0)*0.1352 + 0.3541
            axes[0].imshow(cudafy.get(img).detach().cpu().numpy())
            axes[0].set_title('Original Images (with defects)')

            vqvae.model.eval()
            _, XR, _ = vqvae.model(X)
            img = make_grid(XR, 16).permute(1,2,0)*0.1352 + 0.3541
            axes[1].imshow(cudafy.get(img).detach().cpu().numpy())
            axes[1].set_title('Reconstructed Images (with defects)')        
            break
plot_compare_original()

def random_generate_from_erased_images_GPT2(path_generator='/kaggle/working/out/generator/logs/version_3/checkpoints/epoch=18.ckpt',
                              path_vqvae='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=9.ckpt', nb_examples = 2,test_image=True):
    vqvae = TrainerVQVae.load_from_checkpoint(path_vqvae)
    vqvae = cudafy(vqvae)
    vqvae.eval()
    vqvae.prepare_data()
    erasor = transforms.RandomErasing(p=1., value='random',inplace=False)
    codes = []
    codes = []
    nb = 0
    
    x_example = []
    x_example_erased = []
    loader = vqvae.test_dataloader() if test_image else vqvae.val_dataloader()
    for X, _, _ in loader:
        X = cudafy(X)
        x_example.append(torch.zeros_like(X) + X)
        for iii in range(X.shape[0]):
            X[iii] = erasor(X[iii])
            
        x_example_erased.append(X)
        zinds = vqvae.encode(X)
        if nb==0:
            print('zinds', zinds.shape)
        codes.append(zinds.data.cpu())
        nb += len(zinds)
        if len(codes) >= nb_examples**2:
            break
            
    x_example = torch.cat(x_example)
    x_example_erased = torch.cat(x_example_erased)
    codes = torch.cat(codes)
    print('Code shape', codes.shape, type(codes),'min',codes.min(),'max', codes.max())
    print(np.unique(codes))
    if len(codes) >= nb_examples**2:
        codes = codes[:nb_examples**2]
        x_example = x_example[:nb_examples**2]
        x_example_erased = x_example_erased[:nb_examples**2]
    vocab_size = vqvae.model.num_embeddings + 1
    print(vocab_size,'vocab_size')
    start_token = vqvae.model.num_embeddings
    print(start_token,'start_token')
    codes_ = codes.view(len(codes), -1)
    print(codes_.shape)
    codes_ = torch.cat([(torch.ones(len(codes_), 1).long() * start_token), codes_.long(),], dim=1)
    
    generator = Generator.load_from_checkpoint(path_generator)
    generator = cudafy(generator)
    generator.eval()
    
    input_ids = cudafy(codes_[:,:256//4])
    result = generator.model.generate(input_ids, max_length=generator.hparams.max_length, do_sample=True)
    result = result[:, 1:]
    result = result.contiguous()
    result = result.view(nb_examples**2, generator.hparams.height, generator.hparams.width)
    result[result>=generator.hparams.vocab_size-1] = generator.hparams.vocab_size-2
    with torch.no_grad():
        generated = vqvae.reconstruct_from_code(result)

        fig, axes = plt.subplots(1,4, figsize=(25,25))
        img = make_grid(generated, nb_examples).permute(1,2,0)*0.1352 + 0.3541
        axes[0].imshow(cudafy.get(img).detach().cpu().numpy())
        axes[0].set_title('Generated from the sampled codes')
        vqvae.model.eval()
        _, XR, _ = vqvae.model(generated)
        img = make_grid(XR, nb_examples).permute(1,2,0)*0.1352 + 0.3541
        axes[1].imshow(cudafy.get(img).detach().cpu().numpy())
        axes[1].set_title('Reconstructed from the generated images')
        
        img = make_grid(x_example, nb_examples).permute(1,2,0)*0.1352 + 0.3541
        axes[2].imshow(cudafy.get(img).detach().cpu().numpy())
        axes[2].set_title('Original Image')
        
        img = make_grid(x_example_erased, nb_examples).permute(1,2,0)*0.1352 + 0.3541
        axes[3].imshow(cudafy.get(img).detach().cpu().numpy())
        axes[3].set_title('Original Image with Cutout')
random_generate_from_erased_images_GPT2(path_generator='/kaggle/working/out/generator/logs/version_0/checkpoints/epoch=199.ckpt', path_vqvae='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=299.ckpt', nb_examples = 8)
random_generate_from_erased_images_GPT2(path_generator='/kaggle/working/out/generator/logs/version_0/checkpoints/epoch=199.ckpt', path_vqvae='/kaggle/working/lightning_logs/version_0/checkpoints/epoch=299.ckpt', nb_examples = 8,test_image=False)
