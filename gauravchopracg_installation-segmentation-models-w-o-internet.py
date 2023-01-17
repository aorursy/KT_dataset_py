#.tar file without internet

!pip install ../input/segmentation-modelspytorch-req/efficientnet_pytorch-0.4.0/efficientnet_pytorch-0.4.0/
#.tar file without internet

!pip install ../input/segmentation-modelspytorch-req/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/
#same as git clone segmentation_models.pytorch

!pip install ../input/segmentation-modelspytorch
# for model intialization

!mkdir -p /tmp/.cache/torch/checkpoints/

!cp ../input/efficientnet-pytorch/efficientnet-b0-355c32eb.pth /tmp/.cache/torch/checkpoints/efficientnet-b0-355c32eb.pth
from segmentation_models_pytorch import Unet

model = Unet('efficientnet-b0', encoder_weights='imagenet', classes=4, activation=None)

model