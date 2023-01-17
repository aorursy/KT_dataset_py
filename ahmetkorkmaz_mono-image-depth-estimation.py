import matplotlib.pyplot as plt

import torch.nn.functional as F

import numpy as np

import pandas as pd

from collections import OrderedDict

import sys

#import glob

#import argparse

import PIL.Image as pil

import matplotlib as mpl

import matplotlib.cm as cm

from torchvision import transforms

import torch

import torch.nn as nn

import torchvision.models as models

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def upsample(x):

    """Upsample input tensor by a factor of 2

    """

    return F.interpolate(x, scale_factor=2, mode="nearest")
class DepthDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):

        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels

        self.use_skips = use_skips

        self.upsample_mode = 'nearest'

        self.scales = scales

        self.num_ch_enc = num_ch_enc

        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder

        self.convs = OrderedDict()

        for i in range(4, -1, -1):

            # upconv_0

            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]

            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1

            num_ch_in = self.num_ch_dec[i]

            if self.use_skips and i > 0:

                num_ch_in += self.num_ch_enc[i - 1]

            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:

            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):

        self.outputs = {}

        # decoder

        x = input_features[-1]

        for i in range(4, -1, -1):

            x = self.convs[("upconv", i, 0)](x)

            x = [upsample(x)]

            if self.use_skips and i > 0:

                x += [input_features[i - 1]]

            x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:

                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
class ResnetEncoder(nn.Module):

    """Pytorch module for a resnet encoder

    """

    def __init__(self, num_layers, pretrained, num_input_images=1):

        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,

                   34: models.resnet34,

                   50: models.resnet50,

                   101: models.resnet101,

                   152: models.resnet152}

        if num_layers not in resnets:

            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:

            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)

        else:

            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:

            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):

        self.features = []

        x = (input_image - 0.45) / 0.225

        x = self.encoder.conv1(x)

        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x))

        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))

        self.features.append(self.encoder.layer2(self.features[-1]))

        self.features.append(self.encoder.layer3(self.features[-1]))

        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
class ConvBlock(nn.Module):

    """Layer to perform a convolution followed by ELU

    """

    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)

        self.nonlin = nn.ELU(inplace=True)



    def forward(self, x):

        out = self.conv(x)

        out = self.nonlin(out)

        return out

class Conv3x3(nn.Module):

    """Layer to pad and convolve input

    """

    def __init__(self, in_channels, out_channels, use_refl=True):

        super(Conv3x3, self).__init__()

        if use_refl:

            self.pad = nn.ReflectionPad2d(1)

        else:

            self.pad = nn.ZeroPad2d(1)

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)



    def forward(self, x):

        out = self.pad(x)

        out = self.conv(out)

        return out
def estimate_depth():

    device = torch.device("cpu")

    image_path ='/kaggle/input/safranbolu-sokak/safranbolu_sokak.jpg'

    model_path = "/kaggle/input/mono-odom-640x192"

    print("-> Loading model from ", model_path)

    encoder_path = os.path.join(model_path, "encoder.pth")

    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL

    print("   Loading pretrained encoder")

    encoder = ResnetEncoder(18, False)

    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with

    feed_height = loaded_dict_enc['height']

    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

    encoder.load_state_dict(filtered_dict_enc)

    encoder.to(device)

    encoder.eval()

    #print("   Loading pretrained decoder")

    depth_decoder = DepthDecoder(

        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)

    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)

    depth_decoder.eval()

    print("-> Predicting on {:d} test image")

    # PREDICTING ON EACH IMAGE IN TURN

    with torch.no_grad():

        input_image = pil.open(image_path).convert('RGB')

        original_width, original_height = input_image.size

        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION

        input_image = input_image.to(device)

        features = encoder(input_image)

        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]

        disp_resized = torch.nn.functional.interpolate(

            disp, (original_height, original_width), mode="bilinear", align_corners=True)

        disp_resized_np = disp_resized.squeeze().cpu().numpy()

        vmax = np.percentile(disp_resized_np, 95)

        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)

        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

        im = pil.fromarray(colormapped_im,mode=None)

        input_image=mpl.image.imread('/kaggle/input/safranbolu-sokak/safranbolu_sokak.jpg')

        fig = plt.figure()

        #fig.set_figheight(15)

        fig.set_figwidth(23)

        a = fig.add_subplot(1, 2, 1)

        imgplot = plt.imshow(input_image)

        a.set_title('Original')

        a = fig.add_subplot(1, 2, 2)

        imgplot = plt.imshow(im)

        a.set_title('Depth Estimated')

    print('-> Done!')
if __name__ == '__main__':

    estimate_depth()