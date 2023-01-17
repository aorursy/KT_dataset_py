!cp -r ../input/efficientnetpytorch/ ./efficientnetpytorch

!pip install ./efficientnetpytorch/

!rm -r ./efficientnetpytorch/
%config Completer.use_jedi = False



import os

import json

import functools



import torch

import torchvision

from torch import nn

from torchvision import transforms

from efficientnet_pytorch import EfficientNet

import numpy as np

from tqdm import tqdm

import pandas as pd

from PIL import Image

import albumentations as A

import sklearn.metrics
mode = 'test'

NUM_GRAPHEME_ROOT = 168

NUM_VOWEL_DIACRITIC = 11

NUM_CONSONANT_DIACRITIC = 8

class_map = pd.read_csv('../input/bengaliai-cv19/class_map.csv')

grapheme_root = class_map[class_map['component_type'] == 'grapheme_root']

vowel_diacritic = class_map[class_map['component_type'] == 'vowel_diacritic']

consonant_diacritic = class_map[class_map['component_type'] == 'consonant_diacritic']

grapheme_root_list = grapheme_root['component'].tolist()

vowel_diacritic_list = vowel_diacritic['component'].tolist()

consonant_diacritic_list = consonant_diacritic['component'].tolist()
class BengalModel(nn.Module):

    def __init__(self, backbone, hidden_size=2560, class_num=168*11*7):

        super(BengalModel, self).__init__()

        self.backbone = backbone

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(hidden_size, class_num)

        self.ln = nn.LayerNorm(hidden_size)



        

    def forward(self, inputs):

        bs = inputs.shape[0]

        feature = self.backbone.extract_features(inputs)

        feature_vector = self._avg_pooling(feature)

        feature_vector = feature_vector.view(bs, -1)

        feature_vector = self.ln(feature_vector)



        out = self.fc(feature_vector)

        return out   
class Albumentations:

    def __init__(self, augmentations):

        self.augmentations = A.Compose(augmentations)

    

    def __call__(self, image):

        image = self.augmentations(image=image)['image']

        return image
class TestGraphemeDataset(torch.utils.data.Dataset):

    

    def __init__(self, images, transform=None):

        self.images = images

        self.transform = transform

            

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self, idx):

        np_image = self.images[idx].copy()

        out_image = self.transform(np_image)

        return out_image, idx

 
def label_to_grapheme(grapheme_root, vowel_diacritic, consonant_diacritic):

    if consonant_diacritic == 0:

        if vowel_diacritic == 0:

            return grapheme_root_list[grapheme_root]

        else:

            return grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic]

    elif consonant_diacritic == 1:

        if vowel_diacritic == 0:

            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]

        else:

            return grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic] + consonant_diacritic_list[consonant_diacritic]

    elif consonant_diacritic == 2:

        if vowel_diacritic == 0:

            return consonant_diacritic_list[consonant_diacritic] + grapheme_root_list[grapheme_root]

        else:

            return consonant_diacritic_list[consonant_diacritic] + grapheme_root_list[grapheme_root] + vowel_diacritic_list[vowel_diacritic]

    elif consonant_diacritic == 3:

        if vowel_diacritic == 0:

            return consonant_diacritic_list[consonant_diacritic][:2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic][1:]

        else:

            return consonant_diacritic_list[consonant_diacritic][:2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic][1:] + vowel_diacritic_list[vowel_diacritic]

    elif consonant_diacritic == 4:

        if vowel_diacritic == 0:

            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]

        else:

            if grapheme_root == 123 and vowel_diacritic == 1:

                return grapheme_root_list[grapheme_root] + '\u200d' + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]

            return grapheme_root_list[grapheme_root]  + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]

    elif consonant_diacritic == 5:

        if vowel_diacritic == 0:

            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]

        else:

            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]

    elif consonant_diacritic == 6:

        if vowel_diacritic == 0:

            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic]

        else:

            return grapheme_root_list[grapheme_root] + consonant_diacritic_list[consonant_diacritic] + vowel_diacritic_list[vowel_diacritic]

    elif consonant_diacritic == 7:

        if vowel_diacritic == 0:

            return consonant_diacritic_list[2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[2][::-1]

        else:

            return consonant_diacritic_list[2] + grapheme_root_list[grapheme_root] + consonant_diacritic_list[2][::-1] + vowel_diacritic_list[vowel_diacritic]
device = torch.device("cuda")

class ResnetGenerator(nn.Module):

    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)

    """



    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):

        """Construct a Resnet-based generator

        Parameters:

            input_nc (int)      -- the number of channels in input images

            output_nc (int)     -- the number of channels in output images

            ngf (int)           -- the number of filters in the last conv layer

            norm_layer          -- normalization layer

            use_dropout (bool)  -- if use dropout layers

            n_blocks (int)      -- the number of ResNet blocks

            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero

        """

        assert(n_blocks >= 0)

        super(ResnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:

            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:

            use_bias = norm_layer == nn.InstanceNorm2d



        model = [nn.ReflectionPad2d(3),

                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),

                 norm_layer(ngf),

                 nn.ReLU(True)]



        n_downsampling = 2

        for i in range(n_downsampling):  # add downsampling layers

            mult = 2 ** i

            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),

                      norm_layer(ngf * mult * 2),

                      nn.ReLU(True)]



        mult = 2 ** n_downsampling

        for i in range(n_blocks):       # add ResNet blocks



            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]



        for i in range(n_downsampling):  # add upsampling layers

            mult = 2 ** (n_downsampling - i)

            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),

                                         kernel_size=3, stride=2,

                                         padding=1, output_padding=1,

                                         bias=use_bias),

                      norm_layer(int(ngf * mult / 2)),

                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        model += [nn.Tanh()]



        self.model = nn.Sequential(*model)



    def forward(self, input):

        """Standard forward"""

        return self.model(input)





class ResnetBlock(nn.Module):

    """Define a Resnet block"""



    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        """Initialize the Resnet block

        A resnet block is a conv block with skip connections

        We construct a conv block with build_conv_block function,

        and implement skip connections in <forward> function.

        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf

        """

        super(ResnetBlock, self).__init__()

        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)



    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        """Construct a convolutional block.

        Parameters:

            dim (int)           -- the number of channels in the conv layer.

            padding_type (str)  -- the name of padding layer: reflect | replicate | zero

            norm_layer          -- normalization layer

            use_dropout (bool)  -- if use dropout layers.

            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        """

        conv_block = []

        p = 0

        if padding_type == 'reflect':

            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':

            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':

            p = 1

        else:

            raise NotImplementedError('padding [%s] is not implemented' % padding_type)



        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        if use_dropout:

            conv_block += [nn.Dropout(0.5)]



        p = 0

        if padding_type == 'reflect':

            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':

            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':

            p = 1

        else:

            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]



        return nn.Sequential(*conv_block)



    def forward(self, x):

        """Forward function (with skip connections)"""

        out = x + self.conv_block(x)  # add skip connections

        return out
class EnsembleModel(nn.Module):

    def __init__(self, models):

        super(EnsembleModel, self).__init__()

        self.model1 = models[0]

        self.model2 = models[1]

        

    def forward(self, images):



        return self.model1(images), self.model2(images)
def create_merc(classifier_load_path1, generator_load_path1, classifier_load_path2, generator_load_path2, images):

    MEAN = [0.5, 0.5, 0.5]

    STD = [0.5, 0.5, 0.5]

    IMG_HEIGHT = 224

    IMG_WIDTH = 224

    BATCH_SIZE = 32

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

    generator_b1 = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)

    backbone1 = EfficientNet.from_name('efficientnet-b0')

    classifier1 = BengalModel(backbone1, hidden_size=1280, class_num=NUM_GRAPHEME_ROOT*NUM_VOWEL_DIACRITIC*NUM_CONSONANT_DIACRITIC)



    classifier1.load_state_dict(torch.load(classifier_load_path1))

    generator_b1.load_state_dict(torch.load(generator_load_path1))

    model1 = nn.Sequential(generator_b1, classifier1)

    

    generator_b2 = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)

    backbone2 = EfficientNet.from_name('efficientnet-b0')

    classifier2 = BengalModel(backbone2, hidden_size=1280, class_num=NUM_GRAPHEME_ROOT*NUM_VOWEL_DIACRITIC*NUM_CONSONANT_DIACRITIC)



    classifier2.load_state_dict(torch.load(classifier_load_path2))

    generator_b2.load_state_dict(torch.load(generator_load_path2))

    model2 = nn.Sequential(generator_b2, classifier2)

    model = EnsembleModel([model1, model2])

    model.to(device)

    model.eval()

    

    grapheme_root_map = np.zeros((NUM_GRAPHEME_ROOT*NUM_VOWEL_DIACRITIC*NUM_CONSONANT_DIACRITIC, ), dtype=np.int64)

    vowel_diacritic_map = np.zeros((NUM_GRAPHEME_ROOT*NUM_VOWEL_DIACRITIC*NUM_CONSONANT_DIACRITIC, ), dtype=np.int64)

    consonant_diacritic_map = np.zeros((NUM_GRAPHEME_ROOT*NUM_VOWEL_DIACRITIC*NUM_CONSONANT_DIACRITIC, ), dtype=np.int64)

    for grapheme_root in range(168):

        for vowel_diacritic in range(11):

            for consonant_diacritic in range(8):

                i = (grapheme_root*NUM_VOWEL_DIACRITIC + vowel_diacritic)*NUM_CONSONANT_DIACRITIC + consonant_diacritic

                grapheme_root_map[i] = grapheme_root

                vowel_diacritic_map[i] = vowel_diacritic

                consonant_diacritic_map[i] = consonant_diacritic

    

    preprocess = [

        A.CenterCrop(height=137, width=IMG_WIDTH),

        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),

    ]



    transform = transforms.Compose([

        np.uint8,

        transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),

        np.uint8,

        Albumentations(preprocess),

        transforms.ToTensor(),

        transforms.Normalize(mean=MEAN, std=STD),

    #     transforms.ToPILImage(),

    ])

    

    dataset = TestGraphemeDataset(images, transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    def out2pred(out):

        out1, out2 = out

        softmax = nn.Softmax(dim=1)

        out1 = softmax(out1)

        out2 = softmax(out2)

        out = out1+out2

        

        box_out1 = out1.reshape((-1, NUM_GRAPHEME_ROOT, NUM_VOWEL_DIACRITIC, NUM_CONSONANT_DIACRITIC))

        grapheme_root_out = box_out1.sum(dim=(2, 3))

        vowel_diacritic_out = box_out1.sum(dim=(1, 3))

        consonant_diacritic_out = box_out1.sum(dim=(1, 2))

        box_out2 = out2.reshape((-1, NUM_GRAPHEME_ROOT, NUM_VOWEL_DIACRITIC, NUM_CONSONANT_DIACRITIC))

        grapheme_root_out += box_out2.sum(dim=(2, 3))

        vowel_diacritic_out += box_out2.sum(dim=(1, 3))

        consonant_diacritic_out += box_out2.sum(dim=(1, 2))

        grapheme_root_preds = grapheme_root_out.argmax(dim=1).cpu().numpy()

        vowel_diacritic_preds = vowel_diacritic_out.argmax(dim=1).cpu().numpy()

        consonant_diacritic_preds = consonant_diacritic_out.argmax(dim=1).cpu().numpy()

        preds = (grapheme_root_preds*NUM_VOWEL_DIACRITIC+vowel_diacritic_preds)*NUM_CONSONANT_DIACRITIC+consonant_diacritic_preds

#         confidences, preds = out.max(dim=1)



#         confidences = confidences.cpu().numpy()

#         preds = preds.cpu().numpy()

#         grapheme_root_preds = grapheme_root_map[preds]

#         vowel_diacritic_preds = vowel_diacritic_map[preds]

#         consonant_diacritic_preds = consonant_diacritic_map[preds]

        ret = []

        for p, g, v, co in zip(preds, grapheme_root_preds, vowel_diacritic_preds, consonant_diacritic_preds):

            x = {

                'pred': p,

#                 'confidence': c,

                'grapheme_root': g,

                'vowel_diacritic': v,

                'consonant_diacritic': co,

                'grapheme': label_to_grapheme(g, v, co),

            }

            ret.append(x)

        return ret

        

    

    return model, loader, out2pred

def evaluator(model, loader, out2pred):

    ret = []

    model.eval()

    softmax = nn.Softmax(dim=1)

    for (images, idx) in tqdm(loader):

        images = images.to(device)

        with torch.no_grad():

            out = model(images)

            ret += out2pred(out)

    return ret
def load_images(path):

    image_df = pd.read_parquet(path)

    images = image_df.iloc[:, 1:].values.reshape(-1, 137, 236)

    return images
merc_result = []

images = load_images('../input/bengaliai-cv19/{}_image_data_0.parquet'.format(mode))

model, loader, out2pred = create_merc('../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', '../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', images)

merc_result += evaluator(model, loader, out2pred)
images = load_images('../input/bengaliai-cv19/{}_image_data_1.parquet'.format(mode))

model, loader, out2pred = create_merc('../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', '../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', images)

merc_result += evaluator(model, loader, out2pred)
images = load_images('../input/bengaliai-cv19/{}_image_data_2.parquet'.format(mode))

model, loader, out2pred = create_merc('../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', '../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', images)

merc_result += evaluator(model, loader, out2pred)
images = load_images('../input/bengaliai-cv19/{}_image_data_3.parquet'.format(mode))

model, loader, out2pred = create_merc('../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', '../input/cyclegan-classifier-results/best.pth', '../input/cyclegan-training-results/generator.pth', images)

merc_result += evaluator(model, loader, out2pred)
merc_result_df = pd.DataFrame(merc_result)

def create_inference(merc_result_df):

    ret = []

    for merc_row in merc_result_df.itertuples():

        inference = {}

        inference['grapheme_root'] = merc_row.grapheme_root

        inference['vowel_diacritic'] = merc_row.vowel_diacritic

        inference['consonant_diacritic'] = merc_row.consonant_diacritic

        inference['grapheme'] = merc_row.grapheme

        ret.append(inference)

        

    return pd.DataFrame(ret)
inference = create_inference(merc_result_df)
def submit(inference):

    row_id_list = []

    target_list = []

    for i, row in inference.iterrows():

        row_id_list.append('Test_{}_grapheme_root'.format(i))

        target_list.append(row.grapheme_root)

        row_id_list.append('Test_{}_vowel_diacritic'.format(i))

        target_list.append(row.vowel_diacritic)

        row_id_list.append('Test_{}_consonant_diacritic'.format(i))

        if row.consonant_diacritic == 7:

            target_list.append(2)

        else:

            target_list.append(row.consonant_diacritic)

    raw_submission = {

        'row_id': row_id_list,

        'target': target_list

    }

    submission = pd.DataFrame(raw_submission)

    display(submission)

    submission.to_csv('submission.csv', index=False)
submit(inference)