%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

import torchvision.models as mdls
bs=8

# model = models.densenet201
path = Path('')

path_train = '../input/aptos2019-blindness-detection/train_images'

path_test = '../input/aptos2019-blindness-detection/test_images'
train = pd.read_csv(path/'../input/aptos2019-blindness-detection/train.csv')[:100]

test = pd.read_csv(path/'../input/aptos2019-blindness-detection/sample_submission.csv')

# train.id_code = train.id_code+'.png'
test2 = test.copy()

test2['id_code'] = np.arange(0,test.shape[0],dtype=np.int).astype('str')
tfms = get_transforms(flip_vert=True,max_rotate = 10,max_warp = 0,max_zoom =1.05,max_lighting = 0)
tfms[0]
tfms[1].append(tfms[0][1])

tfms[1].append(tfms[0][2])

# tfms[1].append(tfms[0][5])


tfms[1]
import cv2

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
IMG_SIZE = 400

# use_sigmax = True

def load_ben_color(path, sigmaX=0 ):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    if sigmaX!=0:

        image=cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
# df_train=train.copy()
# from tqdm import tqdm

# from PIL import Image as IM

# imlist = None

# gc.collect()

# imlist = {}



# for i in tqdm(test['id_code']):

#     path_=f"../input/aptos2019-blindness-detection/test_images/{i}.png"

#     image = load_ben_color(path_,sigmaX=30)

# #     print(image)

#     im = IM.fromarray(image)

# #     cv2.imwrite('test/'+i+'.png',image)

# #     print(im)

# #     im.save('test/'+i+'.png')

# #     imlist[i+'jpg'] = image/255

  

# #     imlist.append(image/255)
gc.collect()

# tl = os.listdir('test')
# IM.open('test/'+tl[0])
# iim = IM.fromarray(load_ben_color("../input/aptos2019-blindness-detection/test_images/"+tl[2],sigmaX=30))
# test.head()

def histogram_normalization(image):    

    hist,bins = np.histogram(image.flatten(),256,[0,256])    

    cdf = hist.cumsum()   

    # cdf_normalized = cdf * hist.max()/ cdf.max()    

    cdf_m = np.ma.masked_equal(cdf,0)    

    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())    

    cdf = np.ma.filled(cdf_m,0).astype('uint8')     

    img2 = cdf[image]    

    return img2



def adap_hist_eql(image):

    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    

    return final

def test_img(image):

    im_sz = 1024

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (im_sz, im_sz))

    return image

IMG_SIZE = 400

Test = True





def read_image(path):

#     image = cv2.imread(path)

    image = load_ben_color(path,20)

#     image = resize_image(image)

#    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

#     image=histogram_normalization(image)

    

#     image=cv2.addWeighted (image ,4, cv2.GaussianBlur( image , (0,0), 25) ,-4 ,128)

    

    return image
IMG_SIZE = 400

Test = True





def read_image2(path,k):

#   

    image = None

#     k=2

    if k == 1:

        image =  load_ben_color(path,20)

        

    if k == 2:

        image =  load_ben_color(path,15)

        

    if k == 3:

        image = cv2.imread(path)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image = adap_hist_eql(image)



    if k==4:

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

#     image=histogram_normalization(image)

    

#     image=cv2.addWeighted (image ,4, cv2.GaussianBlur( image , (0,0), 25) ,-4 ,128)

#     print(image)

#     print(k)

    return image
image_dic = {}
from tqdm import tqdm

def load_image_on_ram(p,names,k):

    for i in tqdm(names):

        path= p+i+'.png'

        image = read_image2(path,k)

        image_dic[i] = image
path1 = '../input/aptos2019-blindness-detection/train_images/'

path2 = '../input/aptos2019-blindness-detection/test_images/'
# del image_dic

# gc.collect()
load_image_on_ram(path1,train.id_code.unique(),1)
image_type = lambda x : load_image_on_ram(path2,test.id_code.unique(),x)
# load_image_on_ram(path2,test.id_code.unique(),2)
image_type(2)
def im_b(path=""):

    img = path.split('/')[-1]

    image = image_dic[img]/255

    

    return image

    
def im_n(path=""):

    

#     image = load_ben_color(path+'.png')

    image = cv2.imread(path+'.png')

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255

    

    return image
def im_n2(path=""):

    

    image = load_ben_color(path+'.png',15)/255

    

#     image = read_image2(path+'.png',1)/255

    

    return image
# im_load = im_n2


class MyImageItemList(ImageList):

    

    def open(self,path:PathOrStr)->Image:

#         print(fn)

#         image = load_ben_color(fn,20)/255

        image = im_load(path)

#         img = path.split('/')[-1]

#         image = image_dic[img]/255

#         img = imlist[fn.split('/')[-1]]

        

        xx = vision.Image(px=pil2tensor(image,np.float32))

        return xx
im_load = im_b
# IMG_SIZE = 512

# def _load_format(path, convert_mode, after_open)->Image:

#     sigmax = 20

#     image = cv2.imread(path)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     image = crop_image_from_gray(image)

#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#     image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0), sigmax) ,-4 ,128)

                    

#     return Image(pil2tensor(image, np.float32).div_(255)) #return fastai Image format



# vision.data.open_image = _load_format
def get_data(sz=256):

    data = (MyImageItemList.from_df(df = train,path = '',folder = path1)

           .split_by_rand_pct(.15)

           .label_from_df(cols = 'diagnosis'

#                           ,label_cls=FloatList

                         )

           .add_test(MyImageItemList.from_df(df = test,path='',folder =path2)) 

           .transform(tfms,size=sz,resize_method = ResizeMethod.SQUISH, padding_mode = 'zeros')

           .databunch(bs = 8,num_workers = 4))

    data.normalize(imagenet_stats)

    

    return data

data = get_data(400)
# im_load = im_b
# image_type(1)
# use_sigmax=True

data.show_batch(rows=3,figsize=(8,8),ds_type = DatasetType.Test)
def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, scale:float=1.35) -> Iterator[List[Tensor]]:

    "Computes the outputs for several augmented inputs for TTA"

    dl = learn.dl(ds_type)

    ds = dl.dataset

    old = ds.tfms

#     activ = ifnone(activ, _loss_func2activ(learn.loss_func))

    active = None

    augm_tfm = [o for o in learn.data.train_ds.tfms if o.tfm not in

               (crop_pad, flip_lr, dihedral, zoom)]

    try:

        pbar = master_bar(range(4))

        for i in pbar:

            row = 1 if i&1 else 0

            col = 1 if i&2 else 0

            flip = i&4

            d = {'row_pct':row, 'col_pct':col, 'is_random':False}

            tfm = [*augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]

            if flip: tfm.append(flip_lr(p=1.))

            ds.tfms = tfm

            yield get_preds(learn.model, dl, pbar=pbar, activ=activ)[0]

    finally: ds.tfms = old



Learner.tta_only = _tta_only



# data = (ImageList.from_df(df = train,path = path,folder = path_train,suffix = '.png')

#        .split_by_rand_pct(.15)

#        .label_from_df(cols = 'diagnosis',label_cls=FloatList)

#        .add_test(ImageList.from_df(df = test,path=path,folder = 'test',suffix = '.png')) 

#        .transform(tfms,size=320,resize_method = ResizeMethod.SQUISH,padding_mode = 'zeros')

#        .databunch(bs = bs,num_workers = 4))

# data.normalize(imagenet_stats)
#I could not figure out how to install package in local kernel so i just stole from github =)

#code stolen from https://github.com/lukemelas/EfficientNet-PyTorch





"""

This file contains helper functions for building the model and for loading model parameters.

These helper functions are built to mirror those in the official TensorFlow implementation.

"""







# Parameters for the entire model (stem, all blocks, and head)

GlobalParams = collections.namedtuple('GlobalParams', [

    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',

    'num_classes', 'width_coefficient', 'depth_coefficient',

    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])





# Parameters for an individual model block

BlockArgs = collections.namedtuple('BlockArgs', [

    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',

    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])





# Change namedtuple defaults

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)





def relu_fn(x):

    """ Swish activation function """

    return x * torch.sigmoid(x)





def round_filters(filters, global_params):

    """ Calculate and round number of filters based on depth multiplier. """

    multiplier = global_params.width_coefficient

    if not multiplier:

        return filters

    divisor = global_params.depth_divisor

    min_depth = global_params.min_depth

    filters *= multiplier

    min_depth = min_depth or divisor

    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%

        new_filters += divisor

    return int(new_filters)





def round_repeats(repeats, global_params):

    """ Round number of filters based on depth multiplier. """

    multiplier = global_params.depth_coefficient

    if not multiplier:

        return repeats

    return int(math.ceil(multiplier * repeats))





def drop_connect(inputs, p, training):

    """ Drop connect. """

    if not training: return inputs

    batch_size = inputs.shape[0]

    keep_prob = 1 - p

    random_tensor = keep_prob

    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)

    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor

    return output





def get_same_padding_conv2d(image_size=None):

    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.

        Static padding is necessary for ONNX exporting of models. """

    if image_size is None:

        return Conv2dDynamicSamePadding

    else:

        return partial(Conv2dStaticSamePadding, image_size=image_size)



class Conv2dDynamicSamePadding(nn.Conv2d):

    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2



    def forward(self, x):

        ih, iw = x.size()[-2:]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)





class Conv2dStaticSamePadding(nn.Conv2d):

    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2



        # Calculate padding based on image size and save it

        assert image_size is not None

        ih, iw = image_size if type(image_size) == list else [image_size, image_size]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        else:

            self.static_padding = Identity()



    def forward(self, x):

        x = self.static_padding(x)

        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x





class Identity(nn.Module):

    def __init__(self,):

        super(Identity, self).__init__()



    def forward(self, input):

        return input





########################################################################

############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############

########################################################################





def efficientnet_params(model_name):

    """ Map EfficientNet model name to parameter coefficients. """

    params_dict = {

        # Coefficients:   width,depth,res,dropout

        'efficientnet-b0': (1.0, 1.0, 224, 0.2),

        'efficientnet-b1': (1.0, 1.1, 240, 0.2),

        'efficientnet-b2': (1.1, 1.2, 260, 0.3),

        'efficientnet-b3': (1.2, 1.4, 300, 0.3),

        'efficientnet-b4': (1.4, 1.8, 380, 0.4),

        'efficientnet-b5': (1.6, 2.2, 456, 0.4),

        'efficientnet-b6': (1.8, 2.6, 528, 0.5),

        'efficientnet-b7': (2.0, 3.1, 600, 0.5),

    }

    return params_dict[model_name]





class BlockDecoder(object):

    """ Block Decoder for readability, straight from the official TensorFlow repository """



    @staticmethod

    def _decode_block_string(block_string):

        """ Gets a block through a string notation of arguments. """

        assert isinstance(block_string, str)



        ops = block_string.split('_')

        options = {}

        for op in ops:

            splits = re.split(r'(\d.*)', op)

            if len(splits) >= 2:

                key, value = splits[:2]

                options[key] = value



        # Check stride

        assert (('s' in options and len(options['s']) == 1) or

                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))



        return BlockArgs(

            kernel_size=int(options['k']),

            num_repeat=int(options['r']),

            input_filters=int(options['i']),

            output_filters=int(options['o']),

            expand_ratio=int(options['e']),

            id_skip=('noskip' not in block_string),

            se_ratio=float(options['se']) if 'se' in options else None,

            stride=[int(options['s'][0])])



    @staticmethod

    def _encode_block_string(block):

        """Encodes a block to a string."""

        args = [

            'r%d' % block.num_repeat,

            'k%d' % block.kernel_size,

            's%d%d' % (block.strides[0], block.strides[1]),

            'e%s' % block.expand_ratio,

            'i%d' % block.input_filters,

            'o%d' % block.output_filters

        ]

        if 0 < block.se_ratio <= 1:

            args.append('se%s' % block.se_ratio)

        if block.id_skip is False:

            args.append('noskip')

        return '_'.join(args)



    @staticmethod

    def decode(string_list):

        """

        Decodes a list of string notations to specify blocks inside the network.



        :param string_list: a list of strings, each string is a notation of block

        :return: a list of BlockArgs namedtuples of block args

        """

        assert isinstance(string_list, list)

        blocks_args = []

        for block_string in string_list:

            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        return blocks_args



    @staticmethod

    def encode(blocks_args):

        """

        Encodes a list of BlockArgs to a list of strings.



        :param blocks_args: a list of BlockArgs namedtuples of block args

        :return: a list of strings, each string is a notation of block

        """

        block_strings = []

        for block in blocks_args:

            block_strings.append(BlockDecoder._encode_block_string(block))

        return block_strings





def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,

                 drop_connect_rate=0.2, image_size=None, num_classes=1000):

    """ Creates a efficientnet model. """



    blocks_args = [

        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',

        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',

        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',

        'r1_k3_s11_e6_i192_o320_se0.25',

    ]

    blocks_args = BlockDecoder.decode(blocks_args)



    global_params = GlobalParams(

        batch_norm_momentum=0.99,

        batch_norm_epsilon=1e-3,

        dropout_rate=dropout_rate,

        drop_connect_rate=drop_connect_rate,

        # data_format='channels_last',  # removed, this is always true in PyTorch

        num_classes=num_classes,

        width_coefficient=width_coefficient,

        depth_coefficient=depth_coefficient,

        depth_divisor=8,

        min_depth=None,

        image_size=image_size,

    )



    return blocks_args, global_params





def get_model_params(model_name, override_params):

    """ Get the block args and global params for a given model """

    if model_name.startswith('efficientnet'):

        w, d, s, p = efficientnet_params(model_name)

        # note: all models have drop connect rate = 0.2

        blocks_args, global_params = efficientnet(

            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)

    else:

        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:

        # ValueError will be raised here if override_params has fields not included in global_params.

        global_params = global_params._replace(**override_params)

    return blocks_args, global_params





url_map = {

    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',

    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',

    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',

    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',

    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',

    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth',

}



def load_pretrained_weights(model, model_name, load_fc=True):

    """ Loads pretrained weights, and downloads if loading for the first time. """

    state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:

        model.load_state_dict(state_dict)

    else:

        state_dict.pop('_fc.weight')

        state_dict.pop('_fc.bias')

        res = model.load_state_dict(state_dict, strict=False)

        assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'

    print('Loaded pretrained weights for {}'.format(model_name))

    

    

class MBConvBlock(nn.Module):

    """

    Mobile Inverted Residual Bottleneck Block



    Args:

        block_args (namedtuple): BlockArgs, see above

        global_params (namedtuple): GlobalParam, see above



    Attributes:

        has_se (bool): Whether the block contains a Squeeze and Excitation layer.

    """



    def __init__(self, block_args, global_params):

        super().__init__()

        self._block_args = block_args

        self._bn_mom = 1 - global_params.batch_norm_momentum

        self._bn_eps = global_params.batch_norm_epsilon

        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)

        self.id_skip = block_args.id_skip  # skip connection and drop connect



        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)



        # Expansion phase

        inp = self._block_args.input_filters  # number of input channels

        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels

        if self._block_args.expand_ratio != 1:

            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)

            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)



        # Depthwise convolution phase

        k = self._block_args.kernel_size

        s = self._block_args.stride

        self._depthwise_conv = Conv2d(

            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise

            kernel_size=k, stride=s, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)



        # Squeeze and Excitation layer, if desired

        if self.has_se:

            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))

            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)

            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)



        # Output phase

        final_oup = self._block_args.output_filters

        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)

        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)



    def forward(self, inputs, drop_connect_rate=None):

        """

        :param inputs: input tensor

        :param drop_connect_rate: drop connect rate (float, between 0 and 1)

        :return: output of block

        """



        # Expansion and Depthwise Convolution

        x = inputs

        if self._block_args.expand_ratio != 1:

            x = relu_fn(self._bn0(self._expand_conv(inputs)))

        x = relu_fn(self._bn1(self._depthwise_conv(x)))



        # Squeeze and Excitation

        if self.has_se:

            x_squeezed = F.adaptive_avg_pool2d(x, 1)

            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))

            x = torch.sigmoid(x_squeezed) * x



        x = self._bn2(self._project_conv(x))



        # Skip connection and drop connect

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters

        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:

            if drop_connect_rate:

                x = drop_connect(x, p=drop_connect_rate, training=self.training)

            x = x + inputs  # skip connection

        return x





class EfficientNet(nn.Module):

    """

    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods



    Args:

        blocks_args (list): A list of BlockArgs to construct blocks

        global_params (namedtuple): A set of GlobalParams shared between blocks



    Example:

        model = EfficientNet.from_pretrained('efficientnet-b0')



    """



    def __init__(self, blocks_args=None, global_params=None):

        super().__init__()

        assert isinstance(blocks_args, list), 'blocks_args should be a list'

        assert len(blocks_args) > 0, 'block args must be greater than 0'

        self._global_params = global_params

        self._blocks_args = blocks_args



        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)



        # Batch norm parameters

        bn_mom = 1 - self._global_params.batch_norm_momentum

        bn_eps = self._global_params.batch_norm_epsilon



        # Stem

        in_channels = 3  # rgb

        out_channels = round_filters(32, self._global_params)  # number of output channels

        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)



        # Build blocks

        self._blocks = nn.ModuleList([])

        for block_args in self._blocks_args:



            # Update block input and output filters based on depth multiplier.

            block_args = block_args._replace(

                input_filters=round_filters(block_args.input_filters, self._global_params),

                output_filters=round_filters(block_args.output_filters, self._global_params),

                num_repeat=round_repeats(block_args.num_repeat, self._global_params)

            )



            # The first block needs to take care of stride and filter size increase.

            self._blocks.append(MBConvBlock(block_args, self._global_params))

            if block_args.num_repeat > 1:

                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            for _ in range(block_args.num_repeat - 1):

                self._blocks.append(MBConvBlock(block_args, self._global_params))



        # Head

        in_channels = block_args.output_filters  # output of final block

        out_channels = round_filters(1280, self._global_params)

        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)



        # Final linear layer

        self._dropout = self._global_params.dropout_rate

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)



    def extract_features(self, inputs):

        """ Returns output of the final convolution layer """



        # Stem

        x = relu_fn(self._bn0(self._conv_stem(inputs)))



        # Blocks

        for idx, block in enumerate(self._blocks):

            drop_connect_rate = self._global_params.drop_connect_rate

            if drop_connect_rate:

                drop_connect_rate *= float(idx) / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)



        # Head

        x = relu_fn(self._bn1(self._conv_head(x)))



        return x



    def forward(self, inputs):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """



        # Convolution layers

        x = self.extract_features(inputs)



        # Pooling and final linear layer

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        if self._dropout:

            x = F.dropout(x, p=self._dropout, training=self.training)

        x = self._fc(x)

        return x



    @classmethod

    def from_name(cls, model_name, override_params=None):

        cls._check_model_name_is_valid(model_name)

        blocks_args, global_params = get_model_params(model_name, override_params)

        return EfficientNet(blocks_args, global_params)



    @classmethod

    def from_pretrained(cls, model_name, num_classes=1000):

        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})

        return model



    @classmethod

    def get_image_size(cls, model_name):

        cls._check_model_name_is_valid(model_name)

        _, _, res, _ = efficientnet_params(model_name)

        return res



    @classmethod

    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):

        """ Validates model name. None that pretrained weights are only available for

        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """

        num_models = 4 if also_need_pretrained_weights else 8

        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]

        if model_name.replace('-','_') not in valid_models:

            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
md_ef = EfficientNet.from_pretrained('efficientnet-b5', num_classes=data.c)
# md_ef
!ls ../input
!mkdir models
!cp '../input/ef5-ahe/ef5_ah_400_1.pth' models/



# !cp '../input/ef5-256/ef5_256_1.pth' models/

!cp '../input/ef5-400/ef5_400_1.pth' models/



!cp '../input/ef5-b15/ef5_b15_400_1.pth' models/

!cp '../input/ef5-b15/ef5_b15_320_1.pth' models/



!cp '../input/ef5-b-w-400/ef5_b_400_2.pth' models/

!cp '../input/ef4-b-ls-400/ef5_b_400_1.pth' models/

# !cp '../input/ef5-b-ls-320/ef5_b_320_1.pth' models/

# !cp '../input/b-3000/effi_b_30000_1.pth' models/

# !cp '../input/ef5-b-reg-256/ef5_b_256_1.pth' models/

# !cp '../input/ben_models/effi_5_ben_3.pth' models/
!ls models
# model = torch.load('models/m1.pth')
from sklearn.metrics import cohen_kappa_score

def qk(y_pred, y):

  return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')
# cust_head = create_head(nf = 4096,nc = data.c , lin_ftrs=[2048,1024,512])

learn = Learner(data,md_ef

#                 ,metrics=[qk]

               )

# learn.to_fp16=True

# learn.loss = nn.L1Loss
# learn.load('ef5_b_320_1');
# xx = torch.load('models/m1.pth')
# learn.model.state_dict = xx


# learn.model.load_state_dict(xx)



# learn.unfreeze()

# learn.to_fp16 = True
# learn.save('aptos2')
# learn.lr_find()

# learn.recorder.plot()
!nvidia-smi
# learn.data.batch_size = 16
# learn.to_fp16=True
# learn.fit_one_cycle(4,max_lr = 1e-4)
# learn.fit_one_cycle(3,max_lr = 5e-5)
# learn.save('effi_5_ben_4')
# learn.fit_one_cycle(3,max_lr = 1e-5)
# learn.save('effi_5_ben_5')
# learn.load('effi_5_ben_5')
# test_df = test

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

        print(-loss_partial(self.coef_['x']))



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']

def run_subm(learn=learn, coefficients=[0.5, 1.5, 2.5, 3.5]):

    opt = OptimizedRounder()

    preds,y = learn.TTA(scale=1,ds_type=DatasetType.Test)

    tst_pred = opt.predict(preds, coefficients)

    test_df.diagnosis = tst_pred.astype(int)

    test_df.to_csv('submission.csv',index=False)

    return test_df

#     print ('done')
# test_df = run_subm()
# test_df['diagnosis'].value_counts()
# %%time

# test_preds = np.zeros((test.shape[0],1))





# preds = learn.get_preds(ds_type =DatasetType.Test)



# for i in range(5):

#     print(i)

#     preds_ = learn.get_preds(ds_type =DatasetType.Test)

#     preds[0] += preds_[0]
# !nvidia-smi
# im_load = im_b
image_type(1)
learn.data = get_data(400)
learn.load('ef5_b_400_1');
%%time

preds1 = learn.TTA(beta = .2, scale=1,ds_type=DatasetType.Test)
v1 = torch.tensor([1.0,1.0,.7,.7,1.0])
# preds1[0]
# preds1[0]*v1
learn.load('ef5_b_400_2');
gc.collect()
%%time

preds2 = learn.TTA(beta = .2, scale=1,ds_type=DatasetType.Test)
v2 = torch.tensor([1.0,.7,1.0,.7,1.0])
image_type(2)
learn.data = get_data(400)
gc.collect()
learn.load('ef5_b15_400_1');

gc.collect()
%%time

preds3 = learn.TTA(beta = .2, scale=1,ds_type=DatasetType.Test)
v3 = torch.tensor([1.0,1.0,.5,1.0,1.0])
learn.data = get_data(320)

# im_load = im_n
learn.load('ef5_b15_320_1');
gc.collect()
%%time

preds4 = learn.TTA(beta = .2, scale=1,ds_type=DatasetType.Test)
v4 = torch.tensor([1.0,1.0,.5,1.0,1.0])
image_type(3)

learn.data = get_data(400)
learn.load('ef5_ah_400_1')

gc.collect()
%%time

preds5 = learn.TTA(beta = .2, scale=1,ds_type=DatasetType.Test)
v5 = torch.tensor([1.0,.7,.7,1.0,.7])
image_type(4)
gc.collect()
learn.load('ef5_400_1')

gc.collect()
%%time

preds6 = learn.TTA(beta = .2, scale=1,ds_type=DatasetType.Test)
# preds = preds3[0]+preds4[0]

preds = preds1[0]*v1+preds2[0]*v2+preds3[0]*v3 + preds4[0]*v4 + preds5[0]*v5+preds6[0]
test_preds = torch.argmax(preds,dim=1)
# preds[0] = preds[0]/6
# preds[0]
# coef = [0.5, 1.5, 2.5, 3.5]



# for i, pred in enumerate(preds[0]):

#     if pred < coef[0]:

#         test_preds[i] = 0

#     elif pred >= coef[0] and pred < coef[1]:

#         test_preds[i] = 1

#     elif pred >= coef[1] and pred < coef[2]:

#         test_preds[i] = 2

#     elif pred >= coef[2] and pred < coef[3]:

#         test_preds[i] = 3

#     else:

#         test_preds[i] = 4

test_preds
test['diagnosis'] = np.array(test_preds,dtype = np.int)

# .astype(int)



test.to_csv('submission.csv',index = False)
test['diagnosis'].value_counts()