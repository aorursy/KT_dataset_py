!pip install imagehash
import numpy as np
import pandas as pd
import os
import sys
import cv2
import glob
import fastai
import PIL
from functools import partial
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.basics import *
from fastai.vision import learner

from tqdm import tqdm
import imagehash
from skimage.metrics import structural_similarity
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

tqdm.pandas()

INPUT_PATH = "../input/deepfake-detection-challenge"
VERBOSE = False
EPS = 1e-5
RUN_NOTEBOOK=False

FACES_PATH = 'faces'
os.makedirs(FACES_PATH, exist_ok=True)
os.makedirs("mels", exist_ok=True)

from dfdc_face_extractor import *
from dfdc_fastai_reusables import *

#!pip install ffmpeg - does not puth ffmpeg on path for librosa

!tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz

#replace 20191209 with current version of wheel installed from ffmpeg-static-build
!mv ffmpeg-git-20191209-amd64-static/ffmpeg ffmpeg-git-20191209-amd64-static/ffprobe /usr/local/bin/

import librosa
import soundfile as sf
from skimage import io

def apply_mel_transforms(mels):
    "Normalizes and inverts mel spectrogram"
    mels = np.log(mels + EPS) 
    mels = mels - mels.min()
    img = mels*(255.0/(mels.max()-mels.min()))
    img = img.astype(np.uint8)
    img = np.flip(img, axis=0)
    return img

# https://librosa.github.io/librosa/auto_examples/plot_vocal_separation.html
def extract_voice(y, sr):
    "Extracts voice from background"
    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    return S_foreground


def extract_mels(video_path, basename):
    "Extracts mel spectrograms from video file"
    try:
        command = f"ffmpeg -i {video_path} -ab 192000 -ac 2 -ar 44100 -vn mels/{basename}.wav"
        subprocess.call(command, shell=True)
        audio_data, sample_rate = librosa.load( 'mels/' + basename + '.wav')
        #audio_data, sample_rate = librosa.load(video_path)
        voice_data = extract_voice(audio_data, sample_rate)
        hop_length = 512 # number of samples per time-step in spectrogram
        n_mels = 224 # number of bins in spectrogram. Height of image
        time_steps = 224 # number of time-steps. Width of image

        mel_data = librosa.feature.melspectrogram(S=voice_data, n_mels=n_mels,
                                        n_fft=hop_length*2, hop_length=hop_length)

        mel_image=apply_mel_transforms(mel_data)
        io.imsave("mels/%s.jpg" % (basename), mel_image, quality=100)
    except:
        print(f"Error reading audio from {video_path}")
if RUN_NOTEBOOK==True:
    pair_df = pd.read_csv(f'../input/fakereal-pairs-in-dfdc-test-videos/dfdc_test_video_pairs.csv')

    extractor = DFDCVideoFaceExtractor(backend='CV2')
    pair_df = pair_df[:20]
    for index, row in tqdm(pair_df.iterrows(), total=len(pair_df)):
        video_filename = row["filename"]
        basename, _ = basename_and_ext(video_filename)
        video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
        extractor.extract_faces(video_path, seq_length=10,stride=1, faces_path="faces", margin=1)
        extract_mels(video_path, basename)
        
        video_filename = row["original"]
        basename, _ = basename_and_ext(video_filename)
        video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
        extractor.extract_faces(video_path, seq_length=10,stride=1, faces_path="faces", margin=1)
        extract_mels(video_path, basename)
        

def concat(pils):
    assert (x == pils[0].width for x.width in pils)
    assert (x == pils[0].height for x.height in pils)
    l = len(pils)
    w = pils[0].width
    h = pils[0].height
    dst = PIL.Image.new('RGB', (l*w , h))
    left = 0
    for i in range(l):
        im = pils[i]
        dst.paste(im, (i*w, 0))
    return dst

def image_mse(imageA, imageB):
    "Computes mean squared error between two images"
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.size)
    return err

def adaptiveThreshold(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(image, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 3, 2)

def otsuThreshold(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image,(5,5),0)
    _, img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img


def blur(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = cv2.blur(image, (9,9))
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def gaussian(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = cv2.GaussianBlur(image, (9,9),0)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def apply_filter(rgbimg, func=None):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = func(image)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
def median(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV) # convert to HSV
    image = cv2.medianBlur(image, 9)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def adaptiveThreshold(rgbimg):
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(image, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 3, 2)
    

def sobelxy(rgbimg) :
    image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY) 
    sobel_x = cv2.Sobel(image.astype('float32'), cv2.CV_32F, dx = 1, dy = 0, ksize = 1)
    sobel_y = cv2.Sobel(image.astype('float32'), cv2.CV_32F, dx = 0, dy = 1, ksize = 1)
    blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y,
                          beta=0.5, gamma=0)
    return blended.astype('uint8')

def laplacian(rgbimg):
    #image = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY) 
    image = cv2.Laplacian(rgbimg.astype('float32'),cv2.CV_32F)
    return image.astype('uint8')
    
def dft1(c1):
    dft = cv2.dft(np.float32(c1),flags = cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # save image of the image in the fourier domain.
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

def dft(rgbimg):
    r = rgbimg[:,:,0]
    g = rgbimg[:,:,1]
    b = rgbimg[:,:,1]
    r,g,b=cv2.split(rgbimg)
    r=dft1(r)
    g=dft1(g)
    b=dft1(b)
    merged= cv2.merge((r,g,b))
    return merged.astype('uint8')

def show_pairs(real_faces, fake_faces, real_videoname, fake_videoname, func=None ):
    l = len(real_faces)
    fig,axes = plt.subplots(l,1,figsize=(600,20))
    count = 0
    for i in range(len(real_faces)):
        pil_im1 = real_faces[i]
        im1 = np.array(pil_im1)
        if func != None:
            im1=func(im1)
        h,w = im1.shape[0], im1.shape[1]
        pil_im2 = fake_faces[i]
        im2 = np.array(pil_im2)
        if func != None:
            im2=func(im2)
        mse = image_mse(im1, im2) 
        skssim, diff = structural_similarity(im1,im2,multichannel=True, full=True)
        diff = (diff * 255).astype('uint8')
        pil_im1 = PIL.Image.fromarray(im1)
        pil_im2 = PIL.Image.fromarray(im2)
        diff = PIL.Image.fromarray(diff)
        im3 = concat([pil_im1,pil_im2, diff])
        im1hash = imagehash.whash(pil_im1)
        im2hash = imagehash.whash(pil_im2)
        hash_diff = abs(im1hash-im2hash)
        title = 'real: %s / fake:%s  / mse: %.2f / skssim: %.2f / hash_diff: %d'% \
        (real_videoname, fake_videoname, mse, skssim,hash_diff)
        axes[count].set_title(title)
        axes[count].axis('off')
        axes[count].imshow(im3)
        count+=1
        
def compare_faces(rownum, func=None):
    pair_df = pd.read_csv(f'../input/fakereal-pairs-in-dfdc-test-videos/dfdc_test_video_pairs.csv')
    if rownum > len(pair_df):
        return
    extractor = DFDCVideoFaceExtractor(backend='CV2')
    row = pair_df.iloc[rownum]
    video_filename = row["filename"]
    fake_basename, _ = basename_and_ext(video_filename)
    video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
    fake_frames = extractor.extract_frames(video_path, seq_length=5,stride=1)

    video_filename = row["original"]
    real_basename, _ = basename_and_ext(video_filename)
    video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
    real_frames = extractor.extract_frames(video_path, seq_length=5,stride=1)

    realfaces, fakefaces, framenums = extractor.extract_face_pairs(real_frames, fake_frames, real_basename, margin=.5)
    
    show_pairs(realfaces, fakefaces, real_basename, fake_basename, func=func)
K = 20 
@interact_manual
def compare_faces_interactive():
    global K
    compare_faces(K, func=median)
    K += 1
if RUN_NOTEBOOK==True:
    data = get_deepfakeimagelist_data()
    data.show_batch(dstype=DatasetType.Valid)
if RUN_NOTEBOOK==True:
    # Resnet50 Pretrained

    model_dir = 'models/dfdc-resnet50'
    os.makedirs(model_dir, exist_ok=True)
    learn = cnn_learner(data,
                        models.resnet50,
                        bn_final=True,
                        loss_func=BCEWithLogitsFlat(),
                        pretrained=True,
                        metrics=[DFDCAUROC(),RealLoss(),FakeLoss()],
                        model_dir=model_dir,
                        concat_pool = True
                       )
    #.to_fp16(); requires GPU
    learn.fit(1)
    #learn.fit_one_cycle(3,max_lr=3e-4)
def logits(p):
    return log( p / (1-p))


def gsigmoid( t):
    'Gentle sigmoid function that spreads out the predictions'
    return (t / (1 + abs(t)) + 1.) / 2.

def get_fake_preds(testfaces):
    count = 0
    fakeness = 0.5
    preds = torch.tensor(())
    preds.new_empty((0,2),dtype=torch.float32)

    nf = len(testfaces)

    if(nf > 0):
        faces = pd.Series(testfaces)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", UserWarning)
          test_data = (DeepFakeImageList.from_pils(faces)
                 .split_none()
                 .label_from_array(np.arange(nf))
                 .transform(TFMS)
                 .databunch(bs=nf))\
                 .normalize(imagenet_stats)
        preds = get_preds_from_learner(learn, test_data, sigmoid=False)
        return preds

def get_preds_from_learner(lrnr, test_data, sigmoid=False):
    if (DEVICE != 'cpu'):
        lrnr.to_fp32()
    lrnr.data = test_data
    lrnr.data.valid_dl = lrnr.data.train_dl
    if (DEVICE != 'cpu'):
        lrnr.to_fp16()
    preds,y = lrnr.get_preds(ds_type=DatasetType.Valid)
    preds = preds[np.argsort(y)]
    preds = preds.detach().float().cpu()
    preds = preds.squeeze()
    # Depending on loss function used get_preds returns probability or logits
    if sigmoid == True:
        preds = torch.sigmoid(preds)
    preds = torch.clamp(preds, EPS, 1-EPS)
    return preds.numpy()

def aggregate(preds, framenums):
    fp = pd.DataFrame(columns=['n','label'])
    fp['n'] = framenums
    fp['label'] = preds

    # get fakest face from each frame
    frame_preds = fp.groupby(['n']).agg({'label': ['max']})
    fp_df = pd.DataFrame(frame_preds)
    fp_df.reset_index(inplace=True)
    fp_df.columns = ['n','label']
    fakeness = fp_df['label'].mean()
    
    # spread out the probabilities to eliminate over-confident predictions
    fakeness = gsigmoid(logits(fakeness))
    return fakeness

if RUN_NOTEBOOK==True:
    learn.model.eval();    
    test_df = pd.read_csv(f'{INPUT_PATH}/sample_submission.csv')
    test_df = test_df[:10]
    submission_df = pd.DataFrame(columns=['filename', 'label'])
    total = len(test_df)
    for index, row in tqdm(test_df.iterrows(), total=total):
        video_filename = row['filename']
        basename, _ = basename_and_ext(video_filename)
        video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
        faces, framenums = extract_faces_with_cv2(video_path, basename, seq_length=10,stride=1, output_path=None)
        if(len(faces) == 0):
            submission_df = submission_df.append({'filename': video_filename, 'label': 0.5}, ignore_index = True)
        else:
            preds = get_fake_preds(faces)
            label = aggregate(preds, framenums)
            submission_df = submission_df.append({'filename': video_filename, 'label': label}, ignore_index = True)
    submission_df.to_csv("submission.csv", index=False)
if RUN_NOTEBOOK==True:
    # GAN Critic
    from fastai.vision.gan import *
    def dfdc_critic(in_size:int, n_channels:int, n_features:int=16, n_extra_layers:int=0, **conv_kwargs):
        "Based on fastai basic_critic"
        layers = [conv_layer(n_channels, n_features, 5, 2, 1, leaky=0.2, **conv_kwargs)]#norm_type=None?
        cur_size, cur_ftrs = in_size//2, n_features
        layers.append(nn.Sequential(*[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=0.2, **conv_kwargs) for _ in range(n_extra_layers)]))
        while cur_size > 4:
            layers.append(conv_layer(cur_ftrs, cur_ftrs*2, 3, 2, 1, leaky=0.2, **conv_kwargs))
            cur_ftrs *= 2 ; cur_size //= 2
        layers += [conv_layer(cur_ftrs, 1, 4, padding=0,leaky=0.2, **conv_kwargs), Flatten()]
        return nn.Sequential(*layers)

    critic = dfdc_critic(224,3,64,1 )

    apply_init(critic, nn.init.kaiming_normal_)
    model_dir = 'models/dfdc-critic'
    os.makedirs(model_dir, exist_ok=True)

    critic_learn = Learner(data=data, model=critic, loss_func=FocalLoss(), 
                    model_dir=model_dir,
                    metrics=[DFDCAUROC(), RealLoss(),FakeLoss(),DFDCBceLoss()],
                    )# .to_fp16()

    critic_learn.fit(1)

if RUN_NOTEBOOK==True:
    # EfficientNet
    !pip install efficientnet_pytorch
    from efficientnet_pytorch import EfficientNet
    effnet = EfficientNet.from_pretrained('efficientnet-b5',num_classes=1)
    model_dir = 'models/dfdc-effnet'
    os.makedirs(model_dir, exist_ok=True)

    effnet_learn = Learner(data=data, model=effnet, loss_func=BCEWithLogitsFlat(), 
                    model_dir=model_dir,
                    metrics=[DFDCAUROC(), RealLoss(),FakeLoss()],
                    )# .to_fp16()
    effnet_learn.fit(1)
