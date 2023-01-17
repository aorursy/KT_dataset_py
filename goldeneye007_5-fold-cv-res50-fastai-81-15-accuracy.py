import librosa
import librosa.display
from fastai import *
from fastai.vision import *
import os
import shutil
import matplotlib.pyplot as plt
raw_data_path = Path("../input/esc50-github")
df = pd.read_csv(raw_data_path/'meta/esc50.csv')
fold1 = [f for f in df.loc[df['fold'] == 1].filename]
fold2 = [f for f in df.loc[df['fold'] == 2].filename]
fold3 = [f for f in df.loc[df['fold'] == 3].filename]
fold4 = [f for f in df.loc[df['fold'] == 4].filename]
fold5 = [f for f in df.loc[df['fold'] == 5].filename]
path=Path('/kaggle/working')
os.mkdir(path/'1')
os.mkdir(path/'2')
os.mkdir(path/'3')
os.mkdir(path/'4')
os.mkdir(path/'5')
for file in fold1:
    shutil.copy(raw_data_path/'audio'/file,path/'1')
for file in fold2:
    shutil.copy(raw_data_path/'audio'/file,path/'2')
for file in fold3:
    shutil.copy(raw_data_path/'audio'/file,path/'3')
for file in fold4:
    shutil.copy(raw_data_path/'audio'/file,path/'4')
for file in fold5:
    shutil.copy(raw_data_path/'audio'/file,path/'5')
labels_dict = dict(zip(df['target'],df['category']))
print(labels_dict)
os.mkdir(path/'spec1')
os.mkdir(path/'spec2')
os.mkdir(path/'spec3')
os.mkdir(path/'spec4')
os.mkdir(path/'spec5')
for i in range(1,6):
    os.mkdir(path/Path('spec'+str(i))/'train')
    os.mkdir(path/Path('spec'+str(i))/'valid')
    for label in labels_dict.values():       
        os.mkdir(path/Path('spec'+str(i))/'train'/label)
        os.mkdir(path/Path('spec'+str(i))/'valid'/label)
for i in range(1,6):
    for file in os.listdir(path/str(i)):
        label=labels_dict[int(file[:-4].split("-")[-1])]                
        fig = plt.figure(figsize=[0.96,0.96])
        ax = fig.add_subplot(111)        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        samples, sample_rate = librosa.load(path/str(i)/file,sr=None)
        sg = librosa.feature.melspectrogram(y=samples,sr=44100,n_fft=2048,hop_length=739,n_mels=299,fmin=0.0)
        db_spec = librosa.amplitude_to_db(sg, ref=1.0, amin=1e-05, top_db=80.0)
        librosa.display.specshow(db_spec, fmax=8000)
        file = os.path.join(path,'spec'+str(i),'valid',label,file[:-4]+'.png')
        plt.savefig(file,dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')
for i in range(1,6):
    source_folds = [j for j in range(1,6) if i != j]
    for fold in source_folds:
        for label in labels_dict.values():
            for file in os.listdir(path/Path('spec'+str(fold))/'valid'/label):
                shutil.copy(path/Path('spec'+str(fold))/'valid'/label/file,path/Path('spec'+str(i))/'train'/label)
train = path/'spec1'
valid = path/'spec1'
data = ImageDataBunch.from_folder(path/'spec1',ds_tfms=None,num_workers=1)
data.show_batch(rows=6,figsize=(12,12))
def valid_fold(fold):
    train = path/Path('spec'+str(fold))
    valid = path/Path('spec'+str(fold))
    data = ImageDataBunch.from_folder(path/Path('spec'+str(fold)),ds_tfms=None,num_workers=1)
    learn = cnn_learner(data, models.resnet50, metrics=error_rate).mixup()
    learn.fit_one_cycle(2,2e-03)
    learn.unfreeze()
    learn.fit_one_cycle(35,1e-03)
    learn.recorder.plot_losses()
valid_fold(1)
valid_fold(2)
valid_fold(3)
valid_fold(4)
valid_fold(5)
1-((0.21+0.1625+0.18+0.175+0.215)/5)