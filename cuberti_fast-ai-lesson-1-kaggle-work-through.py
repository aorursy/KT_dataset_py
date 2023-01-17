# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
%%capture
!pip install fastai==0.7.0
!pip install torchtext==0.2.3
# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH)
fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in f else 1) for f in fnames])
img = plt.imread(f'{PATH}{fnames[3]}')
plt.imshow(img);
img.shape
img[:4,:4]
arch = resnet34 #Set model archatecture
#format data using FASTAI ImageClassifierData Class
data = ImageClassifierData.from_names_and_array(
    path = PATH,
    fnames = fnames, #Directory of all image file names
    y = labels, #labels taken from filenames in previous cell
    classes = ['dogs', 'cats'], #set labels
    test_name = 'test', #test directory
    tfms = tfms_from_model(arch, sz)
)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01, 2) #Learning Rate set to 0.01, and n_Epochs are 2
learn.fit(0.01, 2)
# Uncomment the below if you need to reset your precomputed activations
# shutil.rmtree(f'{PATH}tmp', ignore_errors=True)
arch=resnet34
data = ImageClassifierData.from_names_and_array(
    path=PATH, 
    fnames=fnames, 
    y=labels, 
    classes=['dogs', 'cats'], 
    test_name='test', 
    tfms=tfms_from_model(arch, sz)
)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01, 2)
data.classes
data.val_y
log_preds = learn.predict()
log_preds.shape
log_preds[:10] #take a look at the last ten log-predictions
#first column represents log-probability of dogs, second cats`
preds = np.argmax(log_preds, axis=1) #which column is higher? (dogs or cats)
probs = np.exp(log_preds[:,1])
def rand_by_mask(mask): 
    return np.random.choice(np.where(mask)[0], min(len(preds), 4), replace = False)

def rand_by_correct(is_correct):
    return rand_by_mask((preds== data.val_y)==is_correct)

def plots(ims, figsize = (12,6), rows = 1, titles = None):
    f= plt.figure(figsize = figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize = 16)
        plt.imshow(ims[i])

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_titles(idxs, title):
    imgs = [load_img_id(data.val_ds, x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles = title_probs, figsize = (16,8)) if len(imgs)>0 else print("Not Found")
plot_val_with_titles(rand_by_correct(True), "Correctly Classified")
plot_val_with_titles(rand_by_correct(False), "Incorrectly Classified")
def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct):
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)
plot_val_with_titles(most_by_correct(0, True), "Most Cat Like Cats")
plot_val_with_titles(most_by_correct(1, True), "Most Dog Like Dogs")
plot_val_with_titles(most_by_correct(0, False), "Least Cat Like Cats")
plot_val_with_titles(most_by_correct(1, False), "Least Dog Like Dogs")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_titles(most_uncertain, "Most uncertain preds")
learn = ConvLearner.pretrained(arch, data, precompute = True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
lrf = learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom = 1.1)
def get_augs():
    data = ImageClassifierData.from_names_and_array(
        path = PATH,
        fnames = fnames, #Directory of all image file names
        y = labels, #labels taken from filenames in previous cell
        classes = ['dogs', 'cats'], #set labels
        test_name = 'test', #test directory
        tfms = tfms,
        bs=2,
        num_workers=1
    )
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]
ims = np.stack([get_augs() for i in range(6)])
plots(ims, rows=2)
data = ImageClassifierData.from_names_and_array(
    path = PATH,
    fnames = fnames, #Directory of all image file names
    y = labels, #labels taken from filenames in previous cell
    classes = ['dogs', 'cats'], #set labels
    test_name = 'test', #test directory
    tfms = tfms,
) #This reformats the data with the trasforms
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name = TMP_PATH, models_name = MODEL_PATH)
learn.fit(1e-2, 1) #Re train teh model with single epoch
learn.precompute=False 
learn.fit(1e-2, 3, cycle_len=1) #takes some time to run
learn.sched.plot_lr()
learn.unfreeze()
lr = np.array([1e-4, 1e-3, 1e-2])
learn.fit(lr, 3, cycle_len = 1, cycle_mult = 2) #Re-fit model with differential learning rate. Some time is required to run the 6 epochs
learn.sched.plot_lr()
learn.save('224_all')
learn.load('224_all')
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)
plot_val_with_titles(most_by_correct(0, False), "Most incorrect cats")
plot_val_with_titles(most_by_correct(1, False), "Most incorrect dogs")
test_pred  = learn.predict(is_test=True)
test_pred
pred = (np.argmax(test_pred, axis =1))
submission = pd.DataFrame({'id': os.listdir(f'{PATH}test'), 'label': pred})
submission.to_csv('submission.csv')