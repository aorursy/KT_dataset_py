#Source: https://github.com/fastai/fastai/tree/master/courses/dl1
#Reproduced the notebook from Jeremy Howard's amazing Deep Learning course
!ls ../input
#Import the necessary fastai libraries
from fastai.conv_learner import *
from fastai.plots import *
#Points to our data directory where the training data and labels are located
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
from sklearn.metrics import fbeta_score
def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg",f"{PATH}train-jpg/train_12.jpg"]
titles=["haze primary", "agriculture clear primary water", "cloudy"]
plots_from_files(list_paths, titles=titles, maintitle="Multi-label classification")
#Our pretrained neural net architecture
f_model = resnet34

#The performance measure we will use to evaluate our model
metrics=[f2]
#Point to the file that contains our labels (images -> [list of labels])
label_csv = f'{PATH}train_v2.csv'

#Determine how many images are in the training set (first row is column header)
n = len(list(open(label_csv)))-1

#Identify which images will be part of the validation set (80% train, 20% val)
val_idxs = get_cv_idxs(n)
#Helper method to 
def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs)
#This will control how images are scaled before the are passed into the network
sz=64
#Load our data
data = get_data(sz)
#Create a pretrained model from the architecture we specified
learn = ConvLearner.pretrained(f_model, data, metrics=metrics, ps=0.5, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
#Define the learning rate
lr = 0.2
#Train for 5 epochs
learn.fit(lr, 5, cycle_len=1)
#Define learning rated for different parts of our model
lrs = np.array([lr/9,lr/3,lr])
#Important to turn precompute off so we can now take advantage of data augmentation
learn.precompute=False

#Allow the pretrained network to start learning (weights get updated)
learn.unfreeze()

#Train for 3 epochs (first cycle is of lenght 1 epoch, second cycle is of length 2 epochs)
learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)
#Test using the validation set and used Test Time Augmentation
multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)
#Make final prediction
f2(preds,y)
