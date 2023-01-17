import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from fastai.vision import *
import fastai
from fastai.metrics import *
from fastai import *
from os import *
import seaborn as sns
from sklearn.metrics import auc,roc_curve,accuracy_score, roc_auc_score
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
np.random.seed(42)
from glob import glob 
%matplotlib inline
model_path='.'
path='/kaggle/input/histopathologic-cancer-detection/'
train_folder=f'{path}train'
test_folder=f'{path}test'
train_lbl=f'{path}train_labels.csv'

bs=64
num_workers=None 
sz=96
# Programming framework behind the scenes of NVIDIA GPU is CUDA
print(torch.cuda.is_available())
# Check if gpu is enabled
print(torch.backends.cudnn.enabled)
df_train = pd.read_csv(train_lbl)
print(f'Number of labels {len(df_train)}')
# Proportion of classes 
df_train['label'].value_counts(normalize=True)
sns.countplot(x='label',data=df_train)
cancer_cell = df_train[df_train['label']==1].head()
cancer_cell
non_cancer_cell = df_train[df_train['label']==0].head()
non_cancer_cell
plt.subplot(1 , 2 , 1)
img = np.asarray(plt.imread(train_folder+'/'+cancer_cell.iloc[1][0]+'.tif'))
plt.title('METASTATIC CELL TISSUE')
plt.imshow(img)

plt.subplot(1 , 2 , 2)
img = np.asarray(plt.imread(train_folder+'/'+ non_cancer_cell.iloc[1][0]+'.tif'))
plt.title('NON-METASTATIC CELL TISSUE')
plt.imshow(img)

plt.show()
list = os.listdir(test_folder) # dir is your directory path
len(list)
list = os.listdir(train_folder) # dir is your directory path
len(list)
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=1.1,max_lighting=0.05, max_warp=0.)
data = ImageDataBunch.from_csv(path,folder='train',valid_pct=0.3,csv_labels=train_lbl,ds_tfms=tfms, size=90, suffix='.tif',test=test_folder,bs=64)
data.classes
print(data.c, len(data.train_ds), len(data.valid_ds))
stats=data.batch_stats()        
data.normalize(stats)
#data.normalize(imagenet_stats)
#See the classes and labels
data.show_batch(rows=3, figsize=(8,5))
model_dir = "/kaggle/working/tmp/models/"
os.makedirs('/kaggle/working/tmp/models/')
#fastai comes with various models
dir(fastai.vision.models)
#create learner object by passing data bunch, specifying model architecture and metrics to use to evaluate training stats
learner_resnet50 = cnn_learner(data=data, base_arch=models.resnet50,model_dir=model_dir, metrics=[accuracy,error_rate], ps=0.5) #densenet201
lr_find(learner_resnet50)
learner_resnet50.recorder.plot()
defaults.device = torch.device('cuda') # makes sure the gpu is used
learner_resnet50.fit_one_cycle(1, 1e-02)
learner_resnet50.recorder.plot(return_fig=True)
#See how the learning rate and momentum varies with the training and losses
learner_resnet50.recorder.plot_lr(show_moms=True)
learner_resnet50.recorder.plot_losses(show_grid=True)
learner_resnet50.show_results(alpha=1)
#save weights in a file
learner_resnet50.save('stage-1',return_path=True)
#Unfreeze the encoder resnet
learner_resnet50.unfreeze()
lr_find(learner_resnet50)
learner_resnet50.recorder.plot()
#slice suggests is, train the initial layers at start value specified and last layer at the end value specified and interpolate for the rest of the layers
learner_resnet50.fit_one_cycle(1,slice(1e-06,1e-05),pct_start=0.8)
learner_resnet50.recorder.plot_losses()
learner_resnet50.save('stage-2',return_path=True)
#create interpreter object
interp = ClassificationInterpretation.from_learner(learner_resnet50)
#Plot the biggest losses of the model
interp.plot_top_losses(9,figsize=(12,12),heatmap=False)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(15,5))
#To view the list of classes most misclassified as a list
#Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences
interp.most_confused(min_val=2)
pred_val ,y_val = learner_resnet50.get_preds()

def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score

pred_score=auc_score(pred_val ,y_val)
pred_score
pred_score_acc=accuracy(pred_val ,y_val)
pred_score_acc
fpr, tpr, thresholds = roc_curve(y_val.numpy(), pred_val.numpy()[:,1], pos_label=1)
pred_score_auc = auc(fpr, tpr)
print(f'ROC area: {pred_score_auc}')
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
learner_resnet50.export('/kaggle/working/tmp/models/export.pkl')
loaded_learner = load_learner(Path(model_dir))
loaded_learner.data.classes
img, cat = data.train_ds[0]
img.show()
print(cat)
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
print(pred_class, pred_idx,pred_probs)
img, cat = data.valid_ds[1]
img.show()
print(cat)
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
print(pred_class, pred_idx,pred_probs)
img = open_image(Path('../input/test-image/test_img.tif'))
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
img.show()
targets = ['Non-Cancerous','Cancerous'] #since sequence of classes in data is as 0,1
print("Tissue cell is identified as" , targets[pred_idx] , "with probability of", float(pred_probs[pred_idx]*100))
loaded_learner_val = load_learner(Path(model_dir),test=ImageList.from_folder(Path(test_folder)))
pred_test ,y_test = loaded_learner_val.get_preds(ds_type=DatasetType.Test)
sub=pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv').set_index('id')
clean_names = np.vectorize(lambda imgname: str(imgname).split('/')[-1][:-4])
cleaned_names = clean_names(data.test_ds.items).astype(str)
sub.loc[cleaned_names,'label']=pred_test.numpy()[:,1]
sub.to_csv(f'/kaggle/working/submission_{int(pred_score_auc*100)}auc.csv')
predicted_prob_test = pd.read_csv('./submission_98auc.csv')
predicted_prob_test.head(10)