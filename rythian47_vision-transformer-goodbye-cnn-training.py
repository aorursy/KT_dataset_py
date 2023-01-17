!pip install linformer
!pip install axial_attention
import numpy as np 
import pandas as pd 
import glob
import cv2
import os
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
from skimage.color import gray2rgb
import functools
from sklearn.metrics import hamming_loss, accuracy_score

import torch
from torch import nn
from linformer import Linformer
from axial_attention import AxialAttention


from tqdm.auto import tqdm
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import os
class data_config:
    train_csv_path = 'train.csv'
    jpeg_dir ='train-jpegs'
    ids = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']    
    label_lstm = ['pe_present_on_image','negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
        
class visualTrans:  
    model_name="visualT"
    batch_size = 1
    WORKERS = 4
    classes =9
    epochs = 64
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':1e-4,'weight_decay':0.0001}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max':5500,'eta_min':0.00001}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = 'log/cpt'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
CFG = {
    'image_target_cols': [
        'pe_present_on_image', # only image level
    ],
    
    'exam_target_cols': [
        'negative_exam_for_pe', # exam level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ], 
    
    'image_weight': 0.07361963,
    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988],
}
train_csv_path = 'train.csv'
jpeg_dir = 'train-jpegs'
from sklearn.model_selection import StratifiedKFold
def get_fold(train,FOLD_NUM = 5):
    train_image_num_per_patient = train.groupby('StudyInstanceUID')['SOPInstanceUID'].nunique()
    target_cols = [c for i, c in enumerate(train.columns) if i > 2]
    
    train_per_patient_char = pd.DataFrame(index=train_image_num_per_patient.index, columns=['image_per_patient'], data=train_image_num_per_patient.values.copy())
    for t in target_cols:
        train_per_patient_char[t] = train_per_patient_char.index.map(train.groupby('StudyInstanceUID')[t].mean())
        
    
    bin_counts = [40] #, 20]
    digitize_cols = ['image_per_patient'] #, 'pe_present_on_image']
    non_digitize_cols = [c for c in train_per_patient_char.columns if c not in digitize_cols]
    for i, c in enumerate(digitize_cols):
        bin_count = bin_counts[i]
        percentiles = np.percentile(train_per_patient_char[c], q=np.arange(bin_count)/bin_count*100.)
        train_per_patient_char[c+'_digitize'] = np.digitize(train_per_patient_char[c], percentiles, right=False)
        
    train_per_patient_char['key'] = train_per_patient_char[digitize_cols[0]+'_digitize'].apply(str)
    for c in digitize_cols[1:]:
        train_per_patient_char['key'] = train_per_patient_char['key']+'_'+train_per_patient_char[c+'_digitize'].apply(str)
    folds = FOLD_NUM
    kfolder = StratifiedKFold(n_splits=folds, shuffle=True, random_state=719)
    val_indices = [val_indices for _, val_indices in kfolder.split(train_per_patient_char['key'], train_per_patient_char['key'])]
    train_per_patient_char['fold'] = -1
    for i, vi in enumerate(val_indices):
        patients = train_per_patient_char.index[vi]
        train_per_patient_char.loc[patients, 'fold'] = i
    return train_per_patient_char

def split_train_val_lstm(data_config, fold, FOLD_NUM=5):
    main_df = pd.read_csv(data_config.train_csv_path)
    train_df = main_df[data_config.ids+ data_config.label_lstm]
    train_per_patient_char = get_fold(main_df, FOLD_NUM)
    TID = train_per_patient_char[train_per_patient_char.fold!=fold].index
    VID = train_per_patient_char[train_per_patient_char.fold==fold].index
    t_df = train_df[train_df['StudyInstanceUID'].isin(TID)]
    v_df = train_df[train_df['StudyInstanceUID'].isin(VID)]
    return t_df,v_df
t_df,v_df = split_train_val_lstm(data_config,fold=0,FOLD_NUM=5)
path256 = f"{data_config.jpeg_dir}/*/*/*.jpg"
data = glob.glob(path256)
new_df = []
for row in tqdm(data):
    StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID = row.split("/")[-3:]
    num,SOPInstanceUID = SOPInstanceUID.replace(".jpg","").split("_")
    new_df.append([StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID,num])
s_df = pd.DataFrame(new_df)
s_df.columns = list(t_df.columns[:3])+["slice"]
t_df = t_df.merge(s_df,on=list(t_df.columns[:3]),how='left')
v_df = v_df.merge(s_df,on=list(v_df.columns[:3]),how='left')
t = t_df.groupby(list(t_df.columns[:2]))
mini_dfs= []
for i,row in tqdm(t_df.groupby(list(t_df.columns[:2]))):
    if len(row)>400:
        continue
    mini_dfs.append(row.sort_values("slice"))
mini_dfs_val = []
for i,row in tqdm(v_df.groupby(list(v_df.columns[:2]))):
    if len(row)>400:
        continue
    mini_dfs_val.append(row.sort_values("slice"))
class CTDataset(Dataset):
    def __init__(self,df,jpeg_dir,transforms = None, preprocessing=None, size=256, mode='val'):
        self.df_main = df
        self.jpeg_dir = jpeg_dir
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.size=size

    def __getitem__(self, idx):
        mini = self.df_main[idx].values
        all_paths = [f"{self.jpeg_dir}/{row[0]}/{row[1]}/{row[-1]}_{row[2]}.jpg" for row in mini]
        img = [self.transforms(image=cv2.imread(p))['image'] for p in all_paths]
        label = mini[:,3:-1].astype(int)        
            
        if self.preprocessing:
            img = [self.preprocessing(image=im)['image'] for im in img]
        return np.array(img),torch.from_numpy(label[:,0]), torch.from_numpy(label[0,1:])
    
    def __len__(self):
        return len(self.df_main)
def get_training_augmentation(y=256,x=256):
    train_transform = [albu.RandomBrightnessContrast(p=0.3),
                           albu.VerticalFlip(p=0.5),
                           albu.HorizontalFlip(p=0.5),
                           albu.Downscale(p=1.0,scale_min=0.35,scale_max=0.75,),
                           albu.Resize(y, x)]
    return albu.Compose(train_transform)


formatted_settings = {
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],}

def mono_tr(x):
    train_transforms = Compose([ScaleIntensity(), 
                            Resize((image_size, image_size, image_size)), 
                            RandAffine( 
                                      prob=0.5,
                                      translate_range=(5, 5, 5),
                                      rotate_range=(np.pi * 4, np.pi * 4, np.pi * 4),
                                      scale_range=(0.15, 0.15, 0.15),
                                      padding_mode='border'),
                            ToTensor()])
    apply_transform(train_transforms, x)
    
def mono_val(x):
    val_transforms = Compose([ScaleIntensity(), ToTensor()])
    
    apply_transform(val_transforms, x)
    

def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        #albu.Lambda(image=preprocessing_fn_2),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation(y=256,x=256):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(y, x)]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

def norm(img):
    img-=img.min()
    return img/img.max()
preprocessing_fn = functools.partial(preprocess_input, **formatted_settings)
train_dataset = CTDataset(mini_dfs,data_config.jpeg_dir,
                            transforms=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
val_dataset = CTDataset(mini_dfs_val,data_config.jpeg_dir,
                            transforms=get_validation_augmentation(),preprocessing=get_preprocessing(preprocessing_fn))
x,y,y1 = train_dataset[0]
x.shape,y.shape,y1.shape
global view_output
def hook_fn(module, input, output):
    global view_output
    view_output = output
import torch
from einops import rearrange
from torch import nn

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, out_dim, dim, transformer, channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, out_dim)
        )

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
from torch import nn
from torch.nn import functional as F

class TransNET(nn.Module):
    def __init__(self, embed_size= 256, LSTM_UNITS= 64):
        super(TransNET, self).__init__()
        #self.axttn = AxialAttention(dim = 3, dim_index = 1, dim_heads = 16, heads = 1, num_dimensions = 2, sum_axial_out = True)
        self.lin = Linformer(dim = 128, seq_len = 65, depth = 6, heads = 8, k = 256)
        self.vit = ViT(image_size = 256, patch_size = 32, out_dim = 256, dim = 128, transformer = self.lin)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2, bias = False)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2, bias = False)

        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
        self.linear_global = nn.Linear(LSTM_UNITS*2, 9)

    def forward(self, x, lengths=None):
        #embedding = self.axttn(x)
        embedding = self.vit(x)
        b,f = embedding.shape
        embedding = embedding.reshape(1,b,f)
            
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear_pe(hidden)
        output_global = self.linear_global(hidden.mean(1))
        return output,output_global
model = TransNET().cuda()
model_config = visualT
optimizer = eval(model_config.optimizer)(model.parameters(),**model_config.optimizer_parm)
scheduler = eval(model_config.scheduler)(optimizer,**model_config.scheduler_parm)
#loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_fn = eval(model_config.loss_fn)(reduction='none')
label_w = torch.cuda.FloatTensor(CFG['exam_weights']).view(1, -1)
img_w = CFG['image_weight']
import torch
import numpy as np
from tqdm.auto import tqdm
import os
class trainer:
    def __init__(self,loss_fn,model,optimizer,scheduler,config,label_w, img_w):
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.label_w = label_w
        self.img_w = label_w

        
    def batch_train(self, batch_imgs, batch_labels0, batch_labels1, batch_idx):
        batch_imgs, batch_labels0, batch_labels1 = batch_imgs.cuda().float(), batch_labels0.cuda().float(),batch_labels1.cuda().float()
        predicted = self.model(batch_imgs)
        loss0 = self.loss_fn(predicted[0].float().reshape(-1), batch_labels0.reshape(-1))
        loss1 = self.loss_fn(predicted[1].float().reshape(-1), batch_labels1.reshape(-1))
        loss1 = torch.sum(torch.mul(loss1, self.label_w), 1)[0]
        img_num = batch_labels0.shape[1]
        qi = torch.sum(batch_labels0.reshape(-1))/img_num
        loss0 = torch.sum(img_w* qi* loss0)
        loss = loss0 + loss1
        total = label_w.sum() + img_w*qi*img_num
        loss = loss/total
        #print(loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), predicted
    
    def batch_valid(self, batch_imgs,get_fet):
        self.model.eval()
        batch_imgs = batch_imgs.cuda()
        with torch.no_grad():
            predicted = self.model(batch_imgs)
            predicted[0] = torch.sigmoid(predicted[0])
            predicted[1] = torch.sigmoid(predicted[1])
        return predicted
    
    def train_epoch(self, loader):
        self.model.train()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        for batch_idx, (imgs,labels,labels1) in enumerate(tqdm_loader):
            loss, predicted = self.batch_train(imgs[0], labels,labels1, batch_idx)
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
            tqdm_loader.set_description('loss: {:.4} lr:{:.6}'.format(
                    current_loss_mean, self.optimizer.param_groups[0]['lr']))
            self.scheduler.step(batch_idx)
            if batch_idx>10:
                break
        return current_loss_mean
    
    def valid_epoch(self, loader,name="valid"):
        self.model.eval()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        correct = 0
        for batch_idx, (imgs,labels0,labels1) in enumerate(tqdm_loader):
            with torch.no_grad():
                batch_imgs = imgs.cuda().float()[0]
                batch_labels0 = labels0.cuda().float()
                batch_labels1 = labels1.cuda().float()
                
                predicted = self.model(batch_imgs)
                loss0 = self.loss_fn(predicted[0].float().reshape(-1),batch_labels0.float().reshape(-1)) #.item()
                loss1 = self.loss_fn(predicted[1].float().reshape(-1),batch_labels1.float().reshape(-1))#.item()
                loss1 = torch.sum(torch.mul(loss1, self.label_w), 1)[0]
                img_num = batch_labels0.shape[1]
                qi = torch.sum(batch_labels0.reshape(-1))/img_num
                loss0 = torch.sum(img_w* qi* loss0)
                loss = loss0 + loss1
                total = label_w.sum() + img_w*qi*img_num
                loss = loss/total
                hm_1 = hamming_loss(batch_labels1.reshape(-1).detach().cpu(), 
                                    torch.round(torch.sigmoid(predicted[1].reshape(-1).detach().cpu())))
                hm_0 = hamming_loss(batch_labels0.reshape(-1).detach().cpu(), 
                                    torch.round(torch.sigmoid(predicted[0].reshape(-1).detach().cpu())))
                print("Hamming_loss_0:", hm_0)
                print("Hamming_loss_1:", hm_1)
                tqdm_loader.set_description(f"loss : {loss:.4}")
            if batch_idx>10:
                break
        score = 1- loss
        print('metric {}'.format(score))
        return score
    
    def run(self,train_loder,val_loder):
        best_score = -100000
        for e in range(self.config.epochs):
            print("----------Epoch {}-----------".format(e))
            current_loss_mean = self.train_epoch(train_loder)
            score = self.valid_epoch(val_loder)
            if best_score < score:
                best_score = score
                torch.save(self.model.state_dict(),self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name))

    def batch_valid_tta(self, batch_imgs):
        batch_imgs = batch_imgs.cuda()
        predicted = model(batch_imgs)
        tta_flip = [[-1],[-2]]
        for axis in tta_flip:
            predicted += torch.flip(model(torch.flip(batch_imgs, axis)), axis)
        predicted = predicted/(1+len(tta_flip))
        predicted = torch.sigmoid(predicted)
        return predicted.cpu().numpy()
            
    def load_best_model(self):
        if os.path.exists(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)):
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)))
            print("load best model")
        
    def predict(self,imgs_tensor,get_fet = False):
        self.model.eval()
        with torch.no_grad():
            return self.batch_valid(imgs_tensor,get_fet=get_fet)
Trainer = trainer(loss_fn, model, optimizer, scheduler, config=model_config, label_w=label_w, img_w=img_w)
train = DataLoader(train_dataset, batch_size= 1, shuffle=True, num_workers= model_config.WORKERS, pin_memory = False)
val = DataLoader(val_dataset, batch_size= 1 , shuffle= True, num_workers= model_config.WORKERS, pin_memory = False)
Trainer.run(train,val)