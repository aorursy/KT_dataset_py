import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
df_submission = pd.read_csv( '../input/sample_submission.csv' )
print( df_submission.shape )
df_submission.head()
CATEGORY_NAMES = df_submission.columns.values[1:].tolist()

CATEGORY_NAME_TO_ID = {}
for i, name in enumerate( CATEGORY_NAMES ):
    CATEGORY_NAME_TO_ID[name] = i

print( len( CATEGORY_NAME_TO_ID ) )
df_labels = pd.read_csv( '../input/labels.csv' )
df_labels['breed_id'] = df_labels['breed'].map( CATEGORY_NAME_TO_ID )
df_labels.head()
class MyDataset(Dataset):

    def __init__(self, df):
        
        dataset_str = 'train' if df.shape[1] == 2 else 'test'
        
        self.image_dir = Path( '../input' ) / dataset_str
        self.df = df
        
        # 对训练集做数据增强
        if dataset_str == 'train':
            self.preprocess = transforms.Compose( [
                transforms.RandomRotation(5),
                transforms.Resize(256),     # 缩小图像，使得短边为256
                transforms.RandomCrop(224), # 随机裁剪224×224
                #transforms.RandomHorizontalFlip()
            ] )
        else:
            self.preprocess = transforms.Compose( [
                transforms.Resize(256),     # 缩小图像，使得短边为256
                transforms.CenterCrop(224), # 中心裁剪224×224
            ] )
        
        self.preprocess.transforms.append( transforms.ToTensor() )
#         self.preprocess.transforms.append( transforms.Normalize( mean=[0.485, 0.456, 0.406],
#                                                                  std=[0.229, 0.224, 0.225] ) )


    def __getitem__(self, index):

        image_name = self.df.loc[index, 'id'] + '.jpg'
        image_path = self.image_dir / image_name

        X = Image.open( image_path )
        X_p = self.preprocess(X)

        simple_ToTensor = transforms.ToTensor()
        X = simple_ToTensor(X)
        y = self.df.loc[index, 'breed_id']

        return X, X_p, y


    def __len__(self):
        return len( self.df )
df_train = df_labels[['id', 'breed_id']]
df_test = df_submission[['id']]
print( df_train.shape )
print( df_test.shape )
df_temp = df_train.sample(1).copy().reset_index(drop=True)
train_set = MyDataset( df_temp )
train_loader = DataLoader( train_set, batch_size=1, shuffle=False )
image_list = []

for i in range(9):
    for batch_i, data in enumerate( train_loader ):
        batch_X, batch_Xp, batch_y = data
        
        # PyTorch使用transforms.ToTensor后，size为[N, C, H, W]
        batch_X = np.transpose( batch_X.squeeze().numpy(), [1, 2, 0] )
        batch_Xp = np.transpose( batch_Xp.squeeze().numpy(), [1, 2, 0] )
        
        image_list.append( batch_Xp )
        break
print( batch_X.shape )
plt.imshow( batch_X )
fig, axes = plt.subplots( 3, 3, figsize=(9, 9) )

for i, ax in enumerate(axes.flat):
    ax.imshow( image_list[i] )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
