# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from nltk.stem.snowball import SnowballStemmer
import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
train_zip  = ZipFile('../input/avito-demand-prediction/train_jpg.zip')
filenames = train_zip.namelist()[1:]
len(filenames)
train_zip.close()
del train_zip
train_zip  = ZipFile('../input/avito-demand-prediction/test_jpg.zip')
filenames = train_zip.namelist()[1:]
print(len(filenames))
train_zip.close()
del train_zip
train_image_info = pd.read_csv('../input/extract-image-features-train-files/train_img_feat.csv',index_col=0)
train_image_info.shape
train_image_size = pd.read_csv('../input/extract-image-features-train-files/train_img_files_info.csv',index_col=0)
train_image_size.shape
%%time
image_features = train_image_info.merge(train_image_size,on='image',how='left')
image_features.head()
image_features['dim'] = image_features['dim'].apply(lambda x:eval(str(x)))
image_features['colors'] = image_features['colors'].apply(lambda x: eval(str(x)))
image_features['width'] = image_features['dim'].apply(lambda x: x[0])
image_features['height'] = image_features['dim'].apply(lambda x: x[1])
image_features['red_avg'] = image_features['colors'].apply(lambda x: x[0])
image_features['green_avg'] = image_features['colors'].apply(lambda x: x[1])
image_features['blue_avg'] = image_features['colors'].apply(lambda x: x[2])
image_features.drop(columns=['dim','colors','csize'],inplace=True)
image_features.head()
def make_features(image_features):
    image_features['width_height_diff'] = image_features[['width','height']].diff(axis=1)['height']
    image_features['green_blue_diff'] = image_features[['green_avg','blue_avg']].diff(axis=1)['blue_avg']
    image_features['green_red_diff'] = image_features[['green_avg','red_avg']].diff(axis=1)['red_avg']
    image_features['red_blue_diff'] = image_features[['red_avg','blue_avg']].diff(axis=1)['blue_avg']
    image_features['width_height_ratio'] = image_features['width']/image_features['height']
    image_features['total_pixel'] = image_features['width']*image_features['height']
    return image_features
image_features = make_features(image_features)
image_features.head()
def show_corr(df):
    f, ax = plt.subplots(figsize=[10,7])
    sns.heatmap(df.corr(),
                annot=False, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="plasma",ax=ax, linewidths=.5)
    ax.set_title("Dense Features Correlation Matrix")
    plt.savefig('correlation_matrix.png')
show_corr(image_features)
image_features.columns
test_image_yuv = pd.read_csv('../input/script-image-features-test-yuv-multproces/test_jpg_img_feat_saturation.csv',index_col=0)
train_image_yuv = pd.read_csv('../input/script-image-features-train-yuv-multprocess/train_jpg_img_feat_saturation.csv',index_col=0)
yuv_col = [ 'bright_avg', 'u_avg', 'yuv_v_avg', 'bright_std', 'bright_min','birght_max', 'bright_diff']
test_image_yuv.columns
test_image_yuv.rename(columns={'v_avg':'yuv_v_avg'},inplace=True,index=str)
train_image_yuv.rename(columns={'v_avg':'yuv_v_avg'},inplace=True,index=str)
test_image_yuv.columns,train_image_yuv.columns
test_image_yuv.head()
image_features.shape,test_image_yuv.shape,train_image_yuv.shape
# test_image_yuv= test_image_yuv[using_col]
# train_image_yuv= train_image_yuv[using_col]
train_image_yuv.head()
image_features = image_features.merge(train_image_yuv,on='image',how='left')
image_features.head()
train_path_ = '../input/script-image-features-train-hsv-batch%s/train_jpg_img_feat_saturation.csv'
train_ids = [1,15,2,25,3,4]
test_path_ = '../input/script-image-features-test-hsv-batch%s/test_jpg_img_feat_saturation.csv'
test_ids = [1,2]
train_hsv_df = pd.DataFrame()
test_hsv_df = pd.DataFrame()
for i in train_ids:
    p = train_path_%i
    train_hsv_df = train_hsv_df.append(pd.read_csv(p,index_col=0))
    print(p,train_hsv_df.shape)

for i in test_ids:
    p = test_path_%i
    test_hsv_df = test_hsv_df.append(pd.read_csv(p,index_col=0))
    print(p,test_hsv_df.shape)
train_hsv_df.shape,test_hsv_df.shape
train_hsv_df.rename(columns={'v_avg':'hsv_v_avg'},inplace=True,index=str)
test_hsv_df.rename(columns={'v_avg':'hsv_v_avg'},inplace=True,index=str)
image_features = image_features.merge(train_hsv_df,on='image',how='left')
image_features.head()
train_hsv_df.columns
hsv_col = ['hue_avg', 'sst_avg', 'hsv_v_avg', 'sat_std', 'sat_min', 'sat_max', 'sat_diff']
show_corr(image_features)
train_path_ = '../input/script-image-features-train-color-batch%s/train_jpg_img_feat_saturation.csv'
test_path_ = '../input/script-image-features-test-color-multprocess/test_jpg_img_feat_saturation.csv'
train_ids= [1,2,3,4,5,6]
# train_ids= [1,2,3,5,6]
test_color_df = pd.read_csv(test_path_,index_col=0)
train_color_df = pd.DataFrame()
for i in train_ids:
    p = train_path_%i
    x = pd.read_csv(p,index_col=0)
    print(p,x.shape)
    train_color_df = train_color_df.append(x)
train_color_df.shape,test_color_df.shape
del train_image_info,train_image_size
import gc
gc.collect()
image_features = image_features.merge(train_color_df,on='image',how='left')
image_features.head()
color = ['colorfull']
train_std_df = pd.read_csv("../input/script-image-features-train-pil-std-batch1/train_jpg_img_feat_saturation.csv",index_col=0)
test_std_df = pd.read_csv("../input/script-image-features-test-pil-std-batch1/test_jpg_img_feat_saturation.csv",index_col=0)
std_cols = ['r_std', 'g_std', 'b_std', 'r_md', 'g_md', 'b_md']
image_features = image_features.merge(train_std_df,on='image',how='left')
image_features.head()
train_xception_df = pd.read_csv('../input/xception-train-features-starter-include-top/train_xception.csv',index_col=0)
test_xception_df = pd.read_csv('../input/xception-test-features-starter-include-top/test_xception.csv',index_col=0)
train_xception_df.columns = ['image','item_label', 'xception_prob', 'xception_var',
       'xception_nonzero']
test_xception_df.columns = ['image','item_label', 'xception_prob', 'xception_var',
       'xception_nonzero']
train_xception_df.head()
cols = ['item_label', 'xception_prob', 'xception_var',
       'xception_nonzero']
image_features = image_features.merge(train_xception_df,on='image',how='left')
image_features.head()
train_nima_df = pd.read_csv('../input/neural-image-assessment-train-features-starter/train_xception.csv',index_col=0)
test_nima_df = pd.read_csv('../input/neural-image-assessment-test-feature-starter/test_xception.csv',index_col=0)
train_nima_df.shape,test_nima_df.shape
image_features = image_features.merge(train_nima_df,on='image',how='left')
nima_cols = ['mean_nima','std_nima_']
image_features.head()
train_blurr_df = pd.read_csv('../input/blurr-features-train-part1/train_jpg_img_feat_blurr.csv',index_col=0)
train_blurr_df=train_blurr_df.append(pd.read_csv('../input/blurr-features-train-part2/train_jpg_img_feat_blurr.csv',index_col=0))
train_blurr_df=train_blurr_df.append(pd.read_csv('../input/fork-of-blurr-features-train-part/train_jpg_img_feat_blurr.csv',index_col=0))

test_blurr_df = pd.read_csv('../input/blurr-image-features-test/test_jpg_img_feat_blurr.csv',index_col=0)
train_blurr_df.shape,test_blurr_df.shape
train_blurr_df.columns
image_features = image_features.merge(train_blurr_df,on='image',how='left')
image_features.head()
show_corr(image_features)
train_df = pd.read_csv('../input/avito-demand-prediction/train.csv',usecols=['image','item_id',],index_col=0)
%%time
train_df = train_df.reset_index().merge(image_features,on='image',how='left').set_index('item_id')
train_df.head()
image_cols = image_features.columns
image_cols
train_image_features=image_features
image_features.shape
test_image_info = pd.read_csv('../input/extract-image-features-test-file/train_img_feat.csv',index_col=0)
test_file_info = pd.read_csv('../input/extract-image-features-test-file/test_img_files_info.csv',index_col=0)
test_imge_features = test_image_info.merge(test_file_info,on='image',how='left')
def proce_data(image_features):
    image_features['dim'] = image_features['dim'].apply(lambda x:eval(x))
    image_features['colors'] = image_features['colors'].apply(lambda x: eval(x))
    image_features['width'] = image_features['dim'].apply(lambda x: x[0])
    image_features['height'] = image_features['dim'].apply(lambda x: x[1])
    image_features['red_avg'] = image_features['colors'].apply(lambda x: x[0])
    image_features['green_avg'] = image_features['colors'].apply(lambda x: x[1])
    image_features['blue_avg'] = image_features['colors'].apply(lambda x: x[2])
    return image_features

test_imge_features = proce_data(test_imge_features)
test_imge_features = make_features(test_imge_features)
test_imge_features.shape
test_imge_features.head()
test_imge_features.drop(columns=['dim','colors','csize'],inplace=True)
test_imge_features = test_imge_features.merge(test_hsv_df,on='image',how='left')
test_imge_features = test_imge_features.merge(test_color_df,on='image',how='left')
test_imge_features.shape
test_imge_features = test_imge_features.merge(test_image_yuv,on='image',how='left')
test_imge_features.shape
test_imge_features = test_imge_features.merge(test_std_df,on='image',how='left')
test_imge_features = test_imge_features.merge(test_xception_df,on='image',how='left')
test_imge_features = test_imge_features.merge(test_nima_df,on='image',how='left')
test_imge_features = test_imge_features.merge(test_blurr_df,on='image',how='left')
show_corr(test_imge_features)
test_df = pd.read_csv('../input/avito-demand-prediction/test.csv',usecols=['image','item_id','image_top_1','description','title'],index_col=0)
test_df  = test_df.reset_index().merge(test_imge_features,on='image',how='left').set_index('item_id')
test_df.shape
test_df[train_image_features.columns].drop(columns=['image']).to_csv('test_image_features.csv.gzip',compression='gzip')
train_df[image_features.columns].drop(columns=['image']).to_csv('train_image_features.csv.gzip',compression='gzip')
# submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv',index_col=0)
# submission['deal_probability'] = predict_y
# submission.to_csv('rf_image.csv',index=True)