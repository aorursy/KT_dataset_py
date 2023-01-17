from IPython.display import clear_output

!pip install autokeras  # 安装 Auto-Keras

clear_output() # 为了美观清除该单元格输出内容
!unzip -o /kaggle/input/train.zip

!unzip -o /kaggle/input/test.zip

clear_output() # 为了美观清除该单元格输出内容
!ls /kaggle/input/
import pandas as pd



df_train = pd.read_csv("../input/train.csv") # 读取原标注文件

df_train.columns = ['File Name', 'Label'] # 修改文件名

df_train.to_csv("./train/label.csv", index=None) # 写入新标注文件

df_train.head()
df_test = pd.read_csv("../input/sample_submission.csv") # 读取原标注文件

df_test.columns = ['File Name', 'Label'] # 修改文件名

df_test.to_csv("./test/label.csv", index=None) # 写入新标注文件

df_test.head()
import cv2

import numpy as np



def load_images(dir_paths,images_path):

    iter_all_images = (cv2.imread(dir_paths+fn) for fn in images_path)



    # iter_all_images 是一个 generator 类型，将它转换成熟知的 numpy 的列表类型并返回

    for i, image in enumerate(iter_all_images):

        if i == 0:

            # 对all_images 进行初始,并且指定格式

            all_images = np.empty(

                (len(images_path),) + image.shape, dtype=image.dtype)

        all_images[i] = image



    return all_images



 
x_img_path = df_train.iloc[:,0].values

x_dir_path = "train/"

X_train =   load_images(x_dir_path,x_img_path)

y_train =   df_train.iloc[:,1].values



 

x_img_path = df_test.iloc[:,0].values

x_dir_path = "test/"

X_test =   load_images(x_dir_path,x_img_path)

y_test =   df_test.iloc[:,1].values    



X_train.shape, y_train.shape,X_test.shape, y_test.shape
import autokeras as ak



# 实例化模型，max_trials=3 ，表示尝试的最大的 神经网络模型数 

clf = ak.ImageClassifier(max_trials=3) 

# epochs 表示每个模型训练的最大世代数

clf.fit(X_train, y_train,epochs=5,verbose=2)

print("训练完成")
preds= clf.predict(X_test) # 推理

preds
df_pred = pd.read_csv("../input/sample_submission.csv") # 读取原标注文件

df_pred.head()
df_pred['has_cactus'] = preds

df_pred.to_csv("./preds_submission.csv", index=None) # 保存预测文件用于提交

df_pred.head()
!rm -rf ./train

!rm -rf ./test