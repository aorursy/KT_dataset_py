import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import os
import matplotlib.image as mpimg
df_train = pd.read_csv("../input/train.csv")
columns = ["00_Nucleoplasm" ,"01_Nuclear membrane", "02_Nucleoli", "03_Nucleoli fibrillar center" ,"04_Nuclear speckles" ,"05_Nuclear bodies",
"06_Endoplasmic reticulum","07_Golgi apparatus","08_Peroxisomes","09_Endosomes","10_Lysosomes","11_Intermediate filaments" ,"12_Actin filaments" ,
"13_Focal adhesion sites" ,"14_Microtubules" ,"15_Microtubule ends" ,"16_Cytokinetic bridge" ,"17_Mitotic spindle" ,"18_Microtubule organizing center" ,
"19_Centrosome" ,"20_Lipid droplets" ,"21_Plasma membrane" ,"22_Cell junctions" ,"23_Mitochondria" ,"24_Aggresome" ,"25_Cytosol" ,
"26_Cytoplasmic bodies" ,"27_Rods & rings" ]
for column in columns:
    df_train[column] = np.zeros(len(df_train)) 
for row in range(0,len(df_train.Target)):
    for number in df_train.Target[row].split():
        df_train.iloc[row,int(number)+2]=1
df_train["amount"] = [len(row.split()) for row in df_train.Target]
df_train.amount.hist()
df_analysis = pd.DataFrame([(sum(df_train.loc[:,column])) for column in columns])
df_analysis.columns = ["count"]
df_analysis["log10_count"] = [math.log((sum(df_train.loc[:,column])),10) for column in columns]
df_analysis.sort_values(by="count", ascending=False).plot(kind="bar")
sum((df_train.loc[:,"06_Endoplasmic reticulum"] ==1) & (df_train.loc[:,"16_Cytokinetic bridge"] ==1))
sns.heatmap(df_train.loc[:,columns].corr(),cmap = "Paired")
image_folder = "../input/train"
images = os.listdir(image_folder)
images[0]
picture = mpimg.imread(image_folder+"/"+images[0])
print(picture.shape)
colormaps = {1:"Blues", 2:"Greens",3:"Reds",4:"Oranges"}
fig = plt.figure(figsize=(10,10))
columns = 2
rows = 2

for i in range(1, 5):
    img = mpimg.imread(image_folder+"/"+images[i-1])
    a = fig.add_subplot(columns, rows, i)
    a.set_title(images[i-1][-8:-4])
    plt.imshow(img, cmap=colormaps[i])
plt.show()
df_train.head()
for row in df_train.Id:
    df_train.loc["blue"] = mpimg.imread(image_folder+"/"+row+"_blue.png")



import tensorflow as tf
sess = tf.InteractiveSession()
image = tf.image.decode_png(tf.read_file(image_folder+"/"+images[0]), channels=1)
print(sess.run(image))
