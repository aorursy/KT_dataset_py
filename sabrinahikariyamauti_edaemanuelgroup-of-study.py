#importando as bibliotecas
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm_notebook as tqdm

#analisando a pasta que contém os dados
!ls -lh ../input/vehicle/
#analisando subpastas
!ls -lh ../input/vehicle/train/train

#mostrar apenas os 5 primeiros arquivos/imagens do testset
!ls -lh ../input/vehicle/test/testset |head -5
#mostrar as pastas do train
root = '../input/vehicle/train/train'
os.listdir(root)
root = '../input/vehicle/train/train'
i = 0
n_data_train = 0

for root, directories, files in os.walk(root):#percorrer pelos diretórios e pastas
    if(len(files)==0):#ignorar se o núemro de aqrvuio for 0
        pass
    else:
        n_data_train += len(files)#contagem dos arquivos   
print(n_data_train)
root = '../input/vehicle/train/train'
for root, directories, files in os.walk(root):
    if len(files)==0:
        print(directories)
        pass
    else:
        print(root)#exibir dretório
        print(directories)
        print(len(files))#exibir quantidade de arquivos por pastas
rroot = '../input/vehicle/train/train'
hist = []
for root, directories, files in os.walk(root):
    if len(files)==0:
        print(directories)
        pass
    else:
        print(len(files))#exibir quantidade de arquivos por pastas
        hist.append(len(files))
        
hist.value_counts().plot(kind='bar')
test_data = '../input/vehicle/test/testset'
os.listdir(test_data)
for root_test, directories_test, files_test in os.walk(test_data):
    print(root_test)
    #print(directories_test)
    print(len(files_test))
"""
Como os dados de teste são classificados, não aparecem as pastas nomeadas por categoria
a quantidade de arquivo é exibido de uma vez
"""
root = '../input/vehicle/train/train'
for last_name_root, directories, files in os.walk(root):
    if len(files)==0:
        pass
    else:
        checking = "{}:{}" 
        print(checking.format(os.path.basename(last_name_root), len(files)))
"""
Foi mostrado na tela apenas o último nome do diretório que são as categorias,
seguido de quantidade de arquivos por pasta
Para visualizar de forma melhor, o código abaixo foi criado para fazer a plotagem
"""
root = '../input/vehicle/train/train'
list_last_name_root = []#criando lista com nome de categorias para plotar
list_lenfiles = []#criando lista quantidade de arquivos para plotar
for last_name_root, directories, files in os.walk(root):
    if len(files)==0:
        pass
    else:
        #inserindo nomes do último diretório na lista
        list_last_name_root.append(os.path.basename(last_name_root))
        #inserindo quantidade de arquivos de pastas na lista
        list_lenfiles.append(len(files))

print(list_last_name_root)
print(list_lenfiles)
from PIL import Image 
# Open the image form working directory
image = Image.open('../input/vehicle/train/train/Helicopter/002964_10.jpg')
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
# show the image
#image.show()#naõ se o porquê mas não funciona .show() para mim
image

####################

from IPython.display import Image
image = "../input/vehicle/train/train/Helicopter/002964_10.jpg"
Image(image)

####################

"""
Nas duas células abaixo, tentei exibir imagens sequencialmente, mas não consegui fazer isto
"""

import glob
for f in glob.iglob('../input/vehicle/train/train/Helicopter/*'):
    Image(f)
    #images.append(np.array(Image.open(f)))
from PIL import Image 
root = "../input/vehicle/train/train/Ambulance"
for file in sorted(os.listdir(os.path.join(root))):
    data = Image.open(os.path.join(root,file))
    print(data.size)
"""
A seguir são exibidos 4 classes de forma separada e por último, estão exibição de todas as classes
"""
from PIL import Image 
ambulance_w=0
ambulance_h=0
n_ambulance=0
root = "../input/vehicle/train/train/Ambulance"
for file in sorted(os.listdir(os.path.join(root))):
    data = Image.open(os.path.join(root,file)) 
    n_ambulance +=1
    #print(os.path.join(root,file))
    ambulance_w += data.size[0]
    ambulance_h += data.size[1]
print(round(ambulance_w/n_ambulance))
print(round(ambulance_h/n_ambulance))

#######################

car_w=0
car_h=0
n_car=0
root = "../input/vehicle/train/train/Car"
for file in sorted(os.listdir(os.path.join(root))):
    data = Image.open(os.path.join(root,file)) 
    n_car +=1
    #print(os.path.join(root,file))
    car_w += data.size[0]
    car_h += data.size[1]
print(round(car_w/n_car))
print(round(car_h/n_car))

####################

van_w=0
van_h=0
n_van=0
root = "../input/vehicle/train/train/Van"
for file in sorted(os.listdir(os.path.join(root))):
    data = Image.open(os.path.join(root,file)) 
    n_van +=1
    #print(os.path.join(root,file))
    van_w += data.size[0]
    van_h += data.size[1]
print(round(van_w/n_van))
print(round(van_h/n_van))

#########################

snowmobile_w=0
snowmobile_h=0
n_snowmobile=0
root = "../input/vehicle/train/train/Snowmobile"
for file in sorted(os.listdir(os.path.join(root))):
    data = Image.open(os.path.join(root,file)) 
    n_snowmobile +=1
    #print(os.path.join(root,file))
    snowmobile_w += data.size[0]
    snowmobile_h += data.size[1]
print(round(snowmobile_w/n_snowmobile))
print(round(snowmobile_h/n_snowmobile))

###########################

root = "../input/vehicle/train/train"
for directory in sorted(os.listdir(os.path.join(root))):
    print(directory)
    data_w=0
    data_h=0
    n=0
    for file in sorted(os.listdir(os.path.join(root,directory))):
        data = Image.open(os.path.join(root,directory,file))
        n +=1
        data_w += data.size[0]
        data_h += data.size[1]
    print(round(data_w/n), round(data_h/n))
"""
Apresentando todos os dados 
"""
root = "../input/vehicle/train/train"
w_cat=[]
h_cat=[]
cat=[]
for directory in sorted(os.listdir(os.path.join(root))):
    cat.append(os.path.basename(directory))
    data_w=0
    data_h=0
    n=0
    for file in sorted(os.listdir(os.path.join(root,directory))):
        data = Image.open(os.path.join(root,directory,file))
        n +=1
        data_w += data.size[0]
        data_h += data.size[1]
    w_cat.append(round(data_w/n))
    h_cat.append(round(data_h/n))
print(cat)
print(w_cat)
print(h_cat)
    
import os
root = "../input/vehicle/train/train"
data_array=[]
for directory in sorted(os.listdir(os.path.join(root))):
    for file in sorted(os.listdir(os.path.join(root,directory))):
        data = Image.open(os.path.join(root,directory,file))
        data_array.append(np.array(data))
#Exemplo
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,1,2,3])
y = np.array([20,21,22,23])
my_xticks = ['John','Arnold','Mavis','Matt']
plt.xticks(x, my_xticks)
plt.plot(x, y)
plt.show()
"""
#Mostrando dimensões das imagens em shape (OpenCv)
import os
import cv2
root = "../input/vehicle/train/train"
y = []
x_width = []
x_heigh = []
iden = 0
for directory in sorted(os.listdir(root)):
    iden+=1
    y.append(iden)
    for i in sorted(os.listdir(os.path.join(root,directory))):
        data = cv2.imread((os.path.join(root,directory,i)), cv2.IMREAD_UNCHANGED)
        print(data.shape)
        #x_width.append(data.size[0])
        #x_heigh.append(data.size[1])
 
#Mostrando dimensões das imagens em size (PIL) dentro de uma lista
import os
root = "../input/vehicle/train/train"
y = []
x_width_PIL = []
x_heigh_PIL = []
iden = 0
for directory in sorted(os.listdir(root)):
    iden+=1
    y.append(iden)
    for i in sorted(os.listdir(os.path.join(root,directory))):
        data = Image.open(os.path.join(root,directory,i))
        x_width_PIL.append(data.size[0])
        x_heigh_PIL.append(data.size[1])
