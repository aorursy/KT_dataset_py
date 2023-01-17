import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from numpy import random
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train.head()
target = train.label.values
# Drop the label feature
train = train.drop("label",axis=1)
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
del scaler
sns.set(rc={'figure.figsize': (5, 7)})
n = np.random.randint(0, 42000, 1)

digits = train[n]
labels = target[n]

image = digits.reshape(28, 28)
plt.imshow(image, cmap='gray')
plt.axis("off")     
# causes zooming in the x direction,
def matrix_хах(a):                   
    matrix_хах = np.array([[a, 0], 
                           [0, 1]])
    return matrix_хах

# in this example we will transform both coordinates
def matrix_хах_ydy(a, d):            
    matrix_хах_ydy = np.array([[a, 0],  
                               [0, d]])
    return matrix_хах_ydy

# display point about y axis
def revers_on_y(): 
    revers_on_y = np.array([[-1, 0],   
                            [0, 1]])
    return revers_on_y

# display point about x axis
def revers_on_x():
    revers_on_x = np.array([[1 ,0], 
                            [0 ,-1]]) 
    return revers_on_x

# point mapping relative to both x y axes
def revers_on_xy():
    revers_on_xy = np.array([[-1, 0],  
                             [0, -1]])
    return revers_on_xy

# the shift is proportional to x
def shift_on_x(b):
    shift_on_x = np.array([[1, b],  
                           [0, 1]])
    return shift_on_x

# the shift is proportional to y
def shift_on_y(c):
    shift_on_y = np.array([[1, 0],  
                           [c, 1]])
    return shift_on_y

# Turns 90 and 180

def Turn_on90():
    turn_on90 = np.array([[0, 1],  
                          [-1, 0]])
    return turn_on90

def Turn_on180():
    turn_on180 = np.array([[-1, 0],  
                           [0, -1]])
    return turn_on180
random_transformation_matrix = [ 
                                 matrix_хах(np.random.randint(2, 5)), 
    
                                 matrix_хах_ydy(2, 2), 
                                 matrix_хах_ydy(np.random.randint(1, 4), np.random.randint(1, 4)),  
    
                                 revers_on_y(),
                                 revers_on_x(), 
                                 revers_on_xy(), 
    
                                 shift_on_x(np.random.randint(1, 4)), 
                                 shift_on_y(np.random.randint(1, 3)), 
    
                                 Turn_on90(),
                                 Turn_on180()
                               ]
def image_shift(shift_fun):

    Frame_new_alis = []

    for ind_new_alis, mas_digits in tqdm(enumerate(digits)):

        h = shift_fun 
        alis = []
        for d in range(0, len(list(mas_digits)), 28):
            alis.append(mas_digits[d:d+28]) 

        new_alis = np.zeros((200, 200)) 

        for i, t in enumerate(alis):
            for ind_t in range(len(t)):
                new_koar = np.array([i, ind_t]).dot(h) 
                new_alis[abs(new_koar[0])][abs(new_koar[1])] = t[ind_t]

    #mas_DF_newalias_dict = {}
    mas_DF_newalias = []

    for st_di in new_alis: 
        for st_di_i in st_di:
            mas_DF_newalias.append(st_di_i)
    #mas_DF_newalias_dict['{}'.format(di)] = mas_DF_newalias


    #df2 = pd.DataFrame(data=mas_DF_newalias_dict) 
                                                  
    df2 = pd.DataFrame(data=mas_DF_newalias)
    
    #del mas_DF_newalias_dict 
    df2 = df2.T

    dict_abbreviated = {}
    for index_df2 in tqdm(range(df2.shape[0])): 
        abbreviated_mas = []
        for nd_df2 in df2.iloc[index_df2].values.reshape(200, 200):
            for truncated_array in nd_df2[:100]:
                abbreviated_mas.append(truncated_array)
        dict_abbreviated[index_df2] = abbreviated_mas

    df3 = pd.DataFrame(data=dict_abbreviated).T
    df3 = df3.iloc[:,0:10000]
    digits_df3 = df3.values
    
    return digits_df3
image_matrix_хах = image_shift(random_transformation_matrix[0])

image_matrix_хах_ydy1 = image_shift(random_transformation_matrix[1])
image_matrix_хах_ydy2 = image_shift(random_transformation_matrix[2])

image_revers_on_y = image_shift(random_transformation_matrix[3])
image_revers_on_x = image_shift(random_transformation_matrix[4])
image_revers_on_xy = image_shift(random_transformation_matrix[5])

image_shift_on_x = image_shift(random_transformation_matrix[6])
image_shift_on_y = image_shift(random_transformation_matrix[7])

image_Turn_on90 = image_shift(random_transformation_matrix[8])
image_Turn_on180 = image_shift(random_transformation_matrix[9])
plt.grid(False)
sns.set(rc={'figure.figsize': (10, 7)})

f1 = image_matrix_хах.reshape(100, 100)
plt.imshow(f1, cmap = 'gray', interpolation="nearest")
plt.subplot (121)
plt.grid(False)
f2 = image_matrix_хах_ydy1.reshape(100, 100)
plt.imshow(f2, cmap = 'gray', interpolation="nearest")

plt.subplot (122)
plt.grid(False)
f2 = image_matrix_хах_ydy2.reshape(100, 100)
plt.imshow(f2, cmap = 'gray', interpolation="nearest")
plt.subplot (131)
plt.grid(False)
f3 = image_revers_on_y.reshape(100, 100)
plt.imshow(f3, cmap = 'gray', interpolation="nearest")

plt.subplot (132)
plt.grid(False)
f4 = image_revers_on_x.reshape(100, 100)
plt.imshow(f4, cmap ='gray', interpolation="nearest")

plt.subplot (133)
plt.grid(False)
f5 = image_revers_on_xy.reshape(100, 100)
plt.imshow(f5, cmap ='gray', interpolation="nearest")
plt.subplot (121)
plt.grid(False)
f6 = image_shift_on_x.reshape(100, 100)
plt.imshow(f6, cmap ='gray', interpolation="nearest")

plt.subplot (122)
plt.grid(False)
f7 = image_shift_on_y.reshape(100, 100)
plt.imshow(f7, cmap ='gray', interpolation="nearest")
plt.subplot (121)
plt.grid(False)
f8 = image_Turn_on90.reshape(100, 100)
plt.imshow(f8, cmap ='gray', interpolation="nearest")

plt.subplot (122)
plt.grid(False)
f9 = image_Turn_on180.reshape(100, 100)
plt.imshow(f9, cmap ='gray', interpolation="nearest")