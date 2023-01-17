import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
ROOT_DIR = '../input/'
#Load data

train = pd.read_json(ROOT_DIR+"train.json")

print("done!")
train.inc_angle = train.inc_angle.replace('na', 0)

train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

print("done!")
for i in range(10):

    print(' Iceberg : ',i,' ',train['is_iceberg'][i])

print("done!")
def TRTBAND(ID):

    

    AR = abs(x_band3x[ID] * np.sin(train["inc_angle"][ID]))

    

    MAXI = np.max(AR)

    MINI = np.min(AR)

    MED = np.mean(AR)

    STD = np.std(AR)

    

    # Init MAXi et MINi

    MAX1 = MAX2 = MAX3 = MAX4 = MED + 2 * STD

    MIN1 = MIN2 = MIN3 = MIN4 = MED - 2 * STD

    

    # Recherche plus grand MIN et plus petit MAX sur ligne

    for i in range(AR.shape[0]):

        MIN = np.min(AR[i])

        MAX = np.max(AR[i])

        if MAX < MAX1:

            MAX1 = MAX

        if MIN > MIN1:

            MIN1 = MIN

            

    # Recherche plus grand MIN et plus petit MAX sur colonne

    for i in range(AR[:].shape[0]):        

        MIN = np.min(AR[:][i])

        MAX = np.max(AR[:][i])

        if MAX < MAX2:

            MAX2 = MAX

        if MIN > MIN2:

            MIN2 = MIN

            

    # Recherche inversee

    for i in range(AR.shape[0]-1,0,-1):

        MIN = np.min(AR[i])

        MAX = np.max(AR[i])

        if MAX < MAX3:

            MAX3 = MAX

        if MIN > MIN3:

            MIN3 = MIN

            

    for i in range(AR[:].shape[0]-1,0,-1):

        MIN = np.min(AR[:][i])

        MAX = np.max(AR[:][i])

        if MAX < MAX4:

            MAX4 = MAX

        if MIN > MIN4:

            MIN4 = MIN

            

    print(ID," Info image : MAXI=",MAXI," MINI=",MINI," MED=",MED," STD=",STD)

    print(ID," Info recherche : MAXi :",MAX1,MAX2,MAX3,MAX4)

    print(ID," Info recherche : MINi :",MIN1,MIN2,MIN3,MIN4)

    

    MIN = np.max([MIN1,MIN2,MIN3,MIN4])

    MAX = np.min([MAX1,MAX2,MAX3,MAX4])

    #MIN = MED - 2 * STD

    #MAX = MED + 2 * STD

    print(ID,MIN,MAX,train['is_iceberg'][ID])



    plt.imshow(AR)

    plt.show()   



    # Ecreter ligne :

    # - SI [MIN, MAX]  --> 0

    # - SI ]-inf, MIN[ --> MIN

    # - SI ]MAX, +inf[ --> MAX

    for i in range(AR.shape[0]):

        for j in range(AR[i].shape[0]):

            if AR[i][j] >= MIN and AR[i][j] <= MAX:

                AR[i][j] = 0

            else:

                if AR[i][j] < MIN:

                    AR[i][j] = MIN

                else:

                    if AR[i][j] > MAX:

                        AR[i][j] = MAX

                        

    # On elimine les pixels isoles

    for i in range(AR.shape[0]):                    

        for j in range(AR[i].shape[0]):

            if j == 0:

                if AR[i][j+1] == 0 or AR[i][j+2] == 0:

                    AR[i][j] = 0

                continue

            if j == AR.shape[0]-1:

                if AR[i][j-1] == 0 or AR[i][j-2] == 0:

                    AR[i][j] = 0

                continue

            if j < AR.shape[0]-2:

                if (AR[i][j-1] == 0 and AR[i][j+1] == 0) or (AR[i][j-2] == 0 and AR[i][j+2] == 0):

                    AR[i][j] = 0                    

                    

    plt.imshow(AR)

    plt.show()



    # Ecreter colonne

    for j in range(AR.shape[1]):                   

        for i in range(AR[:][j].shape[0]):

            if i == 0:

                if AR[i+1][j] == 0:

                    AR[i][j] = 0

                continue

            if i == AR.shape[0]-1:

                if AR[i-1][j] == 0:

                    AR[i][j] = 0

                continue

            if i < AR.shape[0]-1:

                if (AR[i-1][j] == 0 and AR[i+1][j] == 0):

                    AR[i][j] = 0                

    

    plt.imshow(AR)

    plt.show()        
x_band1x = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])

x_band2x = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

x_band3x = (x_band1x+x_band2x)/2



for ID in range(5):

    TRTBAND(ID)