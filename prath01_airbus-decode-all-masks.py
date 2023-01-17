import numpy as np
import pandas as pd
train_segs = r"/kaggle/input/train_ship_segmentations_v2.csv"
def rle_decode(path_to_segments, shape = (768,768)):
    segs = pd.read_csv(path_to_segments, index_col = 0)
    segNames = np.unique(segs.index)
    allMasks = []
    for segName in segNames:
        masks = []
        segData = segs.loc[segName].dropna()
        if segData.shape[0] != 0:
            if segData.shape[0] > 1:
                obj = [np.array(each.replace(" ", ", ").split(', '), dtype = 'int') for each in segData.EncodedPixels]
            else:
                obj = [np.array(segData.EncodedPixels.replace(" ", ", ").split(', '), dtype = 'int')]
            for eachRLE in obj:
                rle_indexes = eachRLE.reshape((int(eachRLE.shape[0]/2), 2))
                template = np.arange(0,shape[0]*shape[1],1).reshape(shape).T
                for i, rle in enumerate(rle_indexes):
                    runItems = np.arange(rle[0], rle[0] + rle[1])
                    for runItem in runItems:
                        template[template == runItem] = -1
                masks.append((template == -1).astype('int'))
        else:
            masks.append(np.zeros((shape)))
            
        allMasks.append(np.array(masks).max(axis = 0))
    return np.array(allMasks)
        
#masks = rle_decode(train_segs)