import pandas as pd
import numpy as np
from tqdm import tqdm
INPUT = '../input/unet34-submission-tta-0-699-new-public-lb/submission.csv'
OUTPUT = 'submission.csv'
def get_mask(img_id, df, shape = (768,768)):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    px = df.loc[img_id]['EncodedPixels']
    if(type(px) == float): return None
    elif(type(px) == str): px = [px]
    count = 1
    for mask in px:
        if(type(mask) == float):
            if len(px) == 1: return None
            else: continue
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            #keep previous prediction for overlapping pixels
            img[start:start+length] = count*(img[start:start+length] == 0)
        count+=1
    return img.reshape(shape).T

def decode_mask(mask, shape=(768, 768)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if(len(runs) == 0): return np.nan
    runs[runs > shape[0]*shape[1]] = shape[0]*shape[1]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def set_masks(mask):
    n = mask.max()
    result = []
    for i in range(1,n+1):
        result.append(decode_mask(mask == i))
    return result
pred_df = pd.read_csv(INPUT).set_index('ImageId')
pred_df.head()
names = list(set(pred_df.index))
ship_list_dict = []
for name in tqdm(names):
    mask = get_mask(name, pred_df)
    if (not isinstance(mask, np.ndarray) and mask == None) \
      or mask.sum() == 0:# or name in test_names_nothing:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    else:
        encodings = set_masks(mask)
        if(len(encodings) == 0):
            ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
            continue
        
        buf =[]
        for e in encodings:
            if e == e: buf.append(e)
        encodings = buf
        if len(encodings) == 0 : encodings = [np.nan]
        for encoding in encodings:
            ship_list_dict.append({'ImageId':name,'EncodedPixels':encoding})
pred_df_cor = pd.DataFrame(ship_list_dict)
pred_df_cor.to_csv(OUTPUT, index=False)
gr = pred_df_cor.groupby("ImageId")["EncodedPixels"].apply(lambda x: x.isnull().any() and len(x) > 1)  
print(gr.value_counts())  # should all be false  
print(gr[gr])  # these images have predictions and nan rows 