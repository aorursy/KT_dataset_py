import pandas as pd
import glob
from fastai.vision import *
import os
path = '../input/severstal-steel-defect-detection/'
path_train = '../input/severstal-steel-defect-detection/train_images/'
path_test = '../input/severstal-steel-defect-detection/test_images/'
df = pd.read_csv(path + 'train.csv')
df = df[:100]
df.head()
hist = df['ClassId'].hist(bins=4)
df_2 = df.groupby(['ImageId']).agg({'ClassId': lambda x: list(x),'EncodedPixels': lambda x: list(x)})
df_2.head()
df_2['ImageId'] = df_2.index
df_2.head()
df_2 = df_2.reset_index(drop=True)
df_2.head()
df.columns.tolist()
df_2 = df_2[['ImageId', 'ClassId', 'EncodedPixels']]
df_2.head()
def func(x, y):
    a = ['','','','']
    for i, j in zip(x, y):
        a[i-1] = j
        
    return a      

df_2['EncodedPixels'] = df_2.apply(lambda x: func(x['ClassId'], x['EncodedPixels']), axis=1)
df_2.head(10)
img = open_image(path_train + df_2['ImageId'][5])
img.show(figsize=(20,10))
print(img.size)
def rle_decode_3(mask_rle:str, shape=(256, 1600))->NPArrayMask:
    "Return an image array from run-length encoded string `mask_rle` with `shape`."
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1    # Отнимаем от всех стартовых значений 1 т.к. индексация с 0.
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint)
    for a, b in zip(starts, ends): img[a:b] = 1
    img = img.reshape(shape, order='F')
    return img
def open_mask_rle_2(mask_rle:str, shape=(256, 1600))->ImageSegment:
    "Return `ImageSegment` object create from run-length encoded string in `mask_lre` with size in `shape`."
    x = FloatTensor(rle_decode_3(str(mask_rle), shape).astype(np.uint8))
    x = x.view(-1, shape[0], shape[1])
    #return ImageSegment(x)
    return x
def rle_encode_3(img:NPArrayMask, shape=(256, 1600))->str:
    "Return run-length encoding string from `img`."
    pixels = np.concatenate([[0], img.flatten(order='F') , [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

df_2['EncodedPixels'][1]
rle_decode_3(df_2['EncodedPixels'][1][2])
ImageSegment(open_mask_rle_2(df_2['EncodedPixels'][1][2]))
mask = rle_decode_3(df_2['EncodedPixels'][1][2])
type(mask)
rle_encode_3(mask)
fn = path_train + df_2['ImageId'][5]
img = open_image(fn)
img
mask = open_mask_rle_2(df_2['EncodedPixels'].iloc[5][3])
ImageSegment(mask)
mask = open_mask_rle_2(df_2['EncodedPixels'].iloc[5][2])
ImageSegment(mask)
df_2[df_2['ImageId'] == 'db4867ee8.jpg']
df_2['EncodedPixels'].iloc[40]
df_2[df_2['ImageId'] == '000f6bf48.jpg']['EncodedPixels'].item()
for i in df_2[df_2['ImageId'] == '000f6bf48.jpg']['EncodedPixels']:
    print(i)
#fn = path_train + df_2.index['db4867ee8.jpg']
fn = path_train + '000f6bf48.jpg'
#print(fn)
img = open_image(fn)
shape = img.shape[-2:]
#print(shape)
#img = open_image(fn)
final_mask = torch.zeros((1, *shape))
#for i, rle in enumerate(df_2['EncodedPixels'].iloc[5]):
#for i, rle in enumerate(df_2['EncodedPixels'].loc['db4867ee8.jpg']):  
for i, rle in enumerate(df_2[df_2['ImageId'] == '000f6bf48.jpg']['EncodedPixels'].item()):
    #print(rle)
    if isinstance(rle, str):
        mask = open_mask_rle_2(rle)
        #print(mask.shape)
        final_mask += (i + 1) * mask
#mask = open_mask_rle_2(df_train['EncodedPixels'].iloc[0])
final_mask_2 = ImageSegment(final_mask)
_,axs = plt.subplots(3,1, figsize=(20,10))
img.show(ax=axs[0], title='no mask')
img.show(ax=axs[1], y=final_mask_2, title='masked')
final_mask_2.show(ax=axs[2], title='mask only', alpha=1.)
def get_y_fn(fn):
    #print(fn)
    #fn = fn.replace(path_train, '')
    x = df_train[df_train['ImageId'] == fn]['EncodedPixels'].item()
    #print('!')
    return open_mask_rle_2(x)
def get_y_fn_mcl(fn):
    #print(fn)
    final_mask = torch.zeros((1, 256, 1600))
    for i, rle in enumerate(df_2[df_2['ImageId'] == fn]['EncodedPixels'].item()):
    #for i, rle in enumerate(df_2['EncodedPixels'].loc[fn]):    
        if isinstance(rle, str):
            mask = open_mask_rle_2(rle)
            #print('mask=', mask.shape)
            final_mask += (i + 1) * mask
    return ImageSegment(final_mask)
get_y_fn_mcl('000f6bf48.jpg')
class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return get_y_fn_mcl(fn)
    
class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList
path_train
df_2.head()
train_list = (SegmentationItemList
                .from_df(df_2, path_train))
#codes = ['0','1']
train_list = (SegmentationItemList
                .from_df(df_2, path_train)
                .split_by_rand_pct()
                .label_from_df(cols='ImageId', label_cls=SegmentationLabelList, classes=[0, 1, 2, 3, 4])
                #.transform(get_transforms(), size=256,resize_method=ResizeMethod.SQUISH, tfm_y=True)
                .transform(get_transforms(), size=(128, 800),resize_method=ResizeMethod.SQUISH, tfm_y=True)
                #.transform(get_transforms(flip_vert=True), tfm_y=True)
                .databunch(bs=10, num_workers=10))
type(train_list)
train_list.show_batch(rows=3, figsize=(20,10))
def dice_func(input, target):
    smooth = 0
    input = input[:,1,:,:]
    iflat = input.flatten().float()
    tflat = target.flatten().float()
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return intersect / (union-intersect+1.0)
#learn = unet_learner(train_list, models.resnet18, pretrained=False, metrics=[dice], wd=1e-3, model_dir="/tmp/model/")
learn = unet_learner(train_list, models.resnet18, pretrained=False, metrics=[dice], wd=1e-3, model_dir='/kaggle/working/models')
lr_find(learn)
learn.recorder.plot()
learn.fit_one_cycle(10,max_lr = 1e-2)    # 40
lr_find(learn)
learn.recorder.plot()
learn.fit_one_cycle(20,max_lr = slice(1e-3,1e-2))    # 40
learn.unfreeze()
learn.fit_one_cycle(20, max_lr = 1e-3)
#learn.unfreeze()
learn.fit_one_cycle(20, max_lr = 1e-4) 
#learn.fit_one_cycle(20, max_lr = 1e-4)
#learn.fit_one_cycle(20, max_lr = 1e-5)
#learn.save('stage-1')
#learn.export("/kaggle/working/steel-1.pkl")
learn.save('trained_model_1')
learn.export("/kaggle/working/trained_model_1.pkl")
learn = load_learner('/kaggle/working/', 'trained_model_1.pkl')
learn = unet_learner(train_list, models.resnet18, pretrained=False, metrics=[dice], wd=1e-3, model_dir='/kaggle/working/models')
learn = learn.load("trained_model_1")
learn.fit_one_cycle(10,max_lr = 1e-5)    # 40
learn.recorder.plot_lr(show_moms=True)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.show_results()
learn.lr_find()
learn.recorder.plot()
submit = pd.read_csv(path + 'sample_submission.csv', converters={'EncodedPixels': lambda e: ' '})
print(len(submit))
submit.sort_values(by=['ImageId'])
submit.head()
learn.model.cuda()
files = os.listdir(path=path_test)
#a = learn.predict(open_image(path_test + '86c1f219f.jpg'))[1].data.numpy()
a = rle_decode_3(df_2['EncodedPixels'][1][2])
b = a.flatten(order='F')
d = {1: '', 2: '', 3: '', 4: ''}
for start, count in zip (rle_encode_3(a).split(' ')[::2], rle_encode_3(a).split(' ')[1::2]):
    #print(start)
    #print(b[int(start)-1])
    d[b[int(start)-1]] += str(start) + ' ' +  str(count) + ' '
d
d[3] = '294661 251 294917 251 295173 251 295429 251 295685 251 295941 251 296197 251 296453 251 296709 251 296965 251 297221 251 297477 251'

d
img_name = '86c1f219f.jpg'
sub_list = []
for i in d:
    sub_list += [[img_name, i, d[i]]]
sub_list
sub_list = []
for img_name in files:
    #print(img_name)
    pred = learn.predict(open_image(path_test + img_name))[1].data.numpy()
    print(Imagese)
    #print(pred)
    pred_fl = pred.flatten(order='F')
    rle_enc = rle_encode_3(pred).split(' ')
    #print(rle_enc)
    d = {1: '', 2: '', 3: '', 4: ''}
    for start, count in zip (rle_enc[::2], rle_enc[1::2]):
        #print(start)
        #print(pred_fl[int(start)-1])
        d[pred_fl[int(start)-1]] += str(start) + ' ' +  str(count) + ' '
        
    for i in d:
        sub_list += [[img_name, i, d[i]]]

df_2['EncodedPixels'].iloc[0]
df_2['EncodedPixels'].iloc[0][0]
pred = rle_decode_3(df_2['EncodedPixels'].iloc[0][0])
pred_fl = pred.flatten(order='F')
rle_enc = rle_encode_3(pred).split(' ')
rle_enc
d = {1: '', 2: '', 3: '', 4: ''}
for start, count in zip (rle_enc[::2], rle_enc[1::2]):
    print(start)
    print(pred_fl[int(start)-1])
    d[pred_fl[int(start)-1]] += str(start) + ' ' +  str(count) + ' '
d
pred = learn.predict(open_image(path_test + '289d347d9.jpg'))[1].data.numpy()
pred_fl = pred.flatten(order='F')
rle_enc = rle_encode_3(pred).split(' ')

rle_enc
rle_enc
pred_fl[55153+5]
pred_fl[2740-1]
pred_fl[2616-1]
sub_list
submission_df = pd.DataFrame(sub_list, columns=['ImageId', 'EncodedPixels', 'ClassId'])
submission_df.head()
d
b = np.where(a != 0)
b
a[0][38][612]
test_count = len(files)
results = []
def run_length(label_vec):
    encode_list = encode(label_vec)
    index = 1
    class_dict = {}
    for i in encode_list:
        if i[1] != len(codes)-1:
            if i[1] not in class_dict.keys():
                class_dict[i[1]] = []
            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
        index += i[0]
    return class_dict
from itertools import groupby
def encode(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]
codes = ['0','1','2','3', '4'] 
learn.model.cuda()
from tqdm import tqdm
for i, img in tqdm(enumerate(files)):
    img_name = img
    pred = learn.predict(open_image(path_test + img))[1].data.numpy().flatten()
    class_dict = run_length(pred)
    print(class_dict)

for i, img in tqdm(enumerate(files)):
    img_name = img
    pred = learn.predict(open_image(path_test + img))[1].data.numpy().flatten()
    class_dict = run_length(pred)
    if len(class_dict) == 0:
        for i in range(4):
            results.append([img_name+ "_" + str(i+1), ''])
    else:
        for key, val in class_dict.items():
            results.append([img_name + "_" + str(key+1), " ".join(map(str, val))])
        for i in range(4):
            if i not in class_dict.keys():
                results.append([img_name + "_" + str(i+1), ''])
        
        
    if i%20==0:
        print("\r{}/{}".format(i, test_count), end="")
results
files = list(path_test.glob("**/*.jpg"))
def get_predictions(path_test, learn):
    # predicts = get_predictions(path_test, learn)
    learn.model.cuda()
    files = list(path_test.glob("**/*.jpg"))    #<---------- HERE
    test_count = len(files)
    results = []
    for i, img in enumerate(files):
        img_name = img.stem + '.jpg'
        pred = learn.predict(open_image(img))[1].data.numpy().flatten()
        class_dict = run_length(pred)
        if len(class_dict) == 0:
            for i in range(4):
                results.append([img_name+ "_" + str(i+1), ''])
        else:
            for key, val in class_dict.items():
                results.append([img_name + "_" + str(key+1), " ".join(map(str, val))])
            for i in range(4):
                if i not in class_dict.keys():
                    results.append([img_name + "_" + str(i+1), ''])
        
        
        if i%20==0:
            print("\r{}/{}".format(i, test_count), end="")
    return results    

sub_list = get_predictions(path_test, learn)
test_list = (SegmentationItemList
                .from_df(df_2, path_train)
                .split_by_rand_pct()
                .label_from_df(cols='ImageId', label_cls=SegmentationLabelList, classes=[0, 1, 2, 3, 4])
                .transform(get_transforms(), size=256,resize_method=ResizeMethod.SQUISH, tfm_y=True)
                #.transform(get_transforms(flip_vert=True), tfm_y=True)
                .databunch(bs=10, num_workers=10))