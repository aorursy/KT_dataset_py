import dicom # read the dicom files

import os  # do directory operations

import pandas as pd  # nice for data analysis



#data_dir = '../input/sample_images/'

#patients = os.listdir(data_dir)
from subprocess import check_output

print(check_output(["ls","../"]).decode("utf8"))



# the stage1_labels.csv file disappeared 
os.listdir('../input')
data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)
patients[0:5]
# hand-made labels df

#labels=[{'id':'0c0de3749d4fe175b7a5098b060982a1','cancer':1},

#     {'id':'0a0c32c9e08cc2ea76a71649de56be6d','cancer':0},

#     {'id':'0c9d8314f9c69840e25febabb1229fa4','cancer':0},

#     {'id':'0d2fcf787026fece4e57be167d079383','cancer':0},

#     {'id':'0a38e7597ca26f9374f8ea2770ba870d','cancer':0},

#     {'id':'0bd0e3056cbf23a1cb7f0f0b18446068','cancer':0},

#     {'id':'0c37613214faddf8701ca41e6d43f56e','cancer':1},

#     {'id':'0c59313f52304e25d5a7dcf9877633b1','cancer':0},

#     {'id':'0b20184e0cd497028bdd155d9fb42dc9','cancer':0}, # I made this one up

#     {'id':'0c60f4b87afcb3e2dfa65abbbf3ef2f9','cancer':1},

#     {'id':'0d19f1c627df49eb223771c28548350e','cancer':0},

#     {'id':'0d941a3ad6c889ac451caf89c46cb92a','cancer':0},

#     {'id':'0d06d764d3c07572074d468b4cff954f','cancer':1},

#     {'id':'0ddeb08e9c97227853422bd71a2a695e','cancer':0},

#     {'id':'0acbebb8d463b4b9ca88cf38431aac69','cancer':1},

#     {'id':'0de72529c30fe642bc60dcb75c87f6bd','cancer':0},

#     {'id':'0ca943d821204ceb089510f836a367fd','cancer':0},

#     {'id':'0c98fcb55e3f36d0c2b6507f62f4c5f1','cancer':0},

#     {'id':'0a099f2549429d29b32f349e95fb2244','cancer':0},

#     {'id':'00cba091fa4ad62cc3200a657aeb957e','cancer':0}]
#labels_df = pd.DataFrame.from_dict(labels)
# this one no longer works since the input file cannot load

# it works again

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)
labels_df.head()
for patient in patients[:1]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(len(slices), label)

    print(slices[0])
slices[0].pixel_array
import matplotlib.pyplot as plt

import cv2

import numpy as np
IMG_PX_SIZE = 150



for patient in patients[:1]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    fig = plt.figure()

    for num,each_slice in enumerate(slices[:12]):

        y = fig.add_subplot(3,4,num+1)

        new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))

        y.imshow(new_img)

    

    plt.show()
patients[:1]  # I don't know why it looks a little different from OP
import math
# break down a list of items into sublists

def chunks(l,n):

    """Yield successive n-sized chunks from l"""

    for i in range(0, len(l),n):

        yield l[i:i+n]
def mean(l):

    return sum(l) / len(l)
IMG_PX_SIZE = 50

HM_SLICES = 20
for patient in patients:

    try:

        label = labels_df.get_value(patient,'cancer')

        path = data_dir + patient

        slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

        new_slices = []

        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

        chunk_sizes = math.ceil(len(slices) / HM_SLICES)

        for slice_chunk in chunks(slices, chunk_sizes):

            slice_chunk = list(map(mean, zip(*slice_chunk)))  # apply mean to every element in slice_chunk 

            new_slices.append(slice_chunk)

            

        print(len(slices),len(new_slices))

      

    except:

        # pass the patients who don't have labels

        pass
# deal with the problem of unequal 

for patient in patients[:20]:

    try:

        label = labels_df.get_value(patient,'cancer')

        path = data_dir + patient

        slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        new_slices = []

        

        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

        

        chunk_sizes = math.ceil(len(slices)/HM_SLICES)

        

        for slice_chunk in chunks(slices,chunk_sizes):

            slice_chunk = list(map(mean,zip(*slice_chunk)))

            new_slices.append(slice_chunk)

            

        if len(new_slices) == HM_SLICES-1:

            new_slices.append(new_slices[-1])

            

        if len(new_slices) == HM_SLICES-2:

            new_slices.append(new_slices[-1])

            new_slices.append(new_slices[-1])

            

        if len(new_slices) == HM_SLICES+2:

            new_val = list(map(mean,zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))

            del new_slices[HM_SLICES]

            new_slices[HM_SLICES-1] = new_val

            

        if len(new_slices) == HM_SLICES+1:

            new_val = list(map(mean,zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))

            del new_slices[HM_SLICES]

            new_slices[HM_SLICES-1] = new_val

                     

        print(len(slices),len(new_slices))

     

    except Exception as e:

        print(str(e))
labels_df.loc[:'0015ceb851d7251b8f399e39779d1e7d']
labels_df.loc['0b20184e0cd497028bdd155d9fb42dc9']

# There's no such lable in the csv file
patients[5]
#label = labels_df.get_value(patient, 'cancer')

path = data_dir + patients[5]

slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

fig = plt.figure()

for num,each_slice in enumerate(slices[:12]):

    y = fig.add_subplot(3,4,num+1)

    new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))

    y.imshow(new_img)



plt.show()

# seems like a "1"
labels_df.set_value('0b20184e0cd497028bdd155d9fb42dc9','cancer',1)
labels_df.loc['0b20184e0cd497028bdd155d9fb42dc9']

# now there's a value for the 6th patient
# deal with the problem of unequal 

for patient in patients[:20]:

    try:

        label = labels_df.get_value(patient,'cancer')

        path = data_dir + patient

        slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        new_slices = []

        

        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

        

        chunk_sizes = math.ceil(len(slices)/HM_SLICES)

        

        for slice_chunk in chunks(slices,chunk_sizes):

            slice_chunk = list(map(mean,zip(*slice_chunk)))

            new_slices.append(slice_chunk)

            

        if len(new_slices) == HM_SLICES-1:

            new_slices.append(new_slices[-1])

            

        if len(new_slices) == HM_SLICES-2:

            new_slices.append(new_slices[-1])

            new_slices.append(new_slices[-1])

            

        if len(new_slices) == HM_SLICES+2:

            new_val = list(map(mean,zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))

            del new_slices[HM_SLICES]

            new_slices[HM_SLICES-1] = new_val

            

        if len(new_slices) == HM_SLICES+1:

            new_val = list(map(mean,zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))

            del new_slices[HM_SLICES]

            new_slices[HM_SLICES-1] = new_val

                     

        print(len(slices),len(new_slices))

     

    except Exception as e:

        print(str(e))
def process_data(patient,labels_df,img_px_ize=50,hm_slices=20,visualize=False):

    

    label = labels_df.get_value(patient,'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    

    new_slices=[]

    slices = [cv.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]

    

    chunk_sizes = math.ceil(len(slices)/hm_slices)

    for slice_chunk in chunks(slices,chunk_sizes):

        slice_chunk = list(map(mean,zip(*slice_chunk)))

        

    if len(new_slices) == hm_slices-1:

        new_slices.append(new_slices[-1])

        

    if len(new_slices) == hm_slices-2:

        new_slices.append(new_slices[-1])

        new_slices.append(new_slices[-1])

        

    if len(new_slices) == hm_slices+2:

        new_val = lsit(map(mean,zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))

        new_slices[hm_slices-1] = new_val

        

    if visualize:

        fig = plt.figure()

        for num, each_slice in enumerate(new_slices):

            y = fig.add_subplot(4,5,num+1)

            y,imshow(each_slice, cmap='gray')

        plt.show()

        

    if label == 1: label+np.array([0,1])

    elif label == 0: label=np.array([1,0])

        

    return np.array(new_slices),label
import cv2