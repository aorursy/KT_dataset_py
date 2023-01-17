#  -------- Here we load required librations --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
%matplotlib inline
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

# ---- Loading one of the event for data exploration purposes--------
event_id = 'event000001002'
hits, cells, particles, truth =  load_event('../input/train_1/'+event_id)
# Function for coordinate conversion
def cart2spherical(cart):
    r = np.linalg.norm(cart, axis=0)
    theta = np.degrees(np.arccos(cart[2] / r))
    phi = np.degrees(np.arctan2(cart[1], cart[0]))
    return np.vstack((r, theta, phi))

# Convert to spherical coordinate.
xyz = hits.loc[:, ['x', 'y', 'z']].values.transpose()
rtp = cart2spherical(xyz).transpose()
rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))

fig=plt.figure(figsize=(30,10))
ax = fig.add_subplot(131, facecolor='black')
ax.scatter(rtp_df.theta, rtp_df.phi, s=1, color="green")
ax = fig.add_subplot(132, facecolor='black')
ax.scatter(rtp_df.theta, rtp_df.r, s=1, color="green")
ax = fig.add_subplot(133, facecolor='black')
ax.scatter(rtp_df.phi, rtp_df.r, s=1, color="green")
plt.show()
# Function for coordinate conversion
def cart2spherical(cart):
    r = np.linalg.norm(cart, axis=0)
    theta = np.degrees(np.arccos(cart[2] / r))
    phi = np.degrees(np.arctan2(cart[1], cart[0]))
    return np.vstack((r, theta, phi))

#=============================================================================================


# hits, truth, cells - dataframe formats as was specified in the input library
# density 1..100 percents. The initial density can be reduced into smaller subset os hits  
# wsize - subnet neural network window size in  spherical coordinats. Default - 128/128

def CreateSpace(hits, truth, cells, density=100, wsize=(255,255),space_size=(2048,2048)):

    
    print("start")
    rcells=cells.copy()
    rcells.drop_duplicates(subset="hit_id", keep='first', inplace=True)
    glvals = []
    for idx, hts in rcells.iterrows():
        current_hit_id=hts["hit_id"]
        gbl =  cells[cells.hit_id==current_hit_id]
        gml=gbl.where(gbl!=1,0.4)
        gvals = gml["value"]
        glvals.append(gvals.sum())
    hits['values']=glvals    
    print('Space creation...')
        
        
#    for idx, hts in rcells.iterrows():
#        current_hit_id=hts["hit_id"]
#        gbl =  cells[cells.hit_id==current_hit_id]
#        rgbl =  cells[(cells.hit_id==current_hit_id)&(cells.value!=1)]
        #gml=gbl.where(gbl!=1,0.08)
        #gml=gbl
        #vals = gbl["value"]
        #rvals = rgbl["value"]
        #gvals = gml["value"]
        
    
    # making a work copy of hits
    ehits=hits.copy()
    print(ehits.head())
    
    # window size determination
    wsize_x,wsize_y = wsize
    # Evenness check
    if wsize_x % 2 == 0: return # Window has to have odd size in order to have a explisit centre
    if wsize_y % 2 == 0: return
    
    #----- Add spherical coordinates ------------
    xyz = ehits.loc[:, ['x', 'y', 'z']].values.transpose()
    rtp = cart2spherical(xyz).transpose()
    rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))
    ehits['r']  = rtp_df['r']
    ehits['theta']  =rtp_df['theta']
    ehits['phi']  =rtp_df['phi']
    ehits['particle_id']=truth['particle_id']
    
    #  ---- Preparing volume/layers dataframe ---- 
    volume_list=[]
    layer_list=[]
    total_layers=0
    for volume in range (1,19):
        for layer in range (1,15):
            local_df = ehits[(ehits.volume_id==volume)&(ehits.layer_id==layer)]
            # finding a row cound in the filtered dataframe
            count = local_df["r"].count()
            if count!=0:
                total_layers=total_layers+1;
                volume_list.append(volume)
                layer_list.append(layer)
    
    vl_df = pd.DataFrame() 
    vl_df['volumes'] = volume_list
    vl_df['layers'] = layer_list
    
    #  ----- NumPy memory allocation ----------------------
    space_x, space_y = space_size # actual space representation size x
    space_z = total_layers # actual space representation size x
    # Now we need to add some space for padding,
    # later x direction will be populated with zero padding
    # y direction will be populated with circuit padding
    full_space_x = space_x + wsize_x # full array x size with padding in place 
    full_space_y = space_y + wsize_y # full array y size with padding in place 
    a = np.zeros(shape=(full_space_x,full_space_y,space_z), dtype=np.int8)
    
    # Special technological array for easier focus window manipulation
    a_tech = np.zeros(shape=(space_x,space_y), dtype=np.int8)     
    
    # Particle ID array (in order to define particle_id based)
    ap = np.zeros(shape=(full_space_x,full_space_y), dtype=np.int64)
    
    #  ---- Here we reduce the number of hits based on 'density'
    # Filter required number of trajectories
    trajectories = truth.drop_duplicates('particle_id')
    full_size = len(trajectories)
    trajectories = trajectories.drop_duplicates('particle_id')
    if density!=100:
        trajectories = trajectories.sample(int(density*(full_size/100.0)))
    trajectories = trajectories['particle_id']
    trajectories = trajectories.values
    # make addition coloumn with "in play: flag
    in_play_list=[]
    for _, hit in ehits.iterrows():
        part_id=hit['particle_id']
        if part_id in trajectories:
            in_play_list.append(True)
        else:
            in_play_list.append(False)
    ehits['in_play']=in_play_list        
    
    collision_count=0
    #  ---- Here we create 3d space representation  depending also on "in play" field --------
    for z in range (0, space_z):
        v = vl_df.iloc[z]['volumes']
        l = vl_df.iloc[z]['layers']
        l_hits=ehits[(ehits.volume_id==v)&(ehits.layer_id==l)&(ehits.in_play==True)]
        # ----- Convertion hits into space array format --------
        for i, hit in l_hits.iterrows():
            x_idx = int (((hit['phi']+180) /360.0) *space_x)
            y_idx = int ((hit['theta'] /180.0) *space_y)
            z_idx = z
            a[int(wsize_x/2) +x_idx , int(wsize_y/2) + y_idx,z_idx]=255 # intensity in this version is always maximum
            
            a_tech[x_idx,y_idx] = 255 # Technological array
            
            #Collision detection
            #if ap[int(wsize_x/2) +x_idx , int(wsize_y/2) + y_idx] !=0: 
            #    if ap[int(wsize_x/2) +x_idx , int(wsize_y/2) + y_idx] != hit['particle_id']:
            #        collision_count=collision_count+1
                
            # Adding a technological array with partice ids information    
            ap[int(wsize_x/2) +x_idx , int(wsize_y/2) + y_idx] = hit['particle_id']
            
            # Adding space coordinated into common ehits dataframe
            ehits.loc[i,'x_idx']=x_idx
            ehits.loc[i,'y_idx']=y_idx
            ehits.loc[i,'z_idx']=z_idx
            
    #print(collision_count)
    
    return ehits, a, a_tech, ap       

#=============================================================================================


# fsize - focus window size
# max_i - maximum number of iterations (attempts to find a pivot point)
def GeneratePivot(ehits, a_tech, fsize =(25,25), space_size=(2048,2048), max_i=1000):
    space_x, space_y = space_size # actual space representation size x
    #  ----------- Pivot point genaration ------------------------------------------
    # Here we generate a pivot point is 2D coorditates
    random.seed()
    found = False
    for i in range(0,max_i):
        if found == True: break
        x_pivot = random.randint(0,space_x-1)
        y_pivot = random.randint(0,space_y-1)
        # Now we need to extend one pivot point to a focus window, because a probablility of not targeting a point is relatevelly high
        # Usually a fucus window is small (<=30), otherwise this could be computantially expensive
        # From otrher side the focus window can be very small because this will dramatically increase number of iteration on
        # outer computation layer
        fsize_x, fsize_y =  fsize
        fx_start = x_pivot - int(fsize_x / 2)
        fy_start = y_pivot - int(fsize_y / 2)
        if fx_start < 0: fx_start=0
        if fy_start < 0: fy_start=0
        fx_end = x_pivot + int(fsize_x / 2)
        fy_end = y_pivot + int(fsize_y / 2)
        if fx_end > (space_x-1) : fx_end = space_x-1
        if fy_end > (space_y-1) : fy_end = space_y-1
        # Now we start iterating over a focus window trying to find non-black point
        # checking if x_pivot and y_pivot correspond to any particle
        for xf in range (fx_start,fx_end+1):
            if found == True: break
            for yf in range (fy_start,fy_end+1):
                if a_tech[xf,yf]!=0: 
                    x_pivot = xf
                    y_pivot = yf
                    found = True
                    break
                
    return  x_pivot, y_pivot, found    


#=============================================================================================

def GenerateInputInstance(x_pivot, y_pivot, ehits, a, wsize=(255,255)):
    
    # Window size determination
    wsize_x, wsize_y = wsize 
    # Evenness check
    if wsize_x % 2 == 0: return # Window has to have odd size in order to have a explisit centre
    if wsize_y % 2 == 0: return
    # Define window corners
    wstart_x = x_pivot -  int(wsize_x/2)
    wstart_y = y_pivot -  int(wsize_y/2)
    wend_x = x_pivot +  int(wsize_x/2) # inclusevely
    wend_y = y_pivot +  int(wsize_y/2) # inclusevely
    
    # Correcting to coordinates with padding
    wstart_x =  wstart_x + int(wsize_x/2)
    wstart_y = wstart_y +  int(wsize_y/2)
    wend_x = wend_x +  int(wsize_x/2) # inclusevely
    wend_y = wend_y +  int(wsize_y/2) # inclusevely
    
    #print(wstart_x,wstart_y,wend_x,wend_y)
    
    # Resulting array
    a_res = a[wstart_x:wend_x+1,wstart_y:wend_y+1]
    at_res = np.zeros(shape=wsize, dtype=np.int8) 
    
    #print('Start')
    
    # For testing purposes, will need to be deleted after valudation due to computation reasons 
#    for il in range (0,48):
        #print("ZLayer - "+str(il))
#        for ix in range (0,wsize_x):
#            for iy in range (0,wsize_y):
#                if a_res[ix,iy,il]!=0:
#                    print(ix,iy,il)
#                    at_res[ix,iy] = a_res[ix,iy,il]
    #print('End')
    
    return a_res


#=============================================================================================


# The function retrives a particle ID assotiated with a pivot point

def GetParticleID(x_pivot, y_pivot, ap, wsize=(255,255)):

    # Window size determination
    wsize_x, wsize_y = wsize 
    
    # Note 
    # Here some collision effect is possible when several particles belong to same
    # pivot point in 2d space, in this case only one particle_id is return
    # later more sophisticated collision avoidance algorithm might need to be implemented
    # most probably a priority based on layer ID will need to be implemented in CreateSpace function
    particle_id = ap[int(wsize_x/2)+x_pivot,int(wsize_y/2)+y_pivot]
    return particle_id  


#=============================================================================================


# This function generated a true label based on provided pivot point and true particle traectory information

# The true label has the following format (NUMPY ARRAY INT8):

# [0... (wsize_x-1), 0...(wsize_y-1), 0...(48-1)]

# Overall size is wsize_x+wsize_y + 48 bytes

def GenerateLabel(x_pivot,y_pivot,ehits,ap,wsize):

    # Window size determination
    wsize_x, wsize_y = wsize 

    # Evenness check
    if wsize_x % 2 == 0: return # Window has to have odd size in order to have a explisit centre
    if wsize_y % 2 == 0: return

    # Define window corners
    wstart_x = x_pivot -  int(wsize_x/2)
    wstart_y = y_pivot -  int(wsize_y/2)
    

    # Cet a particle ID in a centre point
    pid = GetParticleID(x_pivot,y_pivot,ap,wsize)
    thits = ehits[(ehits.particle_id==pid)&(ehits.in_play==True)]
    
    
    label_array = np.zeros(shape=(wsize_x+wsize_y + 48),dtype=np.int8)

    # ----- Convertion hits into space array format --------
    for _, hit in thits.iterrows():
                x_idx = hit['x_idx']
                y_idx = hit['y_idx']
                z_idx = hit['z_idx']    

                #print(x_idx,y_idx,z_idx)
                
                # conver to local window coordinated
                lx_idx = int(x_idx - wstart_x)
                ly_idx = int(y_idx - wstart_y)
                lz_idx = int(z_idx)

                #print(lx_idx,ly_idx,lz_idx)

                #print(x_idx,wstart_x, lx_idx)
                
                #Convertion to lable
                if((lx_idx>=0) & (lx_idx<wsize_x)):
                    label_array[lx_idx] = 1
                if((ly_idx>=0) & (ly_idx<wsize_y)):
                    label_array[wsize_x + ly_idx] = 1
                label_array[wsize_x+wsize_y+lz_idx] = 1
 
    #print(label_array.size)
    return label_array

#=============================================================================================

def IfPointInlabel(x,y,z,label,wsize):

    # Window size determination
    wsize_x, wsize_y = wsize 
    
    # Local labels
    xlabel = label[0: wsize_x ]
    ylabel = label[wsize_x : wsize_x + wsize_y]
    zlabel = label[wsize_x + wsize_y:wsize_x + wsize_y+48]
    
    res = ((xlabel[x]==1)&(ylabel[y]==1)&(zlabel[z]==1))
    
    return res

#=============================================================================================

def IfPointInlabel(x,y,z,label,wsize):

    # Window size determination
    wsize_x, wsize_y = wsize 
    
    # Local labels
    xlabel = label[0: wsize_x ]
    ylabel = label[wsize_x : wsize_x + wsize_y]
    zlabel = label[wsize_x + wsize_y:wsize_x + wsize_y+48]
    
    res = ((xlabel[x]==1)&(ylabel[y]==1)&(zlabel[z]==1))
    
    return res

#=============================================================================================


def SaveLabeledInstance(start_number,number, ehits, a, a_tech, ap):
    for i in range(start_number,start_number+number):
        x_pivot,y_pivot, found = GeneratePivot(ehits, a_tech)
        InputInstance = GenerateInputInstance(x_pivot,y_pivot,ehits,a)
        label = GenerateLabel(x_pivot,y_pivot,ehits=ehits,ap=ap,wsize=(255,255))
        #np.save('INPUT/255_255_1P_NIP_NI/'+'input'+str(i), InputInstance)
        #np.save('INPUT/255_255_1P_NIP_NI/'+'label'+str(i), label)
        
        np.save('input'+str(i), InputInstance)
        np.save('label'+str(i), label)
    None
event_id = 'event000001000'
hits, cells, particles, truth = load_event('../input/train_1/'+event_id)
ehits, a, a_tech, ap = CreateSpace(hits=hits, truth=truth, cells=cells, density=1, wsize=(255,255),space_size=(2048,2048))

SaveLabeledInstance(0,1000, ehits, a, a_tech, ap)
import keras
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Conv2D
from keras.models import Model
import numpy as np
import gc
#from keras.utils import plot_modell
import random
import matplotlib.pyplot as plt
import time

ZIPPED = False

BATCHSIZE = 5


def GetNextBatch(dir = None, minfn=0, maxfn=10000):
    arraylist=[]
    labellist=[]
    for i in range(0,BATCHSIZE):
        fidx = np.random.randint(minfn,maxfn)
        
        if ZIPPED==False:
            cadr = np.load(dir+'input'+str(fidx)+'.npy').copy()
            label = np.load(dir+'label'+str(fidx)+'.npy').copy()
        if ZIPPED==True:
            loaded = np.load(dir+'data_label_'+str(fidx)+'.npz')
            cadr = loaded['data'].copy()
            label = loaded['label'].copy()
        
        arraylist.append(cadr)
        array = np.stack(arraylist, axis=0)

        labellist.append(label)
        labels = np.stack(labellist, axis=0)

    return array,labels
from keras.layers import Dropout

K.clear_session()
gc.collect()
# -----------------------------------------------
def Acc(y_true, y_pred):
    y_pred=K.round(y_pred)
    res = K.mean(K.square(y_pred - y_true), axis=-1)
    res = K.mean(res, axis=-1)
    res=1-res
    res=res*100
    return res
# -----------------------------------------------
def LossFunction(y_true, y_pred):
    res = K.mean(K.square(y_pred - y_true), axis=-1)
    res = K.mean(res, axis=-1)
    return res

# -------------- Model definition -------------------------------------------------------------------------------------------
inputs1 = Input(shape=(255,255,48))
inputs2 = Input(shape=(48,255,255))

# ----------

x1 = Conv2D(filters = 32, kernel_size = (6,6), strides =(3,3),
                 activation ='relu')(inputs1)
x1 = Flatten()(x1)

# ----------

x2 = Conv2D(filters = 32, kernel_size = (6,6), strides =(3,3),
                 activation ='relu')(inputs2)
x2 = Flatten()(x2)

# ----------

x = keras.layers.concatenate([x1, x2])

x = Dense(558, activation ='relu')(x) 

predictions = Dense(558, activation='sigmoid')(x)


#  ----------- Model compilation -------------------------------------------------------------------------------------------- 

model = Model(inputs=[inputs1,inputs2], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss=LossFunction,
              metrics=[Acc])

# ------------- Model summary print ------------------------------------------------------------------------------------------
model.summary()

tick=time.time()


# ------------------ Test batch generation -----------------------------------------------------------------------------------
#test_batch_input,test_batch_labels  =  GetNextBatch(dir = 'D:/KAGGLE/CERN/INPUT/255_255_100P_IP_I/',minfn=90000, maxfn = 100000)
#test_batch_input1, test_batch_labels  =  GetNextBatch(dir = 'INPUT/255_255_100P_IP_I/',minfn=90000, maxfn = 100000)
test_batch_input1, test_batch_labels  =  GetNextBatch(dir = '',minfn=900, maxfn = 1000)
test_batch_input2 = []
for i in range(0,BATCHSIZE): test_batch_input2.append(test_batch_input1[i].transpose())
test_batch_input2 = np.stack(test_batch_input2, axis=0)

# -------- Model saving and logging parameters -------------------------------------------------------------------------------
ms_counter = 0
testname = 'Test46_Old'
test_acc_list = []
test_true_acc_list = []
train_acc_list = []


# -----------------  Main calculation cycle ----------------------------------------------------------------------------------
for i in range(0,10):
    #---- Printing -------------------------
    print('Step'+str(i))
    # ---- Saving --------------------------
    ms_counter = ms_counter + 1
    if ms_counter>=50:
        ms_counter = 0
        model.save(testname +'_model_'+str(i)+'.mdl')
    # ---- Train batch genaration ----------
#    batch_input,batch_labels  =  GetNextBatch(dir = 'D:/KAGGLE/CERN/INPUT/255_255_100P_IP_I/', minfn=10000, maxfn = 90000)

    #batch_input1, batch_labels  =  GetNextBatch(dir = 'INPUT/255_255_100P_IP_I/', minfn=10000, maxfn = 90000)
    batch_input1, batch_labels  =  GetNextBatch(dir = '', minfn=0, maxfn = 900)
    
    batch_input2 = []
    for i in range(0,BATCHSIZE): batch_input2.append(batch_input1[i].transpose())
    batch_input2 = np.stack(batch_input2, axis=0)
    
    process = model.fit([batch_input1,batch_input2], batch_labels, epochs = 5, 
                        validation_data=([test_batch_input1,test_batch_input2],test_batch_labels))
    #                  batch_size=8) 
    
    acc = process.history["Acc"]
    val_acc = process.history["val_Acc"]
    plt.plot(acc)
    plt.plot(val_acc)
    plt.show()
    
    # - Validation histogram calculation -
    mean_train = np.mean(acc) 
    mean_test = np.mean(val_acc) 
    print('Mean train acc - ', mean_train)
    print('Mean test acc - ', mean_test)
    train_acc_list.append(mean_train)
    test_acc_list.append(mean_test)
    test_true_acc_list.append(acc[0])
    
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.plot(test_true_acc_list)
    
    plt.show()
    
    plt.plot(test_acc_list)
    plt.show()    
    
    tack = time.time()
    
    print("Time from the test start, h: ", (tack-tick)/3600)