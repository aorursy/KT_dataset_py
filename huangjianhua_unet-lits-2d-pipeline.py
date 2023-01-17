# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
This script is used to generate slicewise training data
"""


"""=============================================="""
"""========== COPMUTE WEIGHTMAP ================="""
"""=============================================="""
def find_borders(liver_mask, lesion_mask, res, width=5):
    struct_elem     = np.ones([int(np.clip(width/res[i],2,None)) for i in range(len(liver_mask.shape))])

    ### Locate pixels around liver boundaries
    outer_border = ndi.binary_dilation(liver_mask, struct_elem).astype(int)-liver_mask
    inner_border = liver_mask-ndi.binary_erosion(liver_mask, struct_elem).astype(int)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    ### Generate a weightmap putting weights on pixels close to boundaries
    weight_liv = 0.3
    boundary_weightmap_liver = 0.6*np.clip(np.exp(-0.1*(ndi.morphology.distance_transform_edt(1-total_border))),weight_liv,None)+0.4*liver_mask

    ### Locate pixels around lesion boundaries
    outer_border = ndi.binary_dilation(lesion_mask, struct_elem).astype(int)-lesion_mask
    inner_border = lesion_mask-ndi.binary_erosion(lesion_mask, struct_elem).astype(int)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    ### Generate a weightmap putting weights on pixels close to boundaries
    weight_les = 0.5
    boundary_weightmap_lesion = 0.75*np.clip(np.exp(-0.09*(ndi.morphology.distance_transform_edt(1-total_border))),weight_les,None)+0.25*lesion_mask
    boundary_weightmap_liver  = 0.65*boundary_weightmap_liver + 0.35*boundary_weightmap_lesion

    return boundary_weightmap_liver.astype(np.float16), boundary_weightmap_lesion.astype(np.float16)






"""=============================================="""
"""========== MAIN GENERATION FILE =============="""
"""=============================================="""
def main(opt):
    if not os.path.exists(opt.save_path_4_training_slices): os.makedirs(opt.save_path_4_training_slices)

    assign_file_v    = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_Volumes.csv",    ["Volume","Slice Path"])
    if not opt.is_test_data:
        assign_file_les  = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LesionMasks.csv",["Volume","Slice Path","Has Mask"])
        assign_file_liv  = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LiverMasks.csv", ["Volume","Slice Path","Has Mask"])
        assign_file_wliv = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LiverWmaps.csv", ["Volume","Slice Path"])
        assign_file_wles = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LesionWmaps.csv",["Volume","Slice Path"])

    volumes  = os.listdir(opt.path_2_training_volumes)
    segs, vols = [],[]

    for x in volumes:
        if 'segmentation' in x and not opt.is_test_data: segs.append(x)
        if 'volume' in x: vols.append(x)

    vols.sort()
    segs.sort()

    if not os.path.exists(opt.save_path_4_training_slices):
        os.makedirs(opt.save_path_4_training_slices)


    if not opt.is_test_data:
        volume_iterator = tqdm(zip(vols, segs), position=0, total=len(vols))
    else:
        volume_iterator = tqdm(vols, position=0)


    if opt.is_test_data:
        volume_info = {}


    for i,data_tuple in enumerate(volume_iterator):
        ### ASSIGNING RELEVANT VARIABLES
        if not opt.is_test_data:
            vol, seg = data_tuple
        else:
            vol = data_tuple

        ### LOAD VOLUME AND MASK DATA
        volume_iterator.set_description('Loading Data...')

        volume      = nib.load(opt.path_2_training_volumes+"/"+vol)
        v_name      = vol.split(".")[0]
        res = volume.header.structarr['pixdim'][1:4][[2,0,1]]

        if opt.is_test_data:
            header, affine = volume.header, volume.affine
            volume_info[v_name] = {'header':header, 'affine':affine}

        volume         = np.array(volume.dataobj)
        volume         = volume.transpose(2,0,1)
        save_path_v    = opt.save_path_4_training_slices+"/Volumes/"+v_name
        if not os.path.exists(save_path_v): os.makedirs(save_path_v)

        if not opt.is_test_data:
            segmentation  = np.array(nib.load(opt.path_2_training_volumes+"/"+seg).dataobj)
            segmentation  = segmentation.transpose(2,0,1)

            save_path_lesion_masks        = opt.save_path_4_training_slices+"/LesionMasks/"+v_name
            save_path_liver_masks         = opt.save_path_4_training_slices+"/LiverMasks/"+v_name
            save_path_lesion_weightmaps   = opt.save_path_4_training_slices+"/BoundaryMasksLesion/"+v_name
            save_path_liver_weightmaps    = opt.save_path_4_training_slices+"/BoundaryMasksLiver/"+v_name

            if not os.path.exists(save_path_lesion_masks):     os.makedirs(save_path_lesion_masks)
            if not os.path.exists(save_path_liver_masks):      os.makedirs(save_path_liver_masks)
            if not os.path.exists(save_path_lesion_weightmaps):os.makedirs(save_path_lesion_weightmaps)
            if not os.path.exists(save_path_liver_weightmaps): os.makedirs(save_path_liver_weightmaps)




        if not opt.is_test_data:
            volume_iterator.set_description('Generating Masks...')
            liver_mask, lesion_mask = segmentation>=1,segmentation==2
            volume_iterator.set_description('Generating Weightmaps...')
            weightmap_liver, weightmap_lesion = find_borders(liver_mask, lesion_mask, res)
            volume_slice_iterator = tqdm(zip(volume, liver_mask.astype(np.uint8), lesion_mask.astype(np.uint8), weightmap_liver, weightmap_lesion), position=1, total=len(volume))
        else:
            volume_slice_iterator = tqdm(volume, position=1)


        volume_iterator.set_description('Saving Slices...')

        for idx,data_tuple in enumerate(volume_slice_iterator):
            if not opt.is_test_data:
                (v_slice, liv_slice, les_slice, livmap_slice, lesmap_slice) = data_tuple


                np.save(save_path_liver_weightmaps +"/slice-"+str(idx)+".npy", livmap_slice)
                np.save(save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy", lesmap_slice)
                np.save(save_path_lesion_masks+"/slice-"+str(idx)+".npy", les_slice.astype(np.uint8))
                np.save(save_path_liver_masks +"/slice-"+str(idx)+".npy", liv_slice.astype(np.uint8))

                assign_file_wliv.write([v_name, save_path_liver_weightmaps +"/slice-"+str(idx)+".npy"])
                assign_file_wles.write([v_name, save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy"])
                assign_file_les.write([v_name, save_path_lesion_masks+"/slice-"+str(idx)+".npy", 1 in les_slice.astype(np.uint8)])
                assign_file_liv.write([v_name, save_path_liver_masks +"/slice-"+str(idx)+".npy", 1 in liv_slice.astype(np.uint8)])
            else:
                v_slice = data_tuple

            np.save(save_path_v+"/slice-"+str(idx)+".npy", v_slice.astype(np.int16))
            assign_file_v.write([v_name, save_path_v+"/slice-"+str(idx)+".npy"])

        if opt.is_test_data: pkl.dump(volume_info, open(opt.save_path_4_training_slices+'/volume_nii_info.pkl','wb'))










"""=========================================="""
"""========== ______MAIN______ =============="""
"""=========================================="""
if __name__ == '__main__':

    """=================================="""
    ### LOAD BASIC LIBRARIES
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, os, sys, nibabel as nib, argparse, pickle as pkl
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))
#     sys.path.insert(0, os.getcwd()+'/../Utilities')

#   os.getcwd() 方法用于返回当前工作目录。
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Utilities')
    import General_Utilities as gu
    from tqdm import tqdm
    import scipy.ndimage as ndi


    """===================================="""
    ### GET PATHS
    #Read network and training setup from text file.
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--path_2_training_volumes',     type=str, default='placeholder',
                          help='Path to original LiTS-volumes in nii-format.')
    parse_in.add_argument('--save_path_4_training_slices', type=str, default='placeholder',
                          help='Where to save the 2D-conversion.')
    parse_in.add_argument('--is_test_data', action='store_true',
                          help='Flag to mark if input data is test data or not.')
#     opt = parse_in.parse_args()
    opt = parse_in.parse_known_args()[0]


    """===================================="""
    ### RUN GENERATION
    if 'placeholder' in opt.path_2_training_volumes:
        pt = 'Test_Data' if opt.is_test_data else 'Training_Data'
#         opt.path_2_training_volumes     = os.getcwd()+'/../OriginalData/'+pt
        opt.path_2_training_volumes = '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Original Data/'+pt

    if 'placeholder' in opt.save_path_4_training_slices:
        pt = 'Test_Data_2D' if opt.is_test_data else 'Training_Data_2D'
        # pt = 'Test_Data_2D' if opt.is_test_data else 'Training_Data_2D'
        
#         opt.save_path_4_training_slices = os.getcwd()+'/../LOADDATA/'+pt
        opt.save_path_4_training_slices = './LOADDATA/'+pt

    """===================================="""
    ### RUN GENERATION
    print(os.getcwd())
    main(opt)
# Based on: Expandable script for Lesion Segmentation Base_Unet_Template.py.
# This Variant: Liver-Segmentation
# @author: Karsten Roth - Heidelberg University, 07/11/2017
"""==================================================================================================="""
"""======================= MAIN TRAINING FUNCTION/ALL FUNCTIONALITIES ================================"""
"""==================================================================================================="""
import os

def main(opt):
    """======================================================================================="""
    ### SET SOME DEFAULT PATHS
    if 'placeholder' in opt.Paths['Training_Path']:
        opt.Paths['Training_Path'] = './LOADDATA/Training_Data_2D'
    if 'placeholder' in opt.Paths['Save_Path']:
        foldername                 = 'Standard_Liver_Networks' if opt.Training['data']=='liver' else 'Standard_Lesion_Networks'
        opt.Paths['Save_Path']     = os.getcwd()+'/../SAVEDATA/'+foldername


    """======================================================================================="""
    ### REPRODUCIBILITY
    torch.manual_seed(opt.Training['seed'])
    torch.cuda.manual_seed(opt.Training['seed'])
    np.random.seed(opt.Training['seed'])
    random.seed(opt.Training['seed'])
    torch.backends.cudnn.deterministic = True



    """======================================================================================="""
    ### GPU SETUP
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.Training['gpu'])
    opt.device = torch.device('cuda')




    """======================================================================================="""
    if len(opt.Training['initialization']):
        try:
            ### NETWORK SETUP IF INITIALIZATION IS USED
            init_path   = opt.Training['initialization'] if not 'placeholder' in opt.Training['initialization'] else os.getcwd()+'/..'+opt.Training['initialization'].split('placeholder')[-1]
            network_opt = pkl.load(open(init_path+'/hypa.pkl','rb'))
            network     = netlib.NetworkSelect(network_opt)
            opt.Network = network_opt.Network
        except:
            raise Exception('Error when loading initialization weights! Please make sure that weights exist at {}!'.format(init_path))

    ### LOSS SETUP
    base_loss_func      = nu.Loss_Provider(opt)
    aux_loss_func       = nu.Loss_Provider(opt) if opt.Network['use_auxiliary_inputs'] else None
    opt.Training['use_weightmaps']  = base_loss_func.loss_func.require_weightmaps
    opt.Training['require_one_hot'] = base_loss_func.loss_func.require_one_hot
    opt.Training['num_out_classes'] = 1 if base_loss_func.loss_func.require_single_channel_input else opt.Training['num_classes']


    if len(opt.Training['initialization']):
        try:
            ### ONLY LOAD FEATURE WEIGHTS; FINAL LAYER IS SET UP ACCORDING TO USED LOSS
            checkpoint = torch.load(init_path+'/checkpoint_best_val.pth.tar')
            network.load_state_dict(checkpoint['network_state_dict'])
            if network.output_conv[0].out_channels != opt.Training['num_out_classes']:
                network.output_conv = torch.nn.Sequential(torch.nn.Conv2d(network.output_conv[0].in_channels, opt.Training['num_out_classes'], network.output_conv[0].kernel_size, network.output_conv[0].stride, network.output_conv[0].padding),
                                                          torch.nn.Sigmoid() if opt.Training['num_out_classes']==1 else torch.nn.Softmax(dim=1))

                if opt.Network['use_auxiliary_inputs']:
                    for i in range(len(network.auxiliary_preparators)):
                        in_channels = network.auxiliary_preparators[i].get_aux_output.in_channels
                        kernel_size = network.auxiliary_preparators[i].get_aux_output.kernel_size
                        stride      = network.auxiliary_preparators[i].get_aux_output.stride
                        network.auxiliary_preparators[i].get_aux_output = torch.nn.Conv2d(in_channels, opt.Training['num_out_classes'], kernel_size, stride)
                        network.auxiliary_preparators[i].out_act        = torch.nn.Sigmoid() if opt.Training['num_out_classes']==1 else torch.nn.Softmax(dim=1)
            del checkpoint
        except:
            raise Exception('Error when loading initialization weights! Please make sure that weights exist at {}!'.format(init_path))
    else:
        ### NETWORK SETUP WITHOUT INITIALIZATION
        network = netlib.NetworkSelect(opt)

    network.n_params = nu.gimme_params(network)
    opt.Network['Network_name'] = network.name
    _ = network.to(opt.device)



    """======================================================================================="""
    ### OPTIMIZER SETUP
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['l2_reg'])
    if isinstance(opt.Training['step_size'], list):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.Training['step_size'], gamma=opt.Training['gamma'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.Training['step_size'], gamma=opt.Training['gamma'])



    """======================================================================================="""
    ### TRAINING LOGGING SETUP
    # Set Logging Folder and Save Parameters
    imp.reload(gu)
    gu.logging_setup(opt)
    # Set Logging Dicts
    logging_keys    = ["Train Dice", "Train Loss", "Val Dice"]
    Metrics         = {key:[] for key in logging_keys}
    Metrics['Best Val Dice'] = 0
    # Set CSV Logger
    full_log  = gu.CSVlogger(opt.Paths['Save_Path']+"/log.csv", ["Epoch", "Time", "Training Loss", "Training Dice", "Validation Dice"])



    """======================================================================================="""
    ### TRAINING DATALOADER SETUP
    imp.reload(Data)
    train_dataset, val_dataset = Data.Generate_Required_Datasets(opt)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['num_workers'], batch_size=opt.Training['batch_size'], pin_memory=False, shuffle=True)
    val_data_loader   = torch.utils.data.DataLoader(val_dataset,   num_workers=0, batch_size=1, shuffle=False)





    """======================================================================================="""
    ### START TRAINING
    full_training_start_time = time.time()
    epoch_iter = trange(0,opt.Training['n_epochs'],position=1)
    has_crop   = opt.Training['data']=='lesion'


    for epoch in epoch_iter:
        scheduler.step()
        epoch_iter.set_description("(#{}) Training [lr={}]".format(network.n_params, np.round(scheduler.get_lr(),8)))

        epoch_time = time.time()

        ###### Training ########
        flib.trainer([network,optimizer], train_data_loader, [base_loss_func, aux_loss_func], opt, Metrics, epoch)
        torch.cuda.empty_cache()


        ###### Validation #########
        epoch_iter.set_description('(#{}) Validating...'.format(network.n_params))
        flib.validator(network, val_data_loader, opt, Metrics, epoch)
        torch.cuda.empty_cache()


        ###### Save Training/Best Validation Checkpoint #####
        save_dict = {'epoch': epoch+1, 'network_state_dict':network.state_dict(), 'current_train_time': time.time()-full_training_start_time,
                     'optim_state_dict':optimizer.state_dict(), 'scheduler_state_dict':scheduler.state_dict()}
        # Best Validation Score
        if Metrics['Val Dice'][-1]>Metrics['Best Val Dice']:
            torch.save(save_dict, opt.Paths['Save_Path']+'/checkpoint_best_val.pth.tar')
            Metrics['Best Val Dice'] = Metrics['Val Dice'][-1]
            gu.generate_example_plots_2D(network, train_dataset, val_dataset, opt, has_crop=has_crop, name_append='best_val_dice', n_plots=20, seeds=[111,2222])

        # After Epoch
        torch.save(save_dict, opt.Paths['Save_Path']+'/checkpoint.pth.tar')


        ###### Logging Epoch Data ######
        epoch_iter.set_description('Logging to csv...')
        full_log.write([epoch, time.time()-epoch_time, Metrics["Train Loss"][-1], Metrics["Train Dice"][-1], Metrics["Val Dice"][-1]])


        ###### Generating Summary Plots #######
        epoch_iter.set_description('Generating Summary Plots...')
        sum_title = 'Max Train Dice: {0:2.3f} | Max Val Dice: {1:2.3f}'.format(np.max(Metrics["Train Dice"]), np.max(Metrics["Val Dice"]))
        gu.progress_plotter(np.arange(len(Metrics['Train Loss'])), \
                            Metrics["Train Loss"],Metrics["Train Dice"],Metrics["Val Dice"],
                            opt.Paths['Save_Path']+'/training_results.svg', sum_title)

        _ = gc.collect()

        ###### Generating Sample Plots #######
        epoch_iter.set_description('Generating Sample Plots...')
        gu.generate_example_plots_2D(network, train_dataset, val_dataset, opt, has_crop=has_crop, name_append='end_of_epoch', n_plots=20, seeds=[111,2222])
        torch.cuda.empty_cache()




"""==================================================================================================="""
"""============================= ___________MAIN_____________ ========================================"""
"""==================================================================================================="""
if __name__ == '__main__':


    """===================================="""
    ### LOAD BASIC LIBRARIES
    import warnings
    warnings.filterwarnings("ignore")

    import os,json,sys,gc,time,datetime,imp,argparse
    from tqdm import tqdm, trange
    import torch, random
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Utilities')
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Network_Zoo')
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Train_Networks')
    
    import numpy as np, matplotlib, pickle as pkl
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    import network_zoo as netlib

    import General_Utilities as gu
    import Network_Utilities as nu

    import PyTorch_Datasets as Data
    import Function_Library as flib


    """===================================="""
    ### GET TRAINING SETUPs ###
    #Read network and training setup from text file.
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--base_setup',   type=str, default='Baseline_Parameters.txt',
                                            help='Path to baseline setup-txt which contains all major parameters that most likely will be kept constant during various grid searches.')
    parse_in.add_argument('--search_setup', type=str, default='',
                                            help='Path to search setup-txt, which contains (multiple) variations to the baseline proposed above.')
    parse_in.add_argument('--no_date',      action='store_true', help='Do not use date when logging files.')
    opt = parse_in.parse_args(['--search_setup','Small_UNet_Lesion.txt'])
#     opt = parse_in.parse_args()
    opt = parse_in.parse_known_args()[0]

#     assert opt.search_setup!='', 'Please provide a Variation-Parameter Text File!'


    opt.base_setup   = '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Train_Networks/Training_Setup_Files/'+opt.base_setup
    print(opt.base_setup)
#     print(opt.search_setup)
#     opt.search_setup = '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Train_Networks/Training_Setup_Files/'+opt.search_setup
#     print(opt.search_setup)
    
    training_setups = gu.extract_setup_info(opt)

    for training_setup in tqdm(training_setups, desc='Setup Iteration... ', position=0):
        main(training_setup)
"""
This script is used to generate slicewise training data
@author:Karsten Roth - Heidelberg University, 07/11/2017
"""


"""=============================================="""
"""========== COPMUTE WEIGHTMAP ================="""
"""=============================================="""
def find_borders(liver_mask, lesion_mask, save_path_liver, save_path_lesion, width=5):
    struct_elem     = np.ones([width,width])

    ### Locate pixels around liver boundaries
    ndi.binary_dilation(liver_mask, struct_elem)
    outer_border = ndi.binary_dilation(liver_mask, struct_elem).astype(int)-liver_mask
    inner_border = liver_mask-ndi.binary_erosion(liver_mask, struct_elem).astype(int)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    ### Generate a weightmap putting weights on pixels close to boundaries
    weight = 1/np.sqrt(3)
    boundary_weightmap_liver = 0.75*np.clip(np.exp(-0.02*(ndi.morphology.distance_transform_edt(1-total_border))),weight,None)+0.25*liver_mask


    ### Locate pixels around lesion boundaries
    ndi.binary_dilation(lesion_mask, struct_elem)
    outer_border = ndi.binary_dilation(lesion_mask, struct_elem).astype(int)-lesion_mask
    inner_border = lesion_mask-ndi.binary_erosion(lesion_mask, struct_elem).astype(int)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    ### Generate a weightmap putting weights on pixels close to boundaries
    weight = 1/np.sqrt(3)
    boundary_weightmap_lesion = 0.75*np.clip(np.exp(-0.02*(ndi.morphology.distance_transform_edt(1-total_border))),weight,None)+0.25*lesion_mask
    boundary_weightmap_liver  = 0.65*boundary_weightmap_liver + 0.35*boundary_weightmap_lesion


    np.save(save_path_liver, boundary_weightmap_liver.astype(np.float16))
    np.save(save_path_lesion,boundary_weightmap_lesion.astype(np.float16))







"""=============================================="""
"""========== MAIN GENERATION FILE =============="""
"""=============================================="""
def main(opt):
    if not os.path.exists(opt.save_path_4_training_slices): os.makedirs(opt.save_path_4_training_slices)

    assign_file_v    = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_Volumes.csv",    ["Volume","Slice Path"])
    if not opt.is_test_data:
        assign_file_les  = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LesionMasks.csv",["Volume","Slice Path","Has Mask"])
        assign_file_liv  = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LiverMasks.csv", ["Volume","Slice Path","Has Mask"])
        assign_file_wliv = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LiverWmaps.csv", ["Volume","Slice Path"])
        assign_file_wles = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LesionWmaps.csv",["Volume","Slice Path"])

    volumes  = os.listdir(opt.path_2_training_volumes)
    segs, vols = [],[]

    for x in volumes:
        if 'segmentation' in x and not opt.is_test_data: segs.append(x)
        if 'volume' in x: vols.append(x)

    vols.sort()
    segs.sort()


    if not os.path.exists(opt.save_path_4_training_slices):
        os.makedirs(opt.save_path_4_training_slices)


    if not opt.is_test_data:
        volume_iterator = tqdm(zip(vols, segs), position=0, total=len(vols))
    else:
        volume_iterator = tqdm(vols, position=0)


    if opt.is_test_data:
        volume_info = {}

    for i,data_tuple in enumerate(volume_iterator):
        ### ASSIGNING RELEVANT VARIABLES
        if not opt.is_test_data:
            vol, seg = data_tuple
        else:
            vol = data_tuple

        ### LOAD VOLUME AND MASK DATA
        volume_iterator.set_description('Loading Data...')

        volume      = nib.load(opt.path_2_training_volumes+"/"+vol)
        v_name      = vol.split(".")[0]

        if opt.is_test_data:
            header, affine = volume.header, volume.affine
            volume_info[v_name] = {'header':header, 'affine':affine}

        volume         = np.array(volume.dataobj)
        volume         = volume.transpose(2,0,1)
        save_path_v    = opt.save_path_4_training_slices+"/Volumes/"+v_name
        if not os.path.exists(save_path_v): os.makedirs(save_path_v)

        if not opt.is_test_data:
            segmentation  = np.array(nib.load(opt.path_2_training_volumes+"/"+seg).dataobj)
            segmentation  = segmentation.transpose(2,0,1)

            save_path_lesion_masks        = opt.save_path_4_training_slices+"/LesionMasks/"+v_name
            save_path_liver_masks         = opt.save_path_4_training_slices+"/LiverMasks/"+v_name
            save_path_lesion_weightmaps   = opt.save_path_4_training_slices+"/BoundaryMasksLesion/"+v_name
            save_path_liver_weightmaps    = opt.save_path_4_training_slices+"/BoundaryMasksLiver/"+v_name

            if not os.path.exists(save_path_lesion_masks):     os.makedirs(save_path_lesion_masks)
            if not os.path.exists(save_path_liver_masks):      os.makedirs(save_path_liver_masks)
            if not os.path.exists(save_path_lesion_weightmaps):os.makedirs(save_path_lesion_weightmaps)
            if not os.path.exists(save_path_liver_weightmaps): os.makedirs(save_path_liver_weightmaps)


        volume_iterator.set_description('Generating Weightmaps and saving slices...')

        if not opt.is_test_data:
            volume_slice_iterator = tqdm(zip(volume, segmentation), position=1, total=len(volume))
        else:
            volume_slice_iterator = tqdm(volume, position=1)


        for idx,data_tuple in enumerate(volume_slice_iterator):

            if not opt.is_test_data:
                (v_slice, s_slice) = data_tuple
                liver_mask  = s_slice>=1
                lesion_mask = s_slice==2

                find_borders(liver_mask, lesion_mask, save_path_liver_weightmaps +"/slice-"+str(idx)+".npy", save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy")
                assign_file_wles.write([v_name, save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy"])
                assign_file_wliv.write([v_name, save_path_liver_weightmaps +"/slice-"+str(idx)+".npy"])

                np.save(save_path_lesion_masks+"/slice-"+str(idx)+".npy", lesion_mask.astype(np.uint8))
                np.save(save_path_liver_masks +"/slice-"+str(idx)+".npy", liver_mask.astype(np.uint8))
                assign_file_les.write([v_name, save_path_lesion_masks+"/slice-"+str(idx)+".npy", 1 in lesion_mask.astype(np.uint8)])
                assign_file_liv.write([v_name, save_path_liver_masks +"/slice-"+str(idx)+".npy", 1 in liver_mask.astype(np.uint8)])
            else:
                v_slice = data_tuple

            np.save(save_path_v+"/slice-"+str(idx)+".npy", v_slice.astype(np.int16))
            assign_file_v.write([v_name, save_path_v+"/slice-"+str(idx)+".npy"])

        if opt.is_test_data: pkl.dump(volume_info, open(opt.save_path_4_training_slices+'/volume_nii_info.pkl','wb'))










"""=========================================="""
"""========== ______MAIN______ =============="""
"""=========================================="""
if __name__ == '__main__':

    """=================================="""
    ### LOAD BASIC LIBRARIES
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, os, sys, nibabel as nib, argparse, pickle as pkl
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))    
    sys.path.insert(0,'../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Utilities')
    import General_Utilities as gu
    from tqdm import tqdm
    import scipy.ndimage as ndi


    """===================================="""
    ### GET PATHS
    #Read network and training setup from text file.
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--path_2_training_volumes',     type=str, default='placeholder',
                          help='Path to original LiTS-volumes in nii-format.')
    parse_in.add_argument('--save_path_4_training_slices', type=str, default='placeholder',
                          help='Where to save the 2D-conversion.')
    parse_in.add_argument('--is_test_data', action='store_true',
                          help='Flag to mark if input data is test data or not.')
    opt = parse_in.parse_known_args()[0]


    """===================================="""
    ### RUN GENERATION
    if 'placeholder' in opt.path_2_training_volumes:
        pt = 'Test_Data' if opt.is_test_data else 'Training_Data'
        opt.path_2_training_volumes     = '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Original Data/'+pt
    if 'placeholder' in opt.save_path_4_training_slices:
        pt = 'Test_Data_2D' if opt.is_test_data else 'Training_Data_2D'
        # pt = 'Test_Data_2D' if opt.is_test_data else 'Training_Data_2D'
        opt.save_path_4_training_slices = './LOADDATA/'+pt


    """===================================="""
    ### RUN GENERATION
    main(opt)

# @author: Karsten Roth - Heidelberg University, 07/11/2017


"""======================================="""
"""========= TEST DATASET ================"""
"""======================================="""
import torch.utils.data as data
class TestDataset(data.Dataset):
    def __init__(self, test_data_folder, opt, channel_size=1):
        self.pars = opt
        self.test_volumes = [x for x in os.listdir(test_data_folder+'/Volumes')]
        # self.test_volumes = sorted([x for x in os.listdir(test_data_folder)])
        ### Choose specific volumes
        # self.test_volumes = ['test-volume-42', 'test-volume-5']
        self.test_volume_slices = {key:[test_data_folder+'/Volumes/'+key+'/'+x for x in sorted(os.listdir(test_data_folder+'/Volumes/'+key),key=lambda x: [int(y) for y in x.split('-')[-1].split('.')[0].split('_')])] for key in self.test_volumes}
        self.channel_size = channel_size

        self.iter_data, slice_cluster_collect = [],[]
        for vol in self.test_volumes:
            for i in range(len(self.test_volume_slices[vol])):
                extra_ch  = self.channel_size//2
                low_bound = np.clip(i-extra_ch,0,None).astype(int)
                low_diff  = extra_ch-i
                up_bound  = np.clip(i+extra_ch+1,None,len(self.test_volume_slices[vol])).astype(int)
                up_diff   = i+extra_ch+1-len(self.test_volume_slices[vol])

                vol_slices = self.test_volume_slices[vol][low_bound:up_bound]

                if low_diff>0:
                    extra_slices    = self.test_volume_slices[vol][low_bound+1:low_bound+1+low_diff][::-1]
                    vol_slices      = extra_slices+vol_slices
                if up_diff>0:
                    extra_slices    = self.test_volume_slices[vol][up_bound-up_diff-1:up_bound-1][::-1]
                    vol_slices      = vol_slices+extra_slices

                slice_cluster_collect.append(vol_slices)
                self.iter_data.append((vol,i))

            self.test_volume_slices[vol] = slice_cluster_collect
            slice_cluster_collect = []


        self.n_files = len(self.iter_data)
        self.vol_slice_idx = 0
        self.curr_vol = 0
        self.curr_vol_name = self.test_volumes[0]

    def __getitem__(self, idx):
        VOI, SOI = self.iter_data[idx]
        V2O  = np.concatenate([np.expand_dims(np.load(vol),0) for vol in self.test_volume_slices[VOI][SOI]],axis=0)
        if self.pars.Training['no_standardize']:
            V2O  = gu.normalize(V2O, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            V2O  = gu.normalize(V2O)

        if self.vol_slice_idx==len(self.test_volume_slices[self.test_volumes[self.curr_vol]])-1:
            self.vol_slice_idx = 0
            self.curr_vol_name = self.test_volumes[self.curr_vol]
            self.curr_vol     += 1
            return_data = {'VolSlice':V2O,'end_volume':True}
        else:
            return_data = {'VolSlice':V2O,'end_volume':False}
            self.vol_slice_idx+=1
        return return_data

    def __len__(self):
        return self.n_files




"""======================================="""
"""============= MAIN FUNCTIONALITY ======"""
"""======================================="""
def main(opt):
    ############ Init Network ###################
    network_list = []
    for i,network_setup in enumerate(opt.networks_to_use):
        network_opt = pkl.load(open(network_setup+'/hypa.pkl','rb'))
        network     = netlib.Scaffold_UNet(network_opt)
        checkpoint  = torch.load(network_setup+'/checkpoint_best_val.pth.tar')
        network.load_state_dict(checkpoint['network_state_dict'])
        network_list.append({'network':network, 'settings':network_opt})
        del network, network_opt, checkpoint

    back_device = torch.device('cpu')
    up_device   = torch.device('cuda')


    ############# Set Dataloader ##################
    max_channels    = np.max([item['settings'].Network['channels'] for item in network_list])
    test_dataset    = TestDataset(opt.test_data, network_list[0]['settings'], channel_size=max_channels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    input_slices, vol_segs, volume_info, volume_info_2 = [],[],{},{}
    n_vols = len(test_dataloader.dataset.test_volumes)
    vol_count = 1

    data_iter = tqdm(test_dataloader, position=0)


    ############# Run Test Mask Generation ##################
    for idx,data in enumerate(data_iter):
        ### Getting input slices ###
        data_iter.set_description('Reading... [Vol {}/{}]'.format(vol_count, n_vols))
        input_slices.append(data['VolSlice'])
        end_volume = data['end_volume'].numpy()[0]
        if end_volume:
            prev_vol = test_dataloader.dataset.test_volumes[test_dataloader.dataset.curr_vol-1]
            ### Computing Segmentation ###
            with torch.no_grad():
                vol_segs = np.zeros([512,512,len(input_slices)])
                for net_count,net_dict in enumerate(network_list):

                    network = net_dict['network']
                    network.to(up_device)

                    data_iter.set_description('Segmenting  [Vol {}/{} | Net {}/{}]'.format(vol_count, n_vols, net_count+1, len(opt.networks_to_use)))
                    for slice_idx, input_slice in enumerate(tqdm(input_slices, desc='Running Segmentation...', position=1)):

                        n_ch = net_dict['settings'].Network['channels']
                        input_slice = input_slice[:,max_channels//2-n_ch//2:max_channels//2+n_ch//2+1,:]
                        input_slice = input_slice.type(torch.FloatTensor).to(up_device)
                        seg_pred    = network(input_slice)[0]

                        vol_segs[:,:,slice_idx] += seg_pred.detach().cpu().numpy()[0,-1,:]
                        #if seg_pred.size()[1]==2:
                        #    seg_pred = np.argmax(seg_pred.detach().cpu().numpy(),axis=1)
                        #else:
                        #    seg_pred = seg_pred.detach().cpu().numpy()[0,:]
                        #vol_segs.append(seg_pred)

                    ### Moving Network back to cpu
                    network.to(back_device)
                    torch.cuda.empty_cache()

                ### Finding Biggest Connected Component
                data_iter.set_description('Finding CC... [Vol {}/{} | Net {}/{}]'.format(vol_count, n_vols, net_count+1, len(opt.networks_to_use)))
                #vol_segs = np.vstack(vol_segs).transpose(1,2,0)
                labels   = snm.label(np.round(vol_segs/len(opt.networks_to_use)))
                #labels   = snm.label(np.round(vol_segs/len(opt.networks_to_use)))
                MainCC   = labels[0]==np.argmax(np.bincount(labels[0].flat)[1:])+1
                ### (Optional) Remove thin connections to, most likely, noisy segmentations
                # MainCC = snmo.binary_erosion(MainCC, np.ones((4,4,4)))
                # labels   = snm.label(MainCC)
                # MainCC = labels[0]==np.argmax(np.bincount(labels[0].flat)[1:])+1
                # del labels

                ### (Optional) Resize CC and close minor segmentation holes
                MainCC = snmo.binary_dilation(MainCC, np.ones((5,5,3)))
                MainCC = snmo.binary_erosion(MainCC, np.ones((2,2,3)))

                ### Extract Mask Coordinates for Lesion Segmentation to reduce computations
                MainCC    = MainCC.astype(np.uint8)
                mask_info = np.where(MainCC==1)
                mask_info = {axis_idx:(np.min(x), np.max(x)) for axis_idx,x in enumerate(mask_info)}
                volume_info[prev_vol+'.nii'] = mask_info

                ### Saving Liver Segmentation
                data_iter.set_description('Saving... [Vol {}/{}]'.format(vol_count, n_vols))
                vol_save = opt.save_folder+'/'+prev_vol
                if not os.path.exists(vol_save):
                    os.makedirs(vol_save)

                for slice_i in trange(MainCC.shape[-1], desc='Saving to npy slice...', position=1):
                    np.save(vol_save+'/slice-'+str(slice_i)+'.npy', MainCC[:,:,slice_i])

                pkl.dump(volume_info, open(opt.save_folder+'/liver_bound_dict.pkl','wb'))

                ### Reseting Parameters
                del MainCC
                vol_segs, input_slices = [],[]
                vol_count+=1
                #data_iter.set_description('Completed. [Vol {}/{}]'.format(vol_count, n_vols))








if __name__ == '__main__':
    """======================================="""
    ### LOAD BASIC LIBRARIES
    import os,json,sys,gc,time,datetime,imp,argparse,copy
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Utilities')
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Network_Zoo')

    import numpy as np, pickle as pkl, nibabel as nib
    from tqdm import tqdm,trange

    import pandas as pd
    from collections import OrderedDict
    import scipy.ndimage.measurements as snm
    import scipy.ndimage.morphology as snmo

    import network_zoo as netlib
    import General_Utilities as gu, Network_Utilities as nu

    import PyTorch_Datasets as Data

    import torch
    from torch.utils.data import DataLoader

    ###NOTE: This line is necessary in case of the "too many open files"-Error!
    if int(torch.__version__.split(".")[1])>2:
        torch.multiprocessing.set_sharing_strategy('file_system')


    """=========================================="""
    ### GET INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_folder",    type=str, default='placeholder')
    parser.add_argument("--network_choice",    type=str, default='placeholder')
    parser.add_argument("--test_data",         type=str, default='placeholder')
    parser.add_argument("--save_folder",       type=str, default='placeholder')
    parser.add_argument("--use_all",           action='store_true')
    opt = parse_in.parse_known_args()[0]

#     assert opt.network_choice != 'placeholder' or opt.use_all, 'Please insert name of network to use for liver segmentation!'



    """===================================="""
    ### RUN GENERATION
    if not opt.use_all:
        opt.networks_to_use = [os.getcwd()+'/../SAVEDATA/Standard_Liver_Networks/'+opt.network_choice] if opt.network_folder=='placeholder' else [opt.network_folder+'/'+opt.network_choice]
    else:
        opt.networks_to_use  = [os.getcwd()+'/../SAVEDATA/Standard_Liver_Networks/'+x for x in os.listdir(os.getcwd()+'/../SAVEDATA/Standard_Liver_Networks')]

    if 'placeholder' in opt.test_data:
        opt.test_data        ='./LOADDATA/Test_Data_2D'

    if 'placeholder' in opt.save_folder:
        opt.save_folder      = os.getcwd()+'/../SAVEDATA/Test_Segmentations/Liver_Segmentations'


    """=========================================="""
    ### RUN TEST SEGMENTATIONS
    main(opt)
# @author: Karsten Roth - Heidelberg University, 07/11/2017


"""======================================="""
"""========= TEST DATASET ================"""
"""======================================="""
import torch.utils.data as data
class TestDataset(data.Dataset):
    def __init__(self, test_data_folder, path_2_liver_segmentations, opt, channel_size=1):
        self.pars = opt
        self.test_volumes       = [x for x in os.listdir(test_data_folder+'/Volumes')]
        self.test_volume_slices = {key:[test_data_folder+'/Volumes/'+key+'/'+x for x in sorted(os.listdir(test_data_folder+'/Volumes/'+key),key=lambda x: [int(y) for y in x.split('-')[-1].split('.')[0].split('_')])] for key in self.test_volumes}
        self.test_liver_slices  = {key:[path_2_liver_segmentations+'/'+key+'/'+x for x in sorted(os.listdir(path_2_liver_segmentations+'/'+key),key=lambda x: [int(y) for y in x.split('-')[-1].split('.')[0].split('_')])] for key in self.test_volumes}

        self.channel_size   = channel_size
        self.liver_seg_info = pkl.load(open(path_2_liver_segmentations+'/liver_bound_dict.pkl','rb'))
        self.recon_info     = pkl.load(open(test_data_folder+'/volume_nii_info.pkl','rb'))

        self.iter_data, slice_cluster_collect, liv_cluster_collect = [],[],[]
        for vol in self.test_volumes:
            for i in range(len(self.test_volume_slices[vol])):
                extra_ch  = self.channel_size//2
                low_bound = np.clip(i-extra_ch,0,None).astype(int)
                low_diff  = extra_ch-i
                up_bound  = np.clip(i+extra_ch+1,None,len(self.test_volume_slices[vol])).astype(int)
                up_diff   = i+extra_ch+1-len(self.test_volume_slices[vol])

                vol_slices = self.test_volume_slices[vol][low_bound:up_bound]
                liv_slices = self.test_liver_slices[vol][low_bound:up_bound]

                if low_diff>0:
                    extra_slices    = self.test_volume_slices[vol][low_bound+1:low_bound+1+low_diff][::-1]
                    vol_slices      = extra_slices+vol_slices
                    extra_slices    = self.test_liver_slices[vol][low_bound+1:low_bound+1+low_diff][::-1]
                    liv_slices      = extra_slices+liv_slices
                if up_diff>0:
                    extra_slices    = self.test_volume_slices[vol][up_bound-up_diff-1:up_bound-1][::-1]
                    vol_slices      = vol_slices+extra_slices
                    extra_slices    = self.test_liver_slices[vol][up_bound-up_diff-1:up_bound-1][::-1]
                    liv_slices      = liv_slices+extra_slices

                slice_cluster_collect.append(vol_slices)
                liv_cluster_collect.append(liv_slices)
                self.iter_data.append((vol,i))

            self.test_volume_slices[vol] = slice_cluster_collect
            self.test_liver_slices[vol]  = liv_cluster_collect

            slice_cluster_collect, liv_cluster_collect = [],[]


        self.n_files = len(self.iter_data)
        self.vol_slice_idx = 0
        self.curr_vol = 0
        self.curr_vol_name = self.test_volumes[0]
        self.curr_vol_size = self.liver_seg_info[self.test_volumes[self.curr_vol]+'.nii'][2]

        self.total_num_slices = len(self.test_volume_slices[self.curr_vol_name])

    def __getitem__(self, idx):
        VOI, SOI = self.iter_data[idx]
        V2O  = np.concatenate([np.expand_dims(np.load(vol),0) for vol in self.test_volume_slices[VOI][SOI]],axis=0)

        if self.pars.Training['no_standardize']:
            V2O  = gu.normalize(V2O, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            V2O  = gu.normalize(V2O)

        LMSK  = np.concatenate([np.expand_dims(np.load(vol),0) for vol in self.test_liver_slices[VOI][SOI]],axis=0)

        if self.vol_slice_idx==len(self.test_volume_slices[self.test_volumes[self.curr_vol]])-1:
            self.vol_slice_idx    = 0
            self.curr_vol_name    = self.test_volumes[self.curr_vol]
            self.total_num_slices = len(self.test_volume_slices[self.curr_vol_name])
            self.curr_vol     += 1
            return_data = {'VolSlice':V2O,'end_volume':True}
        else:
            return_data = {'VolSlice':V2O,'end_volume':False}
            self.vol_slice_idx+=1
        return_data['LivMsk'] = LMSK

        return return_data

    def __len__(self):
        return self.n_files





"""======================================="""
"""============= MAIN FUNCTIONALITY ======"""
"""======================================="""
def main(opt):
    ############ Init Network ###################
    network_list = []
    for i,network_setup in enumerate(opt.networks_to_use):
        network_opt = pkl.load(open(network_setup+'/hypa.pkl','rb'))
        network     = netlib.Scaffold_UNet(network_opt)
        checkpoint  = torch.load(network_setup+'/checkpoint_best_val.pth.tar')
        network.load_state_dict(checkpoint['network_state_dict'])
        network_list.append({'network':network, 'settings':network_opt})
        del network, network_opt, checkpoint

    back_device = torch.device('cpu')
    up_device   = torch.device('cuda')



    ############# Set Dataloader ##################
    max_channels    = np.max([item['settings'].Network['channels'] for item in network_list])
    test_dataset    = TestDataset(opt.test_data, opt.path_2_liv_seg, network_list[0]['settings'], channel_size=max_channels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    input_slices, vol_segs, volume_info, liver_masks= [],[],{},[]
    n_vols = len(test_dataloader.dataset.test_volumes)
    vol_count = 1

    data_iter = tqdm(test_dataloader, position=0)


    ############# Run Test Mask Generation ##################
    for idx,data in enumerate(data_iter):
        ### Getting input slices ###
        data_iter.set_description('Reading... [Vol {}/{}]'.format(vol_count, n_vols))
        input_slices.append(data['VolSlice'])
        liver_masks.append(data['LivMsk'].numpy()[0,0,:])
        end_volume = data['end_volume'].numpy()[0]

        if end_volume:
            prev_vol     = test_dataloader.dataset.test_volumes[test_dataloader.dataset.curr_vol-1]
            liver_region = test_dataloader.dataset.liver_seg_info[prev_vol+'.nii'][2]

            with torch.no_grad():
                out_mask     = np.zeros((512,512,len(input_slices)))

                for net_count,net_dict in enumerate(network_list):

                    network = net_dict['network']
                    network.to(up_device)

                    ### Computing Segmentation ###
                    data_iter.set_description('Segmenting... [Vol {}/{} | Net {}/{}]'.format(vol_count, n_vols, net_count+1, len(opt.networks_to_use)))


                    for slice_idx, input_slice in enumerate(tqdm(input_slices, desc='Running Segmentation...', position=1)):
                        if slice_idx in range(np.clip(liver_region[0],0,None),liver_region[1]):

                            n_ch = net_dict['settings'].Network['channels']
                            input_slice = input_slice[:,max_channels//2-n_ch//2:max_channels//2+n_ch//2+1,:]
                            input_slice = input_slice.type(torch.FloatTensor).to(up_device)

                            seg_pred    = network(input_slice)[0]

                            out_mask[:,:,slice_idx] += seg_pred.detach().cpu().numpy()[0,-1,:]

                out_mask    = np.round(out_mask/len(opt.networks_to_use))
                out_mask    = out_mask.astype(np.uint8)

                liver_masks = np.stack(liver_masks,axis=-1).astype(np.uint8)

                ### (optional) Running Post-Processing by removing tiny lesion speckles
                # data_iter.set_description('Post-Processing... [Vol {}/{}]'.format(vol_count, n_vols))
                # out_mask = snm.label(np.round(out_mask))[0].astype(np.uint8)
                # for i in np.unique(out_mask)[1:]:
                #     eqs = out_mask==i
                #     n_zs = len(list(set(list(np.where(eqs)[-1]))))
                #     if n_zs==1:
                #         out_mask[eqs] = 0
                # out_mask = out_mask>0
                # out_mask = out_mask.astype(np.uint8)


                ### Saving Final Segmentation Mask
                data_iter.set_description('Saving... [Vol {}/{}]'.format(vol_count, n_vols))
                #temp_liver_masks = (liver_masks+out_mask)>0
                #labels   = snm.label(temp_liver_masks)
                #MainCC   = labels[0]==np.argmax(np.bincount(labels[0].flat)[1:])+1
                labels   = snm.label(np.round(out_mask/len(opt.networks_to_use)))

                #out_mask = out_mask*MainCC + MainCC
                out_mask = out_mask*liver_masks + liver_masks

                data_header = test_dataloader.dataset.recon_info[prev_vol]['header']
                affine      = test_dataloader.dataset.recon_info[prev_vol]['affine']
                nifti_save_image = nib.Nifti1Image(out_mask, affine=affine)
                nifti_save_image.header['pixdim'] = data_header['pixdim']
                nib.save(nifti_save_image, opt.save_folder+'/test-segmentation-'+prev_vol.split('-')[-1])
                del out_mask

                ### Reseting Parameters
                vol_segs, input_slices, liver_masks = [],[],[]
                vol_count+=1
                data_iter.set_description('Completed. [Vol {}/{}]'.format(vol_count, n_vols))







if __name__ == '__main__':
    """======================================="""
    ### LOAD BASIC LIBRARIES
    import os,json,sys,gc,time,datetime,imp,argparse,copy
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Utilities')
    sys.path.insert(0, '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Network_Zoo')

    import numpy as np, pickle as pkl, nibabel as nib
    from tqdm import tqdm,trange

    import pandas as pd
    from collections import OrderedDict
    import scipy.ndimage.measurements as snm
    import scipy.ndimage.morphology as snmo

    import network_zoo as netlib
    import General_Utilities as gu, Network_Utilities as nu

    import PyTorch_Datasets as Data

    import torch
    from torch.utils.data import DataLoader

    ###NOTE: This line is necessary in case of the "too many open files"-Error!
    if int(torch.__version__.split(".")[1])>2:
        torch.multiprocessing.set_sharing_strategy('file_system')



    """=========================================="""
    ### GET INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_folder",    type=str, default='placeholder')
    parser.add_argument("--network_choice",    type=str, default='placeholder')
    parser.add_argument("--test_data",         type=str, default='placeholder')
    parser.add_argument("--save_folder",       type=str, default='placeholder')
    parser.add_argument("--path_2_liv_seg",    type=str, default='placeholder')
    parser.add_argument("--use_all",           default='placeholder', action='store_true')
    opt = parse_in.parse_known_args()[0]


#     assert opt.network_choice != 'placeholder' or opt.use_all, 'Please insert name of network to use for lesion segmentation!'



    """===================================="""
    ### RUN GENERATION
    if not opt.use_all:
        opt.networks_to_use = [os.getcwd()+'/../SAVEDATA/Standard_Lesion_Networks/'+opt.network_choice] if opt.network_folder=='placeholder' else [opt.network_folder+'/'+opt.network_choice]
    else:
        opt.networks_to_use = [os.getcwd()+'/../SAVEDATA/Standard_Lesion_Networks/'+x for x in os.listdir(os.getcwd()+'/../SAVEDATA/Standard_Lesion_Networks')]


    """===================================="""
    ### ADJUST PLACEHOLDER VALUES IF NOT SPECIFIED
    if 'placeholder' in opt.test_data:
        opt.test_data       = os.getcwd()+'/../LOADDATA/Test_Data_2D'
    if 'placeholder' in opt.save_folder:
        opt.save_folder     = os.getcwd()+'/../SAVEDATA/Test_Segmentations/Test_Submissions'
    if 'placeholder' in opt.path_2_liv_seg:
        opt.path_2_liv_seg  = os.getcwd()+'/../SAVEDATA/Test_Segmentations/Liver_Segmentations'
    if not os.path.exists(opt.save_folder): os.makedirs(opt.save_folder)

    """=========================================="""
    ### RUN TEST SEGMENTATIONS
    main(opt)

"""=================================================="""
### LOAD BASIC LIBRARIES
import argparse, numpy as np, os, sys, torch
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,'../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Utilities')
sys.path.insert(0,'../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Network_Zoo')
import General_Utilities as gu, Network_Utilities as nu
import network_zoo as netlib


"""=================================================="""
### GET NETWORK PARAMETERS
parse_in = argparse.ArgumentParser()
parse_in.add_argument('--base_setup',   type=str, default='Baseline_Parameters.txt',
                                        help='Path to baseline setup-txt which contains all major parameters that most likely will be kept constant during various grid searches.')
# parse_in.add_argument('--search_setup', type=str, default='LiverNetwork_Parameters.txt',
#                                         help='Path to search setup-txt, which contains (multiple) variations to the baseline proposed above.')
# opt = parse_in.parse_known_args()[0]
parse_in.add_argument('--search_setup', type=str, default='',
                                        help='Path to search setup-txt, which contains (multiple) variations to the baseline proposed above.')
opt = parse_in.parse_known_args()[0]


# opt = parse_in.parse_args(["--search_setup","Specific_Setup_Parameters_3D_LesionSegmentation_PC1.txt"])
opt.base_setup   = '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Train_Networks/Training_Setup_Files/'+opt.base_setup
# opt.search_setup = '../input/nii555/LiverLesion_Segmentation/Repository Standard_LiverLesion_Segmentation/Train_Networks/Training_Setup_Files/Baseline_Parameters.txt'
training_setups = gu.extract_setup_info(opt)
opt = training_setups[0]


"""================================================="""
### LOAD NETWORK
opt.Training['num_out_classes'] = 2
network = netlib.NetworkSelect(opt)
network.n_params = nu.gimme_params(network)
opt.Network['Network_name'] = network.name
device = torch.device('cuda')
_ = network.to(device)


### INPUT DATA
input_data   = torch.randn((1,opt.Network['channels'],256,256)).type(torch.FloatTensor).to(device)
network_pred = network(input_data)[0]


"""================================================="""
### SAVE COMPUTATION GRAPH
gu.save_graph(network_pred, "./", "LiverNetwork_Parameters", view=True)