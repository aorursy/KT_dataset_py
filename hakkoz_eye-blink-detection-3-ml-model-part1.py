# import utility functions

from utils_frame_based import *

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt



# define three constants.

# You can later experiment with these constants by changing them to adaptive variables.

EAR_THRESHOLD = 0.21 # eye aspect ratio to indicate blink

EAR_CONSEC_FRAMES = 3 # number of consecutive frames the eye must be below the threshold

SKIP_FIRST_FRAMES = 150 # how many frames we should skip at the beggining
print(os.listdir('../input/blinkdata/eyeblink8/2'))
# create a folder named 'train'

os.mkdir('./train')



# read all videos

directory = "../input/blinkdata/eyeblink8"

subjects=os.listdir(directory)

for subject in subjects:

    video_names=os.listdir(directory+'/'+subject)

    for video_name in video_names:

        clean_name = os.path.splitext(video_name)[0]

        extension = os.path.splitext(video_name)[1]

        if extension=='.avi': 

            file_path = directory+'/'+subject+'/'+video_name

            print(file_path)

            frame_info_df, video_info_dict = process_video_v2(file_path, subject=subject, external_factors=None,facial_actions=clean_name, \

                                                            ear_th=EAR_THRESHOLD, consec_th=EAR_CONSEC_FRAMES, skip_n=SKIP_FIRST_FRAMES)

            frame_info_df.to_pickle('./train/{}_{}_frame_info_df.pkl'.format(subject,clean_name))

            video_info_dict.to_pickle('./train/{}_{}_video_info_df.pkl'.format(subject,clean_name))
# define read_annotations_v2

def read_annotations_v2(input_file, len_video):

    # Read .tag file using readlines() 

    file1 = open(input_file) 

    Lines = file1.readlines() 



    # find "#start" line 

    start_line = 1

    for line in Lines: 

        clean_line=line.strip()

        if clean_line=="#start":

            break

        start_line += 1



    # length of annotations

    len_annot = len(Lines[start_line : -1]) # -1 since last line will be"#end"



    blink_list = [0] * len_video

    closeness_list = [0] * len_video



    # convert tag file to readable format and build "closeness_list" and "blink_list"

    for i in range(len_annot): 

        annotation=Lines[start_line+i].split(':')



        if int(annotation[1]) > 0:

            # it means a new blink

            blink_frame = int(annotation[0])

            blink_list[blink_frame] = 1



        # if current annotation consist fully closed eyes, append it also to "closeness_list" 

        if annotation[3] == "C" and annotation[5] == "C":

            closed_frame = int(annotation[0])

            closeness_list[closed_frame] = 1



        file1.close()



    result_df = pd.DataFrame(list(zip(closeness_list, blink_list)), columns=['closeness_annot', 'blink_annot'])

    return result_df
# full path of a tag file by using read_annotations() utility function

directory = "../input/blinkdata/eyeblink8"

subjects=os.listdir(directory)

for subject in subjects:

    video_names=os.listdir(directory+'/'+subject)

    for video_name in video_names:

        clean_name = os.path.splitext(video_name)[0]

        extension = os.path.splitext(video_name)[1]

        if extension=='.tag': 

            file_path = directory+'/'+subject+'/'+video_name

            print(file_path)

            #length of video

            frame_info_df = pd.read_pickle("./train/" + subject + '_' + clean_name + "_frame_info_df.pkl")

            len_video = len(frame_info_df)

            # read tag file

            annot_df = read_annotations_v2(file_path, len_video)

            annot_df.to_pickle('./train/{}_{}_annotations.pkl'.format(subject,clean_name))
def merge_pickles(directory):

    annots=[]

    frame_infos=[]

    video_infos=[]



    files = os.listdir(directory)

    for file in files:

        clean_name = os.path.splitext(file)[0]

        if clean_name.endswith('annotations'):

            annots.append(file)

        if clean_name.endswith('video_info_df'):

            video_infos.append(file)

        if clean_name.endswith('frame_info_df'):

            frame_infos.append(file)



    for file in annots:

        clean_name = os.path.splitext(file)[0]

        first_part = clean_name[:-12]



        for file2 in frame_infos:

            clean_name2 = os.path.splitext(file2)[0]

            first_part2 = clean_name2[:-14]

            if first_part == first_part2:

                frame_info_df = pd.read_pickle(directory+'/'+file2)

                annotation = pd.read_pickle(directory+'/'+file)

                if len(frame_info_df) !=len(annotation):

                    os.mkdir(directory+'/fix/')

                    os.rename(directory+file, directory+'/fix/'+file)

                    os.rename(directory+file2, directory+'/fix/'+file2)

                    print(file2, len(frame_info_df))

                    print(file, len(annotation))

                else: 

                    result=pd.concat([frame_info_df,annotation], axis=1)

                    result.to_pickle(directory+'/'+first_part+'_merged_df.pkl')
# merge "*_frame_info_df.pkl" with "*_annotations.pkl" by using the function above

merge_pickles("./train")
# append all of pickles ending particular string (i.e. "merged_df") in a directory

def concat_pickles(directory, ending, output_name):

    pickles = os.listdir(directory)

    pickle_list=[]



    for pickle_file in pickles:

        clean_name = os.path.splitext(pickle_file)[0]

        if clean_name.endswith(ending):

            pickle = pd.read_pickle(directory+'/'+pickle_file)

            pickle_list.append(pickle)



    result = pd.concat(pickle_list)

    result.reset_index(inplace=True, drop=True)

    result.to_pickle(directory+'/'+ output_name + '.pkl')
# append all of pickles ending with "merged_df" in ./train

concat_pickles("./train","merged_df","training_set")
# create a folder named 'test'

os.mkdir('./test')



# read all videos

directory = "../input/blinkdata/talkingFace"

files = os.listdir(directory)

for file in files:

    clean_name = os.path.splitext(file)[0]

    extension = os.path.splitext(file)[1]

    if extension=='.avi': 

        file_path = directory+'/'+ file

        print(file_path)

        frame_info_df, video_info_dict = process_video_v2(file_path, subject='talkingFace', external_factors=None,facial_actions=clean_name, \

                                                        ear_th=EAR_THRESHOLD, consec_th=EAR_CONSEC_FRAMES, skip_n=SKIP_FIRST_FRAMES)

        frame_info_df.to_pickle('./test/{}_frame_info_df.pkl'.format(clean_name))

        video_info_dict.to_pickle('./train/{}_video_info_df.pkl'.format(clean_name))



# read tag file

for file in files:

    clean_name = os.path.splitext(file)[0]

    extension = os.path.splitext(file)[1]

    if extension=='.tag': 

        file_path = directory+'/'+ file

        print(file_path)

        #length of video

        frame_info_df = pd.read_pickle('./test/{}_frame_info_df.pkl'.format(clean_name))

        len_video = len(frame_info_df)

        # read tag file

        annot_df = read_annotations_v2(file_path, len_video)

        annot_df.to_pickle('./test/{}_annotations.pkl'.format(clean_name))



# merge annotations and frame_info_df

merge_pickles("./test")



# append all of pickles ending with "merged_df" in ./test

concat_pickles("./test","merged_df","test_set")