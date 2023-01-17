SAMPLE_SUB = "../input/deepfake-detection-challenge/sample_submission.csv"

TRAIN_VIDEOS = "../input/deepfake-detection-challenge/train_sample_videos"

TEST_VIDEOS = "../input/deepfake-detection-challenge/test_videos"

TRAIN_JSON_PATH = "../input/deepfake-detection-challenge/train_sample_videos/metadata.json"
#import packages



import pandas as pd

import numpy as np

import cv2

import seaborn as sns

import matplotlib.pyplot as plt

import os



%matplotlib inline
# check the number of videos present in the train and test data.



n_train_videos = len(os.listdir(TRAIN_VIDEOS))

n_test_videos = len(os.listdir(TEST_VIDEOS))



train_videos = os.listdir(TRAIN_VIDEOS)

test_videos = os.listdir(TEST_VIDEOS)



print("Number of training vidoes: ", n_train_videos - 1)

print("Number of testing videos: ", n_test_videos)
#read the json file



deepfake_labels = pd.read_json(TRAIN_JSON_PATH).T



deepfake_labels.head()
#start with the basic analysis - Missing value analysis



missing_df = pd.DataFrame({"Missing_Count": deepfake_labels.isnull().sum(),

                          "Missing_Percent": round(deepfake_labels.isnull().mean(),2)})

missing_df
#check the distribution of labels

plt.style.use("ggplot")

plt.rcParams['figure.figsize'] = 14,7



deepfake_labels.label.value_counts(normalize = True).plot(kind = "barh")

plt.xlabel("Percentage of labels")

plt.title("Distribution of labels for videos")

plt.show()
# we have already created a list of training data and test data videos



print("Training videos: " ,train_videos[:5])

print("Testing videos: ", test_videos[:5])
train_fake_lst = deepfake_labels[deepfake_labels["label"] == "FAKE"].index.tolist()

train_real_lst = deepfake_labels[deepfake_labels["label"] == "REAL"].index.tolist()
#display frame



def display_frame(video, axis):

    cap = cv2.VideoCapture(video)  

    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    axis.imshow(frame)

    axis.grid(False)

    video_name = video.split("/")[-1]

    axis.set_title("Video: " + video_name, size = 30)
random_num = np.random.randint(0,len(train_fake_lst))

fig,axs = plt.subplots(nrows = 1, ncols=3, figsize=(50,40))

for i in range(3):

    display_frame(os.path.join(TRAIN_VIDEOS, train_fake_lst[random_num]), axs[i])
random_num = np.random.randint(0,len(train_fake_lst))

fig,axs = plt.subplots(nrows = 1, ncols=3, figsize=(50,40))

for i in range(3):

    display_frame(os.path.join(TRAIN_VIDEOS, train_fake_lst[random_num]), axs[i])
random_num = np.random.randint(0,len(train_real_lst))

fig,axs = plt.subplots(nrows = 1, ncols=3, figsize=(50,40))

for i in range(3):

    display_frame(os.path.join(TRAIN_VIDEOS, train_real_lst[random_num]), axs[i])
random_num = np.random.randint(0,len(train_real_lst))

fig,axs = plt.subplots(nrows = 1, ncols=3, figsize=(50,40))

for i in range(3):

    display_frame(os.path.join(TRAIN_VIDEOS, train_real_lst[random_num]), axs[i])
random_num = np.random.randint(0,len(train_real_lst))

fig,axs = plt.subplots(nrows = 1, ncols=3, figsize=(50,40))

for i in range(3):

    display_frame(os.path.join(TRAIN_VIDEOS, train_real_lst[random_num]), axs[i])
#checking the label distribution



deepfake_labels.label.value_counts(normalize = True)
sample_df = pd.read_csv(SAMPLE_SUB)

sample_df.head()
sample_df["label"] = 0.65

sample_df.to_csv("submission.csv", index = False)