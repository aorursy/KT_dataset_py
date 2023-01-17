import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
dataset = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

dataset
newdataset1 = dataset.drop(["video_id", "trending_date"], axis=1) # Drops column video_id and trending_date
newdataset1.to_csv("USvideos_new.csv") # write to file with index
pd.read_csv("USvideos_new.csv") # read file that is saved at previous step
newdataset1.to_csv("USvideos_new2.csv", index=False) # writes to file without index
pd.read_csv("USvideos_new2.csv") # read file that is saved at previous step
dataset = pd.read_html("http://www.contextures.com/xlSampleData01.html", header=0)  # header=0 means, first row is header

dataset
dataset[0] # Shows first table at related internet page