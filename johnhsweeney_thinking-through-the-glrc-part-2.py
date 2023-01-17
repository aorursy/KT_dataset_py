def url_stripper(old_url):
    
    new_url = old_url
    #length = len(old_url)
    position = -1
    done = False
    change = False
    if old_url[position] == '/':
        while not done:
            position -=1
            if old_url[position] == '/':
                done = True
                change = True
    if change:
        new_url = old_url[:position+1]
    
    return new_url
# A simple demonstration of url_stripper

url = 'https://lh3.googleusercontent.com/-SXCAgqmUSCY/TKKFZqwVxxI/AAAAAAAADbw/H440k4K4rlY/w11-h15/'
print('old url: ', url)
url = url_stripper(url)
print('new url: ', url)
import pandas as pd
train_data = pd.read_csv('../input/train.csv') #kaggle version
#train_data = pd.read_csv('./train.csv') # local version

# find out how many unique landmark ids there are, and how many instances of each one
landmark_ids = train_data['landmark_id'].value_counts() 
print('there are', landmark_ids.shape[0], 'unique landmarks')
print('the top 20 landmarks are:')
print('id       count')
print(landmark_ids.head(20))
counts = landmark_ids.values
index = landmark_ids.index
size = 1024
for i in [1024,2048,4096, 8192]:
    percentage = 100*(counts[0:i].sum()/counts.sum())
    print('the top %d landmarks account for %5.2f percent of the training samples' %(i,percentage))
