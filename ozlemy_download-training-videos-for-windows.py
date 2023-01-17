# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import urllib.request as urllib2
import requests
from six.moves.urllib.request import urlretrieve
import sys, os
from glob import glob

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
response_train = urllib2.urlopen('http://storage.googleapis.com/us.data.yt8m.org/2/video/train/index.html')
html = response_train.read()
total_size = response_train.info().get('Content-Length').strip()
total_size = int(total_size)
print(total_size)
url_train = 'http://storage.googleapis.com/us.data.yt8m.org/2/video/train/index.html'
r_train = requests.get(url_train, allow_redirects=True)
open('train_video_list', 'wb').write(r_train.content)
with open('train_video_list') as f:    
    read_data = f.read()
    print(len(read_data))
    print(type(read_data))
    print(read_data[:100])
    read_data_strip = read_data.strip('<html><body>\n')
    read_data_strip = read_data_strip.strip('</body></')
    read_data_split = read_data_strip.split('</a><br/>\n')
    print(read_data_split[:10])
    print()
    print(read_data_split[3843:])
train_filenames = read_data_split[:3844]
print(len(train_filenames))
print(train_filenames[3843:])
train_video_filename_list = []
for fn in train_filenames:
    fn = fn.split('>')
    train_video_filename_list.append(fn[1])
print(len(train_video_filename_list))
print(train_video_filename_list[:5])
last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):    
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)
    if last_percent_reported != percent:
        if percent % 50 == 0:        
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
    last_percent_reported = percent
def download_file_list_numerate(pre_url, filelist, path_to_save, force=False):
    downloaded_filename_list = []
    problem_file_list = []
    for i, filename in enumerate(filelist):
        response_per_file = urllib2.urlopen(pre_url + filename)
        expected_bytes = int(response_per_file.info().get('Content-Length').strip())
        print('Attempting to download:', filename)
        filename_numerate = filename[:7] + '_' + str(i) + filename[7:]
        filename, _ = urlretrieve(pre_url + filename, path_to_save + filename_numerate,
                                  reporthook=download_progress_hook)
        print('\nDownload Complete!')
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            downloaded_filename_list.append(filename)
            print('Found and verified', filename)
        else:
            problem_file_list.append(filename)
            print(filename, ' added to the missing file list')
    return downloaded_filename_list, problem_file_list
               
pre_url = 'http://storage.googleapis.com/us.data.yt8m.org/2/video/train/'
filelist = train_video_filename_list
path_to_save = 'D:/yt8m/v2/video/'
dowloaded_fn, problem_fn = download_file_list_numerate(pre_url, filelist, path_to_save, force=False)
print(len(filelist))
print(len(downloaded_fn))
print(len(problem_fn))