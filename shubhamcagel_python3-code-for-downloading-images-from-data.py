#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing,csv
import urllib.request as urllib2
from PIL import Image
from io import StringIO
import os
import urllib
'''
Here you need to put your own local path from computer
'''
data_file="../input/train.csv"
out_dir="../inputoutputdir"
'''
These will Parse overall Csv file data  
'''
def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header
'''
These will download images from given urls
'''
def DownloadImage(key_url):
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)
  
  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    
    f = open(out_dir+key, 'wb')
    f.write(urllib.request.urlopen(url).read())
    f.close()
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return


def Run():
  key_url_list = ParseData(data_file)
  key_url_list=key_url_list[1023319:]
  pool = multiprocessing.Pool(processes=50)
  pool.map(DownloadImage, key_url_list)
  
if __name__ == '__main__':
  Run()