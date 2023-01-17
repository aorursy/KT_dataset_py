import pandas





list_compute    = ['My NAS', 'Colfax Cluster', "Kaggle's kernel"]

list_files      = 'md5sum(train.7z)', 'md5sum(test.7z)', 'md5sum(additional.7z)', 'md5sum(additional/Type_2/2845.jpg)', 'md5sum(additional/Type_2/5892.jpg)', 'md5sum(additional/Type_2/5893.jpg)', 'md5sum(train/Type_1/1339.jpg)', 'md5sum(additional/Type_1/3068.jpg)', 'md5sum(additional/Type_2/7.jpg)', 'sha1sum(train.7z)', 'sha1sum(test.7z)', 'sha1sum(additional.7z)', 'sha1sum(additional/Type_2/2845.jpg)', 'sha1sum(additional/Type_2/5892.jpg)', 'sha1sum(additional/Type_2/5893.jpg)', 'sha1sum(train/Type_1/1339.jpg)', 'sha1sum(additional/Type_1/3068.jpg)', 'sha1sum(additional/Type_2/7.jpg)'

hash_on_my_nas  = ['4b3fd8c73b6ac21b1e106d5efd8713cd', 'ae9efe79d4efa87c0bf923d86fa5c10f', '3e914c7fe2dcc8a10584b9b7413acec4', 'd41d8cd98f00b204e9800998ecf8427e', 'd41d8cd98f00b204e9800998ecf8427e', 'd41d8cd98f00b204e9800998ecf8427e', 'cc2c4af9200f0e03ccf00f647ceb2adc', '0b8cfe3f0d6256532dade1c1216024d8', '715ae5f8509d6d5eddd98364663ad656', '140233b2fd2b42a68c80d8cf2719392d9ff39d1c', 'a4f1fca9acd5a1a01d6ae4c5fd6c7ca2c2f85f6d', 'cf0066cf2fccf5086c4a540083961b2bf47d5479', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', '239cf581a491df7c9890103efc40a85b6daf82c1', 'd375c65833730d51c3748d64aecee096ff2bdbd5', '75c1d81de299d477e9aa88a41474bc74a3ac447e']

hash_on_colfax  = ['4b3fd8c73b6ac21b1e106d5efd8713cd', 'ae9efe79d4efa87c0bf923d86fa5c10f', '3e914c7fe2dcc8a10584b9b7413acec4', 'd41d8cd98f00b204e9800998ecf8427e', 'd41d8cd98f00b204e9800998ecf8427e', 'd41d8cd98f00b204e9800998ecf8427e', 'cc2c4af9200f0e03ccf00f647ceb2adc', '0b8cfe3f0d6256532dade1c1216024d8', '715ae5f8509d6d5eddd98364663ad656', '140233b2fd2b42a68c80d8cf2719392d9ff39d1c', 'a4f1fca9acd5a1a01d6ae4c5fd6c7ca2c2f85f6d', 'cf0066cf2fccf5086c4a540083961b2bf47d5479', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', '239cf581a491df7c9890103efc40a85b6daf82c1', 'd375c65833730d51c3748d64aecee096ff2bdbd5', '75c1d81de299d477e9aa88a41474bc74a3ac447e']

hash_on_kaggle  = ['Nan', 'Nan', 'Nan', 'd41d8cd98f00b204e9800998ecf8427e', 'd41d8cd98f00b204e9800998ecf8427e', 'd41d8cd98f00b204e9800998ecf8427e', 'cc2c4af9200f0e03ccf00f647ceb2adc', '0b8cfe3f0d6256532dade1c1216024d8', '715ae5f8509d6d5eddd98364663ad656', 'Nan', 'Nan', 'Nan', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'da39a3ee5e6b4b0d3255bfef95601890afd80709', '239cf581a491df7c9890103efc40a85b6daf82c1', 'd375c65833730d51c3748d64aecee096ff2bdbd5', '75c1d81de299d477e9aa88a41474bc74a3ac447e']



pandas.DataFrame([hash_on_my_nas, hash_on_colfax, hash_on_kaggle], index=list_compute, columns=list_files).transpose()




import os

import platform

import subprocess





def res_cmd(cmd):

  return str(subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True).communicate()[0])



def str_md5sum(abspath_dataset_dir, abspath_file):

    str_res = res_cmd('md5sum "%s"' % os.path.join(abspath_dataset_dir, abspath_file))

    return str_res.split("'")[1].replace('\\n', '')



def str_sha1sum(abspath_dataset_dir, abspath_file):

    str_res = res_cmd('sha1sum "%s"' % os.path.join(abspath_dataset_dir, abspath_file))

    return str_res.split("'")[1].replace('\\n', '')





if 'c001' in platform.node():

    # platform.node() => 'c001' or like 'c001-n030' on Colfax

    abspath_dataset_dir = '/data/kaggle/'

elif '.local' in platform.node():

    # platform.node() => '*.local' on my local MacBook Air

    abspath_dataset_dir = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/'

else:

    # For kaggle's kernels environment (docker container?)

    abspath_dataset_dir = '/kaggle/input/'





print(str_md5sum(abspath_dataset_dir, 'train.7z'))                   # 4b3fd8c73b6ac21b1e106d5efd8713cd

print(str_md5sum(abspath_dataset_dir, 'test.7z'))                    # ae9efe79d4efa87c0bf923d86fa5c10f

print(str_md5sum(abspath_dataset_dir, 'additional.7z'))              # 3e914c7fe2dcc8a10584b9b7413acec4

print(str_md5sum(abspath_dataset_dir, 'additional/Type_2/2845.jpg')) # d41d8cd98f00b204e9800998ecf8427e

print(str_md5sum(abspath_dataset_dir, 'additional/Type_2/5892.jpg')) # d41d8cd98f00b204e9800998ecf8427e

print(str_md5sum(abspath_dataset_dir, 'additional/Type_2/5893.jpg')) # d41d8cd98f00b204e9800998ecf8427e

print(str_md5sum(abspath_dataset_dir, 'train/Type_1/1339.jpg'))      # cc2c4af9200f0e03ccf00f647ceb2adc

print(str_md5sum(abspath_dataset_dir, 'additional/Type_1/3068.jpg')) # 0b8cfe3f0d6256532dade1c1216024d8

print(str_md5sum(abspath_dataset_dir, 'additional/Type_2/7.jpg'))    # 715ae5f8509d6d5eddd98364663ad656



print(str_sha1sum(abspath_dataset_dir, 'train.7z'))                   # 140233b2fd2b42a68c80d8cf2719392d9ff39d1c

print(str_sha1sum(abspath_dataset_dir, 'test.7z'))                    # a4f1fca9acd5a1a01d6ae4c5fd6c7ca2c2f85f6d

print(str_sha1sum(abspath_dataset_dir, 'additional.7z'))              # cf0066cf2fccf5086c4a540083961b2bf47d5479

print(str_sha1sum(abspath_dataset_dir, 'additional/Type_2/2845.jpg')) # da39a3ee5e6b4b0d3255bfef95601890afd80709

print(str_sha1sum(abspath_dataset_dir, 'additional/Type_2/5892.jpg')) # da39a3ee5e6b4b0d3255bfef95601890afd80709

print(str_sha1sum(abspath_dataset_dir, 'additional/Type_2/5893.jpg')) # da39a3ee5e6b4b0d3255bfef95601890afd80709

print(str_sha1sum(abspath_dataset_dir, 'train/Type_1/1339.jpg'))      # 239cf581a491df7c9890103efc40a85b6daf82c1

print(str_sha1sum(abspath_dataset_dir, 'additional/Type_1/3068.jpg')) # d375c65833730d51c3748d64aecee096ff2bdbd5

print(str_sha1sum(abspath_dataset_dir, 'additional/Type_2/7.jpg'))    # 75c1d81de299d477e9aa88a41474bc74a3ac447e