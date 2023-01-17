# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import chardet

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
input_dir = "../input/"
output_dir = ""
def get_encoding(file, print_encoding=False, lines_to_read=None):
    '''Guess the encoding of an input file using chardet'''
    with open(file, 'rb') as rawdata:
        encoding_guess = chardet.detect(rawdata.read(lines_to_read))['encoding']
        if print_encoding:
            print("Expected encoding for file '{0}': {1}".format(file, encoding_guess))
        return encoding_guess

def read_file(file, encoding):
    '''Read a file using a specified encoding, and return the result as a string'''
    with open(file, 'rb') as rawdata:
        file_str = rawdata.read()
    file_utf8 = file_str.decode(encoding)
    return file_utf8

def write_as_utf8(data, path):
    '''Write a file to a specified path, using UTF-8 encoding'''
    with open(path, 'w') as f:
        f.write(data)
file_list = os.listdir(input_dir)
for file in file_list:
    # Set input and output paths for this file
    input_path = input_dir + file
    utf8_output_path = output_dir + file
    
    # 'Guess' the file's encoding, read it using that encoding, and then store as UTF-8
    encoding = get_encoding(input_path, print_encoding=True)
    decoded_file = read_file(input_path, encoding)
    write_as_utf8(decoded_file, utf8_output_path)