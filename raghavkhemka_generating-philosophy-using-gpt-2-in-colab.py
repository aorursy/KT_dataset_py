%tensorflow_version 1.x #this works only in google colab
import tensorflow as tf
tf.__version__
import re
file_name = '/content/Beyond Good and Evil.txt'
file1 = open(file_name,"rt",encoding='utf8')
file = file1.readlines()
file1.close()
sentences = []
start = 0
for line in file:
  if line=='PREFACE\n': #to start the text from Preface
    start = 1
  if start!=1:
    continue 
  if line == 'FROM THE HEIGHTS\n':
    break
  if line.startswith('CHAPTER'):
    continue
  # Removing unwanted punctuations
  line = re.sub('--',' ',line)
  words = line.split()
  if len(words)<=2:
    continue
  # Removing numbers from the beginning of the line
  if words[0][0].isnumeric(): 
    newline = ' '.join(words[1:]) +'\n'
  else:
    newline = line
  sentences.append(newline)
# Combining the lines
training_data = ''.join(sentences)
try:
  newfile = open("/content/training_file.txt",'x')
  newfile.close()
except:
  print('Already Exists')
training_file = open("/content/training_file.txt",'w')
training_file.write(training_data)
training_file.close()
!pip install gpt-2-simple
import gpt_2_simple as gpt2
import os

model_name = "124M" #any other model can also be chosen but 124M works fine here
if not os.path.isdir(os.path.join("models", model_name)):
  gpt2.download_gpt2(model_name=model_name)   # model will be saved into current directory under /models/124M/


file_name = '/content/training_file.txt'

    

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=500)
prefix = 'The purpose of life is'
gpt2.generate(sess, model_name='/content/checkpoint/run1',prefix = prefix)