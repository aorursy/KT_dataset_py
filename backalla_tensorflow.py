import os
data_path = "../input/oct2017/OCT2017 /"
train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')
val_path = os.path.join(data_path,'val')
classes = ['DME', 'CNV', 'DRUSEN', 'NORMAL']
print(os.listdir(data_path+"train"))
