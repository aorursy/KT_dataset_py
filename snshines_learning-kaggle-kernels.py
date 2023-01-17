import os

print(os.listdir("../input"))
train_data = open("../input/train.csv", "r").read()
len(train_data)
output_file = open("my_output.txt", "w")
os.listdir(".")
output_file.write("abc\ndef")
output_file.close()
