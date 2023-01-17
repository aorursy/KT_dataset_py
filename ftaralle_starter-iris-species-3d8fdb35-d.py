import os

import pandas as pd
input_folder = '../input'

output_folder = '../output'
print('@root:', input_folder)

for root, dirs, files in os.walk(input_folder):

    for file_name in files:

        print('@file:', os.path.join(input_folder, file_name))

    for dir_name in dirs:

        print('@dir:', os.path.join(input_folder, dir_name))
iris = pd.read_csv('../input/Iris.csv')

display(iris[:10])