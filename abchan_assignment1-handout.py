%matplotlib inline

import IPython.core.display         

# setup output image format (Chrome works best)

IPython.core.display.set_matplotlib_formats("svg")

import matplotlib.pyplot as plt

import matplotlib

from numpy import *

from sklearn import *

from scipy import stats

import csv

random.seed(100)
def read_text_data(fname):

    txtdata = []

    classes = []

    topics  = []

    with open(fname, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        for row in reader:

            # get the text

            txtdata.append(row[0])

            # get the class (convert to integer)

            if len(row)>1:

                classes.append(row[1])

                topics.append(row[2])

    

    if (len(classes)>0) and (len(txtdata) != len(classes)):        

        raise Exception("mismatched length!")

    

    return (txtdata, classes, topics)



def write_csv_kaggle_sub(fname, Y):

    # fname = file name

    # Y is a list/array with class entries

    

    # header

    tmp = [['Id', 'Prediction']]

    

    # add ID numbers for each Y

    for (i,y) in enumerate(Y):

        tmp2 = [(i+1), y]

        tmp.append(tmp2)

        

    # write CSV file

    with open(fname, 'w') as f:

        writer = csv.writer(f)

        writer.writerows(tmp)
# load the data

# (if using Kaggle notebooks you need to include the directory path: /kaggle/input/cs5489-2020b-assignment-1/)

(traintxt, trainY, traintopic) = read_text_data("/kaggle/input/cs5489-2020b-assignment-1/sanders_tweets_train.txt")

(testtxt, _, _)                = read_text_data("/kaggle/input/cs5489-2020b-assignment-1/sanders_tweets_test.txt")



print(len(traintxt))

print(len(testtxt))
classnames = unique(trainY)

print(classnames)
# write your predictions on the test set

i = random.randint(len(classnames), size=len(testtxt))

predY = classnames[i]

write_csv_kaggle_sub("my_submission.csv", predY)
print(trainY[0])

print(traintopic[0])

print(traintxt[0])
# INSERT YOUR CODE HERE