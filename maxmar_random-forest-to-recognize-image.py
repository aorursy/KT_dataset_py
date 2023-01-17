import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier
# Reading the Train and Test Datasets.

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Let's see the shape of the train and test data

print(train.shape, test.shape)
train.isna().any().any()
# dividing the data into the input and output features to train make the model learn based on what to take in and what to throw out.

train_X = train.loc[:, "pixel0":]

train_Y = train.loc[:, "label"]



# Notmailzing the images array to be in the range of 0-1 by dividing them by the max possible value. 

# Here is it 255 as we have 255 value range for pixels of an image. 

train_X = train_X/255.0

test = test/255.0
# Let's make some beautiful plots.

def visualize_digit(row):

    

    digit_array = train_X.loc[row, "pixel0":]

    arr = np.array(digit_array) 



    #.reshape(a, (28,28))

    image_array = np.reshape(arr, (28,28))



    digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

    plt.colorbar(digit_img)

    print("IMAGE LABEL: {}".format(train.loc[row, "label"]))





visualize_digit(55)    
model = RandomForestClassifier(random_state=1)
model.fit(train_X, train_Y)
test['Label'] = model.predict(test)
test['ImageId']=test.index+1
test.loc[:,['ImageId','Label']].head()
test.loc[:,['ImageId','Label']].to_csv('random_forest_sub.csv', index=False)