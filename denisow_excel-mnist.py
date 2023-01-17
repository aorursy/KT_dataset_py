import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
model = pd.read_csv("../input/kannadamnist-avg/model.csv", header=None)



print("Kannada digits models:")

fig = plt.figure()

for i in range(1, 11):

    number = model.iloc[i-1,0:784].values.reshape(28,28)

    ax = fig.add_subplot(2, 5, i)

    plt.imshow(number, cmap=plt.get_cmap('gray'))
sample_submission = pd.read_csv("../input/output/sample_submission.csv")

sample_submission.to_csv('submission.csv',index=False)