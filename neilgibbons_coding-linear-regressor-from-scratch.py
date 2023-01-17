class linReg():

  def __init__(self):

    self.b0 = None

    self.b1 = None

  

  def fit(self, xCoords, yCoords):

    self.b1 = self.get_b1(xCoords,yCoords)

    self.b0 = self.get_b0(xCoords,yCoords)



  def get_b0(self, xCoords, yCoords):

    xMean = self.get_mean(xCoords)

    yMean = self.get_mean(yCoords)

    output = yMean - (self.get_b1(xCoords,yCoords)*xMean)

    return output



  def get_b1(self, xCoords, yCoords):

    return self.get_b1_numerator(xCoords,yCoords)/self.get_b1_denominator(xCoords,yCoords)



  def get_b1_numerator(self, xCoords, yCoords):

    xMean = self.get_mean(xCoords)

    yMean = self.get_mean(yCoords)

    output = 0

    for i in range(0,len(xCoords)):

      output += ((xCoords[i]-xMean)*(yCoords[i]-yMean))

    return output



  def get_b1_denominator(self, xCoords, yCoords):

    xMean = self.get_mean(xCoords)

    output = 0

    for i in range(0,len(xCoords)):

      output += ((xCoords[i]-xMean)**2)

    return output



  def get_mean(self, ls):

    return sum(ls)/len(ls)



  def predict(self, X_test):

    output = []

    for item in X_test:

      output.append(self.b0+item*self.b1)

    return output
import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/home-data-for-ml-course/train.csv.gz")
df.columns
#Want to find column which is the strongest predictor of sale price

simple_df = pd.DataFrame(df['LotArea'],columns=['LotArea'])
simple_df = simple_df.join(df['SalePrice'])
fig = plt.figure(figsize=(1,0.3))

axes = fig.add_axes([0, 0, 10, 10])

axes.scatter(simple_df['LotArea'],simple_df['SalePrice'])
reg = linReg()

reg.fit(simple_df['LotArea'],simple_df['SalePrice'])

preds = reg.predict(simple_df['LotArea'])
#plotting the line on the graph

x1, y1 = [0, simple_df['LotArea'].max()], [preds[0],max(preds)]

axes.plot(x1, y1, marker = 'o', color = 'r')
fig
#red line shows the regressor

#making a submission (it won't be anything special)

test = pd.read_csv("../input/home-data-for-ml-course/test.csv.gz")

simple_test = pd.DataFrame(data=test['LotArea'],columns=['LotArea'])

simple_test.head()

final_preds = reg.predict(simple_test['LotArea'])
test['Id']
submission = pd.DataFrame(data=(final_preds),index=test.index,columns=['SalePrice'])
submission = submission.join(test['Id'])
submission
submission.to_csv('submission.csv', index=False)