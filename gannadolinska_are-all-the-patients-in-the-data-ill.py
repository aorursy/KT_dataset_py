# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# In addition, import...

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import os, shutil



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

train.head()
# Calculate mean percent for each client

clientsPercent = train.groupby('Patient')["Percent"].mean()



# Plot the distribution of mean percent

plt.hist(clientsPercent, bins = 60)

plt.show()
# Define a function lfit, which performs linear fitting and writing of the visualization of the fit to a file

# The fit is done per patient

def lfit (x, y, patient):

    # Detecting which variable is fitted: fraction (percentage), or some absolute one.

    # Note: this is a hack which works on this data only.

    titleAddition = ""

    if y.max() <= 100:

        titleAddition = "Percent_"

    

    # Linear firring with scikit learn

    x = np.array(x).reshape(-1,1)

    y = np.array(y).reshape(-1,1)

    lr = LinearRegression()

    lr.fit(x, y)

    

    # Save the fitted plot to a file

    plt.scatter(x, y, marker = 'o')

    xForPlot = np.array(list(range(x.min(), x.max() + 1))).reshape(-1,1)

    plt.plot(xForPlot, lr.coef_ * xForPlot + lr.intercept_, 'r--')

    plt.savefig(f'/kaggle/working/{titleAddition}{patient.unique()}_linearFit.png')

    plt.clf()

    

    return(float(lr.coef_))
# Clean the folder to which the plots are written before running a round of fitting for all patients

folder = '/kaggle/working/'

for filename in os.listdir(folder):

    file_path = os.path.join(folder, filename)

    try:

        if os.path.isfile(file_path) or os.path.islink(file_path):

            os.unlink(file_path)

        elif os.path.isdir(file_path):

            shutil.rmtree(file_path)

    except Exception as e:

        print('Failed to delete %s. Reason: %s' % (file_path, e))



# Fit for every client

train["slopePercent"] = train.groupby('Patient').apply(lambda x : lfit(x['Weeks'], x['Percent'], x['Patient'])).reindex(train.Patient).values;

train["slopeFVC"] = train.groupby('Patient').apply(lambda x : lfit(x['Weeks'], x['FVC'], x['Patient'])).reindex(train.Patient).values;
plt.hist(np.array(train.groupby('Patient')["slopeFVC"].unique().tolist()).ravel(), bins = 60);
plt.hist(np.array(train.groupby('Patient')["slopePercent"].unique().tolist()).ravel(), bins = 60);