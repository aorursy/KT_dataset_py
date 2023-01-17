# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Let's make some necessary import statements and create a few scatterplots between the different EEG frequency bands to quickly 
#guage whether their strengths are correlated to the study's proprietary measures of mental attention and calmness. 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

EEGData = pd.read_csv("../input/confused-eeg/EEG_data.csv")
EEGData.head()
#In this block of code, I will quickly plot the relationships between the eight different EEG Frequency band strengths and the study's "Attention"
#measure. I'll use the artist layer of matplotlib to make eight separate subplots to do so: 
AttentionEEGPlot = plt.figure()

AttentionEEGSubplot1 = AttentionEEGPlot.add_subplot(4,2,1)
AttentionEEGSubplot2 = AttentionEEGPlot.add_subplot(4,2,2)
AttentionEEGSubplot3 = AttentionEEGPlot.add_subplot(4,2,3)
AttentionEEGSubplot4 = AttentionEEGPlot.add_subplot(4,2,4)
AttentionEEGSubplot5 = AttentionEEGPlot.add_subplot(4,2,5)
AttentionEEGSubplot6 = AttentionEEGPlot.add_subplot(4,2,6)
AttentionEEGSubplot7 = AttentionEEGPlot.add_subplot(4,2,7)
AttentionEEGSubplot8 = AttentionEEGPlot.add_subplot(4,2,8)

AttentionEEGSubplot1.scatter(EEGData['Delta'],EEGData['Attention'])
AttentionEEGSubplot2.scatter(EEGData['Theta'],EEGData['Attention'])
AttentionEEGSubplot3.scatter(EEGData['Alpha1'],EEGData['Attention'])
AttentionEEGSubplot4.scatter(EEGData['Alpha2'],EEGData['Attention'])
AttentionEEGSubplot5.scatter(EEGData['Beta1'],EEGData['Attention'])
AttentionEEGSubplot6.scatter(EEGData['Beta2'],EEGData['Attention'])
AttentionEEGSubplot7.scatter(EEGData['Gamma1'],EEGData['Attention'])
AttentionEEGSubplot8.scatter(EEGData['Gamma2'],EEGData['Attention'])

#The below scatterplots aren't very informative, and I would probably need to change the subplot scaling
#and a number of other variables to make them a little more informative. I think the sampling rate (and the sheer amount
#of datapoints resulting from it) might make it a worthwhile to plot seaborn regression plots within this figure instead. 
#Based on the scatterplots outputted, I'm not expecting particularly strong correlations between any single brainwave frequency
#band and either the measure for calmness or attention. 
AttentionEEGRegressionPlots = plt.figure()

AttentionEEGRegSubplot1 = AttentionEEGRegressionPlots.add_subplot(4,2,1)
AttentionEEGRegSubplot2 = AttentionEEGRegressionPlots.add_subplot(4,2,2)
AttentionEEGRegSubplot3 = AttentionEEGRegressionPlots.add_subplot(4,2,3)
AttentionEEGRegSubplot4 = AttentionEEGRegressionPlots.add_subplot(4,2,4)
AttentionEEGRegSubplot5 = AttentionEEGRegressionPlots.add_subplot(4,2,5)
AttentionEEGRegSubplot6 = AttentionEEGRegressionPlots.add_subplot(4,2,6)
AttentionEEGRegSubplot7 = AttentionEEGRegressionPlots.add_subplot(4,2,7)
AttentionEEGRegSubplot8 = AttentionEEGRegressionPlots.add_subplot(4,2,8)

import seaborn as sns

sns.regplot(EEGData['Delta'],EEGData['Attention'],ax=AttentionEEGRegSubplot1,line_kws = {'color': 'red'})
sns.regplot(EEGData['Theta'],EEGData['Attention'],ax=AttentionEEGRegSubplot2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Alpha1'],EEGData['Attention'],ax=AttentionEEGRegSubplot3,line_kws = {'color': 'red'})
sns.regplot(EEGData['Alpha2'],EEGData['Attention'],ax=AttentionEEGRegSubplot4,line_kws = {'color': 'red'})
sns.regplot(EEGData['Beta1'],EEGData['Attention'],ax=AttentionEEGRegSubplot5,line_kws = {'color': 'red'})
sns.regplot(EEGData['Beta2'],EEGData['Attention'],ax=AttentionEEGRegSubplot6,line_kws = {'color': 'red'})
sns.regplot(EEGData['Gamma1'],EEGData['Attention'],ax=AttentionEEGRegSubplot7,line_kws = {'color': 'red'})
sns.regplot(EEGData['Gamma2'],EEGData['Attention'],ax=AttentionEEGRegSubplot8,line_kws = {'color': 'red'})

plt.ylim(0,100)

#***6/18/2020: The next change I think I will make is to create a total of five different plots, three of which consisting of two different
#subplots with the "1" and "2" subranges of the Alpha, Beta, and Gamma EEG frequency ranges. That will hopefully help solve the: i) weird
#y-axis range issue where there are negative values for some reason and ii) help clear out some of the illegibility issues from 
#the small subplots being really clumped together***.

#(Including this cell) The next five code cells are dedicated to creating the new scatterplots of the 
#different EEG Frequency Bands vs. Attention and Mediation/Confusion data: 
AttentionEEGRegressionPlotv2Delta = plt.figure()

#1) Here's the subplot (just one) for the Delta Wave data: 
AttentionEEGRegressionSubplotv2Delta = AttentionEEGRegressionPlotv2Delta.add_subplot(1,1,1)
sns.regplot(EEGData['Delta'],EEGData['Attention'],ax=AttentionEEGRegressionSubplotv2Delta,line_kws = {'color': 'red'})
sns.regplot(EEGData['Delta'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplotv2Delta,line_kws = {'color': 'green'})

#**Finding: It looks like Delta waves are similarly (negatively) correlated with Attention and Calmness/Mediation.** 
#2) Here's the subplot (again, just one) for the Theta Wave Deta: 
AttentionEEGRegressionPlotv2Theta = plt.figure()

AttentionEEGRegressionSubplotv2Theta = AttentionEEGRegressionPlotv2Theta.add_subplot(1,1,1)
sns.regplot(EEGData['Theta'],EEGData['Attention'],ax=AttentionEEGRegressionSubplotv2Theta,line_kws = {'color': 'red'})
sns.regplot(EEGData['Theta'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplotv2Theta,line_kws = {'color': 'green'})
plt.ylim(0,100)

#**Finding: It looks like Theta waves are similarly (negatively) correlated with Attention and Calmness/Mediation.** 
#3) Here are the new subplots for the Alpha Frequency Bands (1 and 2): 
AttentionEEGRegressionPlotv2Alpha1and2 = plt.figure()

AttentionEEGRegressionSubplot1v2Alpha1and2 = AttentionEEGRegressionPlotv2Alpha1and2.add_subplot(1,2,1)
sns.regplot(EEGData['Alpha1'],EEGData['Attention'],ax=AttentionEEGRegressionSubplot1v2Alpha1and2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Alpha1'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplot1v2Alpha1and2,line_kws = {'color': 'green'})
plt.ylim(0,100)
AttentionEEGRegressionSubplot2v2Alpha1and2 = AttentionEEGRegressionPlotv2Alpha1and2.add_subplot(1,2,2)
sns.regplot(EEGData['Alpha2'],EEGData['Attention'],ax=AttentionEEGRegressionSubplot2v2Alpha1and2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Alpha2'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplot2v2Alpha1and2,line_kws = {'color': 'green'})
plt.ylim(0,100)

#**Finding: Note that for Alpha1 EEG Frequencies, there appears to be a much more significant negative correlation between the strength of 
#Alpha1 EEG Activity and the study's proprietary measure of participant attention than calmness. In the case of Alpha2 EEG strength,
#both the study's measures of calmness and attention were similarly (negatively) correlated.** 
#Here are the new subplots for Beta 1 and 2 EEG Strength vs. Attention and Calmness (Mediation)
AttentionEEGRegressionPlotv2Beta1and2 = plt.figure()

AttentionEEGRegressionSubplot1v2Beta1and2 = AttentionEEGRegressionPlotv2Beta1and2.add_subplot(1,2,1)
sns.regplot(EEGData['Beta1'],EEGData['Attention'],ax=AttentionEEGRegressionSubplot1v2Beta1and2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Beta1'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplot1v2Beta1and2,line_kws = {'color': 'green'})
plt.ylim(0,100)
AttentionEEGRegressionSubplot2v2Beta1and2 = AttentionEEGRegressionPlotv2Beta1and2.add_subplot(1,2,2)
sns.regplot(EEGData['Beta2'],EEGData['Attention'],ax=AttentionEEGRegressionSubplot2v2Beta1and2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Beta2'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplot2v2Beta1and2,line_kws = {'color': 'green'})
plt.ylim(0,100)

#**Finding: It looks like Beta 1 and 2 waves are similarly (negatively) correlated with Attention and Calmness/Mediation.** 
#Here are the new subplots for Gamma 1 and 2 EEG Strength vs. Attention and Calmness (Mediation)

AttentionEEGRegressionPlotv2Gamma1and2 = plt.figure()

AttentionEEGRegressionSubplot1v2Gamma1and2 = AttentionEEGRegressionPlotv2Gamma1and2.add_subplot(1,2,1)
sns.regplot(EEGData['Gamma1'],EEGData['Attention'],ax=AttentionEEGRegressionSubplot1v2Gamma1and2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Gamma1'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplot1v2Gamma1and2,line_kws = {'color': 'green'})
plt.ylim(0,100)
AttentionEEGRegressionSubplot2v2Gamma1and2 = AttentionEEGRegressionPlotv2Gamma1and2.add_subplot(1,2,2)
sns.regplot(EEGData['Gamma2'],EEGData['Attention'],ax=AttentionEEGRegressionSubplot2v2Gamma1and2,line_kws = {'color': 'red'})
sns.regplot(EEGData['Gamma2'],EEGData['Mediation'],ax=AttentionEEGRegressionSubplot2v2Gamma1and2,line_kws = {'color': 'green'})
plt.ylim(0,100)

#**Finding: It looks like Gamma 1 and 2 waves are negatively correlated (strongly) with both Attention and Mediation/Calmness.** 

#***Stack Overflow link for adding legends to seaborn scatterplots: https://stackoverflow.com/questions/53876397/add-legend-to-sns-regplot-and-sns-lmplot***
#Let's make individual residuals plots for each of the EEG frequency bands vs. the study's proprietary measures of mediation and then attention:
AttentionEEGResidualsPlotMediation = plt.figure()

AttentionEEGResidualsPlotMediationDelta = AttentionEEGResidualsPlotMediation.add_subplot(4,2,1)
AttentionEEGResidualsPlotMediationTheta = AttentionEEGResidualsPlotMediation.add_subplot(4,2,2)
AttentionEEGResidualsPlotMediationAlpha1 = AttentionEEGResidualsPlotMediation.add_subplot(4,2,3)
AttentionEEGResidualsPlotMediationAlpha2 = AttentionEEGResidualsPlotMediation.add_subplot(4,2,4)
AttentionEEGResidualsPlotMediationBeta1 = AttentionEEGResidualsPlotMediation.add_subplot(4,2,5)
AttentionEEGResidualsPlotMediationBeta2 = AttentionEEGResidualsPlotMediation.add_subplot(4,2,6)
AttentionEEGResidualsPlotMediationGamma1 = AttentionEEGResidualsPlotMediation.add_subplot(4,2,7)
AttentionEEGResidualsPlotMediationGamma2 = AttentionEEGResidualsPlotMediation.add_subplot(4,2,8)

sns.residplot(EEGData['Delta'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationDelta)
sns.residplot(EEGData['Theta'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationTheta)
sns.residplot(EEGData['Alpha1'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationAlpha1)
sns.residplot(EEGData['Alpha2'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationAlpha2)
sns.residplot(EEGData['Beta1'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationBeta1)
sns.residplot(EEGData['Beta2'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationBeta2)
sns.residplot(EEGData['Gamma1'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationGamma1)
sns.residplot(EEGData['Gamma2'],EEGData['Mediation'],ax=AttentionEEGResidualsPlotMediationGamma2)

#It looks like the residuals plots are mainly random (good) but unfortunately large in the case of the Delta and Theta EEG bands.
#In the case of all of the other bands, a large systematic error line appears. It seems a nonlinear model would be more appropriate here.
#Let's try creating a second order multiple polynomial model and construct residuals plots for that model. 
#I'm going to follow the format (using scikit learn) demonstrated on this Stack Overflow link: https://stackoverflow.com/questions/54891965/multivariate-polynomial-regression-with-python
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

EEGDataSecondOrderMPolyModel = PolynomialFeatures(degree=2)
EEGBandVariables = EEGData[['Delta','Theta','Alpha1','Alpha2','Beta1','Beta2','Gamma1','Gamma2']]
poly_variables = EEGDataSecondOrderMPolyModel.fit_transform(EEGBandVariables)

#poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, results, test_size = 0.3, random_state = 4)

#regression = linear_model.LinearRegression()

#model = regression.fit(poly_var_train, res_train)
#score = model.score(poly_var_test, res_test)