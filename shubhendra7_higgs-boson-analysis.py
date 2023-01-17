import numpy as np 

import pandas as pd 

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
higgs = pd.read_csv('../input/training/training.csv')

pp.ProfileReport(higgs)
new_data= higgs[higgs.columns[1:31]]

new_data.head()



for i in range(0, new_data.shape[1]):

    plt.figure()

    sns.distplot(new_data.iloc[:, i], bins=30,color="#9b59b6")

    #plt.savefig("dataname_{0}.png".format(i))

    

plt.show()
## The below command will command 60 plots, 30 each for class s and b

unique_label = higgs['Label'].unique()

for i in unique_label:

    new_data_label = higgs[higgs['Label'] == i]

    new_data_labels= new_data_label[new_data_label.columns[1:31]]

    for j in range(0, new_data_labels.shape[1]):

        sns.distplot(new_data_labels.iloc[:, j], bins=30, color='#9b59b6').set_title('{} for {}'.format(i,j))

                                                                                    

    #plt.savefig("dataname_{0}.png".format(i))

        plt.show()

    
higgs_select=higgs[["DER_sum_pt","PRI_met_sumet","Label"]]

sns.pairplot(higgs_select, hue="Label").fig.suptitle('Without Log Transformation')

plt.show()



x = higgs_select["DER_sum_pt"].apply(np.log)

y = higgs_select["PRI_met_sumet"].apply(np.log)

z= higgs_select["Label"]



d = {'DER_sum_pt': x, 'PRI_met_sumet': y,'Label':z}

higgs_new = pd.DataFrame(d)

sns.pairplot(higgs_new, hue="Label").fig.suptitle('With Log Transformation')

plt.show()