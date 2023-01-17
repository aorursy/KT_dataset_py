import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



num_months = 17 # between now and the end of next year



model = pd.DataFrame({"Month":             np.array(range(num_months)),

                      "CompSubmitters":    np.zeros(num_months),

                      "CompKernelAuthors": np.zeros(num_months),

                      "CompPenetration":   np.zeros(num_months)})



# initial parameters

model.loc[0, "CompSubmitters"]    = 8000

model.loc[0, "CompPenetration"]   = 0.12

model.loc[0, "CompKernelAuthors"] = model["CompSubmitters"][0]*model["CompPenetration"][0]



comp_submitters_mom_growth_pct = 7

comp_submitter_kernel_penetration_mom_growth_percent = 3



for month in range(1, num_months):

    model.loc[month, "CompSubmitters"]  = model["CompSubmitters"][month-1]*(1+comp_submitters_mom_growth_pct/100)

    model.loc[month, "CompPenetration"] = model["CompPenetration"][month-1]*(1+comp_submitter_kernel_penetration_mom_growth_percent/100)

    if 3<=month<=5: # privacy shock

        model.loc[month, "CompPenetration"] = month/10

    

    model.loc[month, "CompKernelAuthors"] = model["CompSubmitters"][month]*model["CompPenetration"][month]



plot_data = pd.melt(model, "Month", ["CompSubmitters", "CompKernelAuthors"])



sns.set_style("darkgrid")

sns.factorplot(x="Month", y="value", hue="variable", data=plot_data, size=4, aspect=2)

_=plt.ylim(0)
model["NewDatasets"] = 0

model["TotalDatasets"] = 0



model.loc[0, "NewDatasets"]   = 25

model.loc[0, "TotalDatasets"] = 70

model.loc[0, "DatasetKernelAuthors"] = 414



kernel_authors_per_dataset = 6

new_dataset_mom_growth_pct = 20



for month in range(1, num_months):

    model.loc[month, "NewDatasets"] = int(model["NewDatasets"][month-1]*(1+new_dataset_mom_growth_pct/100))

    model.loc[month, "TotalDatasets"] = model["TotalDatasets"][month-1]+model["NewDatasets"][month]

    model.loc[month, "DatasetKernelAuthors"] = model["TotalDatasets"][month]*kernel_authors_per_dataset

    

    if 5<=month<=7: # privacy shock for enabling private kernels on datasets

        kernel_authors_per_dataset += 2



print(model)