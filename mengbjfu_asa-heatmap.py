import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
heatmap_all = pd.read_csv('../input/asa-heatmap-data/ASA_heatmap.csv', skiprows = 1)
heatmap_all
heatmap_all.head()
heatmap_all.tail()
heatmap_all.shape
ready_use_data_density = np.ndarray(shape= (225, 5))



intervals_density = [0.01, 0.04, 0.07, 0.10,

                     0.13, 0.16, 0.19, 0.22,

                     0.25, 0.28, 0.31, 0.34,

                     0.37, 0.40, 0.43]

intervals_height = np.arange(4, 34, 2)

q = 0



for i in range(len(intervals_density)):

    if intervals_density[i] <= 0.01:

        for j in range(len(intervals_height)):

            if intervals_height[j] <= intervals_height[0]:

                df_SelectedByDensity = heatmap_all.loc[(heatmap_all["den"] <= intervals_density[i])

                                                &(heatmap_all["hei"] <= intervals_height[j])]

                ready_use_data_density[q, 0] = intervals_density[i]

                ready_use_data_density[q, 1] = intervals_height[j]

                ready_use_data_density[q, 2] = df_SelectedByDensity["CV_density"].mean()

                ready_use_data_density[q, 3] = df_SelectedByDensity["CV_height"].mean()

                ready_use_data_density[q, 4] = df_SelectedByDensity["CV_volume"].mean()

                q = q + 1

                

            else:

                df_SelectedByDensity = heatmap_all.loc[(heatmap_all["den"] <= intervals_density[i])

                                                        &(heatmap_all["hei"] <= intervals_height[j])

                                                        &(heatmap_all["hei"] > intervals_height[j-1])]

                ready_use_data_density[q, 0] = intervals_density[i]

                ready_use_data_density[q, 1] = intervals_height[j]

                ready_use_data_density[q, 2] = df_SelectedByDensity["CV_density"].mean()

                ready_use_data_density[q, 3] = df_SelectedByDensity["CV_height"].mean()

                ready_use_data_density[q, 4] = df_SelectedByDensity["CV_volume"].mean()

                q = q + 1         

    else:

        for j in range(len(intervals_height)):

            if intervals_height[j] <= intervals_height[0]:

                df_SelectedByDensity = heatmap_all.loc[(heatmap_all["den"] <= intervals_density[i])

                                                        &(heatmap_all["den"] > intervals_density[i-1])

                                                        &(heatmap_all["hei"] <= intervals_height[j])]



                ready_use_data_density[q, 0] = intervals_density[i]

                ready_use_data_density[q, 1] = intervals_height[j]

                ready_use_data_density[q, 2] = df_SelectedByDensity["CV_density"].mean()

                ready_use_data_density[q, 3] = df_SelectedByDensity["CV_height"].mean()

                ready_use_data_density[q, 4] = df_SelectedByDensity["CV_volume"].mean()

                q = q + 1



            else:

                df_SelectedByDensity = heatmap_all.loc[(heatmap_all["den"] <= intervals_density[i])

                                                        &(heatmap_all["den"] > intervals_density[i-1])

                                                        &(heatmap_all["hei"] <= intervals_height[j])

                                                        &(heatmap_all["hei"] > intervals_height[j-1])]

                ready_use_data_density[q, 0] = intervals_density[i]

                ready_use_data_density[q, 1] = intervals_height[j]

                ready_use_data_density[q, 2] = df_SelectedByDensity["CV_density"].mean()

                ready_use_data_density[q, 3] = df_SelectedByDensity["CV_height"].mean()

                ready_use_data_density[q, 4] = df_SelectedByDensity["CV_volume"].mean()

                q = q + 1
# convert ndarray to dataframe

CV_All = pd.DataFrame(ready_use_data_density, columns=["Density", "Height", "CV_Density", "CV_Height", "CV_Volume"])

CV_All
#pivot

CV_Density = CV_All.pivot("Height", "Density", "CV_Density")

CV_Height = CV_All.pivot("Height", "Density", "CV_Height")

CV_Volume = CV_All.pivot("Height", "Density", "CV_Volume")
CV_Density
#plot figures density



sns.set(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.25, color_codes=True, rc=None)

ax = sns.heatmap(CV_Density,

                 linewidths=0.005, linecolor='#aaaaaa',

                 square=True,

                 cmap="Oranges",

                 vmin=0, vmax=0.0005)

plt.gca().invert_yaxis()

plt.title("CV of Density")
ax = sns.heatmap(CV_Height,

                 linewidths=0.005, linecolor='#aaaaaa',

                 square=True,

                 cmap="Oranges",

                 vmin=0, vmax=0.15)

plt.gca().invert_yaxis()

plt.title("CV of Height")