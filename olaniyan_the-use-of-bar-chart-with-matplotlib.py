# import necessary library

import matplotlib.pyplot as plt

import numpy as np
# Restore the rc params from Matplotlib's internal default style

plt.rcdefaults()
plt.style.use(style='ggplot')
# data plot

n_groups=5

rf = (0.587,1.005,0.531,0.572,0.459)

dt = (0.546,1.151,0.718,0.655,0.635)

# create plot

fig,ax=plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.8



rects1 = plt.bar(index,rf,bar_width,alpha=opacity, color= 'b', edgecolor='w',label='RMSE_Random Forest')

rects2 = plt.bar(index + bar_width, dt,bar_width,alpha=opacity,color='r', edgecolor='w',label='RMSE_Decision Tree')

plt.xlabel('Labels')

plt.ylabel('Metrics')

plt.xticks(index+bar_width,('Ca','P','pH','SOC','Sand'))

plt.legend()

plt.tight_layout()

plt.show()