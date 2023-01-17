from random import sample

import matplotlib.pyplot  as plt

import matplotlib.colors as pltc



populations = [1420, 1368, 329, 269, 212, 204] #all populations are in millions.



#generating random colors from the pltc module.

#colors = sample(list(pltc.cnames.values()), len(populations))



#or we could just set the color combinitions

colors = ['#00CED1', '#F4A460', '#808080', '#B8860B', '#FA8072', '#696969']



country = ['China', 'India', 'US', 'Indonesia', 'Brazil', 'Pakistan']



space_slice = [0.05, 0,0,0,0,0] #explode china in the plot



plt.figure(figsize=(6,5))



plt.pie(populations, labels = country, autopct = '1.1f%%', shadow = True, explode = space_slice, colors = colors)



plt.legend(country, loc = (-0.25, 0.7), shadow = True)

plt.show()

