import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax = plt.subplots()
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))
people
y_pos
performance
error
ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()
import pandas as pd
import glob
print(glob.glob("*.csv"))
idr=pd.read_csv('../input/regional-development-index-tunisia/idr_gouv.csv')
idr
idr.head()
idr.describe()
idr['IDR']
idr['gouvernorat']
import numpy as np
gouv=np.asarray(idr['gouvernorat'])
gouv
y_pos=np.arange(len(gouv))
y_pos
idrvar=np.asarray(idr['IDR'])
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(y_pos, idrvar, align='center',
        color='red', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(gouv)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('IDR')
ax.set_title('Regional Development Index (2010)')

plt.show()
decat=pd.read_csv('../input/decathlon/decathlon.csv')
decat.columns
x=np.asarray(decat['100m'])
y=np.asarray(decat['Long.jump'])
plt.scatter(x, y,  alpha=0.5)
plt.xlabel("100m")
plt.ylabel("Long Jump")
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
decat.plot(kind='scatter', x='Discus', y='Long.jump',  s=decat['100m']*10)
plt.show()

from ggplot import *
ggplot(aes(x='Discus', y='Long.jump', color='Competition'), data=decat) +\
    geom_point() +\
    theme_bw() +\
    xlab("Discus") +\
    ylab("Long Jump") +\
    ggtitle("Discus x Long Jump")
