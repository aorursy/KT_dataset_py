# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

plt.plot([1,2,3,1])

plt.ylabel('ylabel')



plt.show()
plt.plot([1,2,3,4,3],[2,4,6,8,2])

plt.title("Plotting X V/s Y plot")

plt.ylabel('Y value')

plt.xlabel('X indices')

plt.show()
plt.plot([2,6,2,1,4,6],[1,4,5,6,2,1],'ro')#red dot -'ro',red square-'rs'

plt.grid()

plt.show()
import numpy as np

t=np.arange(0.,5.,0.4)#0.4 is step size

plt.plot(t,t+2,'g--',label='greendashline(t+2)')

plt.plot(t,t**2,'b^',label='bluetriangle(t**2)')

plt.plot(t+2,2*t,'rs',label='redsquare(2*t)')

plt.plot(t,t**2+4,'yo',label='yellowdot(t**2+4)')

plt.legend()

plt.show()

#ploting multiple line using plot function with linewidth 0.5 and setproperty(setp) for each lines as follow

x1=[2,4,6]

y1=[4,5,6]

x2=[3,4,7,9]

y2=[5,4,8,1]

line=plt.plot(x1,y1,x2,y2)

plt.setp(line[0],linewidth=10,color='y')

plt.setp(line[1],linewidth=4,color='b')

plt.show()
#multiple plotting using figure1 as a work space and subplot denote no of row and column and figure no.

def f(t):

    return np.exp(-t)*np.cos(2*np.pi*t)

t1=np.arange(0.0,5.0,0.01)

t2=np.arange(3.2,9.1,1.1)

plt.figure(1)

plt.subplot(211)#2rows 1 coulmn 1st row figure

plt.plot(t1,f(t1),'b--')

plt.grid()

plt.subplot(212)#2rows 1 coulmn 2nd row figure

plt.plot(f(t2),np.sin(2*np.pi*t2),'r--')



plt.show()

plt.figure(1)

plt.subplot(221)#2rows 1 coulmn 1st figure

plt.plot([1,2,3])

plt.grid()

plt.subplot(222)#2rows 1 coulmn 2nd  figure

plt.plot([4,3,2,1],'r--')

plt.subplot(223)#2rows 1 coulmn 2nd 3rd figure

plt.plot([10,30,20,1],'r^')

plt.grid()

plt.subplot(224)#2rows 1 coulmn 2nd row figure

plt.plot([4,3,5,10.2,9],'rs')



#plt.show()

plt.figure(2)

#2rows 1 coulmn 1st row figure

plt.plot([1,2,3])

#changing property of figure 1

plt.figure(1)

plt.subplot(221)

plt.title('1row 2column 1st figure')



plt.ylabel('just random heading')

plt.show()