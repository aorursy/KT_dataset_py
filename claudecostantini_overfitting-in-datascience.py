#implementing the simple dataset
data = [(1,50),(2,60),(3,250),(4,520),(5,400),(6,650),(7,800),(8,1150),(9,1000)]
#preparing data for dataviz
import numpy as np
x= np.array([tup[0] for tup in data])
y= np.array([tup[1] for tup in data])
import matplotlib.pyplot as plt
%matplotlib inline
fig,ax=plt.subplots(figsize=(10, 6))
ax.plot(x,y,'g^')
fig,ax=plt.subplots(figsize=(10, 6))
xlin=np.linspace(0,10, 100)
y4=-xlin**4+10*xlin**3+2*xlin**2
ax.plot(xlin,y4,'-')
ax.plot(x,y,'g^')
fig,ax=plt.subplots(figsize=(10, 6))
xlin=np.linspace(0,10, 100)
y3=-2*xlin**3+30*xlin**2
ax.plot(xlin,y3,'r-')
ax.plot(x,y,'g^')
fig,ax=plt.subplots(figsize=(10, 6))
xlin=np.linspace(0,10, 100)
y2=20*xlin**2-10*xlin
ax.plot(xlin,y2,'g-')
ax.plot(x,y,'g^')
fig,ax=plt.subplots(figsize=(10, 6))
xlin=np.linspace(0,10, 100)
y4=-xlin**4+10*xlin**3+2*xlin**2
y3=-2*xlin**3+30*xlin**2
y2=20*xlin**2-10*xlin
ax.plot(xlin,y4,'-', label='4th degree model')
ax.plot(xlin,y3,'r-', label='3rd degree model')
ax.plot(xlin,y2,'g-', label='2nd degree model')
ax.plot(x,y,'g^', label='dataset')
ax.legend()
data = [(1,50),(2,25),(3,200),(4,130),(5,200),(6,240)]
x= np.array([tup[0] for tup in data])
y= np.array([tup[1] for tup in data])
fig,ax=plt.subplots(figsize=(10, 6))
xlin=np.linspace(0.8,6, 100)
yd5 = 2935 - 6032.417*xlin + 4352.292*xlin**2 - 1401.042*xlin**3 + 207.7083*xlin**4 - 11.54167*xlin**5
ax.plot(xlin,yd5,'r-')
ax.plot(x,y,'gx')
data = [(1,50),(2,25),(3,200),(4,130),(5,200),(6,240)]
x= np.array([tup[0] for tup in data])
y= np.array([tup[1] for tup in data])
fig,ax=plt.subplots(figsize=(10, 6))
xlin=np.linspace(-0.1,7.2, 100)
yd5 = 2935 - 6032.417*xlin + 4352.292*xlin**2 - 1401.042*xlin**3 + 207.7083*xlin**4 - 11.54167*xlin**5
ax.plot(xlin,yd5,'r-')
ax.plot(x,y,'gx')